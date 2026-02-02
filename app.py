# -*- coding: utf-8 -*-
"""
Personalized AI Financial Advisor Chatbot

Streamlit app with RAG over portfolio CSV, Groq LLM, and tools for stock data,
Sharpe ratio, and portfolio search.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import yfinance as yf

# --- Config & Constants ---
load_dotenv()

REQUIRED_CSV_COLUMNS = ["ticker", "shares", "purchase_price"]
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "llama-3.1-70b-versatile"
CHROMA_COLLECTION = "portfolio"
DEFAULT_RISK_FREE_RATE = 0.02


# --- Custom Tools ---
@tool
def get_stock_price(ticker: str) -> str:
    """Get current stock price and key stats. Input: stock ticker symbol (e.g. AAPL, TSLA)."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please provide a valid stock ticker symbol."
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice") or "N/A"
        high_52 = info.get("fiftyTwoWeekHigh", "N/A")
        low_52 = info.get("fiftyTwoWeekLow", "N/A")
        day_high = info.get("dayHigh", "N/A")
        day_low = info.get("dayLow", "N/A")
        if price != "N/A":
            price = f"${price:.2f}"
        if high_52 != "N/A":
            high_52 = f"${high_52:.2f}"
        if low_52 != "N/A":
            low_52 = f"${low_52:.2f}"
        if day_high != "N/A":
            day_high = f"${day_high:.2f}"
        if day_low != "N/A":
            day_low = f"${day_low:.2f}"
        return (
            f"{ticker}: Current price {price}. "
            f"52-week range: {low_52} - {high_52}. "
            f"Day range: {day_low} - {day_high}."
        )
    except Exception as e:
        return f"Could not fetch data for {ticker}. Please check the symbol and try again."


@tool
def get_stock_info(ticker: str) -> str:
    """Get company info: name, sector, market cap, summary. Input: stock ticker (e.g. AAPL)."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please provide a valid stock ticker symbol."
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get("longName") or info.get("shortName", ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        cap = info.get("marketCap")
        cap_str = f"${cap / 1e9:.2f}B" if cap else "N/A"
        summary = (info.get("longBusinessSummary") or "")[:400]
        if summary:
            summary = summary.rstrip() + "..."
        return (
            f"{ticker} - {name}. Sector: {sector}, Industry: {industry}. "
            f"Market cap: {cap_str}. {summary}"
        )
    except Exception:
        return f"Could not fetch company info for {ticker}."


@tool
def calculate_holding_value(ticker: str, shares: float) -> str:
    """Calculate current market value of a position. Input: ticker symbol and number of shares."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please provide a valid ticker."
    try:
        shares = float(shares)
        if shares <= 0:
            return "Shares must be a positive number."
    except (TypeError, ValueError):
        return "Shares must be a valid number."
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("currentPrice") or stock.info.get("regularMarketPrice")
        if price is None:
            hist = stock.history(period="5d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        if price is None:
            return f"Could not get current price for {ticker}."
        value = shares * price
        return f"{shares:.2f} shares of {ticker} @ ${price:.2f} = ${value:,.2f}."
    except Exception:
        return f"Error calculating value for {ticker}."


@tool
def calculate_sharpe_ratio(
    annual_return: float, annual_volatility: float, risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> str:
    """Calculate Sharpe ratio (risk-adjusted return). Inputs: annual_return (e.g. 0.15 for 15%), annual_volatility (e.g. 0.20 for 20%). Optional: risk_free_rate (default 2%)."""
    try:
        ret = float(annual_return)
        vol = float(annual_volatility)
        rf = float(risk_free_rate) if risk_free_rate is not None else DEFAULT_RISK_FREE_RATE
    except (TypeError, ValueError):
        return "Please provide numbers for annual_return and annual_volatility (e.g. 0.15, 0.20)."
    if vol <= 0:
        return "Volatility must be positive."
    sharpe = (ret - rf) / vol
    return (
        f"Sharpe ratio: {sharpe:.2f}. "
        f"(Return {ret*100:.1f}%, Volatility {vol*100:.1f}%, Risk-free {rf*100:.1f}%. Higher is better.)"
    )


# --- Agent Prompt ---
AGENT_PROMPT_TEMPLATE = """You are a helpful, honest financial advisor. Use the tools when needed.
- Answer from the user's portfolio when they upload one (use retrieve_portfolio).
- Use get_stock_price, get_stock_info, calculate_holding_value for market data.
- Use calculate_sharpe_ratio for risk-adjusted return questions.
- Be clear when something is an estimate or requires real data you don't have.
- Never give specific tax or legal advice; suggest consulting a professional when relevant.

Question: {input}
Thought: {agent_scratchpad}"""
agent_prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)


# --- Helpers ---
def validate_portfolio_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """Check that DataFrame has required columns. Returns (ok, error_message)."""
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        return False, f"CSV must have columns: {', '.join(REQUIRED_CSV_COLUMNS)}. Missing: {', '.join(missing)}."
    if df.empty:
        return False, "CSV has no rows."
    for col in REQUIRED_CSV_COLUMNS:
        if df[col].isna().all():
            return False, f"Column '{col}' has no values."
    try:
        pd.to_numeric(df["shares"], errors="raise")
        pd.to_numeric(df["purchase_price"], errors="raise")
    except (TypeError, ValueError) as e:
        return False, f"Columns 'shares' and 'purchase_price' must be numbers. {e}"
    return True, ""


def build_portfolio_docs(df: pd.DataFrame) -> list[Document]:
    """Build RAG documents from portfolio DataFrame."""
    docs = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        shares = row["shares"]
        purchase_price = row["purchase_price"]
        cost_basis = float(shares) * float(purchase_price)
        text = (
            f"Ticker: {ticker}, Shares: {shares}, Purchase Price: ${purchase_price:.2f}, "
            f"Cost Basis: ${cost_basis:,.2f}"
        )
        docs.append(Document(page_content=text, metadata={"source": "portfolio.csv", "ticker": ticker}))
    return docs


# --- Streamlit UI ---
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Personalized AI Financial Advisor")
st.caption("Upload your portfolio CSV, then ask about holdings, prices, and risk.")

# Sidebar
with st.sidebar:
    st.header("Portfolio")
    uploaded_file = st.file_uploader("Upload CSV", type="csv", help="Columns: ticker, shares, purchase_price")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            ok, err = validate_portfolio_csv(df)
            if not ok:
                st.error(err)
            else:
                st.success("Portfolio loaded.")
                st.dataframe(df.head(), use_container_width=True)
                total_cost = (df["shares"].astype(float) * df["purchase_price"].astype(float)).sum()
                st.metric("Positions", len(df))
                st.metric("Total cost basis", f"${total_cost:,.2f}")

                docs = build_portfolio_docs(df)
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                if "vectorstore" in st.session_state:
                    del st.session_state["vectorstore"]
                st.session_state["vectorstore"] = Chroma.from_documents(
                    docs, embeddings, collection_name=CHROMA_COLLECTION
                )
                if "agent_executor" in st.session_state:
                    del st.session_state["agent_executor"]

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        if "agent_executor" in st.session_state:
            del st.session_state["agent_executor"]
        st.rerun()

    with st.expander("CSV format"):
        st.code("ticker,shares,purchase_price\nAAPL,10,150.50\nTSLA,5,200.00", language="csv")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set. Add it to your .env file.")
        st.stop()
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.7, api_key=api_key)
    tools = [get_stock_price, get_stock_info, calculate_holding_value, calculate_sharpe_ratio]

    @tool
    def retrieve_portfolio(query: str) -> str:
        """Search the user's uploaded portfolio (holdings, tickers, shares, cost). Use this when they ask about their portfolio or holdings."""
        if "vectorstore" not in st.session_state:
            return "No portfolio has been uploaded yet. Ask the user to upload a CSV with ticker, shares, purchase_price."
        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 5})
        results = retriever.invoke(query)
        return "\n".join([doc.page_content for doc in results]) if results else "No matching portfolio data."

    tools.append(retrieve_portfolio)
    agent = create_react_agent(llm, tools, agent_prompt)
    memory = ConversationBufferMemory(memory_key="chat_history")
    st.session_state["agent_executor"] = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

# Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about your portfolio or the market (e.g. What's my AAPL worth?)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state["agent_executor"].invoke({"input": user_input})
                out = response.get("output", "I couldn't generate a response.")
                st.markdown(out)
                st.session_state.messages.append({"role": "assistant", "content": out})
            except Exception as e:
                err_msg = "Something went wrong. Please try again or rephrase your question."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
