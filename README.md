
# Personalized Financial Advisor Chatbot with RAG

## What is This?
A simple AI chatbot for personal finance. Upload your portfolio CSV, ask questions, get smart answers.


## Key Concepts
- **AI Agent**: Like a robot that decides steps (e.g., "Search data → Get price → Calculate").
- **RAG**: Makes AI accurate by searching your file first.
- **Tools**: Extra powers like fetching stock prices.

## How to Run Locally 
1. Open Anaconda Prompt/Terminal.
2. Navigate to folder: `cd path/to/financial-advisor-rag` (replace path).
3. Install libraries: `pip install -r requirements.txt` (takes 5–10 min first time).
4. Run app: `streamlit run app.py`
5. Browser opens http://localhost:8501 — upload sample CSV, chat!

## Troubleshooting
- Error "No Groq key"? Check .env.
- Slow? First run downloads models — wait.
- Upload fails? CSV must have columns: ticker,shares,purchase_price.
- Crashes? Restart terminal, rerun pip.


## Extensions (For Fun/Learning)
- Add news sentiment: New tool with web search.
- Better Sharpe: Fetch real historical data from yfinance.
- PDFs: Use unstructured to handle bank statements.

Disclaimer: Educational only. Not financial advice.
