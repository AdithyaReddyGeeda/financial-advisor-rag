# Personalized AI Financial Advisor

A Streamlit chatbot that uses an AI agent + RAG to answer questions about your portfolio and the market. Upload a portfolio CSV, then ask about holdings, prices, and risk.

## What it does

- **RAG**: Indexes your portfolio CSV so the AI can answer from your actual holdings.
- **Tools**: Fetches stock prices (yfinance), company info, calculates holding values and Sharpe ratio.
- **Agent**: Uses Groq (Llama) to decide when to call tools and how to answer.

## Setup

1. **Clone and enter the project**
   ```bash
   cd financial-advisor-rag
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   # or:  .venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Groq API key**
   - Get a key at [console.groq.com](https://console.groq.com/).
   - Create a `.env` file in the project root with:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

## Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. Upload a CSV, then ask questions (e.g. “What’s the price of Tesla?”, “What’s in my portfolio?”).

## Portfolio CSV format

Your CSV must have these columns:

| Column          | Description     |
|-----------------|-----------------|
| `ticker`        | Stock symbol    |
| `shares`        | Number of shares |
| `purchase_price`| Price per share |

Example:

```csv
ticker,shares,purchase_price
AAPL,10,150.50
TSLA,5,200.00
```

## Troubleshooting

| Issue | What to do |
|-------|------------|
| “GROQ_API_KEY not set” | Add `GROQ_API_KEY=...` to `.env` and restart the app. |
| “CSV must have columns…” | Use exactly `ticker`, `shares`, `purchase_price`. |
| First run is slow | Embedding model and dependencies download on first use. |
| Import or runtime errors | Use Python 3.11 or 3.12 if you hit compatibility issues. |

## Disclaimer

For educational use only. Not financial, tax, or legal advice.
