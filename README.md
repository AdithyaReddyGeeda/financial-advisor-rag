# financial-advisor-rag

# Personalized Financial Advisor Chatbot with RAG

## What is This?
A simple AI chatbot for personal finance. Upload your portfolio CSV, ask questions, get smart answers.


## Key Concepts
- **AI Agent**: Like a robot that decides steps (e.g., "Search data → Get price → Calculate").
- **RAG**: Makes AI accurate by searching your file first.
- **Tools**: Extra powers like fetching stock prices.

## How to Run Locally (On Your Laptop)
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

## Deployment (Make It Online for Free)
1. Sign up GitHub (free).
2. Create repo, upload all files.
3. Go to https://huggingface.co/spaces → New Space → Streamlit template.
4. Link GitHub repo.
5. Add secret: GROQ_API_KEY in HF settings.
6. App lives at https://huggingface.co/spaces/yourname/financial-advisor-rag — share link!

## Extensions (For Fun/Learning)
- Add news sentiment: New tool with web search.
- Better Sharpe: Fetch real historical data from yfinance.
- PDFs: Use unstructured to handle bank statements.

Disclaimer: Educational only. Not financial advice.
