# ORB Trading AI Agent 🤖📈

AI agent built with **phidata** and **Groq (LLaMA 3.3)** that analyzes stocks using Opening Range Breakout (ORB) strategy, gap fill probabilities, ML pattern recognition, and generates complete day trading plans — all through a modern FastAPI + HTMX web interface.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![phidata](https://img.shields.io/badge/phidata-2.7-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-green)
![FastAPI](https://img.shields.io/badge/FastAPI-HTMX-red)

---

## What it does

Ask anything about a stock in natural language and the agent will:

- Fetch **real market data** from Yahoo Finance
- Analyze **gap fill probabilities** by historical size buckets
- Compute **ORB breakout statistics** for 5m, 15m, and 30m windows
- Run **ML regime detection** (KMeans + Random Forest via `ml_engine.py`)
- Query the **pattern learning database** (`trading_db.py`) for historical win rates
- Generate a **complete trading plan** with entry, target, stop, and R:R levels
- Fetch **latest news headlines**
- **Compare two tickers** side by side

---

## Architecture

```
├── agent.py          ← phidata agent + 8 tools
├── server.py         ← FastAPI + HTMX web UI
├── ml_engine.py      ← ML Pattern Engine (KMeans, Random Forest, LOF)
├── trading_db.py     ← Pattern learning database
└── .env              ← API keys (never commit this)
```

### Tools

| Tool | Description |
|------|-------------|
| `analizar_ticker` | Price, trend, EMAs 20/50/200, ATR, 60d range |
| `analisis_gap` | Historical fill rates by gap size (tiny → huge) |
| `analisis_orb` | Breakout % and avg extension for 5m/15m/30m |
| `prediccion_ml` | KMeans regime + RF/GB bias prediction |
| `stats_base_datos` | Historical CALL/PUT win rates from trading_db |
| `plan_del_dia` | Full day plan combining all data sources |
| `noticias` | Yahoo Finance RSS headlines |
| `comparar_tickers` | Side-by-side comparison of two symbols |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/jumaster23/Trading-AI-Agent
cd Trading-AI-Agent
```

### 2. Install dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install phidata groq openai yfinance fastapi uvicorn python-dotenv scikit-learn pandas numpy
```

### 3. Create `.env` file

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com)

### 4. Run

```bash
uv run python server.py
```

Open **http://localhost:7860**

---

## Usage

### Web UI
```bash
uv run python server.py
```

### Terminal chat
```bash
uv run python agent.py
```

### Quick commands
```bash
uv run python agent.py --ticker SPY --plan   # full trading plan
uv run python agent.py --ticker QQQ --ml     # ML analysis
uv run python agent.py --ticker AAPL --news  # market news
```

---

## Example

```
You: Dame un plan para SPY hoy

Agent: SPY — Trading Plan
  Gap: +0.23% (small, bullish) — 67% historical fill rate
  ORB 15m bias: breaks UP 58% of the time
  Key levels: $578.50 (prev close) · $580.20 (ORB high) · $575.10 (support)
  Setup: Long above $580.20 → target $583.10 · stop $579.30 · R:R 2.4x
  ML Regime: UPTREND · Bias 62% bullish
```

---

## Tech Stack

- **[phidata](https://docs.phidata.com)** — Agent framework + tool calling
- **[Groq](https://groq.com)** — LLaMA 3.3 70B inference (free tier)
- **[FastAPI](https://fastapi.tiangolo.com)** — Backend server
- **[HTMX](https://htmx.org)** — Frontend without JavaScript frameworks
- **[yfinance](https://github.com/ranaroussi/yfinance)** — Market data
- **scikit-learn** — KMeans, Random Forest, Local Outlier Factor

---

## Disclaimer

This project is for educational purposes only. Past statistics do not guarantee future results. Not financial advice.