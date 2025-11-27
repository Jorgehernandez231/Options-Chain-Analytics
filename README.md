# SPX Terminal · Options Chain Analytics

Interactive dashboard for analyzing **S&P 500 (SPX) options** using historical snapshots: volatility skew, open interest, gamma/GEX, flows, spreads and term structure.

Built with **Python + Streamlit + Neon PostgreSQL** as an end-to-end analytical system for data ingestion, storage and visualization.

---
## 🌐 Live Dashboard

Access the deployed dashboard here:

👉 https://options-chain-analyticsgit-y9sm2rbd7anavfyy26yk8k.streamlit.app/

---

## 📡 Data Source

All SPX options data is sourced from CBOE’s delayed quotes service:

https://www.cboe.com/delayed_quotes/spx/quote_table

The project downloads option chain data from this source, cleans and normalizes it, and stores it as historical snapshots in a PostgreSQL database for analysis.

---

## 🚀 Features

### 📚 Historical Snapshots
- Load any stored snapshot (`run_ts`)
- Compare two runs (today vs yesterday or any two dates)
- Track how positioning changes over time

### 📈 Volatility Analytics
- IV skew by strike or moneyness (Calls vs Puts)
- Multi-expiry skew overlays
- ATM IV term structure
- 3D volatility surface (moneyness × days to expiry × IV)

### 🧭 Positioning & Flow Analysis
- Open Interest and volume by strike
- Day-over-day OI change (flows)
- Net positioning tilt (Call OI − Put OI)
- Daily summary by expiration (volume, OI, gamma, GEX)

### ⚡ Dealer Gamma & GEX
- Gamma exposure by strike
- Cumulative gamma curve
- Gamma flip zone estimation

### 🧩 Spread Detection (Heuristic)
- Vertical spread detection
- Iron condor patterns

### 💬 Built-in Help
Every tab includes a **"What am I seeing?"** section with interpretation and usage guidance.

---

## 🧱 Architecture

### ETL Pipeline

Script: `spx_to_neon_final.py`

The ETL pipeline performs the following:

- Downloads the SPX options chain from CBOE  
- Normalizes calls/puts into a long format  
- Cleans prices, IV, volume and OI fields  
- Stores a snapshot in PostgreSQL with a timestamp  

Table schema:

spx_chain(
run_ts TIMESTAMPTZ,
expiration_date DATE,
strike NUMERIC,
cp TEXT,
bid NUMERIC,
ask NUMERIC,
last NUMERIC,
volume NUMERIC,
oi NUMERIC,
iv NUMERIC
)


Each execution creates a **complete snapshot** of the option chain.

---

### Database

- Backend: Neon PostgreSQL
- Stores all snapshots
- Enables:
  - Historical comparison
  - Flow analysis
  - Summary metrics

---

### Dashboard

Main application: `dashboard.py`

- Built using Streamlit and Plotly
- Queries database through SQLAlchemy
- Includes:
  - Snapshot selector
  - Expiration filters
  - Strike-range filters
  - Interactive charts

---

### Automation

GitHub Actions workflow:

`.github/workflows/daily_spx.yml`

Runs the ETL pipeline on a schedule to keep the database updated.

---

## 📂 Project Structure

`.
├── dashboard.py
├── spx_to_neon_final.py
├── requirements.txt
├── .streamlit
│ └── secrets.toml
├── .github
│ └── workflows
│ └── daily_spx.yml
└── README.md`

---

## ⚙️ Installation

### Clone Repository

git clone https://github.com/Jorgehernandez231/Options-Chain-Analytics.git
cd Options-Chain-Analytics

---

### Environment Setup

python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

---

### Configuration

Create `.streamlit/secrets.toml`:

DB_USER = "your_user"
DB_PASS = "your_password"
DB_HOST = "your_host"
DB_NAME = "your_db"

Optional ETL connection string:

NEON_URL = "postgresql://user:pass@host/dbname?sslmode=require"

---

### Load Data

python spx_to_neon_final.py

---

### Run Dashboard

streamlit run dashboard.py


---

## 👤 Intended Audience

- Traders & market analysts
- Finance & data science students
- Developers learning financial analytics
- Researchers exploring volatility & market structure

---

## 🧠 Methodology

- Gamma is approximated using Black–Scholes
- Dealer exposure is inferred from open interest
- Flows are snapshot changes, not tick-level trades
- No proprietary market-maker data is used

---

## ⚠️ Disclaimer

This project is for **educational and analytical purposes only**.  
It is not investment advice.

---

## 📘 Glossary

IV — Implied Volatility  
Skew — Volatility by strike  
Gamma — Delta sensitivity  
GEX — Gamma exposure  
OI — Open Interest  
Tilt — Call OI minus Put OI  
Term structure — Volatility by expiration  
Gamma flip — Where net gamma changes sign  

---

## 🙌 Credits

Data sourced from CBOE.  
Inspired by quantitative finance and volatility research communities.
