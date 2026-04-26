# Options Terminal • SPX / NDX / VIX

Interactive options analytics dashboard built with **Python, Streamlit, Plotly, PostgreSQL/Neon, and GitHub Actions**.

The app tracks and visualizes options-chain data for:

- **SPX** — S&P 500 Index Options
- **NDX** — Nasdaq-100 Index Options
- **VIX** — Cboe Volatility Index Options

It includes volatility skew, open interest, volume, gamma exposure, delta analytics, term structure, historical flows, probable levels, and rule-based market summaries.

---

## Live App

[Open the Streamlit App](https://options-chain-analyticsgit-y9sm2rbd7anavfyy26yk8k.streamlit.app/)

---

## Screenshots

### Dashboard Overview

![Dashboard Overview](screenshots/01_dashboard_overview.png)

### Gamma Exposure

![Gamma Exposure](screenshots/02_gamma_exposure.png)

### Delta Analytics

![Delta Analytics](screenshots/03_delta_analytics.png)

### Multi-Expiry Skew Overlay

![Multi-Expiry Skew Overlay](screenshots/04_skew_overlay.png)

### VIX Probable Levels

![VIX Probable Levels](screenshots/05_vix_probable_levels.png)

---

## Main Features

### Multi-Symbol Options Dashboard

The dashboard supports:

- SPX
- NDX
- VIX

Each product can be selected directly from the sidebar.

---

### Historical Snapshots

Options-chain data is stored in a PostgreSQL/Neon database using a multi-symbol table:

```sql
options_chain
```

Each snapshot is identified by:

```sql
symbol + run_ts
```

This allows the dashboard to compare current and previous snapshots.

---

### Core Analytics

The dashboard includes:

- Quick Market Read
- Delta Analytics
- IV Skew
- Open Interest & Volume
- Dealer Gamma Exposure approximation
- Term Structure
- Raw Contracts Table
- Multi-Expiry Skew Overlay
- OI Change / Flows
- Positioning Tilt
- Spread Detector
- Skew: Current vs Previous Snapshot
- 3D Volatility Surface
- Summary by Expiration
- Probable Levels

---

## Quick Market Read

The **Quick Market Read** tab gives a rule-based overview of the selected symbol.

It summarizes:

- Gamma regime
- ATM IV
- Days to expiration
- 1σ expected move
- Total gamma exposure
- Flow bias
- Largest OI strikes
- Largest volume strikes
- Biggest OI changes

This section is deterministic and rule-based. It is not an AI-generated trading signal.

---

## Delta Analytics

The **Delta Analytics** tab uses Cboe-provided delta values to analyze the chain by sensitivity instead of only strike.

It includes:

- 25-delta call IV
- 25-delta put IV
- 25-delta risk reversal
- IV by signed delta
- OI by delta bucket
- Volume by delta bucket
- Delta bucket summary

This is useful for understanding skew, wing pricing, and where positioning is concentrated by option sensitivity.

---

## Gamma Exposure

The dashboard uses stored Cboe-provided gamma when available.

If stored gamma is missing, it falls back to a Black-Scholes gamma approximation.

Approximate GEX formula:

```text
GEX = - gamma × open_interest × contract_multiplier × spot²
```

Important: this is an approximation and should not be interpreted as observed dealer positioning.

---

## Database Schema

The main table is:

```sql
options_chain
```

Recommended schema:

```sql
CREATE TABLE IF NOT EXISTS options_chain (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    run_ts TIMESTAMPTZ NOT NULL,
    underlying_px DOUBLE PRECISION,
    expiration_date DATE NOT NULL,
    strike DOUBLE PRECISION NOT NULL,
    cp TEXT NOT NULL,
    last DOUBLE PRECISION,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    oi DOUBLE PRECISION,
    iv DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION,
    option_net DOUBLE PRECISION
);
```

Recommended indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_options_chain_symbol_run
ON options_chain(symbol, run_ts DESC);

CREATE INDEX IF NOT EXISTS idx_options_chain_symbol_exp_strike
ON options_chain(symbol, expiration_date, strike, cp);

CREATE UNIQUE INDEX IF NOT EXISTS uq_options_chain_snapshot_contract
ON options_chain(symbol, run_ts, expiration_date, strike, cp);
```

---

## ETL Pipeline

The ETL script:

```text
spx_to_neon_final.py
```

Fetches delayed options-chain data from Cboe for:

- SPX
- NDX
- VIX

It stores the data in Neon/PostgreSQL with:

- symbol
- run timestamp
- underlying price
- expiration date
- strike
- call/put side
- bid/ask/last
- volume
- open interest
- implied volatility
- delta
- gamma

The ETL uses:

```python
SYMBOLS_TO_RUN = ["SPX", "NDX", "VIX"]
```

The ETL also includes data-quality checks and deduplication before inserting into the database.

---

## GitHub Actions Automation

The project includes a scheduled GitHub Actions workflow:

```text
.github/workflows/daily_options_etl.yml
```

The workflow:

1. Checks out the repository
2. Sets up Python
3. Installs dependencies
4. Runs the ETL script
5. Inserts the latest SPX, NDX and VIX snapshots into Neon

The workflow can also be triggered manually from the GitHub Actions tab.

---

## Environment Variables / Secrets

### GitHub Actions

Add this repository secret:

```text
NEON_URL
```

or map your existing secret name to the environment variable:

```yaml
env:
  NEON_URL: ${{ secrets.N_URL }}
```

Example format:

```text
postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
```

---

### Streamlit Secrets

The dashboard expects these Streamlit secrets:

```toml
DB_USER = "your_user"
DB_PASS = "your_password"
DB_HOST = "your_host"
DB_NAME = "your_database"
```

The ETL can also use:

```toml
NEON_URL = "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require"
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Jorgehernandez231/Options-Chain-Analytics.git
cd Options-Chain-Analytics
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Git Bash

```bash
source venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
python -m streamlit run dashboard.py
```

---

## Requirements

Main libraries:

```text
streamlit
pandas
numpy
plotly
SQLAlchemy
psycopg2-binary
statsmodels
requests
```

---

## Project Structure

```text
Options-Chain-Analytics/
│
├── dashboard.py
├── spx_to_neon_final.py
├── requirements.txt
├── README.md
│
├── screenshots/
│   ├── 01_dashboard_overview.png
│   ├── 02_gamma_exposure.png
│   ├── 03_delta_analytics.png
│   ├── 04_skew_overlay.png
│   └── 05_vix_probable_levels.png
│
└── .github/
    └── workflows/
        └── daily_options_etl.yml
```

---

## Data Source

Options data is fetched from Cboe delayed quote resources.

Quote table pages:

- SPX: https://www.cboe.com/delayed_quotes/spx/quote_table
- NDX: https://www.cboe.com/delayed_quotes/ndx/quote_table
- VIX: https://www.cboe.com/delayed_quotes/vix/quote_table

---

## Interpretation Notes

This dashboard is designed for options analytics and educational/research purposes.

Important limitations:

- Gamma exposure is approximate.
- Stored gamma depends on the data provider.
- Open interest updates are not necessarily real-time.
- VIX options should be interpreted differently from SPX/NDX equity-index options.
- Probable levels assume simplified volatility-based calculations.

The dashboard does not provide financial advice or trading recommendations.

---

## Author

**Jorge del Cristo Hernández García**

Junior Data Analyst focused on financial markets, options analytics, Python, SQL, Streamlit and data visualization.

GitHub: [Jorgehernandez231](https://github.com/Jorgehernandez231)
