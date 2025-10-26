# 🧠 Market Wars Simulator
### Wealth-Generating Strategy Arena  
**Simulate, Trade, Compete, Evolve.**

---

## 📘 Overview

**Market Wars Simulator** is a **synthetic financial market engine** where multiple trading strategies ("agents") compete under **realistic conditions** — regime shifts, shocks, slippage, and trading costs.  
It provides a *battlefield* to test, compare, and evolve wealth-generation algorithms.

The project currently covers **Stage 1–2** of the full simulator roadmap:
- Stage 1 → Market simulation + trading engine  
- Stage 2 → Four functional strategy agents + comparative analytics  

Upcoming stages will introduce adaptive optimization (Genetic Algorithms) and a Streamlit dashboard.

---

## 🧩 Core Concepts

| Component | Purpose |
|------------|----------|
| **Market Generator** | Creates price paths using a 3-state regime model (Bull, Bear, Flat) with Poisson shock events |
| **Execution Engine** | Simulates realistic trade fills (spreads, slippage, commissions, leverage) |
| **Agents (Strategies)** | Competing trading algorithms that decide target portfolio weights each timestep |
| **Portfolio & Broker** | Manage cash, positions, P&L, and equity over time |
| **Metrics Engine** | Evaluates CAGR, Sharpe, Volatility, Max Drawdown, Calmar, Hit Rate |
| **Visualization Layer** | Plots price + regimes, trade markers, equity curves, and league tables |

---

## 🏗️ Repository Structure
```
market-wars-simulator/
├── market/
│   ├── generator.py          # Regime-based price path generator
│   └── visualize.py           # Price + regime visualization helpers
├── engine/
│   ├── broker.py              # Order execution: spreads, slippage, commissions
│   └── portfolio.py           # Cash, positions, P&L, margin logic
├── agents/
│   ├── base.py                # Abstract agent interface
│   ├── buyhold.py             # Buy & Hold baseline
│   ├── momentum.py            # Momentum (MA crossover) strategy
│   ├── meanrev.py             # Mean Reversion (z-score) strategy
│   └── voltarget.py           # Volatility Targeting (EWMA) strategy
├── eval/
│   └── metrics.py             # Metrics utilities (used inside demos)
├── opt/
│   └── ga.py                  # (Stage 3) Genetic Algorithm tuner [placeholder]
├── demo_run_momentum.py       # Basic momentum test
├── demo_analyze_momentum.py   # Trade markers + performance report
├── demo_compare_agents.py     # Multi-agent comparison + league table
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone and setup
```bash
git clone https://github.com/<yourname>/market-wars-simulator.git
cd market-wars-simulator
python -m venv .venv
source .venv/bin/activate       # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `streamlit`
- `scikit-learn`   *(optional for later GA utilities)*

---

## 🎮 Quick Start

### 1️⃣ Run a single-agent momentum simulation
```bash
python demo_run_momentum.py
```

Generates a 2-year synthetic market, executes a Momentum agent, and plots price + equity + drawdown curves.

---

### 2️⃣ Visualize trade markers and performance report
```bash
python demo_analyze_momentum.py
```

**Shows:**
- 🟩 Price + regimes + shocks
- 🟢 Buy / 🔴 Sell trade markers
- 💰 Equity & Drawdown
- 🧾 Printed performance metrics (CAGR, Sharpe, MaxDD, etc.)

**Output example:**
```
Trades:       165
CAGR  : -0.0755
Sharpe: -0.5432
MaxDD : -19.84%
HitRate: 44.24%
```

---

### 3️⃣ Compare multiple strategies (Stage 2)
```bash
python demo_compare_agents.py
```

Plots all agent equity curves on the same market and prints a league table:

| Agent      | CAGR   | AnnVol | Sharpe | MaxDD   | Calmar | HitRate | Trades |
|------------|--------|--------|--------|---------|--------|---------|--------|
| Momentum   | 0.0367 | 0.1174 | 0.3660 | -18.4%  | 0.20   | 51.7%   | 298    |
| BuyHold    | ...    | ...    | ...    | ...     | ...    | ...     | ...    |
| MeanRev    | ...    | ...    | ...    | ...     | ...    | ...     | ...    |
| VolTarget  | ...    | ...    | ...    | ...     | ...    | ...     | ...    |

**CSV outputs:**
- `compare_league.csv`
- `compare_equity_curves.csv`

---

## 📈 Sample Visualization

*(Insert your screenshots here once generated)*

- 🟩 **Market with Regimes & Shocks**
- 📊 **Agent Equity Curves**
- 💰 **Drawdown Plot**

---

## 📊 Stage 2 — Achievements

| Area                  | Highlights                                                                 |
|-----------------------|----------------------------------------------------------------------------|
| **Market Engine**     | Markov-chain regime model (Bull, Bear, Flat) + stochastic shocks           |
| **Broker / Portfolio**| Spread, slippage, commission, leverage, margin accounting                  |
| **Agents Implemented**| Buy & Hold, Momentum, Mean Reversion, Vol Targeting                        |
| **Metrics**           | CAGR, Volatility, Sharpe, MaxDD, Calmar, Hit Rate                          |
| **Visualization**     | Regime bands, trade markers, equity curves, drawdowns                      |
| **Comparative Analysis** | League table across strategies with CSV export                          |

---

## 💡 Key Insights

| Regime            | Best Strategy              | Notes                                  |
|-------------------|----------------------------|----------------------------------------|
| Calm Bull         | Momentum / Buy & Hold      | Long trends, low noise                 |
| Choppy Sideways   | Mean Reversion             | Frequent reversals, short-term alpha   |
| Crisis            | Vol Targeting              | Survives by cutting exposure           |
| All               | —                          | No single strategy wins everywhere     |

**Lesson:** Robust wealth generation comes from adaptive multi-strategy systems, not single-style trading.

---

## 🧠 Quant Takeaways

- **Regime modeling** introduces realism (trend persistence, volatility clustering).
- **Slippage & spread** transform toy models into plausible simulations.
- **Sharpe and Calmar** reveal efficiency, not just profit.
- **Combining uncorrelated agents** = smoother equity curve.

---

## 🧭 Next Stage — Roadmap

| Stage | Focus | Key Files |
|-------|-------|-----------|
| **3** | **Optimization & Adaptation** — Genetic Algorithm for auto-tuning strategy parameters (maximize Sharpe, penalize DD & turnover) | `opt/ga.py`, `demo_optimize_agents.py` |
| **4** | **Interactive Dashboard** — Streamlit app with controls, scenario presets, league table, downloadable reports | `app.py` |
| **5** | **Packaging & Demo** — Final polish, screenshots, and pitch materials | `README.md`, demo script |

---

## 🏁 Summary

You now have a fully functional quantitative simulation lab where:

- **Markets evolve realistically**,
- **Agents trade with real-world frictions**,
- **Performance is measured with institutional metrics**,
- **Strategies can be compared objectively**.

**Next up** → teach them to adapt and evolve.

Stage 3 introduces a Genetic Algorithm tuner that learns optimal parameters per regime, demonstrating "intelligent wealth generation".

---

## 📄 License

MIT License © 2025 Your Name

---

## 👤 Author

**Your Name**  
Quant Developer | Strategy Simulation Researcher  
*[LinkedIn / GitHub / Email (optional)]*

---

## 🏷️ Keywords

<<<<<<< HEAD
`finance` `quant` `trading-simulator` `machine-learning` `genetic-algorithm` `streamlit` `wealth-generation`
=======
`finance` `quant` `trading-simulator` `machine-learning` `genetic-algorithm` `streamlit` `wealth-generation`
>>>>>>> 8a9cefd (Adding updated files)
