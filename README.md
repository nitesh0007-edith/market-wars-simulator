# ⚔️ Market Wars: Strategy Arena & Wealth Generator

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-FF4B4B.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Dashboard]([https://img.shields.io/badge/status-active-success.svg](https://github.com/Dharundp6/GUTS))

**A full-stack trading strategy simulator with market regimes, risk controls, transaction costs, and a live Genetic Algorithm optimizer.**

[Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-demo-cli) • [Dashboard](#️-streamlit-dashboard) • [Documentation](#-project-structure)

</div>

---

## 🌍 Overview

**Market Wars Simulator** is a comprehensive platform for exploring how different trading strategies perform in realistic, regime-driven markets. It simulates **price dynamics**, **broker execution**, and **portfolio accounting**, while evaluating each agent on **risk-adjusted returns**.

A powerful **Genetic Algorithm (GA)** evolves strategy parameters — either globally or per-regime — and you can visualize everything through an intuitive **Streamlit dashboard**.

### 🎯 What Makes This Special?

- 🏦 **Realistic Market Microstructure** — Hidden Markov regimes, transaction costs, slippage
- 🧠 **Intelligent Agents** — Momentum, mean-reversion, volatility targeting, and buy-hold strategies
- 🧬 **Evolutionary Optimization** — GA-powered parameter tuning with regime awareness
- 📊 **Professional Analytics** — Sharpe ratio, Calmar ratio, drawdown analysis, and more
- 🎨 **Beautiful Visualizations** — Interactive charts with regime coloring and trade markers

---

## 🧩 Key Features

### 🏦 Market & Regimes

- Hidden-Markov chain of **Bull / Bear / Flat** states
- Regime-specific drift (`μ`) and volatility (`σ`)
- **Shock events** via Poisson jumps (crashes or gaps)
- Adjustable transition matrix & shock probability

### ⚙️ Execution Engine & Costs

- Full **broker layer**: commission, bid/ask spread, slippage
- Portfolio with **cash, positions, equity, leverage & drawdown stops**
- Realistic **PnL accounting** per trade

### 🧠 Trading Agents

| Agent | Core Logic | Style |
|:------|:-----------|:------|
| **BuyHold** | Passive hold at fixed weight | Baseline benchmark |
| **Momentum** | Moving-average cross or breakout | Trend-following |
| **MeanRev** | z-score mean reversion with stops | Contrarian |
| **VolTarget** | Target volatility using EWMA scaling | Risk control |

### 📊 Evaluation Metrics

- **CAGR** — Compound Annual Growth Rate
- **Annualized Volatility** — Risk measurement
- **Sharpe Ratio** — Risk-adjusted returns
- **Max Drawdown** — Largest peak-to-trough decline
- **Calmar Ratio** — Return vs drawdown
- **Hit Rate** — Win percentage
- **Turnover** — Trading frequency

### 🧬 Genetic Algorithm

- Population-based search for optimal parameters
- **Global optimization** across full market history
- **Per-regime optimization** — adaptive parameters that change with market conditions
- Live progress bar & fitness chart in Streamlit
- JSON export of tuned parameters and fitness history

### 🔄 Regime-Aware Adaptive Simulation

- **"Apply per-regime genes live"** button dynamically switches agent parameters as regimes change
- Observe adaptive equity curve and risk metrics in real-time
- Compare static vs adaptive strategy performance

---

## 🖥️ Streamlit Dashboard

**Launch the dashboard:**

```bash
streamlit run app.py
```

### 🎛️ Dashboard Features

**📌 Sidebar Controls:**
- Choose market preset (Calm Bull, Choppy Sideways, Crisis)
- Adjust broker costs, agent selection, GA settings
- Configure simulation parameters

**🔘 Action Buttons:**
- ▶️ **Run Simulation** — Execute full market simulation
- 🧬 **Run GA** — Launch genetic algorithm optimization
- ⚙️ **Apply per-regime genes** — Enable adaptive trading live

**📈 Main Panels:**
- **Price Chart** — Regime-colored with shock event markers
- **Agent Equity Curves** — Multi-strategy comparison
- **League Table** — Sortable metrics for all agents
- **Download Results** — Export CSV/JSON data
- **GA Fitness History** — Evolution progress plots

---

## 🧱 Project Structure

```
market-wars-simulator/
├── market/
│   ├── generator.py          # Regime-based market simulator
│   └── visualize.py          # Plot helpers (regime colors, shocks)
├── engine/
│   ├── broker.py             # Costs: spread, commission, slippage
│   └── portfolio.py          # Portfolio accounting (cash, equity)
├── agents/
│   ├── base.py               # Base agent interface
│   ├── buyhold.py            # Buy and hold strategy
│   ├── momentum.py           # Trend following strategy
│   ├── meanrev.py            # Mean reversion strategy
│   └── voltarget.py          # Volatility targeting strategy
├── eval/
│   └── metrics.py            # Risk metrics & league table
├── opt/
│   └── ga.py                 # Genetic Algorithm core
├── tests/
│   ├── test_opt_ga.py        # GA unit tests
│   └── test_per_regime_apply.py  # Regime adaptation tests
├── app.py                    # Streamlit dashboard (main entry)
├── demo_run_market.py        # Market generator demo
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/market-wars-simulator.git
cd market-wars-simulator
```

### 2️⃣ Create Virtual Environment

```bash
# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

---

## ⚡ Quick Demo (CLI)

Run the market generator test:

```bash
python demo_run_market.py
```

This generates a 2-year synthetic price path with regime colors and shock markers.

**Output:** `demo_market_path.csv`

---

## 📊 Example Results

### Momentum Strategy — Calm Bull Market

| Metric | Value |
|:-------|:------|
| **CAGR** | +3.67% |
| **Ann. Volatility** | 11.7% |
| **Sharpe Ratio** | 0.36 |
| **Max Drawdown** | -18.4% |
| **Calmar Ratio** | 0.19 |
| **Hit Rate** | 51.7% |
| **Trades** | 298 |

---

## 🧬 Genetic Algorithm Example

### Configuration

| Parameter | Value |
|:----------|:------|
| **Population** | 12 |
| **Generations** | 10 |
| **Elite Size** | 2 |
| **Mutation Rate** | 0.2 |
| **Fitness Function** | Sharpe - 2×Drawdown - 0.5×Turnover |

### Outputs

- `ga_results/momentum_calm_bull_regime_0.json`
- `ga_results/momentum_calm_bull_regime_1.json`
- Downloadable best gene JSON from UI

---

## 🧪 Testing

Run the unit tests:

```bash
pytest -q
```

**Test Coverage:**
- ✅ GA core execution & callback
- ✅ Regime-aware simulation validation
- ✅ Portfolio accounting accuracy
- ✅ Broker cost calculations

---

## 🧭 Market Presets

| Preset | Behavior | Use Case |
|:-------|:---------|:---------|
| 🟢 **Calm Bull** | Stable uptrend, low volatility | Benchmark for long bias |
| 🟡 **Choppy Sideways** | Range-bound, frequent regime changes | Test strategy adaptability |
| 🔴 **Crisis Mode** | High volatility, negative drift, frequent shocks | Stress test & drawdown handling |

---

## 🧠 How Strategy Evolution Works

### 1. **Simulation**
Market generates regime-driven prices → Agents execute trades → P&L tracked

### 2. **Evaluation**
Compute Sharpe ratio, drawdown, volatility, and turnover metrics

### 3. **Optimization**
- GA mutates strategy parameters (e.g., MA lengths, z-score thresholds)
- Fitness = `Sharpe - α×Drawdown - β×Turnover`
- Elite strategies preserved, weak ones eliminated

### 4. **Adaptation**
- **Global GA**: Single optimized parameter set
- **Per-Regime GA**: Different parameters per market regime

### 5. **Live Mode**
Apply per-regime genes dynamically to observe adaptive trading behavior

---

## 🎨 Visualization Examples

| Chart Type | Description |
|:-----------|:------------|
| **Price Chart** | Regime-colored background with shock event markers |
| **Equity Curves** | Multi-agent performance comparison over time |
| **League Table** | Ranked by Sharpe, Calmar, and drawdown metrics |
| **GA Fitness Plot** | Fitness evolution across generations |
| **Regime Adaptation** | Dynamic parameter switching visualization |

---

## 📁 Data Outputs

| File | Description |
|:-----|:------------|
| `demo_market_path.csv` | Simulated market price path |
| `league_table.csv` | Strategy comparison metrics |
| `equity_curves.csv` | Agent equity curves over time |
| `ga_results/*.json` | GA optimization results per regime |
| `regime_aware_sim_*.csv` | Adaptive simulation results |

---

## 🧭 Roadmap

### Planned Features

- [ ] **Reinforcement Learning Agent** — Discrete L/S/Flat actions with policy optimization
- [ ] **Multi-Asset Simulation** — Correlated regimes for portfolio strategies
- [ ] **Live Monte Carlo Visualizer** — Real-time GA evolution display
- [ ] **Persistent Database** — SQLite/Feather for result storage
- [ ] **Battle Arena Leaderboard** — Web-based competition platform
- [ ] **Options Strategies** — Delta hedging and volatility trading
- [ ] **Backtesting Framework** — Historical data integration

---

## 🏆 Why This Project Stands Out

### 🎯 For Hackathon Judges

**1. Domain Expertise**
- Realistic market microstructure modeling
- Professional-grade risk metrics and accounting

**2. Technical Depth**
- Hidden Markov Models for regime simulation
- Genetic Algorithm optimization with callbacks
- Modular, testable, maintainable codebase

**3. Innovation**
- Per-regime adaptive parameter evolution
- Live simulation with dynamic strategy switching
- Multi-objective fitness optimization

**4. User Experience**
- Beautiful Streamlit dashboard
- Clear visualizations and downloadable results
- Multiple preset scenarios for instant exploration

**5. Practical Value**
- Educational tool for quantitative finance
- Research platform for strategy development
- Portfolio optimization framework

---

## 🧰 Tech Stack

| Category | Libraries |
|:---------|:----------|
| **Simulation** | `numpy`, `pandas` |
| **Visualization** | `matplotlib`, `streamlit` |
| **Optimization** | Custom `opt/ga.py` |
| **Evaluation** | `scikit-learn` (optional utils) |
| **Testing** | `pytest` |

---

## 🧑‍💻 Authors

**Team Market Wars**

👤 **Nitesh Ranjan Singh** — Quant Dev & Simulation  
👤 **Dharun Prasanth** — Optimization & AI Chatbot

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and extend.

See [LICENSE](LICENSE) file for details.

---

## 💡 Pitch (30-Second Version)

> *"We built a trading battle arena where strategies compete in synthetic markets with real-world complexity. Each market has bull/bear/sideways regimes and crash events. Agents range from momentum to mean-reversion, all facing realistic costs. A Genetic Algorithm evolves each strategy's parameters and adapts them to changing market conditions. The dashboard shows clear equity curves, drawdowns, and metrics. Downloadable JSONs ensure reproducibility. Our simulator blends market realism with AI adaptation — a true playground for strategy evolution."*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Contact

Have questions or suggestions? Feel free to open an issue or reach out!

- **GitHub Issues**: [Create an issue](https://github.com/<your-username>/market-wars-simulator/issues)
- **Email**: your.email@example.com

---

<div align="center">

### 📈 Market Wars — Learn, Simulate, Evolve.

**Made with ❤️ by Team Market Wars**

⭐ Star this repo if you find it useful!

</div>
