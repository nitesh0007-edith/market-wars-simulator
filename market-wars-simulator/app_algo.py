import os
import json
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from market.generator import generate_path
from market.visualize import plot_market
from engine.broker import Broker
from engine.portfolio import Portfolio
from agents.buyhold import BuyHold
from agents.momentum import Momentum
from agents.meanrev import MeanRev
from agents.voltarget import VolTarget
from eval.metrics import build_league, trade_stats_from_trades_df, equity_metrics
from opt.ga import simple_ga

import streamlit as st

def render_app():
    """
    Render the Market Wars / Strategy Arena Streamlit UI.
    Call this from the parent app to embed the sub-app.
    """
    # If your app had global top-level Streamlit commands, move them here.
    # Example:
    st.markdown("<h2>Market Wars — Strategy Arena</h2>", unsafe_allow_html=True)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    layout="wide",
    page_title="Market Wars",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ==================== ENHANCED STYLING ====================
st.markdown("""
<style>
    /* Global Theme */
    .stApp { 
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #0f1419 0%, #1a1f3a 100%);
        border-right: 2px solid rgba(59, 130, 246, 0.2);
        box-shadow: 5px 0 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #60a5fa !important;
        text-shadow: 0 0 10px rgba(96, 165, 250, 0.3);
    }
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(59, 130, 246, 0.25);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-header h1 {
        color: white;
        margin: 0;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .hero-header p {
        color: rgba(255, 255, 255, 0.95);
        margin: 0.8rem 0 0;
        font-size: 1.2rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1.5rem 0;
    }
    
    .section-header h2 {
        color: #60a5fa !important;
        margin: 0 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    .section-header p {
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 20, 35, 0.8), rgba(26, 31, 58, 0.8));
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #60a5fa;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1;
    }
    
    .metric-value.positive { color: #34d399; }
    .metric-value.negative { color: #f87171; }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stDownloadButton>button {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    /* Info/Success/Error Boxes */
    .stAlert {
        border-radius: 10px !important;
        border-left: 4px solid !important;
    }
    
    /* Glossary Styles */
    .glossary-container {
        background: linear-gradient(135deg, rgba(15, 20, 35, 0.6), rgba(26, 31, 58, 0.6));
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .glossary-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .glossary-header h2 {
        color: #60a5fa;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .glossary-search {
        max-width: 600px;
        margin: 0 auto 2rem;
    }
    
    .glossary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    
    .glossary-card {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.9), rgba(15, 20, 35, 0.9));
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.15);
        transition: all 0.3s ease;
    }
    
    .glossary-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2);
    }
    
    .glossary-term {
        color: #60a5fa;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .glossary-short {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1rem;
        font-style: italic;
    }
    
    .glossary-body {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.8), rgba(15, 20, 35, 0.8));
        border-radius: 8px;
        color: #94a3b8;
        padding: 0.8rem 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border-color: transparent !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Input Fields */
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background: rgba(15, 20, 35, 0.8) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* Charts Container */
    .chart-container {
        background: linear-gradient(135deg, rgba(15, 20, 35, 0.5), rgba(26, 31, 58, 0.5));
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.15);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 1px solid rgba(59, 130, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== HERO HEADER ====================
st.markdown("""
<div class="hero-header">
  <h1>📈 Market Wars</h1>
  <p>Advanced Trading Strategy Arena • Multi-Regime Markets • Genetic Algorithm Optimization</p>
</div>
""", unsafe_allow_html=True)

# ==================== GLOSSARY DATA ====================
GLOSSARY = {
    "CAGR": {
        "short": "Compound Annual Growth Rate — annualized compounded return.",
        "body": "**Definition:** Annualized compounded return over the period.\n\n**Formula:** `(FinalEquity/InitialEquity)^(252 / nDays) - 1`\n\n**Why it matters:** Normalizes growth to an annual rate so you can compare strategies run over different lengths."
    },
    "AnnVol": {
        "short": "Annualized Volatility — standard deviation of returns scaled to a year.",
        "body": "**Definition:** Standard deviation of daily returns × √252.\n\n**Why it matters:** Measures variability (risk). Lower vol for the same return is better."
    },
    "Sharpe": {
        "short": "Return per unit risk (Annualized Sharpe Ratio).",
        "body": "**Definition:** `(CAGR - r_f) / AnnVol` (risk-free rate = 0 by default).\n\n**Why it matters:** Widely-used risk-adjusted performance metric—higher is better."
    },
    "MaxDD": {
        "short": "Maximum Drawdown — largest peak-to-trough loss.",
        "body": "**Definition:** Largest percentage decline from a peak to subsequent trough during the period.\n\n**Why it matters:** Captures the worst equity shock — crucial for capital preservation."
    },
    "Calmar": {
        "short": "CAGR / |MaxDD| — return adjusted for drawdown.",
        "body": "**Definition:** `CAGR / |MaxDD|`\n\n**Why it matters:** Focuses on return relative to the worst drawdown — 'return per unit of pain'."
    },
    "HitRate": {
        "short": "Proportion of winning trades.",
        "body": "**Definition:** `Wins / TotalTrades`\n\n**Why it matters:** Shows strategy consistency. High hit rate alone doesn't guarantee profitability."
    },
    "Turnover": {
        "short": "Total traded value relative to capital.",
        "body": "**Definition:** `Total traded value / initial capital`\n\n**Why it matters:** Proxy for trading intensity — high turnover increases costs and market impact."
    },
    "Slippage": {
        "short": "Execution price movement vs ideal price.",
        "body": "**Definition:** Modeled as `k * |Δweight|` applied to execution price.\n\n**Why:** Models market impact—important for larger orders."
    },
    "Fitness (GA)": {
        "short": "Objective function used by the Genetic Algorithm.",
        "body": "**Definition:** `Fitness = Sharpe - α * max(0, -MaxDD) - β * Turnover`\n\n**Why:** Balances reward (Sharpe) against risk (drawdown) and cost (turnover)."
    },
    "Per-Regime GA": {
        "short": "Run GA separately for each market regime.",
        "body": "**Definition:** Split market by regime (Bull/Bear/Flat) and optimize parameters independently.\n\n**Why:** Markets are non-stationary — parameters that work in Bull may fail in Bear."
    }
}

# ==================== HELPER FUNCTIONS ====================
preset_map = {
    "calm_bull": {
        "name": "Calm Bull",
        "desc": "Steady upward trend with modest volatility",
        "seed": 1, "n_steps": 252*2, "shock_prob": 0.002,
        "trans": [[0.94,0.03,0.03],[0.10,0.80,0.10],[0.15,0.15,0.70]],
        "mu_by_reg": [0.18, -0.06, 0.02], "sigma_by_reg": [0.12,0.30,0.08]
    },
    "choppy_sideways": {
        "name": "Choppy Sideways",
        "desc": "High regime switching and range-bound behaviour",
        "seed": 11, "n_steps": 252*2, "shock_prob": 0.01,
        "trans": [[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7]],
        "mu_by_reg": [0.08, -0.04, 0.0], "sigma_by_reg": [0.16,0.22,0.12]
    },
    "crisis": {
        "name": "Crisis Mode",
        "desc": "High volatility + frequent shocks",
        "seed": 7, "n_steps": 252*2, "shock_prob": 0.03, "muJ": -0.1, "sigmaJ": 0.12,
        "mu_by_reg": [0.06, -0.25, 0.0], "sigma_by_reg": [0.14,0.45,0.12]
    }
}

def gene_to_agent(agent_name, gene):
    if agent_name == "Momentum":
        fast = int(gene.get('fast', 20))
        slow = int(gene.get('slow', 100))
        weight = float(gene.get('weight', 1.0))
        trailing = gene.get('trailing_stop', None)
        trailing = None if trailing is None else float(trailing)
        return Momentum(fast=fast, slow=slow, weight=weight, trailing_stop=trailing)
    if agent_name == "MeanRev":
        window = int(gene.get('window', 20))
        zentry = float(gene.get('zentry', 1.0))
        zexit = float(gene.get('zexit', 0.5))
        weight = float(gene.get('weight', 1.0))
        return MeanRev(window=window, zentry=zentry, zexit=zexit, weight=weight)
    if agent_name == "VolTarget":
        ewma = float(gene.get('ewma', 0.94))
        targ = float(gene.get('target_ann_vol', 0.12))
        ml = float(gene.get('max_leverage', 2.0))
        return VolTarget(ewma_alpha=ewma, target_ann_vol=targ, max_leverage=ml)
    if agent_name == "BuyHold":
        w = float(gene.get('weight', 1.0))
        return BuyHold(weight=w)
    raise ValueError("Unknown agent " + str(agent_name))

GA_GENE_BOUNDS = {
    "Momentum": {"fast": (5,40,True), "slow": (30,200,True), "weight": (0.1,2.0,False), "trailing_stop": (0.0,0.3,False)},
    "MeanRev": {"window": (5,60,True), "zentry": (0.5,3.0,False), "zexit": (0.1,1.5,False), "weight": (0.1,2.0,False)},
    "VolTarget": {"ewma": (0.85,0.995,False), "target_ann_vol": (0.05,0.30,False), "max_leverage": (1.0,4.0,False)},
    "BuyHold": {"weight": (0.1,2.0,False)}
}

def fitness_on_price_series(gene, agent_name, price_series, broker_cfg, initial_cash=100000.0):
    agent = gene_to_agent(agent_name, gene)
    broker = Broker(**broker_cfg)
    port = Portfolio(initial_cash=initial_cash, broker=broker)
    agent.reset()
    prices = list(map(float, price_series))
    for t, p in enumerate(prices):
        try:
            target = agent.decide(t, prices[:t+1], port)
        except Exception:
            target = 0.0
        port.step(p, float(target))
    eq = port.snapshot_dataframe().get('equity', pd.Series([initial_cash]*len(prices))).astype(float)
    em = equity_metrics(eq)
    trades_df = port.snapshot_dataframe()
    ts = trade_stats_from_trades_df(trades_df)
    sharpe = em.get('Sharpe', -999.0)
    maxdd = em.get('MaxDD', -0.0)
    turnover = ts.get('Turnover', 0.0) if ts.get('Turnover', 0.0) is not None else (ts.get('TotalTraded', 0.0) / (initial_cash+1e-9))
    alpha = 2.0; beta = 0.5
    fitness = sharpe - alpha * max(0.0, -maxdd) - beta * turnover
    fitness += 0.001 * (em.get('CAGR', 0.0) or 0.0)
    return float(fitness)

# ==================== MAIN APP LAYOUT ====================
# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Simulation", "🧬 Genetic Algorithm", "📊 Results & Analysis", "📚 Knowledge Base"])

# ==================== TAB 1: SIMULATION ====================
with tab1:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown('<div class="section-header"><h2>⚙️ Configuration</h2></div>', unsafe_allow_html=True)
        
        with st.expander("🌍 Market Preset", expanded=True):
            preset_choice = st.selectbox("Choose Market Type", list(preset_map.keys()), 
                                        format_func=lambda k: preset_map[k]['name'])
            st.caption(preset_map[preset_choice].get("desc", ""))
        
        with st.expander("💼 Broker Settings", expanded=True):
            commission = st.number_input("Commission per trade ($)", value=1.0, step=0.5)
            spread = st.number_input("Spread (fraction)", value=0.0005, format="%.6f")
            slippage_k = st.number_input("Slippage k", value=0.003, format="%.6f")
            max_leverage = st.number_input("Max leverage", value=2.0, step=0.5)
        
        with st.expander("🤖 Trading Agents", expanded=True):
            run_buyhold = st.checkbox("Buy & Hold", value=True)
            run_momentum = st.checkbox("Momentum", value=True)
            run_meanrev = st.checkbox("Mean Reversion", value=True)
            run_voltarget = st.checkbox("Vol Targeting", value=True)
        
        with st.expander("🎨 Display Options"):
            show_trade_markers = st.checkbox("Show trade markers", value=False)
        
        if st.button("🚀 Run Simulation", key="run_sim", use_container_width=True):
            st.session_state.run_sim_trigger = True
    
    with col_right:
        st.markdown('<div class="section-header"><h2>📈 Market Preview</h2><p>Current market scenario visualization</p></div>', unsafe_allow_html=True)
        
        # Show preview of selected preset
        preset_cfg = preset_map[preset_choice]
        preview_df = generate_path(
            n_steps=min(252, preset_cfg.get("n_steps", 252)),
            seed=preset_cfg.get("seed", 0),
            trans=np.array(preset_cfg["trans"]),
            mu_by_reg=np.array(preset_cfg["mu_by_reg"]),
            sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]),
            shock_prob=preset_cfg.get("shock_prob", 0.005),
            muJ=preset_cfg.get("muJ", 0.0),
            sigmaJ=preset_cfg.get("sigmaJ", 0.08),
            S0=100.0
        )
        
        fig_preview, ax_preview = plt.subplots(figsize=(12, 5))
        plot_market(preview_df, title=f"Preview: {preset_cfg['name']}", ax=ax_preview, show=False)
        st.pyplot(fig_preview)

# ==================== RUN SIMULATION LOGIC ====================
if st.session_state.get("run_sim_trigger"):
    with st.spinner("Running simulation..."):
        preset_cfg = preset_map[preset_choice]
        df = generate_path(
            n_steps=preset_cfg.get("n_steps", 252*2),
            seed=preset_cfg.get("seed", 0),
            trans=np.array(preset_cfg["trans"]),
            mu_by_reg=np.array(preset_cfg["mu_by_reg"]),
            sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]),
            shock_prob=preset_cfg.get("shock_prob", 0.005),
            muJ=preset_cfg.get("muJ", 0.0),
            sigmaJ=preset_cfg.get("sigmaJ", 0.08),
            S0=100.0
        ).reset_index(drop=True)

        broker_cfg = {"commission": commission, "spread": spread, "slippage_k": slippage_k, "max_leverage": max_leverage}

        agents = []
        if run_buyhold: agents.append(BuyHold(weight=1.0))
        if run_momentum: agents.append(Momentum(fast=20, slow=100, weight=1.0))
        if run_meanrev: agents.append(MeanRev(window=20, zentry=1.0, zexit=0.5, weight=1.0))
        if run_voltarget: agents.append(VolTarget(ewma_alpha=0.94, target_ann_vol=0.12, max_leverage=max_leverage))

        results = []
        for agent in agents:
            port = Portfolio(initial_cash=100000.0, broker=Broker(**broker_cfg))
            agent.reset()
            price_history = []
            equity = []
            for t, p in enumerate(df['price'].values):
                price_history.append(float(p))
                target = agent.decide(t, price_history, port)
                e, _ = port.step(p, float(target))
                equity.append(e)
            trades_df = port.snapshot_dataframe()
            res = {
                "agent": agent.name,
                "equity": pd.Series(equity),
                "trades_df": trades_df,
                "n_trades": int((trades_df['delta_units'].fillna(0) != 0).sum()) if not trades_df.empty else 0,
                "total_traded": float(getattr(port, "total_traded", 0.0)),
                "total_commission": float(getattr(port, "total_commission", 0.0))
            }
            results.append(res)

        league = build_league(results)
        st.session_state['last_sim'] = {"df": df, "results": results, "league": league, "broker_cfg": broker_cfg}
        st.session_state.run_sim_trigger = False
    st.success("✅ Simulation completed successfully!")
    st.rerun()

# ==================== TAB 2: GENETIC ALGORITHM ====================
with tab2:
    col_ga_left, col_ga_right = st.columns([1, 2])
    
    with col_ga_left:
        st.markdown('<div class="section-header"><h2>🧬 GA Settings</h2></div>', unsafe_allow_html=True)
        
        ga_agent_choice = st.selectbox("Agent to optimize", ["Momentum", "MeanRev", "VolTarget", "BuyHold"])
        ga_pop = st.number_input("Population", min_value=4, max_value=64, value=8, step=2)
        ga_gens = st.number_input("Generations", min_value=2, max_value=80, value=8, step=1)
        ga_elite = st.number_input("Elites", min_value=1, max_value=max(1, int(ga_pop)-1), value=2, step=1)
        ga_mut = st.slider("Mutation rate", 0.0, 1.0, 0.2, 0.05)
        ga_seed = st.number_input("GA seed", value=0, step=1)
        per_regime_ga = st.checkbox("Per-regime GA (optimize separately)", value=False)
        
        if st.button("🧬 Run GA Optimization", key="run_ga", use_container_width=True):
            st.session_state.run_ga_trigger = True
    
    with col_ga_right:
        st.markdown('<div class="section-header"><h2>📖 GA Explanation</h2></div>', unsafe_allow_html=True)
        st.markdown("""
        The Genetic Algorithm optimizes strategy parameters through evolution:
        
        - **Population**: Number of parameter sets tested each generation
        - **Generations**: How many evolution cycles to run
        - **Elite**: Top performers preserved unchanged
        - **Mutation**: Random parameter variation rate
        - **Per-Regime**: Optimize separately for Bull/Bear/Flat markets
        
        The fitness function balances **Sharpe ratio**, **drawdown**, and **turnover**.
        """)

# ==================== GA EXECUTION LOGIC ====================
if st.session_state.get("run_ga_trigger"):
    preset_cfg = preset_map[preset_choice]
    broker_cfg = {"commission":commission, "spread":spread, "slippage_k":slippage_k, "max_leverage":max_leverage}
    agent_name = ga_agent_choice
    gene_bounds = GA_GENE_BOUNDS.get(agent_name)
    
    if gene_bounds is None:
        st.error("No GA gene bounds for this agent.")
    else:
        st.markdown('<div class="section-header"><h2>🔄 Optimization in Progress</h2></div>', unsafe_allow_html=True)
        
        def normalize_gene(g):
            gf = {}
            for k, (lo, hi, is_int) in gene_bounds.items():
                val = g.get(k, None)
                if val is None:
                    val = np.random.randint(lo, hi+1) if is_int else np.random.uniform(lo, hi)
                gf[k] = int(round(val)) if is_int else float(val)
            if 'trailing_stop' in gf and gf['trailing_stop'] == 0.0:
                gf['trailing_stop'] = None
            return gf

        def run_ga_on_series(price_series, label="global"):
            progress = st.progress(0)
            status = st.empty()
            st.session_state['ga_history'] = []

            def fitness_wrapper(g):
                gf = normalize_gene(g)
                return fitness_on_price_series(gf, agent_name, price_series, broker_cfg, initial_cash=100000.0)

            def on_gen(gen_idx, best_fit_gen, best_gene_gen):
                pct = int((gen_idx + 1) / max(1, int(ga_gens)) * 100)
                progress.progress(pct)
                status.text(f"[{label}] Gen {gen_idx+1}/{ga_gens} best {best_fit_gen:.6f}")
                hist = st.session_state.get('ga_history', [])
                hist.append(float(best_fit_gen))
                st.session_state['ga_history'] = hist
                st.session_state['ga_best_partial'] = best_gene_gen

            best_gene, history = simple_ga(
                fitness_wrapper,
                gene_bounds,
                pop_size=int(ga_pop),
                gens=int(ga_gens),
                elite=int(ga_elite),
                mut_rate=float(ga_mut),
                seed=int(ga_seed),
                verbose=False,
                on_generation=on_gen
            )
            progress.progress(100)
            status.empty()
            return best_gene, history

        if per_regime_ga:
            st.info("🔄 Per-regime GA: splitting market and optimizing each regime segment...")
            df_full = generate_path(
                n_steps=preset_cfg.get("n_steps", 252*2),
                seed=preset_cfg.get("seed", 0),
                trans=np.array(preset_cfg["trans"]),
                mu_by_reg=np.array(preset_cfg["mu_by_reg"]),
                sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]),
                shock_prob=preset_cfg.get("shock_prob", 0.005),
                muJ=preset_cfg.get("muJ", 0.0),
                sigmaJ=preset_cfg.get("sigmaJ", 0.08),
                S0=100.0
            ).reset_index(drop=True)

            if 'regime' not in df_full.columns:
                st.error("Market generator must include 'regime' column for per-regime GA.")
            else:
                segments = {}
                start = 0
                for i in range(1, len(df_full)):
                    if df_full.loc[i, 'regime'] != df_full.loc[start, 'regime']:
                        seg = df_full.loc[start:i-1].copy()
                        rid = int(seg['regime'].iloc[0])
                        segments.setdefault(rid, []).append(seg)
                        start = i
                seg = df_full.loc[start:len(df_full)-1].copy()
                rid = int(seg['regime'].iloc[0])
                segments.setdefault(rid, []).append(seg)

                per_reg_results = {}
                os.makedirs("ga_results", exist_ok=True)
                for rid, segs in segments.items():
                    big = pd.concat(segs, ignore_index=True)
                    price_series = big['price'].values.tolist()
                    st.write(f"🎯 Optimizing regime {rid} ({len(price_series)} bars)")
                    best_gene, history = run_ga_on_series(price_series, label=f"Regime {rid}")
                    per_reg_results[int(rid)] = {"best_gene": best_gene, "history": history}
                    out_path = os.path.join("ga_results", f"{agent_name}_{preset_choice}_regime_{rid}.json")
                    with open(out_path, "w") as f:
                        json.dump({"agent":agent_name,"preset":preset_choice,"regime":int(rid),"best_gene":best_gene,"history":history}, f, indent=2)
                    st.success(f"✅ Saved {out_path}")

                st.session_state['last_ga_result'] = {"agent":agent_name,"preset":preset_choice,"per_regime":True,"results":per_reg_results,"config":{"pop":ga_pop,"gens":ga_gens,"elite":ga_elite,"mut":ga_mut,"seed":ga_seed}}
                st.success("🎉 Per-regime GA complete!")

        else:
            st.info("🔄 Global GA: optimizing on full market path...")
            df_short = generate_path(
                n_steps=252, seed=preset_cfg.get("seed",0),
                trans=np.array(preset_cfg["trans"]),
                mu_by_reg=np.array(preset_cfg["mu_by_reg"]),
                sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]),
                shock_prob=preset_cfg.get("shock_prob", 0.005),
                muJ=preset_cfg.get("muJ", 0.0),
                sigmaJ=preset_cfg.get("sigmaJ", 0.08),
                S0=100.0
            )
            price_series = df_short['price'].values.tolist()
            best_gene, history = run_ga_on_series(price_series, label="Global")
            st.session_state['last_ga_result'] = {"agent":agent_name,"preset":preset_choice,"per_regime":False,"best_gene":best_gene,"history":history,"config":{"pop":ga_pop,"gens":ga_gens,"elite":ga_elite,"mut":ga_mut,"seed":ga_seed}}
            st.success("🎉 Global GA finished!")

        st.session_state.run_ga_trigger = False
        st.rerun()

# ==================== TAB 3: RESULTS & ANALYSIS ====================
with tab3:
    st.markdown('<div class="section-header"><h2>📊 Performance Analysis</h2><p>Comprehensive results dashboard</p></div>', unsafe_allow_html=True)
    
    # Simulation Results
    if st.session_state.get("last_sim"):
        sim = st.session_state['last_sim']
        df = sim['df']
        results = sim['results']
        league = sim['league']
        
        st.markdown("### 🏆 Strategy Leaderboard")
        st.dataframe(
            league.style.format({
                "CAGR":"{:.2%}", "AnnVol":"{:.2%}", "Sharpe":"{:.3f}",
                "MaxDD":"{:.2%}", "Calmar":"{:.3f}", "HitRate":"{:.2%}"
            }),
            use_container_width=True,
            height=300
        )
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Download League CSV",
                data=league.reset_index().to_csv(index=False).encode('utf-8'),
                file_name="league.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            eq_df = pd.DataFrame({r['agent']: r['equity'].values for r in results})
            st.download_button(
                "📥 Download Equity CSV",
                data=eq_df.to_csv(index=False).encode('utf-8'),
                file_name="equity.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("### 📈 Visual Analysis")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0a0e27')
            ax.set_facecolor('#0f1419')
            plot_market(df, title="Market Path", ax=ax, show=False)
            if show_trade_markers:
                colors = ['#34d399','#f87171','#60a5fa','#c084fc']
                for i, r in enumerate(results):
                    td = r['trades_df']
                    if td is None or td.empty: continue
                    buy_idx = td[td['delta_units']>0].index.values
                    sell_idx = td[td['delta_units']<0].index.values
                    exec_prices = td['exec_price'].values if 'exec_price' in td.columns else td['price'].values
                    if len(buy_idx): ax.scatter(buy_idx, exec_prices[buy_idx], marker='^', color=colors[i%len(colors)], s=50, label=f"{r['agent']} Buy", alpha=0.7)
                    if len(sell_idx): ax.scatter(sell_idx, exec_prices[sell_idx], marker='v', color=colors[i%len(colors)], s=50, label=f"{r['agent']} Sell", alpha=0.7)
                ax.legend(fontsize='small', facecolor='#0f1419', edgecolor='#3b82f6')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            fig2.patch.set_facecolor('#0a0e27')
            ax2.set_facecolor('#0f1419')
            colors_eq = ['#60a5fa','#c084fc','#34d399','#f472b6']
            for i, r in enumerate(results):
                ax2.plot(r['equity'].values, label=r['agent'], linewidth=2.5, color=colors_eq[i%len(colors_eq)])
            ax2.legend(facecolor='#0f1419', edgecolor='#3b82f6')
            ax2.set_title("Agent Equity Curves", color='#60a5fa', fontsize=14, fontweight='bold')
            ax2.tick_params(colors='#94a3b8')
            ax2.spines['bottom'].set_color('#3b82f6')
            ax2.spines['left'].set_color('#3b82f6')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("ℹ️ Run a simulation first to see results here!")
    
    # GA Results
    ga_res = st.session_state.get('last_ga_result', None)
    if ga_res:
        st.markdown("---")
        st.markdown('<div class="section-header"><h2>🧬 GA Optimization Results</h2></div>', unsafe_allow_html=True)
        
        col_ga_res1, col_ga_res2 = st.columns([1, 1])
        
        with col_ga_res1:
            st.markdown("#### Configuration")
            config_display = {k: v for k, v in ga_res.items() if k not in ['results', 'history', 'best_gene']}
            st.json(config_display)
            
            if ga_res.get('best_gene'):
                st.markdown("#### Best Parameters Found")
                st.json(ga_res['best_gene'])
        
        with col_ga_res2:
            if ga_res.get('per_regime'):
                st.markdown("#### Per-Regime Fitness Evolution")
                for rid, info in ga_res['results'].items():
                    fig, ax = plt.subplots(figsize=(8, 3))
                    fig.patch.set_facecolor('#0a0e27')
                    ax.set_facecolor('#0f1419')
                    ax.plot(info.get('history', []), marker='o', color='#60a5fa', linewidth=2)
                    ax.set_title(f"Regime {rid} Fitness Evolution", color='#60a5fa', fontweight='bold')
                    ax.set_xlabel("Generation", color='#94a3b8')
                    ax.set_ylabel("Fitness", color='#94a3b8')
                    ax.tick_params(colors='#94a3b8')
                    ax.grid(True, alpha=0.2, color='#3b82f6')
                    ax.spines['bottom'].set_color('#3b82f6')
                    ax.spines['left'].set_color('#3b82f6')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
            else:
                hist = ga_res.get('history', [])
                if hist:
                    st.markdown("#### Fitness Evolution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor('#0a0e27')
                    ax.set_facecolor('#0f1419')
                    ax.plot(hist, marker='o', color='#60a5fa', linewidth=2.5, markersize=8)
                    ax.set_title("GA Best Fitness per Generation", color='#60a5fa', fontsize=14, fontweight='bold')
                    ax.set_xlabel("Generation", color='#94a3b8', fontsize=11)
                    ax.set_ylabel("Fitness Score", color='#94a3b8', fontsize=11)
                    ax.tick_params(colors='#94a3b8')
                    ax.grid(True, alpha=0.2, color='#3b82f6')
                    ax.spines['bottom'].set_color('#3b82f6')
                    ax.spines['left'].set_color('#3b82f6')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
        
        st.download_button(
            "📥 Download GA Results JSON",
            data=json.dumps(ga_res, indent=2),
            file_name=f"ga_{ga_res.get('agent')}_{ga_res.get('preset')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        if ga_res.get('per_regime'):
            if st.button("🎯 Apply Per-Regime Genes & Run Adaptive Simulation", use_container_width=True):
                with st.spinner("Running regime-aware simulation..."):
                    preset_cfg = preset_map[preset_choice]
                    broker_cfg = st.session_state.get('last_sim', {}).get('broker_cfg', {"commission":commission,"spread":spread,"slippage_k":slippage_k,"max_leverage":max_leverage})
                    df_full = generate_path(
                        n_steps=preset_cfg.get("n_steps",252*2), seed=preset_cfg.get("seed",0),
                        trans=np.array(preset_cfg["trans"]), mu_by_reg=np.array(preset_cfg["mu_by_reg"]),
                        sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]), shock_prob=preset_cfg.get("shock_prob",0.005),
                        muJ=preset_cfg.get("muJ",0.0), sigmaJ=preset_cfg.get("sigmaJ",0.08), S0=100.0
                    ).reset_index(drop=True)

                    if 'regime' not in df_full.columns:
                        st.error("Market missing 'regime' column.")
                    else:
                        per_map = {int(k): v['best_gene'] for k,v in ga_res['results'].items()}
                        port = Portfolio(initial_cash=100000.0, broker=Broker(**broker_cfg))
                        equity = []
                        price_hist = []
                        current_agent = None
                        current_regime = None
                        for t, row in df_full.iterrows():
                            price = float(row['price'])
                            price_hist.append(price)
                            rid = int(row['regime'])
                            gene = per_map.get(rid, None)
                            agent = gene_to_agent(ga_res['agent'], gene) if gene is not None else BuyHold(weight=1.0)
                            if current_regime != rid:
                                agent.reset()
                                current_agent = agent
                                current_regime = rid
                            try:
                                target = current_agent.decide(t, price_hist, port)
                            except Exception:
                                target = 0.0
                            e, _ = port.step(price, float(target))
                            equity.append(e)
                        
                        eqs = pd.Series(equity)
                        metrics = equity_metrics(eqs)
                        
                        st.success("✅ Regime-aware simulation complete!")
                        
                        fig_adaptive, ax_adaptive = plt.subplots(figsize=(12, 5))
                        fig_adaptive.patch.set_facecolor('#0a0e27')
                        ax_adaptive.set_facecolor('#0f1419')
                        ax_adaptive.plot(eqs.values, color='#34d399', linewidth=2.5)
                        ax_adaptive.set_title("Adaptive Strategy Equity Curve", color='#60a5fa', fontsize=14, fontweight='bold')
                        ax_adaptive.tick_params(colors='#94a3b8')
                        ax_adaptive.grid(True, alpha=0.2, color='#3b82f6')
                        st.pyplot(fig_adaptive)
                        
                        st.json(metrics)
                        
                        df_out = pd.DataFrame({"price": df_full['price'].values, "regime": df_full['regime'].values, "equity": eqs.values})
                        st.download_button(
                            "📥 Download Adaptive Results",
                            data=df_out.to_csv(index=False).encode('utf-8'),
                            file_name=f"adaptive_{ga_res['agent']}_{preset_choice}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

# ==================== TAB 4: KNOWLEDGE BASE ====================
with tab4:
    st.markdown("""
    <div class="glossary-container">
        <div class="glossary-header">
            <h2>📚 Knowledge Base & Metrics Glossary</h2>
            <p style="color: #94a3b8; font-size: 1.1rem;">
                Comprehensive guide to understanding trading metrics and optimization concepts
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search functionality
    col_search, col_download = st.columns([3, 1])
    with col_search:
        query = st.text_input("🔍 Search glossary", placeholder="Try 'Sharpe', 'drawdown', 'turnover'...", label_visibility="collapsed")
    with col_download:
        md = "# Market Wars Glossary\n\n"
        for k in GLOSSARY.keys():
            md += f"## {k}\n\n**Short:** {GLOSSARY[k]['short']}\n\n{GLOSSARY[k]['body']}\n\n---\n\n"
        st.download_button(
            "📥 Download Guide",
            data=md,
            file_name="market_wars_glossary.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    # Filter glossary
    keys = list(GLOSSARY.keys())
    if query:
        q = query.strip().lower()
        keys = [k for k in keys if q in k.lower() or q in GLOSSARY[k]['short'].lower() or q in GLOSSARY[k]['body'].lower()]
    
    # Display as grid
    st.markdown('<div class="glossary-grid">', unsafe_allow_html=True)
    
    cols_per_row = 2
    for i in range(0, len(keys), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(keys):
                k = keys[i + j]
                with cols[j]:
                    with st.expander(f"**{k}**", expanded=False):
                        st.markdown(f"<div class='glossary-short'>{GLOSSARY[k]['short']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='glossary-body'>{GLOSSARY[k]['body']}</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional resources section
    st.markdown("---")
    st.markdown('<div class="section-header"><h2>📖 Additional Resources</h2></div>', unsafe_allow_html=True)
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.markdown("""
        ### 🎯 Quick Start Guide
        1. **Choose a market preset** from the Simulation tab
        2. **Configure broker settings** (commission, slippage)
        3. **Select trading agents** to compete
        4. **Run simulation** and analyze results
        5. **Optimize with GA** for better performance
        """)
    
    with col_res2:
        st.markdown("""
        ### 🧬 GA Best Practices
        - Start with **population=8, gens=8** for quick tests
        - Use **per-regime GA** for non-stationary markets
        - Higher **mutation rate** for exploration
        - **Elite count** preserves best solutions
        - Monitor fitness evolution for convergence
        """)
    
    with col_res3:
        st.markdown("""
        ### 📊 Interpreting Results
        - **Sharpe > 1.0**: Good risk-adjusted returns
        - **MaxDD < -20%**: High drawdown risk
        - **Calmar > 1.0**: Strong drawdown-adjusted return
        - **High Turnover**: Watch transaction costs
        - Compare **multiple metrics** together
        """)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p>🚀 Market Wars Simulator — Advanced Multi-Regime Trading Strategy Platform</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
        Powered by Genetic Algorithms • Real-time Market Simulation • Comprehensive Performance Analytics
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    render_app()