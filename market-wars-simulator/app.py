# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from market.generator import generate_path
from market.visualize import plot_market
from engine.broker import Broker
from engine.portfolio import Portfolio
from agents.buyhold import BuyHold
from agents.momentum import Momentum
from agents.meanrev import MeanRev
from agents.voltarget import VolTarget
from eval.metrics import build_league, trade_stats_from_trades_df, equity_metrics

# GA
from opt.ga import simple_ga

st.set_page_config(layout="wide", page_title="Market Wars Simulator", initial_sidebar_state="expanded")
st.title("Market Wars — Strategy League (Stage 3 + GA)")

# ---------------- Sidebar: market + execution ----------------
st.sidebar.header("Market Preset")
preset_choice = st.sidebar.selectbox("Preset", ["calm_bull", "choppy_sideways", "crisis"])

preset_map = {
    "calm_bull": {
        "name": "calm_bull", "seed": 1, "n_steps": 252*2, "shock_prob": 0.002,
        "trans": [[0.94,0.03,0.03],[0.10,0.80,0.10],[0.15,0.15,0.70]],
        "mu_by_reg": [0.18, -0.06, 0.02], "sigma_by_reg": [0.12,0.30,0.08]
    },
    "choppy_sideways": {
        "name": "choppy_sideways", "seed": 11, "n_steps": 252*2, "shock_prob": 0.01,
        "trans": [[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7]],
        "mu_by_reg": [0.08, -0.04, 0.0], "sigma_by_reg": [0.16,0.22,0.12]
    },
    "crisis": {
        "name": "crisis", "seed": 7, "n_steps": 252*2, "shock_prob": 0.03, "muJ": -0.1, "sigmaJ": 0.12,
        "mu_by_reg": [0.06, -0.25, 0.0], "sigma_by_reg": [0.14,0.45,0.12]
    }
}

st.sidebar.header("Execution (Broker)")
commission = st.sidebar.number_input("Commission per trade ($)", value=1.0, step=0.5)
spread = st.sidebar.number_input("Spread (fraction)", value=0.0005, format="%.6f")
slippage_k = st.sidebar.number_input("Slippage k (fraction per |Δw|)", value=0.003, format="%.6f")
max_leverage = st.sidebar.number_input("Max leverage", value=2.0, step=0.5)

st.sidebar.header("Agents")
run_buyhold = st.sidebar.checkbox("Buy & Hold", value=True)
run_momentum = st.sidebar.checkbox("Momentum", value=True)
run_meanrev = st.sidebar.checkbox("Mean Reversion", value=True)
run_voltarget = st.sidebar.checkbox("Vol Targeting", value=True)

# per-agent trade marker toggles
st.sidebar.header("Display Options")
show_trade_markers = st.sidebar.checkbox("Show per-agent trade markers (top chart)", value=False)

# ---------------- GA controls ----------------
st.sidebar.header("Genetic Algorithm (optimize agent)")
ga_agent_choice = st.sidebar.selectbox("Agent to optimize", ["Momentum", "MeanRev", "VolTarget", "BuyHold"])
ga_pop = st.sidebar.number_input("GA population", min_value=4, max_value=40, value=8, step=2)
ga_gens = st.sidebar.number_input("GA generations", min_value=2, max_value=40, value=8, step=1)
ga_elite = st.sidebar.number_input("GA elites", min_value=1, max_value=ga_pop-1, value=2, step=1)
ga_mut = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.2, 0.05)
ga_seed = st.sidebar.number_input("GA seed", value=0, step=1)

# Run simulation / GA buttons
run_sim = st.sidebar.button("Run Simulation")
run_ga = st.sidebar.button("Run GA (optimize selected agent)")

# ---------------- helper: build agent from gene ----------------
def gene_to_agent(agent_name, gene):
    if agent_name == "Momentum":
        # gene: fast, slow, weight (floats/ints)
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

# gene bounds for GA per agent
GA_GENE_BOUNDS = {
    "Momentum": {
        "fast": (5, 40, True),
        "slow": (30, 200, True),
        "weight": (0.1, 2.0, False),
        "trailing_stop": (0.0, 0.3, False)  # 0 disables if set to 0
    },
    "MeanRev": {
        "window": (5, 60, True),
        "zentry": (0.5, 3.0, False),
        "zexit": (0.1, 1.5, False),
        "weight": (0.1, 2.0, False)
    },
    "VolTarget": {
        "ewma": (0.85, 0.995, False),
        "target_ann_vol": (0.05, 0.30, False),
        "max_leverage": (1.0, 4.0, False)
    },
    "BuyHold": {
        "weight": (0.1, 2.0, False)
    }
}

# ---------------- fitness wrapper ----------------
def fitness_for_gene(gene, agent_name, preset_cfg, broker_cfg, initial_cash=100000.0, quick_days=252):
    """
    Build agent from gene, run a short sim (quick_days) on the market preset,
    return a scalar fitness = sharpe - alpha*maxdd - beta*turnover_penalty.
    """
    # build agent
    agent = gene_to_agent(agent_name, gene)
    # build market (use shorter path to speed up GA)
    df = generate_path(
        n_steps=quick_days,
        seed=preset_cfg.get("seed", 0),
        trans=np.array(preset_cfg["trans"]) if "trans" in preset_cfg else None,
        mu_by_reg=np.array(preset_cfg["mu_by_reg"]) if "mu_by_reg" in preset_cfg else None,
        sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]) if "sigma_by_reg" in preset_cfg else None,
        shock_prob=preset_cfg.get("shock_prob", 0.005),
        muJ=preset_cfg.get("muJ", 0.0),
        sigmaJ=preset_cfg.get("sigmaJ", 0.08),
        S0=100.0
    )
    broker = Broker(**broker_cfg)
    port = Portfolio(initial_cash=initial_cash, broker=broker)
    agent.reset()
    price_series = []
    for t, price in enumerate(df['price'].values):
        price_series.append(float(price))
        try:
            target = agent.decide(t, price_series, port)
        except Exception:
            target = 0.0
        port.step(price, float(target))

    eq = port.snapshot_dataframe().get('equity', pd.Series([initial_cash]*len(df))).astype(float)
    # compute equity metrics
    em = equity_metrics(eq)
    # trade stats
    trades_df = port.snapshot_dataframe()
    ts = trade_stats_from_trades_df(trades_df)
    # fitness: baseline sharpe, penalize large drawdown and high turnover
    sharpe = em.get('Sharpe', -999.0)
    maxdd = em.get('MaxDD', -0.0)
    turnover = ts.get('Turnover', 0.0) if ts.get('Turnover', 0.0) is not None else (ts.get('TotalTraded', 0.0) / (initial_cash+1e-9))
    # penalties
    alpha = 2.0
    beta = 0.5
    fitness = sharpe - alpha * max(0.0, -maxdd) - beta * turnover
    # small tie-breaker: prefer higher CAGR
    fitness += 0.001 * (em.get('CAGR', 0.0) if em.get('CAGR') is not None else 0.0)
    return float(fitness)

# ---------------- Run Simulation / GA Actions ----------------
if run_sim:
    preset_cfg = preset_map[preset_choice]
    df = generate_path(
        n_steps=preset_cfg.get("n_steps", 252),
        seed=preset_cfg.get("seed", 0),
        trans=np.array(preset_cfg["trans"]) if "trans" in preset_cfg else None,
        mu_by_reg=np.array(preset_cfg["mu_by_reg"]) if "mu_by_reg" in preset_cfg else None,
        sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]) if "sigma_by_reg" in preset_cfg else None,
        shock_prob=preset_cfg.get("shock_prob", 0.005),
        muJ=preset_cfg.get("muJ", 0.0),
        sigmaJ=preset_cfg.get("sigmaJ", 0.08),
        S0=100.0
    )

    broker_cfg = {"commission": commission, "spread": spread, "slippage_k": slippage_k, "max_leverage": max_leverage}

    # instantiate agents and track per-agent trades (for marker overlay)
    agents = []
    if run_buyhold:
        agents.append(BuyHold(weight=1.0))
    if run_momentum:
        agents.append(Momentum(fast=20, slow=100, weight=1.0))
    if run_meanrev:
        agents.append(MeanRev(window=20, zentry=1.0, zexit=0.5, weight=1.0))
    if run_voltarget:
        agents.append(VolTarget(ewma_alpha=0.94, target_ann_vol=0.12, max_leverage=max_leverage))

    results = []
    for agent in agents:
        port = Portfolio(initial_cash=100000.0, broker=Broker(**broker_cfg))
        agent.reset()
        price_series = []
        equity = []
        for t, price in enumerate(df['price'].values):
            price_series.append(float(price))
            target = agent.decide(t, price_series, port)
            e, info = port.step(price, float(target))
            equity.append(e)
        res = {
            "agent": agent.name,
            "equity": pd.Series(equity),
            "trades_df": port.snapshot_dataframe(),
            "n_trades": len(port.snapshot_dataframe()[port.snapshot_dataframe()['delta_units'] != 0]),
            "total_traded": float(getattr(port, "total_traded", 0.0)),
            "total_commission": float(getattr(port, "total_commission", 0.0))
        }
        results.append(res)

    # league + display
    league = build_league(results)
    st.subheader("League Table")
    st.dataframe(league.style.format({
        "CAGR": "{:.2%}", "AnnVol": "{:.2%}", "Sharpe": "{:.3f}",
        "MaxDD": "{:.2%}", "Calmar": "{:.3f}", "HitRate": "{:.2%}",
        "TotalTraded": "${:,.0f}", "TotalCommission": "${:,.2f}"
    }))

    # plot market + optional trade markers
    col1, col2 = st.columns([2,3])
    with col1:
        st.subheader("Market Path")
        fig, ax = plt.subplots(figsize=(8,4))
        plot_market(df, title="Market", ax=ax, show=False)
        if show_trade_markers:
            # overlay markers for each agent using exec_price/time from trades_df
            colors = ['g','r','b','m','c','y']
            for i, r in enumerate(results):
                td = r['trades_df']
                if td is None or td.empty:
                    continue
                buy_idx = td[td['delta_units'] > 0].index.values
                sell_idx = td[td['delta_units'] < 0].index.values
                exec_prices = td['exec_price'].values if 'exec_price' in td.columns else td['price'].values
                # plot buys
                if len(buy_idx):
                    ax.scatter(buy_idx, exec_prices[buy_idx], marker='^', color=colors[i%len(colors)], s=40, label=f"{r['agent']} Buy", zorder=6)
                if len(sell_idx):
                    ax.scatter(sell_idx, exec_prices[sell_idx], marker='v', color=colors[i%len(colors)], s=40, label=f"{r['agent']} Sell", zorder=6)
            ax.legend(fontsize='small')
        st.pyplot(fig)
    with col2:
        st.subheader("Equity Curves")
        fig2, ax2 = plt.subplots(figsize=(10,4))
        for r in results:
            ax2.plot(r['equity'].values, label=r['agent'])
        ax2.legend()
        ax2.set_title("Agent Equity Curves")
        st.pyplot(fig2)

    # CSV downloads
    csv_league = league.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("Download League CSV", data=csv_league, file_name="league_table.csv", mime="text/csv")

    eq_df = pd.DataFrame({r['agent']: r['equity'].values for r in results})
    csv_eq = eq_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Equity Curves CSV", data=csv_eq, file_name="equity_curves.csv", mime="text/csv")

# ---------------- Run GA optimization ----------------
if run_ga:
    preset_cfg = preset_map[preset_choice]
    broker_cfg = {"commission": commission, "spread": spread, "slippage_k": slippage_k, "max_leverage": max_leverage}
    agent_name = ga_agent_choice

    st.subheader(f"Running GA for {agent_name}")
    st.write(f"Population: {ga_pop}, Generations: {ga_gens}, Elites: {ga_elite}, MutRate: {ga_mut}")

    gene_bounds = GA_GENE_BOUNDS.get(agent_name)
    if gene_bounds is None:
        st.error("No GA gene bounds for selected agent.")
    else:
        # wrap fitness to match simple_ga signature
        def fitness_wrapper(gene):
            # ensure ints where needed (simple_ga may pass floats)
            gene_fixed = {}
            for k, (lo, hi, is_int) in gene_bounds.items():
                val = gene.get(k, None)
                if val is None:
                    # default random in bounds
                    val = np.random.randint(lo, hi+1) if is_int else np.random.uniform(lo, hi)
                if is_int:
                    gene_fixed[k] = int(round(val))
                else:
                    gene_fixed[k] = float(val)
            # handle trailing_stop=0 => disable (None)
            if 'trailing_stop' in gene_fixed and gene_fixed['trailing_stop'] == 0.0:
                gene_fixed['trailing_stop'] = None
            return fitness_for_gene(gene_fixed, agent_name, preset_cfg, broker_cfg, initial_cash=100000.0, quick_days=252)

        start_time = time.time()
        with st.spinner("Running GA... this may take a while depending on pop/generations"):
            best_gene, history = simple_ga(fitness_wrapper, gene_bounds, pop_size=int(ga_pop), gens=int(ga_gens), elite=int(ga_elite), mut_rate=float(ga_mut), seed=int(ga_seed), verbose=False)
        elapsed = time.time() - start_time
        st.success(f"GA finished in {elapsed:.1f}s. Best fitness: {history[-1]:.4f}")
        st.write("Best gene (raw):")
        st.json(best_gene)

        # show before vs after using the best gene
        st.write("Running pre/post comparison on full preset (2yrs)...")
        # baseline agent (default)
        default_agent = {"Momentum": Momentum(fast=20, slow=100, weight=1.0, trailing_stop=None),
                         "MeanRev": MeanRev(window=20, zentry=1.0, zexit=0.5, weight=1.0),
                         "VolTarget": VolTarget(ewma_alpha=0.94, target_ann_vol=0.12, max_leverage=max_leverage),
                         "BuyHold": BuyHold(weight=1.0)}[agent_name]

        # run baseline
        df_full = generate_path(
            n_steps=preset_cfg.get("n_steps", 252*2),
            seed=preset_cfg.get("seed", 0),
            trans=np.array(preset_cfg["trans"]) if "trans" in preset_cfg else None,
            mu_by_reg=np.array(preset_cfg["mu_by_reg"]) if "mu_by_reg" in preset_cfg else None,
            sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]) if "sigma_by_reg" in preset_cfg else None,
            shock_prob=preset_cfg.get("shock_prob", 0.005),
            muJ=preset_cfg.get("muJ", 0.0),
            sigmaJ=preset_cfg.get("sigmaJ", 0.08),
            S0=100.0
        )

        # run baseline and tuned agent
        def run_agent_full(agent, df, broker_cfg):
            port = Portfolio(initial_cash=100000.0, broker=Broker(**broker_cfg))
            agent.reset()
            price_series = []
            equity = []
            for t, price in enumerate(df['price'].values):
                price_series.append(float(price))
                target = agent.decide(t, price_series, port)
                e, info = port.step(price, float(target))
                equity.append(e)
            trades_df = port.snapshot_dataframe()
            return pd.Series(equity), trades_df

        base_eq, base_td = run_agent_full(default_agent, df_full, broker_cfg)
        tuned_agent = gene_to_agent(agent_name, best_gene)
        tuned_eq, tuned_td = run_agent_full(tuned_agent, df_full, broker_cfg)

        base_metrics = equity_metrics(base_eq)
        tuned_metrics = equity_metrics(tuned_eq)
        st.write("Baseline metrics:")
        st.json(base_metrics)
        st.write("Tuned metrics:")
        st.json(tuned_metrics)

        # plot equity comparison
        fig3, ax3 = plt.subplots(figsize=(10,4))
        ax3.plot(base_eq.values, label="Baseline")
        ax3.plot(tuned_eq.values, label="Tuned")
        ax3.legend()
        ax3.set_title(f"Pre vs Post GA: {agent_name}")
        st.pyplot(fig3)

        # downloadable tuned gene + metrics
        out = {"best_gene": best_gene, "tuned_metrics": tuned_metrics, "baseline_metrics": base_metrics, "history": history}
        json_data = json.dumps(out, indent=2)
        st.download_button("Download GA result (JSON)",data=json_data,file_name=f"ga_{agent_name}_result.json",mime="application/json")
