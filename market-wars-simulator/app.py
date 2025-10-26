# app.py (complete final)
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

# GA (expects simple_ga with on_generation callback)
from opt.ga import simple_ga

# ==================== PAGE CONFIG & STYLING ====================
st.set_page_config(
    layout="wide",
    page_title="Market Wars",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg,#0f172a 0%,#0b1220 100%); }
    [data-testid="stSidebar"]{ background: linear-gradient(180deg,#0b1220 0%,#0f172a 100%); border-right:1px solid #1f2937; }
    .main-header{ background: linear-gradient(90deg,#3b82f6 0%,#8b5cf6 100%); padding:1.4rem; border-radius:12px; margin-bottom:1.2rem; box-shadow:0 10px 30px rgba(59,130,246,0.18); text-align:center;}
    .main-header h1{ color:white; margin:0; font-size:2.4rem; font-weight:800; }
    .main-header p{ color:#e6eefc; margin:0.4rem 0 0; }
    .metric-card{ background:linear-gradient(135deg,#0b1220,#111827); border-radius:8px; padding:0.9rem; border:1px solid #1f2937; color:#e6eefc;}
    .stButton>button{ background:linear-gradient(90deg,#3b82f6,#8b5cf6); color:white; border-radius:8px; padding:0.6rem 1rem; font-weight:700; }
    .stDownloadButton>button{ background:linear-gradient(90deg,#10b981,#059669); color:white; border-radius:8px; padding:0.5rem 0.9rem; font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>📈 Market Wars</h1>
  <p>Strategy Arena • Regimes • Shocks • GA Optimization</p>
</div>
""", unsafe_allow_html=True)

# ==================== PRESET CONFIG ====================
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

# ==================== HELPERS: agent gene <-> instantiation ====================
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

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Configuration")
    preset_choice = st.selectbox("Preset", list(preset_map.keys()), format_func=lambda k: preset_map[k]['name'])
    st.caption(preset_map[preset_choice].get("desc", ""))

    st.markdown("---")
    st.subheader("Execution (Broker)")
    commission = st.number_input("Commission per trade ($)", value=1.0, step=0.5)
    spread = st.number_input("Spread (fraction)", value=0.0005, format="%.6f")
    slippage_k = st.number_input("Slippage k (fraction per |Δw|)", value=0.003, format="%.6f")
    max_leverage = st.number_input("Max leverage", value=2.0, step=0.5)

    st.markdown("---")
    st.subheader("Agents")
    run_buyhold = st.checkbox("Buy & Hold", value=True)
    run_momentum = st.checkbox("Momentum", value=True)
    run_meanrev = st.checkbox("Mean Reversion", value=True)
    run_voltarget = st.checkbox("Vol Targeting", value=True)

    st.markdown("---")
    st.subheader("Display")
    show_trade_markers = st.checkbox("Show per-agent trade markers", value=False)

    st.markdown("---")
    st.subheader("Genetic Algorithm")
    ga_agent_choice = st.selectbox("Agent to optimize", ["Momentum", "MeanRev", "VolTarget", "BuyHold"])
    ga_pop = st.number_input("Population", min_value=4, max_value=64, value=8, step=2)
    ga_gens = st.number_input("Generations", min_value=2, max_value=80, value=8, step=1)
    ga_elite = st.number_input("Elites", min_value=1, max_value=max(1, int(ga_pop)-1), value=2, step=1)
    ga_mut = st.slider("Mutation rate", 0.0, 1.0, 0.2, 0.05)
    ga_seed = st.number_input("GA seed", value=0, step=1)
    per_regime_ga = st.checkbox("Per-regime GA (optimize separately)", value=False)

    st.markdown("---")
    st.button("Run Simulation", key="run_sim")
    st.button("Run GA", key="run_ga")

# ==================== RUN SIMULATION ====================
if st.session_state.get("run_sim"):
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
    st.success("Simulation finished and stored in session.")

# Display simulation results
if st.session_state.get("last_sim"):
    sim = st.session_state['last_sim']
    df = sim['df']; results = sim['results']; league = sim['league']
    st.markdown("## Simulation Results")
    st.dataframe(league.style.format({
        "CAGR":"{:.2%}", "AnnVol":"{:.2%}", "Sharpe":"{:.3f}",
        "MaxDD":"{:.2%}", "Calmar":"{:.3f}", "HitRate":"{:.2%}"
    }), use_container_width=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.download_button("Download League CSV", data=league.reset_index().to_csv(index=False).encode('utf-8'),
                           file_name="league.csv", mime="text/csv")
    with col2:
        eq_df = pd.DataFrame({r['agent']: r['equity'].values for r in results})
        st.download_button("Download Equity CSV", data=eq_df.to_csv(index=False).encode('utf-8'),
                           file_name="equity.csv", mime="text/csv")

    st.markdown("### Market & Equity")
    c1, c2 = st.columns([1,1])
    with c1:
        fig, ax = plt.subplots(figsize=(10,5))
        plot_market(df, title="Market Path", ax=ax, show=False)
        if show_trade_markers:
            colors = ['g','r','b','m','c','y']
            for i, r in enumerate(results):
                td = r['trades_df']
                if td is None or td.empty: continue
                buy_idx = td[td['delta_units']>0].index.values
                sell_idx = td[td['delta_units']<0].index.values
                exec_prices = td['exec_price'].values if 'exec_price' in td.columns else td['price'].values
                if len(buy_idx): ax.scatter(buy_idx, exec_prices[buy_idx], marker='^', color=colors[i%len(colors)], s=40, label=f"{r['agent']} Buy")
                if len(sell_idx): ax.scatter(sell_idx, exec_prices[sell_idx], marker='v', color=colors[i%len(colors)], s=40, label=f"{r['agent']} Sell")
            ax.legend(fontsize='small')
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(10,5))
        colors_eq = ['#60a5fa','#c084fc','#f472b6','#34d399']
        for i, r in enumerate(results):
            ax2.plot(r['equity'].values, label=r['agent'], linewidth=2.5, color=colors_eq[i%len(colors_eq)])
        ax2.legend()
        ax2.set_title("Agent Equity Curves")
        st.pyplot(fig2)

# ==================== GA: global or per-regime ====================
if st.session_state.get("run_ga"):
    preset_cfg = preset_map[preset_choice]
    broker_cfg = {"commission":commission, "spread":spread, "slippage_k":slippage_k, "max_leverage":max_leverage}
    agent_name = ga_agent_choice
    gene_bounds = GA_GENE_BOUNDS.get(agent_name)
    if gene_bounds is None:
        st.error("No GA gene bounds for this agent.")
    else:
        st.markdown("## Genetic Algorithm")
        st.write(f"Optimizing {agent_name} — pop {ga_pop} gens {ga_gens} elite {ga_elite} mut {ga_mut}")

        # normalize gene helper
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

        # GA runner on a price series with live progress
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

        # per-regime branch
        if per_regime_ga:
            st.info("Per-regime GA: splitting full market and optimizing on each regime segment.")
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
                # collect contiguous segments by regime and concat per regime
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
                    st.write(f"Optimizing regime {rid} ({len(price_series)} bars)")
                    best_gene, history = run_ga_on_series(price_series, label=f"Regime {rid}")
                    per_reg_results[int(rid)] = {"best_gene": best_gene, "history": history}
                    out_path = os.path.join("ga_results", f"{agent_name}_{preset_choice}_regime_{rid}.json")
                    with open(out_path, "w") as f:
                        json.dump({"agent":agent_name,"preset":preset_choice,"regime":int(rid),"best_gene":best_gene,"history":history}, f, indent=2)
                    st.success(f"Saved {out_path}")

                st.session_state['last_ga_result'] = {"agent":agent_name,"preset":preset_choice,"per_regime":True,"results":per_reg_results,"config":{"pop":ga_pop,"gens":ga_gens,"elite":ga_elite,"mut":ga_mut,"seed":ga_seed}}
                st.success("Per-regime GA complete and saved.")

        # global GA branch
        else:
            st.info("Global GA: optimizing on a 1-year generated path (fast fitness).")
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
            st.success("Global GA finished and saved in session.")

# ==================== GA RESULT DISPLAY & APPLY PER-REGIME ====================
ga_res = st.session_state.get('last_ga_result', None)
if ga_res:
    st.markdown("## GA Results")
    st.json({k:v for k,v in ga_res.items() if k != 'results'})

    # show histories
    if ga_res.get('per_regime'):
        for rid, info in ga_res['results'].items():
            fig, ax = plt.subplots(figsize=(6,2))
            ax.plot(info.get('history', []), marker='o')
            ax.set_title(f"Regime {rid} fitness")
            st.pyplot(fig)
    else:
        hist = ga_res.get('history', [])
        if hist:
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(hist, marker='o', color='#60a5fa')
            ax.set_title("GA best fitness per generation")
            st.pyplot(fig)

    st.download_button("Download GA session JSON", data=json.dumps(ga_res, indent=2), file_name=f"ga_{ga_res.get('agent')}_{ga_res.get('preset')}.json", mime="application/json")

    # Apply per-regime genes live (if per-regime present)
    if ga_res.get('per_regime'):
        if st.button("Apply per-regime genes (live) & run regime-aware sim"):
            preset_cfg = preset_map[preset_choice]
            broker_cfg = st.session_state.get('last_sim', {}).get('broker_cfg', {"commission":commission,"spread":spread,"slippage_k":slippage_k,"max_leverage":max_leverage})
            df_full = generate_path(
                n_steps=preset_cfg.get("n_steps",252*2), seed=preset_cfg.get("seed",0),
                trans=np.array(preset_cfg["trans"]), mu_by_reg=np.array(preset_cfg["mu_by_reg"]),
                sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]), shock_prob=preset_cfg.get("shock_prob",0.005),
                muJ=preset_cfg.get("muJ",0.0), sigmaJ=preset_cfg.get("sigmaJ",0.08), S0=100.0
            ).reset_index(drop=True)

            if 'regime' not in df_full.columns:
                st.error("Market missing 'regime' column. Can't run regime-aware sim.")
            else:
                st.info("Running regime-aware sim...")
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
                    # reinit agent when regime changes
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
                st.success("Regime-aware simulation finished.")
                st.line_chart(eqs)
                st.json(metrics)
                df_out = pd.DataFrame({"price": df_full['price'].values, "regime": df_full['regime'].values, "equity": eqs.values})
                st.download_button("Download regime-aware CSV", data=df_out.to_csv(index=False).encode('utf-8'), file_name=f"regime_aware_{ga_res['agent']}_{preset_choice}.csv", mime="text/csv")
                # ---------------- Glossary (user-accessible) ----------------

# ---------------- Glossary (user-accessible) ----------------


GLOSSARY = {
    "CAGR": {
        "short": "Compound Annual Growth Rate — annualized compounded return.",
        "body": (
            "**Definition:** Annualized compounded return over the period.\n\n"
            "**Formula:** `(FinalEquity/InitialEquity)^(252 / nDays) - 1`.\n\n"
            "**Why it matters:** Normalizes growth to an annual rate so you can compare strategies run over different lengths. "
            "Good for long-term performance comparisons."
        )
    },
    "AnnVol": {
        "short": "Annualized Volatility — standard deviation of returns scaled to a year.",
        "body": (
            "**Definition:** Standard deviation of daily returns × √252.\n\n"
            "**Why it matters:** Measures variability (risk). Lower vol for the same return is better."
        )
    },
    "Sharpe": {
        "short": "Return per unit risk (Ann. Sharpe).",
        "body": (
            "**Definition:** `(CAGR - r_f) / AnnVol` (we set risk-free `r_f = 0` by default).\n\n"
            "**Why it matters:** Widely-used risk-adjusted performance metric—higher is better. "
            "Good baseline for GA fitness but should be combined with drawdown checks."
        )
    },
    "MaxDD": {
        "short": "Maximum Drawdown — largest peak-to-trough loss.",
        "body": (
            "**Definition:** Largest percentage decline from a peak to subsequent trough during the period.\n\n"
            "**Why it matters:** Captures the worst equity shock — crucial for investor psychology and capital preservation."
        )
    },
    "Calmar": {
        "short": "CAGR / |MaxDD| — return adjusted for drawdown.",
        "body": (
            "**Definition:** `CAGR / |MaxDD|`.\n\n"
            "**Why it matters:** Focuses on return relative to the worst drawdown — useful to judge 'return per unit of pain'."
        )
    },
    "HitRate": {
        "short": "Proportion of winning trades.",
        "body": (
            "**Definition:** `Wins / TotalTrades`.\n\n"
            "**Why it matters:** Shows strategy consistency. Note: a high hit rate alone doesn't imply profitability (pay attention to average win vs loss)."
        )
    },
    "Turnover": {
        "short": "Total traded value relative to capital.",
        "body": (
            "**Definition:** `Total traded value / initial capital`.\n\n"
            "**Why it matters:** Proxy for trading intensity — high turnover increases costs and market impact. Penalized in our GA."
        )
    },
    "TotalTraded": {
        "short": "Sum of all trade values.",
        "body": "Reflects how much notional the strategy traded in total. Useful for liquidity assessment."
    },
    "TotalCommission": {
        "short": "Sum of commissions paid.",
        "body": "Real cash cost of trading; used to calculate net returns."
    },
    "Slippage": {
        "short": "Execution price movement vs ideal price.",
        "body": (
            "**Definition:** Modeled here as `k * |Δweight|` applied to execution price.\n\n"
            "**Why:** Models market impact—important for larger orders and high-turnover strategies."
        )
    },
    "Fitness (GA)": {
        "short": "Objective function used by the Genetic Algorithm.",
        "body": (
            "**Definition (default):** `Fitness = Sharpe - α * max(0, -MaxDD) - β * Turnover`.\n\n"
            "**Why:** Balances reward (Sharpe) against risk (drawdown) and cost (turnover). "
            "α and β are tunable to prioritize drawdown aversion vs trading efficiency."
        )
    },
    "Per-Regime GA": {
        "short": "Run GA separately for each market regime.",
        "body": (
            "**Definition:** Split the market by regime (Bull/Bear/Flat) and optimize parameters independently for each slice.\n\n"
            "**Why:** Markets are non-stationary — parameters that work in Bull may fail in Bear. "
            "Per-regime GA produces adaptive agents that change behavior with the regime."
        )
    },
    "Rolling Sharpe / Ulcer Index": {
        "short": "Stability & downside pain metrics.",
        "body": (
            "Rolling Sharpe shows performance stability through time. Ulcer Index captures drawdown depth and duration — more focused on downside risk than volatility."
        )
    },
    "VaR (optional)": {
        "short": "Value at Risk — extreme loss quantile.",
        "body": "Used for tail-risk assessment. Not enabled by default, but easy to add for stress-testing."
    },
    "Why These Metrics Together": {
        "short": "Combining multiple metrics gives a robust picture.",
        "body": (
            "Sharpe measures average risk-adjusted returns, MaxDD measures worst capital loss, Calmar relates return to drawdown, "
            "and Turnover/Commission capture operational costs. The GA uses a composite fitness so strategies can't 'game' a single metric."
        )
    }
}

def _render_glossary_ui():
    st.markdown("## 📘 Glossary — Metrics & Optimization Terms")
    st.write("Search or expand terms to learn what each metric means and why it matters for trading strategies.")

    # search box
    query = st.text_input("Search glossary (type a metric or keyword)", value="", help="Try 'Sharpe', 'drawdown', 'turnover'")

    # Build filtered list
    keys = list(GLOSSARY.keys())
    if query:
        q = query.strip().lower()
        keys = [k for k in keys if q in k.lower() or q in GLOSSARY[k]['short'].lower() or q in GLOSSARY[k]['body'].lower()]

    # Download full glossary as markdown
    if st.button("Download full glossary (Markdown)"):
        md = "# Market Wars Glossary\n\n"
        for k in GLOSSARY.keys():
            md += f"## {k}\n\n**Short:** {GLOSSARY[k]['short']}\n\n{GLOSSARY[k]['body']}\n\n"
        st.download_button("Download glossary.md", data=md, file_name="glossary.md", mime="text/markdown")

    # render each as an expander
    for k in keys:
        with st.expander(f"{k} — {GLOSSARY[k]['short']}", expanded=False):
            st.markdown(GLOSSARY[k]['body'])


# Sidebar toggle for glossary — single checkbox with unique key
if 'show_glossary' not in st.session_state:
    st.session_state['show_glossary'] = False

# Use one checkbox, give it a unique key so Streamlit doesn't auto-duplicate IDs
show_glossary_widget_value = st.sidebar.checkbox(
    "Show Glossary (metrics & GA)",
    value=st.session_state.get('show_glossary', False),
    key="show_glossary_checkbox"
)

# Sync the widget value into session_state so the rest of the app can use it
st.session_state['show_glossary'] = bool(show_glossary_widget_value)


# Render glossary in main area if show_glossary True
if st.session_state.get('show_glossary', False):
    st.markdown("---")
    _render_glossary_ui()
    st.markdown("---")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<div style='text-align:center;color:#94a3b8;padding:1rem 0;'>Market Wars Simulator — Stage 4+ — Per-regime GA & live application</div>", unsafe_allow_html=True)
