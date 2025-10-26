# demo_compare_agents.py
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

# ---- metrics helper ----
def compute_metrics_from_equity(eqs):
    rets = eqs.pct_change().fillna(0)
    n = len(eqs)
    years = n / 252.0 if n > 0 else np.nan
    total_return = eqs.iloc[-1] / eqs.iloc[0] if eqs.iloc[0] != 0 else np.nan
    cagr = total_return ** (1/years) - 1 if years > 0 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / (ann_vol + 1e-12)
    cummax = eqs.cummax()
    dd = (eqs - cummax) / cummax
    maxdd = dd.min()
    calmar = cagr / (-maxdd + 1e-12) if maxdd < 0 else np.nan
    hit_rate = (rets > 0).sum() / max(1, (rets != 0).sum())
    return {"CAGR": cagr, "AnnVol": ann_vol, "Sharpe": sharpe,
            "MaxDD": maxdd, "Calmar": calmar, "HitRate": hit_rate}

# ---- runner for a single agent ----
def run_agent_on_path(agent, df, broker_cfg, initial_cash=100000.0):
    broker = Broker(**broker_cfg)
    port = Portfolio(initial_cash=initial_cash, broker=broker)
    agent.reset()
    price_series = []
    equity = []
    for t, price in enumerate(df['price'].values):
        price_series.append(float(price))
        try:
            target = agent.decide(t, price_series, port)
        except Exception:
            target = 0.0
        e, info = port.step(price, float(target))
        equity.append(e)
    eqs = pd.Series(equity)
    metrics = compute_metrics_from_equity(eqs)
    # additional stats
    trades_df = port.snapshot_dataframe()
    n_trades = len(trades_df[trades_df['delta_units'] != 0])
    total_traded = getattr(port, "total_traded", trades_df['trade_value'].sum() if 'trade_value' in trades_df else 0.0)
    total_comm = getattr(port, "total_commission", trades_df['commission'].sum() if 'commission' in trades_df else 0.0)
    return {
        "agent": agent.name,
        "equity": eqs,
        "metrics": metrics,
        "trades_df": trades_df,
        "n_trades": int(n_trades),
        "total_traded": float(total_traded),
        "total_commission": float(total_comm)
    }

# ---- compare multiple agents on a preset ----
def compare_agents(preset_cfg, agents_list, broker_cfg=None, initial_cash=100000.0, save_prefix="compare"):
    broker_cfg = broker_cfg or {"commission": 1.0, "spread": 0.0005, "slippage_k": 0.003, "max_leverage": 2.0}
    # generate the market once
    df = generate_path(
        n_steps=preset_cfg.get("n_steps", 252),
        seed=preset_cfg.get("seed", 0),
        trans=np.array(preset_cfg["trans"]) if "trans" in preset_cfg else None,
        mu_by_reg=np.array(preset_cfg["mu_by_reg"]) if "mu_by_reg" in preset_cfg else None,
        sigma_by_reg=np.array(preset_cfg["sigma_by_reg"]) if "sigma_by_reg" in preset_cfg else None,
        shock_prob=preset_cfg.get("shock_prob", 0.005),
        muJ=preset_cfg.get("muJ", 0.0),
        sigmaJ=preset_cfg.get("sigmaJ", 0.08),
        S0=preset_cfg.get("S0", 100.0)
    )

    results = []
    for agent in agents_list:
        out = run_agent_on_path(agent, df, broker_cfg, initial_cash=initial_cash)
        results.append(out)

    # Plot price + equity curves
    fig, axs = plt.subplots(2, 1, figsize=(14,9), sharex=True,
                            gridspec_kw={'height_ratios':[1.2,1]})
    plot_market(df, title=f"Market (preset={preset_cfg.get('name','preset')})", ax=axs[0], show=False)

    # equity overlay
    for res in results:
        axs[1].plot(res['equity'].values, label=res['agent'])
    axs[1].set_ylabel("Equity")
    axs[1].set_title("Agent Equity Curves")
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    # build league table
    rows = []
    for res in results:
        m = res['metrics']
        rows.append({
            "Agent": res['agent'],
            "CAGR": m.get("CAGR"),
            "AnnVol": m.get("AnnVol"),
            "Sharpe": m.get("Sharpe"),
            "MaxDD": m.get("MaxDD"),
            "Calmar": m.get("Calmar"),
            "HitRate": m.get("HitRate"),
            "Trades": res['n_trades'],
            "TotalTraded": res['total_traded'],
            "Commission": res['total_commission']
        })
    league = pd.DataFrame(rows).set_index("Agent")
    # sort by Sharpe then CAGR
    league = league.sort_values(by=["Sharpe", "CAGR"], ascending=False)
    # save outputs
    league.to_csv(f"{save_prefix}_league.csv")
    # save equity curves CSV (one column per agent)
    eq_df = pd.DataFrame({res['agent']: res['equity'].values for res in results})
    eq_df.to_csv(f"{save_prefix}_equity_curves.csv", index=False)

    # print nicely
    pd.options.display.float_format = '{:,.4f}'.format
    print("\n=== League Table (sorted by Sharpe) ===")
    print(league)
    print(f"\nSaved {save_prefix}_league.csv and {save_prefix}_equity_curves.csv\n")

    return df, results, league

# ---- main ----
if __name__ == "__main__":
    # preset to use (change to calm_bull or crisis to test)
    PRESET = {
        "name": "choppy_sideways",
        "seed": 11, "n_steps": 252*2, "shock_prob": 0.01,
        "trans": [[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7]],
        "mu_by_reg": [0.08, -0.04, 0.0], "sigma_by_reg": [0.16,0.22,0.12]
    }

    # instantiate agents (tweak params if needed)
    agents_list = [
        BuyHold(weight=1.0),
        Momentum(fast=20, slow=100, weight=1.0, trailing_stop=None),
        MeanRev(window=20, zentry=1.0, zexit=0.5, weight=1.0),
        VolTarget(ewma_alpha=0.94, target_ann_vol=0.12, max_leverage=2.0)
    ]

    df, results, league = compare_agents(PRESET, agents_list, broker_cfg={"commission":1.0,"spread":0.0005,"slippage_k":0.003,"max_leverage":2.0}, initial_cash=100000.0)
