# demo_run_momentum.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from market.generator import generate_path
from market.visualize import plot_market
from engine.broker import Broker
from engine.portfolio import Portfolio
from agents.momentum import Momentum

# small metrics helper
def compute_metrics_from_equity(eqs):
    rets = eqs.pct_change().fillna(0)
    n = len(eqs)
    years = n / 252.0
    total_return = eqs.iloc[-1] / eqs.iloc[0]
    cagr = total_return ** (1/years) - 1 if years > 0 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / (ann_vol + 1e-12)
    cummax = eqs.cummax()
    dd = (eqs - cummax) / cummax
    maxdd = dd.min()
    calmar = cagr / (-maxdd + 1e-12) if maxdd < 0 else np.nan
    hit_rate = (rets > 0).sum() / max(1, (rets != 0).sum())
    return {
        "CAGR": cagr, "AnnVol": ann_vol, "Sharpe": sharpe,
        "MaxDD": maxdd, "Calmar": calmar, "HitRate": hit_rate
    }

def run_momentum_demo(preset="choppy_sideways"):
    PRESETS = {
        "calm_bull": {
            "seed": 1, "n_steps": 252*2, "shock_prob": 0.002,
            "mu_by_reg": [0.18, -0.06, 0.02]
        },
        "choppy_sideways": {
            "seed": 11, "n_steps": 252*2, "shock_prob": 0.01,
            "trans": [[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7]],
            "mu_by_reg": [0.08, -0.04, 0.0], "sigma_by_reg": [0.16,0.22,0.12]
        },
        "crisis": {
            "seed": 7, "n_steps": 252*2, "shock_prob": 0.03, "muJ": -0.1, "sigmaJ": 0.12,
            "mu_by_reg": [0.06, -0.25, 0.0], "sigma_by_reg": [0.14,0.45,0.12]
        }
    }

    cfg = PRESETS[preset]

    # generate market
    df = generate_path(
        n_steps=cfg.get("n_steps", 252),
        seed=cfg.get("seed", 0),
        trans=np.array(cfg["trans"]) if "trans" in cfg else None,
        mu_by_reg=np.array(cfg["mu_by_reg"]) if "mu_by_reg" in cfg else None,
        sigma_by_reg=np.array(cfg["sigma_by_reg"]) if "sigma_by_reg" in cfg else None,
        shock_prob=cfg.get("shock_prob", 0.005),
        muJ=cfg.get("muJ", 0.0),
        sigmaJ=cfg.get("sigmaJ", 0.08),
        S0=100.0
    )

    # set up execution and portfolio
    broker = Broker(commission=1.0, spread=0.0005, slippage_k=0.003, max_leverage=2.0)
    port = Portfolio(initial_cash=100000.0, broker=broker)

    # instantiate a Momentum agent (tune windows / weight as you like)
    # Note: agent.decide(t, price_series, portfolio) returns target_weight
    agent = Momentum(fast=20, slow=100, weight=1.0)

    # run simulation loop
    price_series = []
    equity_series = []
    agent_price_history = []  # to pass price_series slices
    for t, price in enumerate(df['price'].values):
        price_series.append(price)
        target_w = agent.decide(t, price_series, port)
        equity_after, info = port.step(price, target_w)
        equity_series.append(equity_after)
        agent_price_history.append({
            "step": t, "price": price, "regime": int(df['regime'].iloc[t]),
            "shock": int(df['shock'].iloc[t]), "target_w": target_w,
            "equity": equity_after
        })

    eqs = pd.Series(equity_series)
    metrics = compute_metrics_from_equity(eqs)

    # produce plots
    fig, axs = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    # top: market with regimes
    plot_market(df, title=f"Market (preset={preset})", ax=axs[0], show=False)
    # bottom: equity curve
    axs[1].plot(eqs.values, lw=1.5, label=f"Momentum Equity")
    axs[1].set_title("Equity Curve")
    axs[1].set_ylabel("Equity")
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    # drawdown plot
    cummax = eqs.cummax()
    dd = (eqs - cummax) / cummax
    plt.figure(figsize=(12,3))
    plt.plot(dd.values, label="Drawdown")
    plt.axhline(0, color='k', lw=0.5)
    plt.title("Drawdown")
    plt.ylabel("Drawdown")
    plt.show()

    # save results
    trades_df = pd.DataFrame(agent_price_history)
    trades_df.to_csv("momentum_run_trades.csv", index=False)
    eqs.to_csv("momentum_equity.csv", index=False)

    print("Metrics:")
    for k,v in metrics.items():
        if v is None:
            print(f"  {k}: {v}")
        else:
            # format percentages for readability
            if k in ("CAGR","AnnVol","Sharpe","Calmar"):
                print(f"  {k}: {v:.4f}")
            elif k == "MaxDD":
                print(f"  {k}: {v:.4%}")
            else:
                print(f"  {k}: {v}")

    print("Saved momentum_run_trades.csv and momentum_equity.csv")
    return df, port, metrics

if __name__ == "__main__":
    # change preset to "calm_bull" or "crisis" to try different worlds
    df, portfolio, metrics = run_momentum_demo(preset="choppy_sideways")
