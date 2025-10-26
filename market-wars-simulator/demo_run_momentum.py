# demo_analyze_momentum.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from market.generator import generate_path
from market.visualize import plot_market
from engine.broker import Broker
from engine.portfolio import Portfolio
from agents.momentum import Momentum

def compute_metrics_from_equity(eqs):
    rets = eqs.pct_change().fillna(0)
    n = len(eqs)
    years = n / 252.0
    total_return = eqs.iloc[-1] / eqs.iloc[0] if eqs.iloc[0] != 0 else np.nan
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
        "MaxDD": maxdd, "Calmar": calmar, "HitRate": hit_rate,
        "AvgDailyRet": rets.mean()
    }

def run_and_analyze(preset_cfg, agent, broker_cfg=None, initial_cash=100000.0,
                    out_trades="momentum_run_trades.csv", out_eq="momentum_equity.csv"):
    # generate market
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

    broker_cfg = broker_cfg or {"commission": 1.0, "spread": 0.0005, "slippage_k": 0.003, "max_leverage": 2.0}
    broker = Broker(**broker_cfg)
    port = Portfolio(initial_cash=initial_cash, broker=broker)

    price_series = []
    equity_series = []
    # run sim
    for t, price in enumerate(df['price'].values):
        price_series.append(float(price))
        target_w = agent.decide(t, price_series, port)
        equity_after, info = port.step(price, target_w)
        equity_series.append(equity_after)

    eqs = pd.Series(equity_series)
    metrics = compute_metrics_from_equity(eqs)

    # save trade history from portfolio
    trades_df = port.snapshot_dataframe().copy()
    trades_df['step'] = trades_df.index
    trades_df.to_csv(out_trades, index=False)
    eqs.to_csv(out_eq, index=False)

    # --- PLOT: market (top) + equity (bottom) with trade markers overlay ---
    fig, axs = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    plot_market(df, title="Market (preset)", ax=axs[0], show=False)

    # equity curve
    axs[1].plot(eqs.values, lw=1.5, label=f"Momentum Equity")
    axs[1].set_title("Equity Curve")
    axs[1].set_ylabel("Equity")

    # plot buy/sell markers using trades_df delta_units
    if not trades_df.empty:
        buy_idx = trades_df[trades_df['delta_units'] > 0].index.values
        sell_idx = trades_df[trades_df['delta_units'] < 0].index.values
        # get prices at those steps (use exec_price when available)
        exec_prices = trades_df['exec_price'].values
        # plot on top axis (price chart)
        # convert steps to x coords relative to price series
        price_ax = axs[0]
        # buys: green triangle up
        if len(buy_idx):
            x_buys = buy_idx
            y_buys = exec_prices[buy_idx]
            price_ax.scatter(x_buys, y_buys, marker='^', color='g', s=70, label='Buy', zorder=6)
        if len(sell_idx):
            x_sells = sell_idx
            y_sells = exec_prices[sell_idx]
            price_ax.scatter(x_sells, y_sells, marker='v', color='r', s=70, label='Sell', zorder=6)

        # also annotate number of trades on equity plot
        axs[1].text(0.01, 0.95, f"Trades: {len(buy_idx)+len(sell_idx)} | Total Traded Value: ${port.total_traded:,.0f}\nTotal Commission: ${port.total_commission:,.2f}",
                    transform=axs[1].transAxes, fontsize=9, va='top', bbox=dict(boxstyle="round", fc="wheat", alpha=0.4))

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

    # print the compact performance summary
    print("\n=== Performance Summary ===")
    print(f"Start equity: ${float(eqs.iloc[0]):,.2f}")
    print(f"End equity:   ${float(eqs.iloc[-1]):,.2f}")
    print(f"Net P/L:      ${float(eqs.iloc[-1] - eqs.iloc[0]):,.2f}")
    print(f"Trades:       {len(trades_df[trades_df['delta_units']!=0])}")
    print(f"Total traded value: ${port.total_traded:,.0f}")
    print(f"Total commission:   ${port.total_commission:,.2f}")
    print("")
    # metrics nicely formatted
    def pf(k,v):
        if k in ("CAGR","AnnVol","Sharpe","Calmar"):
            return f"{v:.4f}"
        if k == "MaxDD":
            return f"{v:.2%}"
        if k == "HitRate":
            return f"{v:.2%}"
        return f"{v}"
    for k in ("CAGR","AnnVol","Sharpe","MaxDD","Calmar","HitRate"):
        print(f"{k:6s}: {pf(k, metrics.get(k))}")
    print("===========================\n")

    return df, port, trades_df, eqs, metrics

if __name__ == "__main__":
    """
    PRESET = {
        "name": "choppy_sideways",
        "seed": 11, "n_steps": 252*2, "shock_prob": 0.01,
        "trans": [[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7]],
        "mu_by_reg": [0.08, -0.04, 0.0], "sigma_by_reg": [0.16,0.22,0.12]
    }
    agent = Momentum(fast=20, slow=100, weight=1.0, trailing_stop=None)
    df, port, trades_df, eqs, metrics = run_and_analyze(PRESET, agent)
    """
    
    # Calm Bull market preset — smooth upward regime
    PRESET_CALM_BULL = {
        "name": "calm_bull",
        "seed": 1,
        "n_steps": 252 * 2,
        "shock_prob": 0.002,
        "trans": [[0.94,0.03,0.03],
                  [0.10,0.80,0.10],
                  [0.15,0.15,0.70]],
        "mu_by_reg": [0.18, -0.06, 0.02],
        "sigma_by_reg": [0.12, 0.30, 0.08]
    }

    # Same momentum agent — unchanged logic
    agent = Momentum(fast=20, slow=100, weight=1.0, trailing_stop=None)

    # Run simulation + visualization + performance report
    df, port, trades_df, eqs, metrics = run_and_analyze(PRESET_CALM_BULL, agent)
