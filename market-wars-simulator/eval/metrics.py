# eval/metrics.py
import numpy as np
import pandas as pd

def equity_metrics(equity_series, trading_days=252):
    """
    Compute standard annualized metrics from an equity pd.Series (index = step order).
    Returns dict with CAGR, AnnVol, Sharpe, MaxDD, Calmar, HitRate.
    """
    eq = pd.Series(equity_series).reset_index(drop=True).astype(float)
    if eq.empty:
        return {"CAGR": np.nan, "AnnVol": np.nan, "Sharpe": np.nan,
                "MaxDD": np.nan, "Calmar": np.nan, "HitRate": np.nan}

    rets = eq.pct_change().fillna(0.0)
    n = len(eq)
    years = n / float(trading_days) if n > 0 else np.nan

    # CAGR
    start = float(eq.iloc[0]) if eq.iloc[0] != 0 else np.nan
    end = float(eq.iloc[-1])
    total_ret = (end / start) if start and not np.isnan(start) else np.nan
    cagr = total_ret ** (1.0 / years) - 1.0 if years > 0 and not np.isnan(total_ret) else np.nan

    ann_vol = rets.std(ddof=0) * np.sqrt(trading_days)
    sharpe = (rets.mean() * trading_days) / (ann_vol + 1e-12)

    cummax = eq.cummax()
    drawdowns = (eq - cummax) / cummax
    maxdd = drawdowns.min()

    calmar = cagr / (-maxdd + 1e-12) if maxdd < 0 and not np.isnan(cagr) else np.nan

    hit_rate = (rets > 0).sum() / max(1, (rets != 0).sum())

    return {
        "CAGR": float(cagr),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(maxdd),
        "Calmar": float(calmar),
        "HitRate": float(hit_rate)
    }

def trade_stats_from_trades_df(trades_df):
    """
    trades_df: DataFrame produced by Portfolio.snapshot_dataframe()
    Expected columns: delta_units, exec_price, trade_value, commission (may vary)
    Returns dict: Trades, TotalTraded, TotalCommission, Turnover (total abs delta weight if available)
    """
    if trades_df is None or trades_df.empty:
        return {"Trades": 0, "TotalTraded": 0.0, "TotalCommission": 0.0, "Turnover": 0.0}

    df = trades_df.copy()
    # number of executed trades where delta_units != 0
    trades = int((df['delta_units'].fillna(0) != 0).sum())

    # trade_value column (engine.portfolio sets trade_value)
    total_traded = float(df['trade_value'].sum()) if 'trade_value' in df.columns else float(df.get('trade_value', 0.0).sum() if hasattr(df, 'trade_value') else 0.0)

    # commission column
    if 'commission' in df.columns:
        total_comm = float(df['commission'].sum())
    else:
        total_comm = 0.0

    # turnover: if delta_abs_weight exists, sum it; else approximate from trade_value / avg equity
    turnover = float(df['delta_abs_weight'].sum()) if 'delta_abs_weight' in df.columns else np.nan

    return {"Trades": trades, "TotalTraded": total_traded, "TotalCommission": total_comm, "Turnover": turnover}

def build_league(results, trading_days=252):
    """
    Convert a list of result dicts (from demo runner) into a league table DataFrame.
    Each result dict expected to contain:
      - 'agent' (name)
      - 'equity' (pd.Series)
      - 'trades_df' (pd.DataFrame) OR portfolio-level totals in keys 'total_traded','total_commission','n_trades'
    Returns league DataFrame indexed by Agent and sorted by Sharpe descending.
    """
    rows = []
    for r in results:
        name = r.get('agent') or r.get('name') or "Agent"
        eq = r.get('equity')
        metrics = equity_metrics(eq, trading_days=trading_days)
        # trades
        trades_df = r.get('trades_df', None)
        trade_stats = trade_stats_from_trades_df(trades_df)
        # fallback if runner provided totals
        n_trades = r.get('n_trades', trade_stats.get('Trades', 0))
        total_traded = r.get('total_traded', trade_stats.get('TotalTraded', 0.0))
        total_comm = r.get('total_commission', trade_stats.get('TotalCommission', 0.0))
        turnover = trade_stats.get('Turnover', r.get('turnover', np.nan))

        row = {
            "Agent": name,
            "CAGR": metrics.get('CAGR'),
            "AnnVol": metrics.get('AnnVol'),
            "Sharpe": metrics.get('Sharpe'),
            "MaxDD": metrics.get('MaxDD'),
            "Calmar": metrics.get('Calmar'),
            "HitRate": metrics.get('HitRate'),
            "Trades": int(n_trades),
            "TotalTraded": float(total_traded),
            "TotalCommission": float(total_comm),
            "Turnover": float(turnover) if not np.isnan(turnover) else np.nan
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Agent")
    # sort by Sharpe then CAGR
    df = df.sort_values(by=["Sharpe", "CAGR"], ascending=False)
    return df
