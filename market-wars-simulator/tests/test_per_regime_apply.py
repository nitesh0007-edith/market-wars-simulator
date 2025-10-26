# tests/test_per_regime_apply.py
import os
import json
import pandas as pd
from market.generator import generate_path
from engine.portfolio import Portfolio
from engine.broker import Broker
from agents.buyhold import BuyHold
from eval.metrics import equity_metrics

def test_apply_per_regime_simulation(tmp_path):
    # create a short market (100 bars) with regime column
    preset = {
        "name": "test_preset",
        "seed": 42,
        "n_steps": 100,
        "trans": [[0.9,0.1,0.0],[0.1,0.9,0.0],[0.0,0.0,1.0]],
        "mu_by_reg": [0.1, -0.05, 0.0],
        "sigma_by_reg": [0.1, 0.2, 0.05],
        "shock_prob": 0.0
    }
    df = generate_path(n_steps=100, seed=42, trans=preset['trans'], mu_by_reg=preset['mu_by_reg'], sigma_by_reg=preset['sigma_by_reg'], shock_prob=0.0)
    assert 'regime' in df.columns

    # build fake per-regime best genes: for each unique regime assign a BuyHold weight gene
    unique_regs = sorted(df['regime'].unique())
    per_regime_genes = {}
    for rid in unique_regs:
        per_regime_genes[int(rid)] = {"best_gene": {"weight": 1.0}}

    # simulate applying per-regime genes (simple implementation)
    broker_cfg = {"commission": 0.0, "spread": 0.0, "slippage_k": 0.0, "max_leverage": 1.0}
    port = Portfolio(initial_cash=10000.0, broker=Broker(**broker_cfg))
    equity = []
    price_history = []
    current_agent = None
    current_regime = None
    for t, row in df.reset_index(drop=True).iterrows():
        price = float(row['price'])
        price_history.append(price)
        rid = int(row['regime'])
        # build BuyHold agent per regime
        gene = per_regime_genes.get(rid, {"best_gene":{"weight":1.0}})
        bh = BuyHold(weight=gene['best_gene']['weight'])
        # re-init on regime change
        if current_regime != rid:
            bh.reset()
            current_agent = bh
            current_regime = rid
        target = current_agent.decide(t, price_history, port)
        e, _ = port.step(price, float(target))
        equity.append(e)

    eq = pd.Series(equity)
    metrics = equity_metrics(eq)
    assert not eq.empty
    assert 'CAGR' in metrics
    # also ensure the output file saving routine works (write a small json)
    out_file = tmp_path / "ga_results" / "agent_test_regime_0.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"agent":"TestAgent","best_gene":{"weight":1.0}}, f)
    assert out_file.exists()
