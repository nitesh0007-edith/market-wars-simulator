# demo_run_market.py
import argparse
import numpy as np
import pandas as pd
from market.generator import generate_path
from market.visualize import plot_market

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
        "seed": 42, "n_steps": 252*2, "shock_prob": 0.03, "muJ": -0.1, "sigmaJ": 0.12,
        "mu_by_reg": [0.06, -0.25, 0.0], "sigma_by_reg": [0.14,0.45,0.12]
    }
}

def run_and_save(cfg, out_csv="demo_market_path.csv", title=None):
    df = generate_path(
        n_steps=cfg.get("n_steps", 252),
        seed=cfg.get("seed", 0),
        trans=np.array(cfg["trans"]) if "trans" in cfg else None,
        mu_by_reg=np.array(cfg["mu_by_reg"]) if "mu_by_reg" in cfg else None,
        sigma_by_reg=np.array(cfg["sigma_by_reg"]) if "sigma_by_reg" in cfg else None,
        shock_prob=cfg.get("shock_prob", 0.005),
        muJ=cfg.get("muJ", 0.0),
        sigmaJ=cfg.get("sigmaJ", 0.08),
        S0=cfg.get("S0", 100.0)
    )
    df.to_csv(out_csv, index=False)
    plot_market(df, title=title or "Synthetic Market Path")
    print(f"Saved {out_csv}")
    return df

def main():
    p = argparse.ArgumentParser(description="Generate & visualize synthetic market paths")
    p.add_argument("--preset", type=str, default="choppy_sideways",
                   choices=list(PRESETS.keys()), help="preset scenario")
    p.add_argument("--seed", type=int, default=None, help="override preset seed")
    p.add_argument("--steps", type=int, default=None, help="override number of steps")
    p.add_argument("--out", type=str, default="demo_market_path.csv", help="output CSV filename")
    args = p.parse_args()

    cfg = PRESETS[args.preset].copy()
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.steps is not None:
        cfg['n_steps'] = args.steps

    title = f"Preset: {args.preset} (seed={cfg.get('seed')})"
    run_and_save(cfg, out_csv=args.out, title=title)

if __name__ == "__main__":
    main()
