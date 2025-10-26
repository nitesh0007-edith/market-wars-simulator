# opt/ga_regime.py
"""
Per-regime GA runner:
- Splits generated path into regime slices
- Optimizes agent parameters separately on each regime slice
- Saves best genes per regime to disk (via JSON)
"""

import os
import json
from opt.ga import simple_ga
from copy import deepcopy

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def run_ga_on_slice(fitness_fn, gene_bounds, pop_size=8, gens=8, elite=2, mut_rate=0.2, seed=0, on_generation=None):
    # simple wrapper reusing simple_ga; returns best_gene, history
    best_gene, history = simple_ga(fitness_fn, gene_bounds, pop_size=pop_size, gens=gens,
                                   elite=elite, mut_rate=mut_rate, seed=seed, verbose=False,
                                   on_generation=on_generation)
    return best_gene, history

def optimize_per_regime(agent_name, preset_cfg, build_fitness_fn, gene_bounds,
                        pop_size=8, gens=8, elite=2, mut_rate=0.2, seed=0,
                        out_dir="ga_results", on_generation=None):
    """
    agent_name: str
    preset_cfg: preset dict used to generate market
    build_fitness_fn(regime_slice_df, agent_name, gene) -> fitness float
      - called for each regime slice with gene dict
    gene_bounds: bounds dict as used by simple_ga
    Returns: dict of {regime_name: {best_gene:..., history:[...]}}
    """
    _ensure_dir(out_dir)
    results = {}
    # generate full path
    from market.generator import generate_path
    df = generate_path(
        n_steps=preset_cfg.get("n_steps", 252*2),
        seed=preset_cfg.get("seed", 0),
        trans=preset_cfg.get("trans"),
        mu_by_reg=preset_cfg.get("mu_by_reg"),
        sigma_by_reg=preset_cfg.get("sigma_by_reg"),
        shock_prob=preset_cfg.get("shock_prob", 0.005),
        muJ=preset_cfg.get("muJ", 0.0),
        sigmaJ=preset_cfg.get("sigmaJ", 0.08),
        S0=preset_cfg.get("S0", 100.0)
    )
    # split by contiguous regimes
    df = df.reset_index(drop=True)
    df['idx'] = df.index
    slices = []
    if 'regime' not in df.columns:
        raise ValueError("market.generator must include 'regime' column")
    start = 0
    for i in range(1, len(df)):
        if df.loc[i, 'regime'] != df.loc[start, 'regime']:
            slices.append(df.loc[start:i-1].copy())
            start = i
    slices.append(df.loc[start:len(df)-1].copy())

    # group slices by regime id
    by_regime = {}
    for s in slices:
        rid = int(s['regime'].iloc[0])
        by_regime.setdefault(rid, []).append(s)

    # for each regime, create a concatenated slice (optionally multiple slices can be combined)
    for regime, segs in by_regime.items():
        # combine segments into one df to maximize sample length
        reg_df = segs[0].copy()
        if len(segs) > 1:
            reg_df = pd.concat(segs, ignore_index=True)
        # build fitness closure that runs fitness_fn on this regime slice
        def fitness_gene_wrapper(gene):
            return build_fitness_fn(reg_df, agent_name, gene)
        # run GA on this regime slice
        best_gene, history = run_ga_on_slice(fitness_gene_wrapper, gene_bounds,
                                             pop_size=pop_size, gens=gens, elite=elite, mut_rate=mut_rate, seed=seed,
                                             on_generation=on_generation)
        results[regime] = {"best_gene": best_gene, "history": history}
        # persist to disk
        out_path = os.path.join(out_dir, f"{agent_name}_preset_{preset_cfg.get('name','preset')}_regime_{regime}.json")
        with open(out_path, "w") as f:
            json.dump({"agent": agent_name, "preset": preset_cfg.get('name','preset'), "regime":int(regime),
                       "best_gene": best_gene, "history": history}, f, indent=2)
    return results
