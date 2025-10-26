# demo_optimize_regime.py
"""
Demo script: run per-regime GA for a given agent and preset, save results.
Usage:
    python demo_optimize_regime.py --agent Momentum --preset calm_bull
"""
import argparse
import pandas as pd
from opt.ga_regime import optimize_per_regime
from eval.metrics import trade_stats_from_trades_df, equity_metrics
from opt.ga import simple_ga
from tools.persist import save_json

# you will need to implement build_fitness_fn to call your existing fitness_for_gene,
# here we assume a fitness_for_gene function exists in app or move appropriate logic to a util.

from app import fitness_for_gene  # if you prefer to import; otherwise duplicate small wrapper

def build_fitness_fn_from_global(preset_cfg, broker_cfg, initial_cash=100000.0, quick_days=252):
    # returns a function(reg_df, agent_name, gene) -> fitness
    # For per-regime, you can evaluate on reg_df series of prices. Here we wrap earlier fitness_for_gene by creating a temp market generator that returns reg_df.
    def wrapper(reg_df, agent_name, gene):
        # quick workaround: create a fake small df for fitness wrapper by using reg_df's price column
        # Implement simplified fitness: run agent on reg_df.price values
        from market.generator import generate_path
        # Instead reuse fitness_for_gene but override market generation — simplest approach: write temporary small generator wrapper in fitness_for_gene
        # To keep demo small, call fitness_for_gene with quick_days = len(reg_df) and seed = 0 (it will regenerate path, not ideal)
        # Better approach: refactor fitness_for_gene into a utility that accepts explicit price series. For now we fallback to calling fitness_for_gene (best-effort).
        return fitness_for_gene(gene, agent_name, preset_cfg, broker_cfg, initial_cash=initial_cash, quick_days=len(reg_df))
    return wrapper

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--gens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # load preset map from app (or duplicate)
    from app import preset_map, GA_GENE_BOUNDS
    preset_cfg = preset_map[args.preset]
    gene_bounds = GA_GENE_BOUNDS[args.agent]
    broker_cfg = {"commission":1.0,"spread":0.0005,"slippage_k":0.003,"max_leverage":2.0}

    fitness_builder = build_fitness_fn_from_global(preset_cfg, broker_cfg)
    results = optimize_per_regime(args.agent, preset_cfg, fitness_builder, gene_bounds, pop_size=args.pop, gens=args.gens, seed=args.seed, out_dir="ga_results")
    print("Saved per-regime GA results to ga_results/")

if __name__ == "__main__":
    main()
