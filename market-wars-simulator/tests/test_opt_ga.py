# tests/test_opt_ga.py
import os
import json
import tempfile
from opt.ga import simple_ga

def test_simple_ga_runs_and_calls_callback():
    # trivial fitness: maximize sum of gene values
    gene_bounds = {
        "a": (0, 10, True),
        "b": (0.0, 1.0, False)
    }

    def fitness_fn(g):
        return float(g['a']) + float(g['b'])

    called = {"count": 0}
    def on_gen(idx, best_fit, best_gene):
        # should be called gens times
        called["count"] += 1
        assert isinstance(best_fit, float)
        assert isinstance(best_gene, dict)

    best_gene, history = simple_ga(fitness_fn, gene_bounds, pop_size=6, gens=4, elite=1, mut_rate=0.2, seed=1, on_generation=on_gen)
    assert isinstance(best_gene, dict)
    assert len(history) == 4
    assert called["count"] == 4
