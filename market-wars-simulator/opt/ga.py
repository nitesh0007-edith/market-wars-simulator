# opt/ga.py
import random
import numpy as np
from copy import deepcopy

def init_population(gene_bounds, pop_size, rng):
    pop = []
    for _ in range(pop_size):
        g = {}
        for k, (lo, hi, is_int) in gene_bounds.items():
            if is_int:
                g[k] = rng.randint(lo, hi)
            else:
                g[k] = rng.uniform(lo, hi)
        pop.append(g)
    return pop

def mutate_gene(gene, gene_bounds, rng, mut_rate=0.2):
    child = deepcopy(gene)
    for k, (lo, hi, is_int) in gene_bounds.items():
        if rng.random() < mut_rate:
            if is_int:
                child[k] = rng.randint(lo, hi)
            else:
                child[k] = rng.uniform(lo, hi)
    return child

def crossover(g1, g2, rng):
    child = {}
    for k in g1.keys():
        child[k] = g1[k] if rng.random() < 0.5 else g2[k]
    return child

def simple_ga(fitness_fn, gene_bounds, pop_size=12, gens=8, elite=2, mut_rate=0.2, seed=0, verbose=False):
    """
    Simple generational GA:
      - gene_bounds: dict of {name: (low,high,is_int)}
      - fitness_fn: callable(gene_dict) -> float (higher better)
    Returns: best_gene, history (best fitness per generation)
    """
    rng = random.Random(seed)
    pop = init_population(gene_bounds, pop_size, rng)
    history = []
    best_gene = None
    best_fit = -1e12

    for gen in range(gens):
        scored = []
        for gene in pop:
            try:
                fit = fitness_fn(gene)
            except Exception as e:
                if verbose:
                    print("Fitness error:", e)
                fit = -1e12
            scored.append((fit, gene))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_fit_gen, best_gene_gen = scored[0]
        history.append(best_fit_gen)
        if best_fit_gen > best_fit:
            best_fit = best_fit_gen
            best_gene = deepcopy(best_gene_gen)
        if verbose:
            print(f"[GA] gen {gen+1}/{gens} best fit {best_fit_gen:.6f}")

        # elitism
        new_pop = [deepcopy(scored[i][1]) for i in range(min(elite, len(scored)))]
        # fill rest
        while len(new_pop) < pop_size:
            # tournament-like selection: pick two parents from top half
            p1 = rng.choice(scored[:max(1, len(scored)//2)])[1]
            p2 = rng.choice(scored[:max(1, len(scored)//2)])[1]
            child = crossover(p1, p2, rng)
            child = mutate_gene(child, gene_bounds, rng, mut_rate=mut_rate)
            new_pop.append(child)
        pop = new_pop
    return best_gene, history
