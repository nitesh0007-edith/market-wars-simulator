# opt/ga.py
"""
Simple Genetic Algorithm utilities for Market Wars.
Provides a generational GA with an optional on_generation callback
that receives (gen_index, best_fitness_in_gen, best_gene_in_gen).
"""

import random
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

def simple_ga(fitness_fn,
              gene_bounds,
              pop_size=12,
              gens=8,
              elite=2,
              mut_rate=0.2,
              seed=0,
              verbose=False,
              on_generation=None):
    """
    Simple generational GA with optional on_generation callback.

    Parameters
    ----------
    fitness_fn : callable(gene_dict) -> float
        Returns a scalar fitness (higher is better).
    gene_bounds : dict
        { 'param': (low, high, is_int), ... }
    pop_size : int
    gens : int
    elite : int
    mut_rate : float
    seed : int
    verbose : bool
    on_generation : callable(gen_index, best_fit_in_gen, best_gene_in_gen) or None

    Returns
    -------
    best_gene : dict
    history : list of best_fitness per generation
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
            scored.append((fit, deepcopy(gene)))

        # sort by fitness desc
        scored.sort(key=lambda x: x[0], reverse=True)
        best_fit_gen, best_gene_gen = scored[0]
        history.append(float(best_fit_gen))

        # update global best
        if best_fit_gen > best_fit:
            best_fit = best_fit_gen
            best_gene = deepcopy(best_gene_gen)

        if verbose:
            print(f"[GA] gen {gen+1}/{gens} best_fit={best_fit_gen:.6f}")

        # callback for UI progress / logging
        if on_generation is not None:
            try:
                on_generation(gen, float(best_fit_gen), deepcopy(best_gene_gen))
            except Exception:
                # never allow callback to stop the algorithm
                if verbose:
                    print("on_generation callback raised an exception, continuing...")

        # elitism: copy top `elite` genes unchanged
        new_pop = [deepcopy(scored[i][1]) for i in range(min(elite, len(scored)))]

        # fill with children
        while len(new_pop) < pop_size:
            # pick parents from top half (tournament-like)
            top_half = scored[:max(1, len(scored)//2)]
            p1 = random.choice(top_half)[1]
            p2 = random.choice(top_half)[1]
            child = crossover(p1, p2, random)
            child = mutate_gene(child, gene_bounds, random, mut_rate=mut_rate)
            new_pop.append(child)

        pop = new_pop

    return best_gene, history
