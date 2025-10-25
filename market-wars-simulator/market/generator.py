# market/generator.py
import numpy as np
import pandas as pd

def generate_path(n_steps=252, seed=0,
                  trans=None,
                  mu_by_reg=None, sigma_by_reg=None,
                  shock_prob=0.005, muJ=0.0, sigmaJ=0.08,
                  S0=100.0, dt=1/252):
    """
    Returns DataFrame with columns: price, ret, regime (0=bull,1=bear,2=flat), shock
    """
    rng = np.random.RandomState(seed)

    # default transition matrix: stay with high prob, small chance to switch
    if trans is None:
        trans = np.array([
            [0.94, 0.04, 0.02],
            [0.05, 0.92, 0.03],
            [0.03, 0.03, 0.94]
        ])

    # annualized mu and sigma per regime
    if mu_by_reg is None:
        mu_by_reg = np.array([0.12, -0.10, 0.01])
    if sigma_by_reg is None:
        sigma_by_reg = np.array([0.12, 0.30, 0.08])

    # sample regime path (Markov chain)
    regime = np.zeros(n_steps, dtype=int)
    regime[0] = rng.choice(3)
    for t in range(1, n_steps):
        regime[t] = rng.choice(3, p=trans[regime[t-1]])

    # gaussian shocks + Poisson jumps
    eps = rng.normal(size=n_steps)
    shocks = rng.binomial(1, shock_prob, size=n_steps)
    jumps = rng.normal(loc=muJ, scale=sigmaJ, size=n_steps) * shocks

    # build returns
    ret = np.zeros(n_steps)
    for t in range(n_steps):
        mu = mu_by_reg[regime[t]] * dt
        sigma = sigma_by_reg[regime[t]] * np.sqrt(dt)
        ret[t] = mu + sigma * eps[t] + jumps[t]

    prices = S0 * np.exp(np.cumsum(ret))

    df = pd.DataFrame({
        "price": prices,
        "ret": ret,
        "regime": regime,
        "shock": shocks
    })
    return df
