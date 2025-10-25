# market/visualize.py
import matplotlib.pyplot as plt
import numpy as np

COLORS = {0: "#e6ffed", 1: "#ffe6e6", 2: "#fff7e6"}

def regime_color(reg):
    return COLORS.get(int(reg), "#f0f0f0")

def plot_market(df, title="Synthetic Market Path", ax=None, show=True):
    price = df['price'].values
    regimes = df['regime'].values
    shocks = df['shock'].values
    t = np.arange(len(df))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,5))
    else:
        fig = ax.figure

    # background for contiguous regime runs
    start = 0
    for i in range(1, len(regimes)+1):
        if i == len(regimes) or regimes[i] != regimes[start]:
            ax.axvspan(start, i-1, color=regime_color(regimes[start]), alpha=0.35, linewidth=0)
            start = i

    ax.plot(t, price, lw=1.5, label="Price")
    shock_idx = (shocks == 1).nonzero()[0]
    if len(shock_idx):
        ax.scatter(shock_idx, price[shock_idx], marker='x', s=40, color='k', label='Shock', zorder=5)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
