# agents/buyhold.py
from .base import Agent

class BuyHold(Agent):
    name = "BuyHold"

    def __init__(self, weight=1.0):
        """
        Simple buy-and-hold benchmark.
        weight: proportion of equity to hold (1.0 = full long)
        """
        self.weight = float(weight)

    def reset(self):
        pass

    def decide(self, t, price_series, portfolio):
        # always aim for constant weight
        return float(self.weight)
