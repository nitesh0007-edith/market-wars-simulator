# agents/momentum.py
from .base import Agent
import numpy as np

class Momentum(Agent):
    name = "Momentum"
    def __init__(self, fast=20, slow=100, weight=1.0, trailing_stop=None):
        """
        fast, slow: windows for moving averages (ints)
        weight: target weight when signal is long (float, e.g. 1.0)
        trailing_stop: fraction (e.g. 0.10 for 10%) to cut exposure from peak while long; None disables
        """
        self.fast = int(fast)
        self.slow = int(slow)
        self.weight = float(weight)
        self.trailing_stop = None if trailing_stop is None else float(trailing_stop)
        self.last_target = 0.0
        self.peak_price = None

    def reset(self):
        self.last_target = 0.0
        self.peak_price = None

    def decide(self, t, price_series, portfolio):
        """
        price_series: sequence-like of historical prices up to current t (includes current price)
        portfolio: Portfolio object (can be used for current weight / position)
        returns target_weight in [-max_leverage, +max_leverage] (we use long/flat here)
        """
        if t < self.slow or len(price_series) < self.slow:
            # not enough data to evaluate
            # ensure we update peak if we already hold
            if portfolio is not None:
                curr_w = portfolio.current_weight(price_series[-1])
                if curr_w > 0 and self.peak_price is None:
                    self.peak_price = price_series[-1]
            return 0.0

        prices = np.asarray(price_series)
        sma_fast = prices[-self.fast :].mean()
        sma_slow = prices[-self.slow :].mean()

        # base signal: long if fast > slow, otherwise flat
        signal = self.weight if sma_fast > sma_slow else 0.0

        # trailing stop logic (only applies if we currently are long or going long)
        curr_price = float(prices[-1])
        curr_w = portfolio.current_weight(curr_price) if portfolio is not None else 0.0

        if curr_w > 0:
            # we currently hold long; update peak
            if self.peak_price is None:
                self.peak_price = curr_price
            else:
                if curr_price > self.peak_price:
                    self.peak_price = curr_price
            # if trailing stop is set and price dropped below threshold, cut exposure
            if self.trailing_stop is not None:
                if curr_price < self.peak_price * (1.0 - self.trailing_stop):
                    signal = 0.0
                    # reset peak so we don't immediately re-enter on tiny bounce
                    self.peak_price = None

        # If signal says go long now and we were not long, set initial peak
        if signal > 0 and curr_w <= 0:
            self.peak_price = curr_price

        self.last_target = signal
        return float(signal)
