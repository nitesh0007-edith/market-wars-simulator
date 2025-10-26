# agents/meanrev.py
from .base import Agent
import numpy as np

class MeanRev(Agent):
    name = "MeanRev"

    def __init__(self, window=20, zentry=1.0, zexit=0.5, weight=1.0):
        """
        window: lookback (returns)
        zentry: z-score threshold to enter
        zexit: threshold to exit (smaller to avoid flip-flopping)
        weight: target absolute weight when fully long/short
        """
        self.window = int(window)
        self.zentry = float(zentry)
        self.zexit = float(zexit)
        self.weight = float(weight)
        self.position = 0.0

    def reset(self):
        self.position = 0.0

    def decide(self, t, price_series, portfolio):
        if t < self.window + 1:
            return 0.0

        prices = np.asarray(price_series)
        # use log returns
        rets = np.diff(np.log(prices[-(self.window+1):]))
        mu = rets.mean()
        sigma = rets.std(ddof=0) + 1e-12
        z = (rets[-1] - mu) / sigma

        # entry rules (mean reversion): if last return is extreme, take opposite side
        target = 0.0
        if z > self.zentry:
            target = -self.weight
        elif z < -self.zentry:
            target = self.weight
        else:
            # if currently in a position, wait until z moves closer to mean (exit)
            if self.position > 0 and z > -self.zexit:
                target = self.position  # keep
            elif self.position < 0 and z < self.zexit:
                target = self.position
            else:
                target = 0.0

        # store last chosen (portfolio will execute)
        self.position = float(target)
        return float(target)
