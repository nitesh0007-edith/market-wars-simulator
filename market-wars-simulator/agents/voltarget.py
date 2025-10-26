# agents/voltarget.py
from .base import Agent
import numpy as np

class VolTarget(Agent):
    name = "VolTarget"

    def __init__(self, ewma_alpha=0.94, target_ann_vol=0.12, max_leverage=2.0):
        """
        ewma_alpha: smoothing for EWMA volatility (0..1)
        target_ann_vol: desired annualized volatility (e.g., 0.12)
        max_leverage: clamp on absolute exposure
        """
        self.ewma_alpha = float(ewma_alpha)
        self.target_ann_vol = float(target_ann_vol)
        self.max_leverage = float(max_leverage)
        self.sigma_hat = None
        self.prev_price = None

    def reset(self):
        self.sigma_hat = None
        self.prev_price = None

    def decide(self, t, price_series, portfolio):
        if len(price_series) < 2:
            return 0.0

        prices = price_series
        # simple log return for last step
        ret = np.log(prices[-1]) - np.log(prices[-2])
        if self.sigma_hat is None:
            # initialize with absolute return
            self.sigma_hat = abs(ret)
        else:
            # EWMA on variance
            self.sigma_hat = (self.ewma_alpha * (self.sigma_hat**2) + (1.0 - self.ewma_alpha) * (ret**2))**0.5

        # annualize assuming 252 trading days
        sigma_ann = self.sigma_hat * (252**0.5) + 1e-12
        raw_w = self.target_ann_vol / sigma_ann
        # clamp to leverage limits
        w = max(-self.max_leverage, min(self.max_leverage, raw_w))
        return float(w)
