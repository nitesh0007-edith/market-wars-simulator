# engine/broker.py
import numpy as np

class Broker:
    """
    Simple broker model:
      - commission: fixed per trade ($)
      - spread: fraction (half-spread applied to direction)
      - slippage_k: slippage proportional to abs(delta_weight)
      - max_leverage: absolute cap on portfolio weight
    """
    def __init__(self, commission=0.0, spread=0.0005, slippage_k=0.002, max_leverage=2.0):
        self.commission = float(commission)
        self.spread = float(spread)
        self.slippage_k = float(slippage_k)
        self.max_leverage = float(max_leverage)

    def execution_price(self, mid_price, target_weight, current_weight):
        """
        Return (exec_price, trade_slippage_fraction, delta_abs_weight)
        exec_price = mid_price * (1 + sign(delta_w)*(spread + slippage))
        """
        delta_w = target_weight - current_weight
        if delta_w == 0:
            return mid_price, 0.0, 0.0
        sign = np.sign(delta_w)
        slippage = self.slippage_k * abs(delta_w)
        # total fractional move applied to mid price
        frac = sign * (self.spread + slippage)
        exec_price = mid_price * (1.0 + frac)
        return float(exec_price), float(slippage), float(abs(delta_w))

    def trade_cost(self, exec_price, units):
        """
        Fixed commission (per trade) + simple proportional fee could be extended.
        Here commission applied if units != 0.
        """
        trade_value = abs(units) * exec_price
        commission = self.commission if abs(units) > 0 else 0.0
        return float(commission), float(trade_value)
