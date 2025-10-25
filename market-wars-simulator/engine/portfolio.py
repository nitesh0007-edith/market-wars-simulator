# engine/portfolio.py
import numpy as np
import pandas as pd

class Portfolio:
    """
    Single-asset portfolio (USD base).
    Tracks: cash, position (units), equity, history (list of dicts).
    Rebalances toward target_weight each step using broker for execution.
    """
    def __init__(self, initial_cash=100000.0, broker=None):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.position = 0.0  # units of asset
        self.equity = float(initial_cash)
        self.broker = broker
        self.history = []    # list of dicts with daily snapshots
        # cumulative metrics
        self.total_commission = 0.0
        self.total_traded = 0.0

    def current_notional(self, price):
        return self.position * price

    def current_weight(self, price):
        if self.equity == 0:
            return 0.0
        return (self.position * price) / self.equity

    def mark_to_market(self, price):
        self.equity = self.cash + self.position * price
        return self.equity

    def _enforce_leverage(self, target_weight):
        if self.broker is None:
            return target_weight
        ml = self.broker.max_leverage
        return max(-ml, min(ml, target_weight))

    def step(self, price, target_weight):
        """
        Rebalance toward target_weight (in [-max_leverage, max_leverage]).
        Returns (equity_after, info_dict)
        """
        # safety
        if self.equity <= 0:
            # forced flat: liquidate everything at current price
            info = self._liquidate(price, reason="zero_equity")
            self.history.append(info)
            return self.equity, info

        # enforce leverage cap
        target_weight = self._enforce_leverage(target_weight)
        curr_weight = self.current_weight(price)

        # if broker is provided, compute execution price and slippage
        if self.broker is None:
            exec_price = price
            slippage = 0.0
            delta_abs_w = abs(target_weight - curr_weight)
        else:
            exec_price, slippage, delta_abs_w = self.broker.execution_price(price, target_weight, curr_weight)

        # compute desired notional and units
        desired_notional = target_weight * self.equity
        desired_units = 0.0 if exec_price == 0 else (desired_notional / exec_price)
        delta_units = desired_units - self.position

        # apply trade: cash change, commission, update position
        commission, trade_value = (0.0, 0.0)
        if self.broker is not None and abs(delta_units) > 0:
            commission, trade_value = self.broker.trade_cost(exec_price, delta_units)

        # update cash and position
        # buying positive delta_units costs cash; selling increases cash
        self.cash -= delta_units * exec_price
        # subtract commission
        self.cash -= commission
        self.position += delta_units

        # update running totals
        self.total_commission += commission
        self.total_traded += trade_value

        # mark-to-market
        self.equity = self.cash + self.position * price

        info = {
            "price": float(price),
            "exec_price": float(exec_price),
            "target_weight": float(target_weight),
            "curr_weight": float(curr_weight),
            "delta_units": float(delta_units),
            "trade_value": float(trade_value),
            "commission": float(commission),
            "slippage_frac": float(slippage),
            "delta_abs_weight": float(delta_abs_w),
            "cash": float(self.cash),
            "position": float(self.position),
            "equity": float(self.equity)
        }
        self.history.append(info)

        # basic margin-call logic: if equity goes negative, liquidate and set equity to 0
        if self.equity <= 0:
            liq = self._liquidate(price, reason="negative_equity")
            # replace last history with liquidate info to be explicit
            self.history[-1] = liq
            return self.equity, liq

        return self.equity, info

    def _liquidate(self, price, reason="liquidate"):
        """
        Forcefully close position at given price. Update cash/equity and return info dict.
        """
        # sell all units
        trade_value = abs(self.position) * price
        commission = 0.0
        if self.broker is not None and abs(self.position) > 0:
            commission, _ = self.broker.trade_cost(price, -self.position)  # cost for closing
        self.cash += -self.position * price  # if position positive, selling increases cash
        self.cash -= commission
        self.position = 0.0
        self.equity = self.cash
        info = {
            "price": float(price),
            "exec_price": float(price),
            "target_weight": 0.0,
            "curr_weight": 0.0,
            "delta_units": 0.0,
            "trade_value": float(trade_value),
            "commission": float(commission),
            "slippage_frac": 0.0,
            "delta_abs_weight": 0.0,
            "cash": float(self.cash),
            "position": float(self.position),
            "equity": float(self.equity),
            "liquidation_reason": reason
        }
        self.history.append(info)
        return info

    def snapshot_dataframe(self):
        """Return history as a pandas DataFrame"""
        return pd.DataFrame(self.history)
