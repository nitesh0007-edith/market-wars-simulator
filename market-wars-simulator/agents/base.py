# agents/base.py

class Agent:
    """Base class for trading agents. Defines common interface."""

    name = "BaseAgent"

    def reset(self):
        """Reset any internal state before a new simulation."""
        pass

    def decide(self, t, price_series, portfolio):
        """
        Decide target weight for current time step.
        Args:
            t (int): current time index
            price_series (list or np.array): all prices up to now (including current)
            portfolio (Portfolio): current portfolio state
        Returns:
            float: target portfolio weight in [-1, +1] or beyond if leverage allowed
        """
        raise NotImplementedError("decide() must be implemented by subclasses")
