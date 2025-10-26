"""
Investment simulation utilities
"""
import numpy as np


def simulate_investment(df, investment_amount, entry_date, exit_date):
    """
    Simulate investment returns based on historical data

    Args:
        df: DataFrame with stock data
        investment_amount: Amount to invest
        entry_date: Date to enter the position
        exit_date: Date to exit the position

    Returns:
        Dictionary with simulation results
    """
    # Filter data for the period
    mask = (df['Timestamp'].dt.date >= entry_date) & (df['Timestamp'].dt.date <= exit_date)
    period_df = df.loc[mask].copy()

    if len(period_df) < 2:
        return None

    # Entry and exit prices
    entry_price = period_df.iloc[0]['Close']
    exit_price = period_df.iloc[-1]['Close']

    # Calculate shares purchased
    shares = investment_amount / entry_price

    # Calculate returns
    exit_value = shares * exit_price
    profit_loss = exit_value - investment_amount
    return_pct = (profit_loss / investment_amount) * 100

    # Calculate additional metrics
    period_df['Value'] = shares * period_df['Close']
    max_value = period_df['Value'].max()
    min_value = period_df['Value'].min()

    max_gain = ((max_value - investment_amount) / investment_amount) * 100
    max_loss = ((min_value - investment_amount) / investment_amount) * 100

    # Daily returns
    period_df['Daily_Return'] = period_df['Close'].pct_change()
    avg_daily_return = period_df['Daily_Return'].mean() * 100
    volatility = period_df['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized

    return {
        'entry_date': period_df.iloc[0]['Timestamp'],
        'exit_date': period_df.iloc[-1]['Timestamp'],
        'entry_price': entry_price,
        'exit_price': exit_price,
        'shares': shares,
        'investment': investment_amount,
        'exit_value': exit_value,
        'profit_loss': profit_loss,
        'return_pct': return_pct,
        'max_value': max_value,
        'min_value': min_value,
        'max_gain': max_gain,
        'max_loss': max_loss,
        'avg_daily_return': avg_daily_return,
        'volatility': volatility,
        'period_df': period_df
    }
