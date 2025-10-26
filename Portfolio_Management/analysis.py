"""
Core analysis functions for S&P 500 data
Contains 4 main analyses:
1. Trend Analysis - Moving averages with buy/sell signals
2. Volatility Analysis - Risk assessment and market stability
3. Support & Resistance - Key price levels for entry/exit
4. Risk-Return Metrics - Sharpe ratio, drawdown, performance
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def trend_analysis(df):
    """
    📈 Trend Analysis - Moving averages with buy/sell signals

    Args:
        df: DataFrame with columns ['Timestamp', 'Close']

    Returns:
        DataFrame with moving averages and signals, plus plotly figure
    """
    df = df.copy()

    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    # Generate buy/sell signals
    # Buy: SMA_20 crosses above SMA_50
    # Sell: SMA_20 crosses below SMA_50
    df['Signal'] = 0
    df.loc[(df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1)), 'Signal'] = 1  # Buy
    df.loc[(df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1)), 'Signal'] = -1  # Sell

    # Current trend
    current_trend = "Bullish 📈" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "Bearish 📉"

    # Create plot
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price and Moving Averages', 'MACD'),
        vertical_spacing=0.1
    )

    # Price and moving averages
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SMA_50'], name='SMA 50', line=dict(color='green', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SMA_200'], name='SMA 200', line=dict(color='red', dash='dot')), row=1, col=1)

    # Buy/sell signals
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]

    fig.add_trace(go.Scatter(x=buy_signals['Timestamp'], y=buy_signals['Close'],
                            mode='markers', name='Buy Signal',
                            marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals['Timestamp'], y=sell_signals['Close'],
                            mode='markers', name='Sell Signal',
                            marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Signal_Line'], name='Signal Line', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Bar(x=df['Timestamp'], y=df['MACD_Histogram'], name='MACD Histogram', marker_color='gray'), row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_layout(height=800, showlegend=True, title_text="Trend Analysis")

    # Summary statistics
    summary = {
        'Current Trend': current_trend,
        'SMA 20': f"${df['SMA_20'].iloc[-1]:.2f}",
        'SMA 50': f"${df['SMA_50'].iloc[-1]:.2f}",
        'SMA 200': f"${df['SMA_200'].iloc[-1]:.2f}",
        'MACD': f"{df['MACD'].iloc[-1]:.2f}",
        'Signal Line': f"{df['Signal_Line'].iloc[-1]:.2f}",
        'Total Buy Signals': len(buy_signals),
        'Total Sell Signals': len(sell_signals)
    }

    return df, fig, summary


def volatility_analysis(df):
    """
    📊 Volatility Analysis - Risk assessment and market stability

    Args:
        df: DataFrame with columns ['Timestamp', 'Close', 'High', 'Low']

    Returns:
        DataFrame with volatility metrics, plus plotly figure
    """
    df = df.copy()

    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility metrics
    df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100  # Annualized
    df['Volatility_50'] = df['Returns'].rolling(window=50).std() * np.sqrt(252) * 100

    # Average True Range (ATR)
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100

    # Current volatility level
    current_vol = df['Volatility_20'].iloc[-1]
    avg_vol = df['Volatility_20'].mean()
    vol_status = "High 🔴" if current_vol > avg_vol * 1.2 else "Normal 🟢" if current_vol < avg_vol * 0.8 else "Moderate 🟡"

    # Create plot
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=('Price with Bollinger Bands', 'Volatility (20-day & 50-day)', 'Average True Range (ATR)'),
        vertical_spacing=0.1
    )

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['BB_Upper'], name='BB Upper', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['BB_Middle'], name='BB Middle', line=dict(color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['BB_Lower'], name='BB Lower', line=dict(color='green', dash='dash')), row=1, col=1)

    # Volatility
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Volatility_20'], name='Volatility 20', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Volatility_50'], name='Volatility 50', line=dict(color='purple')), row=2, col=1)

    # ATR
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['ATR_14'], name='ATR 14', line=dict(color='brown')), row=3, col=1)

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=3, col=1)
    fig.update_layout(height=900, showlegend=True, title_text="Volatility Analysis")

    # Summary statistics
    summary = {
        'Current Volatility': f"{current_vol:.2f}%",
        'Average Volatility': f"{avg_vol:.2f}%",
        'Volatility Status': vol_status,
        'ATR (14-day)': f"${df['ATR_14'].iloc[-1]:.2f}",
        'BB Width': f"{df['BB_Width'].iloc[-1]:.2f}%",
        'Max Returns': f"{df['Returns'].max() * 100:.2f}%",
        'Min Returns': f"{df['Returns'].min() * 100:.2f}%",
        'Std Dev (Returns)': f"{df['Returns'].std() * 100:.2f}%"
    }

    return df, fig, summary


def support_resistance(df, window=20):
    """
    🎯 Support & Resistance - Key price levels for entry/exit

    Args:
        df: DataFrame with columns ['Timestamp', 'Close', 'High', 'Low']
        window: Rolling window for finding local peaks and troughs

    Returns:
        DataFrame with support/resistance levels, plus plotly figure
    """
    df = df.copy()

    # Find local maxima (resistance) and minima (support)
    df['Local_Max'] = df['High'].rolling(window=window, center=True).max()
    df['Local_Min'] = df['Low'].rolling(window=window, center=True).min()

    df['Is_Resistance'] = (df['High'] == df['Local_Max']) & (df['High'].shift(1) < df['High']) & (df['High'].shift(-1) < df['High'])
    df['Is_Support'] = (df['Low'] == df['Local_Min']) & (df['Low'].shift(1) > df['Low']) & (df['Low'].shift(-1) > df['Low'])

    # Get resistance and support levels
    resistance_levels = df[df['Is_Resistance']]['High'].values
    support_levels = df[df['Is_Support']]['Low'].values

    # Cluster nearby levels (within 1% of each other)
    def cluster_levels(levels, threshold=0.01):
        if len(levels) == 0:
            return []
        sorted_levels = np.sort(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        clusters.append(np.mean(current_cluster))
        return clusters

    resistance_clusters = cluster_levels(resistance_levels)[-5:]  # Top 5 resistance levels
    support_clusters = cluster_levels(support_levels)[-5:]  # Top 5 support levels

    # Current price position
    current_price = df['Close'].iloc[-1]

    # Create plot
    fig = go.Figure()

    # Price
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Close'], name='Close Price', line=dict(color='blue')))

    # Resistance levels
    for i, level in enumerate(resistance_clusters):
        fig.add_hline(y=level, line_dash="dash", line_color="red",
                     annotation_text=f"R{i+1}: ${level:.2f}",
                     annotation_position="right")

    # Support levels
    for i, level in enumerate(support_clusters):
        fig.add_hline(y=level, line_dash="dash", line_color="green",
                     annotation_text=f"S{i+1}: ${level:.2f}",
                     annotation_position="right")

    # Mark resistance and support points
    resistance_points = df[df['Is_Resistance']]
    support_points = df[df['Is_Support']]

    fig.add_trace(go.Scatter(x=resistance_points['Timestamp'], y=resistance_points['High'],
                            mode='markers', name='Resistance Points',
                            marker=dict(color='red', size=8, symbol='triangle-down')))
    fig.add_trace(go.Scatter(x=support_points['Timestamp'], y=support_points['Low'],
                            mode='markers', name='Support Points',
                            marker=dict(color='green', size=8, symbol='triangle-up')))

    fig.update_layout(
        title='Support & Resistance Levels',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        showlegend=True
    )

    # Summary statistics
    nearest_resistance = min([r for r in resistance_clusters if r > current_price], default=None)
    nearest_support = max([s for s in support_clusters if s < current_price], default=None)

    summary = {
        'Current Price': f"${current_price:.2f}",
        'Nearest Resistance': f"${nearest_resistance:.2f}" if nearest_resistance else "N/A",
        'Nearest Support': f"${nearest_support:.2f}" if nearest_support else "N/A",
        'Resistance Levels': len(resistance_clusters),
        'Support Levels': len(support_clusters),
        'Distance to Resistance': f"{((nearest_resistance - current_price) / current_price * 100):.2f}%" if nearest_resistance else "N/A",
        'Distance to Support': f"{((current_price - nearest_support) / current_price * 100):.2f}%" if nearest_support else "N/A"
    }

    return df, fig, summary, resistance_clusters, support_clusters


def risk_return_metrics(df, risk_free_rate=0.04):
    """
    ⚖️ Risk-Return Metrics - Sharpe ratio, drawdown, performance

    Args:
        df: DataFrame with columns ['Timestamp', 'Close']
        risk_free_rate: Annual risk-free rate (default: 4%)

    Returns:
        Dictionary with metrics and plotly figure
    """
    df = df.copy()

    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod() - 1

    # Performance metrics
    total_return = df['Cumulative_Returns'].iloc[-1] * 100
    annualized_return = ((1 + df['Cumulative_Returns'].iloc[-1]) ** (252 / len(df)) - 1) * 100

    # Volatility
    volatility = df['Returns'].std() * np.sqrt(252) * 100  # Annualized

    # Sharpe Ratio
    excess_return = annualized_return / 100 - risk_free_rate
    sharpe_ratio = excess_return / (volatility / 100) if volatility != 0 else 0

    # Maximum Drawdown
    df['Cumulative_Max'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Cumulative_Max']) / df['Cumulative_Max'] * 100
    max_drawdown = df['Drawdown'].min()

    # Calmar Ratio (Annualized return / Max drawdown)
    calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    winning_days = len(df[df['Returns'] > 0])
    total_days = len(df[df['Returns'].notna()])
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0

    # Best and worst days
    best_day = df['Returns'].max() * 100
    worst_day = df['Returns'].min() * 100

    # Average win/loss
    avg_win = df[df['Returns'] > 0]['Returns'].mean() * 100
    avg_loss = df[df['Returns'] < 0]['Returns'].mean() * 100

    # Create plot
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        subplot_titles=('Cumulative Returns', 'Drawdown'),
        vertical_spacing=0.1
    )

    # Cumulative returns
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Cumulative_Returns'] * 100,
                            name='Cumulative Returns', line=dict(color='green'),
                            fill='tozeroy'), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Drawdown'],
                            name='Drawdown', line=dict(color='red'),
                            fill='tozeroy'), row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Returns (%)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_layout(height=700, showlegend=True, title_text="Risk-Return Analysis")

    # Summary metrics
    summary = {
        'Total Return': f"{total_return:.2f}%",
        'Annualized Return': f"{annualized_return:.2f}%",
        'Volatility (Annual)': f"{volatility:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2f}%",
        'Calmar Ratio': f"{calmar_ratio:.2f}",
        'Win Rate': f"{win_rate:.2f}%",
        'Best Day': f"{best_day:.2f}%",
        'Worst Day': f"{worst_day:.2f}%",
        'Avg Win': f"{avg_win:.2f}%",
        'Avg Loss': f"{avg_loss:.2f}%"
    }

    return df, fig, summary
