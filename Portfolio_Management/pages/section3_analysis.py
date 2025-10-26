"""
Section 3: S&P 500 Technical Analysis
"""
import streamlit as st
import pandas as pd
from analysis import trend_analysis, volatility_analysis, support_resistance, risk_return_metrics


def render(df):
    """Render Section 3: S&P 500 Technical Analysis"""

    st.header("📈 Section 3: S&P 500 Technical Analysis")

    # Check if we have data
    if df.empty:
        st.warning("⚠️ No data available. Data file not found.")
        st.info("""
        **To use technical analysis:**
        1. Run: `python fetch_data_csv.py`
        2. This will create: `data/sp500_data.csv`
        3. Refresh this page (F5 or click Refresh Data in sidebar)
        4. Data will be automatically loaded

        **Data Requirements:**
        - File location: `data/sp500_data.csv`
        - Required columns: Timestamp, Open, High, Low, Close, Volume
        - Data will be preloaded automatically when you run the app
        """)
        st.stop()

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trend Analysis",
        "📊 Volatility Analysis",
        "🎯 Support & Resistance",
        "⚖️ Risk-Return Metrics",
        "📋 Raw Data"
    ])

    # Tab 1: Trend Analysis
    with tab1:
        st.header("📈 Trend Analysis")
        st.write("Moving averages and buy/sell signals based on technical indicators")

        with st.spinner("Calculating trend analysis..."):
            df_trend, fig_trend, summary_trend = trend_analysis(df)

        # Display summary
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Summary")
            for key, value in list(summary_trend.items())[:4]:
                st.metric(key, value)
        with col2:
            st.subheader("📈 Signals")
            for key, value in list(summary_trend.items())[4:]:
                st.metric(key, value)

        # Display chart
        st.plotly_chart(fig_trend, use_container_width=True)

        # Interpretation
        with st.expander("ℹ️ How to Interpret"):
            st.markdown("""
            **Moving Averages:**
            - **SMA 20**: Short-term trend (20 periods)
            - **SMA 50**: Medium-term trend (50 periods)
            - **SMA 200**: Long-term trend (200 periods)

            **Signals:**
            - **Buy Signal** (🔺): When SMA 20 crosses above SMA 50
            - **Sell Signal** (🔻): When SMA 20 crosses below SMA 50

            **MACD:**
            - Positive MACD = Bullish momentum
            - Negative MACD = Bearish momentum
            - MACD crossing Signal Line = Trend change
            """)

    # Tab 2: Volatility Analysis
    with tab2:
        st.header("📊 Volatility Analysis")
        st.write("Risk assessment and market stability indicators")

        with st.spinner("Calculating volatility analysis..."):
            df_vol, fig_vol, summary_vol = volatility_analysis(df)

        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Volatility", summary_vol['Current Volatility'])
        with col2:
            st.metric("Average Volatility", summary_vol['Average Volatility'])
        with col3:
            st.metric("Volatility Status", summary_vol['Volatility Status'])
        with col4:
            st.metric("ATR (14-day)", summary_vol['ATR (14-day)'])

        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BB Width", summary_vol['BB Width'])
        with col2:
            st.metric("Max Returns", summary_vol['Max Returns'])
        with col3:
            st.metric("Min Returns", summary_vol['Min Returns'])
        with col4:
            st.metric("Std Dev", summary_vol['Std Dev (Returns)'])

        # Display chart
        st.plotly_chart(fig_vol, use_container_width=True)

        # Interpretation
        with st.expander("ℹ️ How to Interpret"):
            st.markdown("""
            **Volatility:**
            - High volatility = Higher risk and potential reward
            - Low volatility = More stable, less dramatic price movements

            **Bollinger Bands:**
            - Price touching upper band = Potentially overbought
            - Price touching lower band = Potentially oversold
            - Wide bands = High volatility
            - Narrow bands = Low volatility (potential breakout coming)

            **ATR (Average True Range):**
            - Measures market volatility
            - Higher ATR = More volatile market
            - Used for setting stop-loss levels
            """)

    # Tab 3: Support & Resistance
    with tab3:
        st.header("🎯 Support & Resistance Levels")
        st.write("Key price levels for entry and exit decisions")

        with st.spinner("Calculating support & resistance..."):
            df_sr, fig_sr, summary_sr, resistance_levels, support_levels = support_resistance(df)

        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", summary_sr['Current Price'])
        with col2:
            st.metric("Nearest Resistance", summary_sr['Nearest Resistance'])
            st.caption(f"Distance: {summary_sr['Distance to Resistance']}")
        with col3:
            st.metric("Nearest Support", summary_sr['Nearest Support'])
            st.caption(f"Distance: {summary_sr['Distance to Support']}")

        # Display chart
        st.plotly_chart(fig_sr, use_container_width=True)

        # Display levels
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔴 Resistance Levels")
            for i, level in enumerate(resistance_levels, 1):
                st.write(f"**R{i}:** ${level:.2f}")

        with col2:
            st.subheader("🟢 Support Levels")
            for i, level in enumerate(support_levels, 1):
                st.write(f"**S{i}:** ${level:.2f}")

        # Interpretation
        with st.expander("ℹ️ How to Interpret"):
            st.markdown("""
            **Support Levels (Green):**
            - Price levels where buying pressure overcomes selling pressure
            - Price tends to bounce up from these levels
            - Good entry points for long positions

            **Resistance Levels (Red):**
            - Price levels where selling pressure overcomes buying pressure
            - Price tends to fall back from these levels
            - Good exit points or short entry points

            **Trading Strategy:**
            - Buy near support levels
            - Sell near resistance levels
            - Breakout above resistance = Strong bullish signal
            - Breakdown below support = Strong bearish signal
            """)

    # Tab 4: Risk-Return Metrics
    with tab4:
        st.header("⚖️ Risk-Return Metrics")
        st.write("Comprehensive performance and risk assessment")

        with st.spinner("Calculating risk-return metrics..."):
            df_risk, fig_risk, summary_risk = risk_return_metrics(df)

        # Display summary
        st.subheader("📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", summary_risk['Total Return'])
        with col2:
            st.metric("Annualized Return", summary_risk['Annualized Return'])
        with col3:
            st.metric("Volatility", summary_risk['Volatility (Annual)'])
        with col4:
            st.metric("Sharpe Ratio", summary_risk['Sharpe Ratio'])

        st.subheader("📉 Risk Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Max Drawdown", summary_risk['Max Drawdown'])
        with col2:
            st.metric("Calmar Ratio", summary_risk['Calmar Ratio'])
        with col3:
            st.metric("Win Rate", summary_risk['Win Rate'])
        with col4:
            st.metric("Best Day", summary_risk['Best Day'])

        st.subheader("📈 Win/Loss Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Worst Day", summary_risk['Worst Day'])
        with col2:
            st.metric("Avg Win", summary_risk['Avg Win'])
        with col3:
            st.metric("Avg Loss", summary_risk['Avg Loss'])

        # Display chart
        st.plotly_chart(fig_risk, use_container_width=True)

        # Interpretation
        with st.expander("ℹ️ How to Interpret"):
            st.markdown("""
            **Sharpe Ratio:**
            - Measures risk-adjusted return
            - > 1 = Good, > 2 = Very Good, > 3 = Excellent
            - Higher is better

            **Max Drawdown:**
            - Maximum peak-to-trough decline
            - Indicates worst-case loss scenario
            - Lower (less negative) is better

            **Calmar Ratio:**
            - Return divided by maximum drawdown
            - Measures return per unit of risk
            - Higher is better

            **Win Rate:**
            - Percentage of profitable periods
            - > 50% = More winning days than losing days
            """)

    # Tab 5: Raw Data
    with tab5:
        st.header("📋 Raw Data")
        st.write(f"Displaying {len(df):,} data points")

        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_rows = st.selectbox("Number of rows to display", [10, 25, 50, 100, "All"], index=1)
        with col2:
            download_csv = st.download_button(
                label="📥 Download CSV",
                data=df.to_csv(index=False),
                file_name=f"sp500_data_{df['Timestamp'].min().date()}_{df['Timestamp'].max().date()}.csv",
                mime="text/csv"
            )

        # Display data
        if show_rows == "All":
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df.head(show_rows), use_container_width=True)

        # Data statistics
        with st.expander("📊 Data Statistics"):
            st.write(df.describe())
