"""
Section 2: Investment Simulator for Top 5 Stocks
User selects stock, enters investment amount, and runs simulation
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from utils.simulator import simulate_investment


# Top 5 stocks data
TOP_5_STOCKS = {
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "description": "Leading smartphone and computer manufacturer",
        "current_price": 185.50,
        "color": "#A2AAAD"
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "description": "Software, cloud computing, and services",
        "current_price": 420.25,
        "color": "#00A4EF"
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "sector": "Consumer/Cloud",
        "description": "E-commerce and cloud computing giant",
        "current_price": 175.80,
        "color": "#FF9900"
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "sector": "Technology",
        "description": "Internet services and advertising",
        "current_price": 142.30,
        "color": "#4285F4"
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "sector": "Automotive/Energy",
        "description": "Electric vehicles and clean energy",
        "current_price": 245.75,
        "color": "#E31937"
    }
}


def render(df):
    """Render Section 2: Investment Simulator for Top 5 Stocks"""

    st.header("💰 Section 2: Investment Simulator")
    st.write("Simulate your investment in top US stocks and see potential returns")

    # Check if we have data
    if df.empty:
        st.warning("⚠️ No data available. Data file not found.")
        st.info("""
        **To use the simulator:**
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

    # Step 1: Stock Selection
    st.subheader("📊 Step 1: Select Your Stock")

    # Display stock cards
    cols = st.columns(5)

    for idx, (symbol, info) in enumerate(TOP_5_STOCKS.items()):
        with cols[idx]:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; border: 2px solid {info['color']}; border-radius: 10px; background-color: rgba(0,0,0,0.02);'>
                <h3 style='color: {info['color']};'>{symbol}</h3>
                <p style='font-size: 0.9em;'><b>{info['name']}</b></p>
                <p style='font-size: 0.8em; color: gray;'>{info['sector']}</p>
                <p style='font-size: 1.2em; font-weight: bold;'>${info['current_price']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Stock selector
    selected_stock = st.selectbox(
        "🎯 Choose a stock to simulate:",
        options=list(TOP_5_STOCKS.keys()),
        format_func=lambda x: f"{x} - {TOP_5_STOCKS[x]['name']} (${TOP_5_STOCKS[x]['current_price']:.2f})",
        help="Select one of the top 5 US stocks to run investment simulation"
    )

    stock_info = TOP_5_STOCKS[selected_stock]

    # Display selected stock info
    st.info(f"""
    **Selected:** {stock_info['name']} ({selected_stock})

    **Sector:** {stock_info['sector']}

    **Description:** {stock_info['description']}

    **Current Price:** ${stock_info['current_price']:.2f}
    """)

    st.markdown("---")

    # Step 2: Investment Parameters
    st.subheader("💵 Step 2: Enter Investment Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Investment Amount**")
        investment_amount = st.number_input(
            "Enter amount ($)",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=500.0,
            help="How much money do you want to invest?",
            label_visibility="collapsed"
        )

        # Calculate shares
        estimated_shares = investment_amount / stock_info['current_price']
        st.caption(f"📌 You can buy approximately **{estimated_shares:.4f} shares** at current price")

        # Investment strategy
        strategy = st.radio(
            "Investment Strategy",
            ["Long-term Hold", "Short-term Trade", "Custom Period"],
            help="Choose your investment timeframe"
        )

    with col2:
        st.write("**Investment Period**")
        min_date = df['Timestamp'].min().date()
        max_date = df['Timestamp'].max().date()

        if strategy == "Long-term Hold":
            default_entry = min_date
            default_exit = max_date
            st.caption("📅 Using full date range for long-term simulation")
        elif strategy == "Short-term Trade":
            default_entry = max(min_date, max_date - timedelta(days=7))
            default_exit = max_date
            st.caption("📅 Using last 7 days for short-term simulation")
        else:
            default_entry = min_date
            default_exit = max_date
            st.caption("📅 Select custom date range below")

        entry_date = st.date_input(
            "Entry Date (Buy)",
            value=default_entry,
            min_value=min_date,
            max_value=max_date,
            help="When do you want to buy the stock?"
        )

        exit_date = st.date_input(
            "Exit Date (Sell)",
            value=default_exit,
            min_value=min_date,
            max_value=max_date,
            help="When do you want to sell the stock?"
        )

    # Date validation
    if entry_date >= exit_date:
        st.error("⚠️ Exit date must be after entry date!")
        st.stop()

    holding_period = (exit_date - entry_date).days
    st.info(f"📊 **Holding Period:** {holding_period} days ({holding_period/30:.1f} months)")

    st.markdown("---")

    # Step 3: Risk Tolerance
    st.subheader("⚠️ Step 3: Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate",
            help="How much risk are you willing to take?"
        )

    with col2:
        stop_loss = st.number_input(
            "Stop Loss (%)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
            help="Automatically sell if loss exceeds this percentage"
        )

    with col3:
        take_profit = st.number_input(
            "Take Profit (%)",
            min_value=0.0,
            max_value=200.0,
            value=20.0,
            step=5.0,
            help="Automatically sell if profit reaches this percentage"
        )

    st.markdown("---")

    # Step 4: Run Simulation
    st.subheader("🚀 Step 4: Run Simulation")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    if run_button:
        with st.spinner(f"Running simulation for {selected_stock}..."):
            results = simulate_investment(df, investment_amount, entry_date, exit_date)

            if results is None:
                st.error("❌ Not enough data for the selected period. Please choose a different date range.")
            else:
                # Display results
                st.success(f"✅ Simulation Complete for {stock_info['name']}!")

                # Key results
                st.subheader("📊 Simulation Results")

                # Main metrics
                col1, col2, col3, col4 = st.columns(4)

                profit_emoji = "📈" if results['profit_loss'] >= 0 else "📉"
                profit_color = "green" if results['profit_loss'] >= 0 else "red"

                with col1:
                    st.markdown(f"<h3 style='color: {profit_color};'>{profit_emoji} {results['profit_loss']:+.2f} USD</h3>", unsafe_allow_html=True)
                    st.caption("Total Profit/Loss")
                    st.metric("Return", f"{results['return_pct']:+.2f}%", delta=None)

                with col2:
                    st.metric("Initial Investment", f"${results['investment']:,.2f}")
                    st.metric("Final Value", f"${results['exit_value']:,.2f}")

                with col3:
                    st.metric("Shares Purchased", f"{results['shares']:.4f}")
                    st.metric("Entry Price", f"${results['entry_price']:.2f}")

                with col4:
                    st.metric("Exit Price", f"${results['exit_price']:.2f}")
                    price_change_pct = ((results['exit_price'] - results['entry_price']) / results['entry_price']) * 100
                    st.metric("Price Change", f"{price_change_pct:+.2f}%")

                # Risk metrics
                st.subheader("⚠️ Risk Analysis")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    gain_emoji = "🟢" if results['max_gain'] > 0 else "⚪"
                    st.metric(f"{gain_emoji} Max Gain", f"{results['max_gain']:+.2f}%")
                    st.caption(f"Highest value: ${results['max_value']:,.2f}")

                with col2:
                    loss_emoji = "🔴" if results['max_loss'] < 0 else "⚪"
                    st.metric(f"{loss_emoji} Max Loss", f"{results['max_loss']:+.2f}%")
                    st.caption(f"Lowest value: ${results['min_value']:,.2f}")

                with col3:
                    st.metric("Avg Daily Return", f"{results['avg_daily_return']:+.3f}%")
                    st.caption("Average daily percentage change")

                with col4:
                    vol_emoji = "🔥" if results['volatility'] > 30 else "⚡" if results['volatility'] > 20 else "✅"
                    st.metric(f"{vol_emoji} Volatility", f"{results['volatility']:.2f}%")
                    st.caption("Annual volatility (risk)")

                # Stop Loss / Take Profit Analysis
                st.subheader("🎯 Stop Loss & Take Profit Analysis")

                stop_loss_triggered = results['max_loss'] <= -stop_loss
                take_profit_triggered = results['max_gain'] >= take_profit

                col1, col2 = st.columns(2)

                with col1:
                    if stop_loss_triggered:
                        st.error(f"""
                        🛑 **Stop Loss Would Have Triggered!**
                        - Your stop loss: -{stop_loss}%
                        - Maximum loss reached: {results['max_loss']:.2f}%
                        - You would have been protected from larger losses
                        """)
                    else:
                        st.success(f"""
                        ✅ **Stop Loss Not Triggered**
                        - Your stop loss: -{stop_loss}%
                        - Maximum loss: {results['max_loss']:.2f}%
                        - Your investment stayed within risk limits
                        """)

                with col2:
                    if take_profit_triggered:
                        st.success(f"""
                        🎉 **Take Profit Would Have Triggered!**
                        - Your target: +{take_profit}%
                        - Maximum gain reached: {results['max_gain']:.2f}%
                        - You would have locked in profits
                        """)
                    else:
                        st.warning(f"""
                        ⏳ **Take Profit Not Reached**
                        - Your target: +{take_profit}%
                        - Maximum gain: {results['max_gain']:.2f}%
                        - Consider adjusting your profit target
                        """)

                # Portfolio value over time
                st.subheader("📈 Portfolio Value Over Time")
                period_df = results['period_df']

                fig = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f'{selected_stock} Portfolio Value', 'Daily Returns (%)'),
                    vertical_spacing=0.1
                )

                # Portfolio value line
                fig.add_trace(
                    go.Scatter(
                        x=period_df['Timestamp'],
                        y=period_df['Value'],
                        name='Portfolio Value',
                        line=dict(color=stock_info['color'], width=3),
                        fill='tozeroy',
                        fillcolor=f"rgba({int(stock_info['color'][1:3], 16)}, {int(stock_info['color'][3:5], 16)}, {int(stock_info['color'][5:7], 16)}, 0.2)"
                    ),
                    row=1, col=1
                )

                # Initial investment line
                fig.add_hline(
                    y=investment_amount,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Initial: ${investment_amount:,.2f}",
                    row=1, col=1
                )

                # Stop loss line
                stop_loss_value = investment_amount * (1 - stop_loss / 100)
                fig.add_hline(
                    y=stop_loss_value,
                    line_dash="dot",
                    line_color="red",
                    annotation_text=f"Stop Loss: ${stop_loss_value:,.2f}",
                    row=1, col=1
                )

                # Take profit line
                take_profit_value = investment_amount * (1 + take_profit / 100)
                fig.add_hline(
                    y=take_profit_value,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Take Profit: ${take_profit_value:,.2f}",
                    row=1, col=1
                )

                # Daily returns bar chart
                colors = ['green' if x > 0 else 'red' for x in period_df['Daily_Return'].fillna(0)]
                fig.add_trace(
                    go.Bar(
                        x=period_df['Timestamp'],
                        y=period_df['Daily_Return'] * 100,
                        name='Daily Return %',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )

                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
                fig.update_yaxes(title_text="Return (%)", row=2, col=1)
                fig.update_layout(height=700, showlegend=True, hovermode='x unified')

                st.plotly_chart(fig, use_container_width=True)

                # Investment summary table
                st.subheader("📋 Detailed Investment Summary")

                summary_data = {
                    "Metric": [
                        "Stock Symbol",
                        "Company Name",
                        "Entry Date",
                        "Exit Date",
                        "Holding Period",
                        "Entry Price",
                        "Exit Price",
                        "Price Change",
                        "Shares Purchased",
                        "Investment Amount",
                        "Final Value",
                        "Profit/Loss ($)",
                        "Return (%)",
                        "Avg Daily Return",
                        "Volatility (Annual)",
                        "Max Potential Gain",
                        "Max Potential Loss",
                        "Risk Tolerance",
                        "Stop Loss Setting",
                        "Take Profit Setting"
                    ],
                    "Value": [
                        selected_stock,
                        stock_info['name'],
                        results['entry_date'].strftime('%Y-%m-%d %H:%M'),
                        results['exit_date'].strftime('%Y-%m-%d %H:%M'),
                        f"{(results['exit_date'] - results['entry_date']).days} days",
                        f"${results['entry_price']:.2f}",
                        f"${results['exit_price']:.2f}",
                        f"${results['exit_price'] - results['entry_price']:.2f} ({price_change_pct:+.2f}%)",
                        f"{results['shares']:.4f}",
                        f"${results['investment']:,.2f}",
                        f"${results['exit_value']:,.2f}",
                        f"${results['profit_loss']:+,.2f}",
                        f"{results['return_pct']:+.2f}%",
                        f"{results['avg_daily_return']:+.3f}%",
                        f"{results['volatility']:.2f}%",
                        f"{results['max_gain']:+.2f}% (${results['max_value']:,.2f})",
                        f"{results['max_loss']:+.2f}% (${results['min_value']:,.2f})",
                        risk_tolerance,
                        f"-{stop_loss}%",
                        f"+{take_profit}%"
                    ]
                }
                st.table(pd.DataFrame(summary_data))

                # Investment Recommendation
                st.subheader("💡 Investment Analysis & Recommendation")

                if results['return_pct'] > 0:
                    recommendation_color = "success"
                    recommendation_icon = "✅"
                    recommendation_title = "Profitable Investment"
                else:
                    recommendation_color = "error"
                    recommendation_icon = "❌"
                    recommendation_title = "Loss-Making Investment"

                # Determine performance rating
                if results['return_pct'] > 20:
                    rating = "Excellent"
                elif results['return_pct'] > 10:
                    rating = "Good"
                elif results['return_pct'] > 0:
                    rating = "Moderate"
                else:
                    rating = "Poor"

                # Volatility assessment
                if results['volatility'] > 30:
                    vol_rating = "High Risk"
                elif results['volatility'] > 20:
                    vol_rating = "Moderate Risk"
                else:
                    vol_rating = "Low Risk"

                # Display recommendation
                if recommendation_color == "success":
                    st.success(f"""
                    {recommendation_icon} **{recommendation_title}**

                    **Performance Rating:** {rating}
                    - Total Return: ${results['profit_loss']:+,.2f} ({results['return_pct']:+.2f}%)
                    - Annualized Return: ~{(results['return_pct'] * 365 / holding_period):.2f}% (estimated)

                    **Risk Assessment:** {vol_rating}
                    - Volatility: {results['volatility']:.2f}%
                    - Max Drawdown: {results['max_loss']:.2f}%

                    **Recommendation:**
                    - This investment would have been profitable
                    - {'Consider taking profits if volatility is high' if results['volatility'] > 25 else 'Relatively stable investment'}
                    - {'Stop loss would have protected you' if stop_loss_triggered else 'Good risk management'}
                    """)
                else:
                    st.error(f"""
                    {recommendation_icon} **{recommendation_title}**

                    **Performance Rating:** {rating}
                    - Total Loss: ${results['profit_loss']:,.2f} ({results['return_pct']:.2f}%)

                    **Risk Assessment:** {vol_rating}
                    - Volatility: {results['volatility']:.2f}%
                    - Max Drawdown: {results['max_loss']:.2f}%

                    **Recommendation:**
                    - This investment would have resulted in a loss
                    - Consider reviewing market conditions during this period
                    - {'Stop loss would have limited your losses' if stop_loss_triggered else 'Adjust stop loss settings'}
                    - Wait for better entry points or consider diversification
                    """)

                # Additional insights
                st.info(f"""
                **📊 Key Insights for {selected_stock}:**
                - Best single-day gain: {(period_df['Daily_Return'].max() * 100):.2f}%
                - Worst single-day loss: {(period_df['Daily_Return'].min() * 100):.2f}%
                - Profitable days: {len(period_df[period_df['Daily_Return'] > 0])} / {len(period_df)}
                - Win rate: {(len(period_df[period_df['Daily_Return'] > 0]) / len(period_df) * 100):.1f}%
                """)

    # Help section
    with st.expander("❓ How to Use This Simulator"):
        st.markdown("""
        **Step-by-Step Guide:**

        1. **Select a Stock:** Choose from the top 5 US stocks (AAPL, MSFT, AMZN, GOOGL, TSLA)

        2. **Enter Investment Amount:** Specify how much you want to invest ($100 - $1,000,000)

        3. **Choose Investment Period:**
           - Long-term Hold: Full date range available
           - Short-term Trade: Last 7 days
           - Custom Period: Select specific dates

        4. **Set Risk Parameters:**
           - Stop Loss: Automatic sell when losses exceed this %
           - Take Profit: Automatic sell when profits reach this %

        5. **Run Simulation:** Click the button to see results

        **Understanding Results:**
        - **Green metrics** = Profitable
        - **Red metrics** = Loss
        - **Volatility** = How much the price fluctuates (risk indicator)
        - **Max Gain/Loss** = Best and worst moments during holding period

        **⚠️ Important Notes:**
        - This is a backtesting simulation using historical data
        - Past performance doesn't guarantee future results
        - Always do your own research before investing
        - Consider consulting a financial advisor
        """)
