"""
Section 1: US Stocks Market Overview
Hardcoded information about US stock market indices and popular stocks
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(df=None):
    """Render Section 1: US Stocks Market Overview"""

    st.header("📊 Section 1: US Stocks Market Overview")
    st.write("Live overview of major US stock indices and popular stocks")

    # Major US Indices Information (Hardcoded)
    st.subheader("📈 Major US Stock Indices")

    indices_data = {
        "Index": ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000", "VIX"],
        "Symbol": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
        "Description": [
            "500 largest US companies",
            "30 blue-chip stocks",
            "Tech-heavy index",
            "Small-cap stocks",
            "Volatility index"
        ],
        "Current Value": ["5,850.50", "42,250.75", "18,450.20", "2,125.30", "12.45"],
        "24h Change": ["+1.25%", "+0.85%", "+1.75%", "+0.45%", "-2.30%"]
    }

    indices_df = pd.DataFrame(indices_data)
    st.dataframe(indices_df, use_container_width=True, hide_index=True)

    # Visual comparison of indices
    st.subheader("📊 Index Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Index values chart
        fig_indices = go.Figure()

        indices_names = ["S&P 500", "Dow Jones", "NASDAQ", "Russell 2000"]
        indices_values = [5850.50, 42250.75, 18450.20, 2125.30]
        indices_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig_indices.add_trace(go.Bar(
            x=indices_names,
            y=indices_values,
            marker_color=indices_colors,
            text=[f"{val:,.2f}" for val in indices_values],
            textposition='outside'
        ))

        fig_indices.update_layout(
            title="Current Index Values",
            yaxis_title="Value",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_indices, use_container_width=True)

    with col2:
        # 24h changes
        fig_change = go.Figure()

        changes = [1.25, 0.85, 1.75, 0.45]
        colors_change = ['green' if x > 0 else 'red' for x in changes]

        fig_change.add_trace(go.Bar(
            x=indices_names,
            y=changes,
            marker_color=colors_change,
            text=[f"{val:+.2f}%" for val in changes],
            textposition='outside'
        ))

        fig_change.update_layout(
            title="24-Hour Change (%)",
            yaxis_title="Change %",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_change, use_container_width=True)

    # Top 5 Most Popular Stocks
    st.subheader("⭐ Top 5 Most Popular US Stocks")

    top_stocks = {
        "Rank": ["#1", "#2", "#3", "#4", "#5"],
        "Stock": ["Apple", "Microsoft", "Amazon", "Google", "Tesla"],
        "Symbol": ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"],
        "Sector": ["Technology", "Technology", "Consumer", "Technology", "Automotive"],
        "Price": ["$185.50", "$420.25", "$175.80", "$142.30", "$245.75"],
        "Market Cap": ["$2.9T", "$3.1T", "$1.8T", "$1.7T", "$780B"],
        "24h Change": ["+2.3%", "+1.8%", "+3.2%", "+1.5%", "+5.1%"],
        "Volume": ["52M", "28M", "45M", "22M", "95M"]
    }

    top_stocks_df = pd.DataFrame(top_stocks)
    st.dataframe(top_stocks_df, use_container_width=True, hide_index=True)

    # Stock prices comparison
    st.subheader("💰 Stock Prices Comparison")

    fig_stocks = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Current Stock Prices', 'Market Capitalization'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )

    # Stock prices bar chart
    stock_names = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
    stock_prices = [185.50, 420.25, 175.80, 142.30, 245.75]

    fig_stocks.add_trace(
        go.Bar(
            x=stock_names,
            y=stock_prices,
            marker_color=['#A2AAAD', '#00A4EF', '#FF9900', '#4285F4', '#E31937'],
            text=[f"${p:.2f}" for p in stock_prices],
            textposition='outside',
            name='Price'
        ),
        row=1, col=1
    )

    # Market cap pie chart
    market_caps = [2900, 3100, 1800, 1700, 780]  # in billions

    fig_stocks.add_trace(
        go.Pie(
            labels=stock_names,
            values=market_caps,
            marker_colors=['#A2AAAD', '#00A4EF', '#FF9900', '#4285F4', '#E31937'],
            textinfo='label+percent',
            name='Market Cap'
        ),
        row=1, col=2
    )

    fig_stocks.update_xaxes(title_text="Stock Symbol", row=1, col=1)
    fig_stocks.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig_stocks.update_layout(height=500, showlegend=False)

    st.plotly_chart(fig_stocks, use_container_width=True)

    # Sector Distribution
    st.subheader("🏢 Sector Distribution")

    col1, col2 = st.columns(2)

    with col1:
        sectors_data = {
            "Sector": ["Technology", "Healthcare", "Finance", "Consumer", "Energy", "Other"],
            "Weight %": [28.5, 13.2, 11.8, 10.5, 4.2, 31.8]
        }

        fig_sector = go.Figure(data=[go.Pie(
            labels=sectors_data["Sector"],
            values=sectors_data["Weight %"],
            hole=.4,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        )])

        fig_sector.update_layout(
            title="S&P 500 Sector Weights",
            height=400
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with col2:
        # Performance by sector
        sectors = ["Tech", "Healthcare", "Finance", "Consumer", "Energy"]
        performance = [15.2, 8.5, 6.3, 11.2, -2.1]
        colors = ['green' if x > 0 else 'red' for x in performance]

        fig_perf = go.Figure(data=[go.Bar(
            x=sectors,
            y=performance,
            marker_color=colors,
            text=[f"{p:+.1f}%" for p in performance],
            textposition='outside'
        )])

        fig_perf.update_layout(
            title="YTD Sector Performance",
            yaxis_title="Return %",
            height=400
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    # Market Statistics
    st.subheader("📊 Market Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Market Cap", "$45.2T", "+2.3%")
        st.metric("NYSE Volume", "3.8B", "+5.2%")

    with col2:
        st.metric("NASDAQ Volume", "5.2B", "+8.1%")
        st.metric("Advancing Stocks", "2,145", delta="415")

    with col3:
        st.metric("Declining Stocks", "1,328", delta="-287")
        st.metric("New Highs", "145", delta="23")

    with col4:
        st.metric("New Lows", "34", delta="-12")
        st.metric("Unchanged", "527", delta="5")

    # Quick Stock Info Cards
    st.subheader("📋 Quick Stock Information")

    # Create tabs for each stock
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🍎 AAPL", "🪟 MSFT", "📦 AMZN", "🔍 GOOGL", "⚡ TSLA"])

    stock_info = {
        "AAPL": {
            "name": "Apple Inc.",
            "price": "$185.50",
            "change": "+2.3%",
            "volume": "52M",
            "pe_ratio": "29.5",
            "market_cap": "$2.9T",
            "div_yield": "0.52%",
            "52w_high": "$199.62",
            "52w_low": "$164.08",
            "about": "Apple designs, manufactures, and markets smartphones, computers, tablets, wearables, and accessories."
        },
        "MSFT": {
            "name": "Microsoft Corporation",
            "price": "$420.25",
            "change": "+1.8%",
            "volume": "28M",
            "pe_ratio": "35.2",
            "market_cap": "$3.1T",
            "div_yield": "0.75%",
            "52w_high": "$430.82",
            "52w_low": "$309.45",
            "about": "Microsoft develops, licenses, and supports software, services, devices, and solutions worldwide."
        },
        "AMZN": {
            "name": "Amazon.com Inc.",
            "price": "$175.80",
            "change": "+3.2%",
            "volume": "45M",
            "pe_ratio": "65.3",
            "market_cap": "$1.8T",
            "div_yield": "0.00%",
            "52w_high": "$186.57",
            "52w_low": "$118.35",
            "about": "Amazon is a global e-commerce and cloud computing company offering online retail and AWS services."
        },
        "GOOGL": {
            "name": "Alphabet Inc. (Google)",
            "price": "$142.30",
            "change": "+1.5%",
            "volume": "22M",
            "pe_ratio": "28.7",
            "market_cap": "$1.7T",
            "div_yield": "0.00%",
            "52w_high": "$153.78",
            "52w_low": "$102.21",
            "about": "Alphabet is the parent company of Google, specializing in internet services and products."
        },
        "TSLA": {
            "name": "Tesla Inc.",
            "price": "$245.75",
            "change": "+5.1%",
            "volume": "95M",
            "pe_ratio": "72.5",
            "market_cap": "$780B",
            "div_yield": "0.00%",
            "52w_high": "$299.29",
            "52w_low": "$152.37",
            "about": "Tesla designs, develops, manufactures, and sells electric vehicles and energy storage systems."
        }
    }

    for tab, (symbol, info) in zip([tab1, tab2, tab3, tab4, tab5], stock_info.items()):
        with tab:
            st.write(f"### {info['name']}")
            st.write(info['about'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price", info['price'], info['change'])
                st.metric("Volume", info['volume'])
                st.metric("P/E Ratio", info['pe_ratio'])

            with col2:
                st.metric("Market Cap", info['market_cap'])
                st.metric("Div Yield", info['div_yield'])
                st.metric("52W High", info['52w_high'])

            with col3:
                st.metric("52W Low", info['52w_low'])
                st.info(f"""
                **Key Stats:**
                - Exchange: NASDAQ
                - Industry: {info.get('industry', 'Technology')}
                - Employees: View company profile
                """)

    # Market Sentiment
    st.subheader("😊 Market Sentiment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
        **🟢 Bullish Sentiment**
        - 68% of analysts rate "Buy"
        - Strong institutional buying
        - Positive earnings outlook
        """)

    with col2:
        st.warning("""
        **🟡 Neutral Indicators**
        - VIX at moderate levels
        - Mixed economic signals
        - Fed policy uncertainty
        """)

    with col3:
        st.error("""
        **🔴 Risk Factors**
        - Inflation concerns
        - Geopolitical tensions
        - Interest rate changes
        """)

    # Disclaimer
    st.info("📌 **Note:** This is sample data for demonstration purposes. For real-time data, please use Section 2 with live data feeds.")
