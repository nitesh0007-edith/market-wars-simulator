"""
Investment Analysis Dashboard - Main Entry Point
Modular structure with separate pages for each section
Data is preloaded from SPY_5min_2months_FullData.csv
Includes AI Investment Chatbot powered by Google Gemini
"""
import streamlit as st
import pandas as pd
from datetime import timedelta
from utils.data_loader import load_data_from_file
from utils.chatbot import render_chatbot_popup
from pages import section1_overview, section2_simulator, section3_analysis

# Page configuration
st.set_page_config(
    page_title="Investment Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application"""

    # Header
    st.markdown('<div class="main-header">📊 Investment Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Render investment chatbot popup
    render_chatbot_popup()

    # Load data once at startup (cached)
    df_full, error_msg = load_data_from_file()

    # Initialize session state for section selection
    if 'selected_section' not in st.session_state:
        st.session_state.selected_section = "Section 1"

    # Main Navigation - 3 Sections with Button-like Interface
    st.markdown("### Choose a Section:")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Section 1: Overview", use_container_width=True, type="primary" if st.session_state.selected_section == "Section 1" else "secondary"):
            st.session_state.selected_section = "Section 1"

    with col2:
        if st.button("💰 Section 2: Investment Simulator", use_container_width=True, type="primary" if st.session_state.selected_section == "Section 2" else "secondary"):
            st.session_state.selected_section = "Section 2"

    with col3:
        if st.button("📈 Section 3: S&P 500 Analysis", use_container_width=True, type="primary" if st.session_state.selected_section == "Section 3" else "secondary"):
            st.session_state.selected_section = "Section 3"

    st.markdown("---")

    # Section 1 doesn't need data - it's hardcoded
    if st.session_state.selected_section == "Section 1":
        section1_overview.render(None)

    # Sections 2 and 3 need data
    else:
        # Initialize df
        df = pd.DataFrame()

        # Sidebar - Data Info and Settings
        with st.sidebar:
            st.header("⚙️ Data Settings")

            # Show data status
            if error_msg:
                st.error(f"❌ {error_msg}")
                st.info("""
                **To load data:**
                1. Place your CSV file as: `SPY_5min_2months_FullData.csv`
                2. File should be in the project root folder
                3. Refresh this page
                """)
            else:
                st.success(f"✅ Data loaded: {len(df_full):,} records")
                st.info(f"**Date Range:**\n{df_full['Timestamp'].min().strftime('%Y-%m-%d %H:%M')} to\n{df_full['Timestamp'].max().strftime('%Y-%m-%d %H:%M')}")

                # Date range selector
                st.subheader("📅 Date Range Filter")
                min_date = df_full['Timestamp'].min().date()
                max_date = df_full['Timestamp'].max().date()

                # Default to last 30 days
                default_start = max(min_date, max_date - timedelta(days=30))

                start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

                if start_date > end_date:
                    st.error("Error: Start date must be before end date")
                    st.stop()

                # Filter data by date range
                mask = (df_full['Timestamp'].dt.date >= start_date) & (df_full['Timestamp'].dt.date <= end_date)
                df = df_full.loc[mask].copy()

                st.markdown("---")
                st.info(f"**Filtered Data:**\n\n📊 {len(df):,} records\n\n📅 {(df['Timestamp'].max() - df['Timestamp'].min()).days} days")

                # Actions
                st.subheader("🔄 Actions")
                if st.button("Refresh Data"):
                    st.cache_data.clear()
                    st.rerun()

                # Download button for filtered data
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=df.to_csv(index=False),
                    file_name=f"sp500_data_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )

        # Render selected section with data
        if st.session_state.selected_section == "Section 2":
            section2_simulator.render(df)

        elif st.session_state.selected_section == "Section 3":
            section3_analysis.render(df)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>Investment Analysis Dashboard | Data from Polygon.io | Built with Streamlit</p>
        <p>⚠️ This is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
