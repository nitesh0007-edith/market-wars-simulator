"""
Data loading utilities - Auto-load from data folder
"""
import streamlit as st
import pandas as pd
import os

# Data file path
CSV_FILE = "SPY_5min_2months_FullData.csv"


@st.cache_data
def load_data_from_file():
    """
    Automatically load data from CSV file in the project folder

    Returns:
        pandas DataFrame with stock data
    """
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(), f"Data file not found: {CSV_FILE}. Please place your CSV file in the project root folder."

    try:
        df = pd.read_csv(CSV_FILE)

        # Convert Timestamp and remove nanosecond precision for Arrow compatibility
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.floor('s')

        # Ensure numeric columns are proper types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Validate required columns
        required_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in df.columns for col in required_cols):
            return df, None
        else:
            return pd.DataFrame(), f"Missing required columns. Found: {', '.join(df.columns.tolist())}"

    except Exception as e:
        return pd.DataFrame(), f"Error reading file: {e}"
