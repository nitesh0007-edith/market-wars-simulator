"""
Data loading utilities - Auto-load from data folder
"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Data file name
CSV_FILENAME = "SPY_5min_2months_FullData.csv"

# Try multiple possible locations for the CSV file
def find_csv_file():
    """Find the CSV file in multiple possible locations"""
    possible_paths = [
        # Path 1: Relative to this file (utils/data_loader.py)
        Path(__file__).parent.parent / CSV_FILENAME,
        # Path 2: Current working directory
        Path.cwd() / "Portfolio_Management" / CSV_FILENAME,
        # Path 3: Just in current directory
        Path.cwd() / CSV_FILENAME,
        # Path 4: Parent directory
        Path(__file__).parent.parent.parent / "Portfolio_Management" / CSV_FILENAME,
    ]

    for path in possible_paths:
        if path.exists():
            return str(path.resolve())

    return None


@st.cache_data
def load_data_from_file():
    """
    Automatically load data from CSV file in the project folder

    Returns:
        pandas DataFrame with stock data, error message
    """
    # Find CSV file
    csv_file = find_csv_file()

    if not csv_file:
        return pd.DataFrame(), f"Data file '{CSV_FILENAME}' not found in any expected location."

    try:
        df = pd.read_csv(csv_file)

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
