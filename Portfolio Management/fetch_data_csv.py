"""
Fetch S&P 500 data from Polygon.io API and save to CSV
No database required - just simple CSV files!
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# 🔑 Your Polygon.io API key
API_KEY = "VhIWsZpW1u2mrveC9bWq4jgLpYsWzsdw"  # 🔑 UPDATE THIS IF NEEDED

# 🏦 Use SPY (S&P 500 ETF) — SPX requires a paid plan
SYMBOL = "SPY"

# 📁 Data directory
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "sp500_data.csv")


def fetch_polygon_data(symbol, start_date, end_date, timespan="5", multiplier="minute"):
    """
    Fetch data from Polygon.io API

    Args:
        symbol: Stock ticker symbol (e.g., 'SPY')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        timespan: Time span (minute, hour, day, week, month, quarter, year)
        multiplier: Time multiplier (e.g., '5' for 5-minute intervals)

    Returns:
        pandas DataFrame with stock data
    """
    print(f"📥 Fetching {symbol} data from {start_date} to {end_date}...")

    # 🔗 Polygon API URL for aggregate bars
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': API_KEY
    }

    try:
        # 📥 Fetch data
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes

        data = response.json()

        # Check if results exist
        if 'results' not in data or not data['results']:
            print(f"⚠️ No data returned from API")
            print(f"API Response: {data}")
            return pd.DataFrame()

        results = data['results']
        print(f"✅ Fetched {len(results)} data points")

        # 🧹 Process data
        df = pd.DataFrame(results)

        # Convert timestamp and rename columns with clear names
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        df.rename(columns={
            "t": "Timestamp",
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
            "vw": "VWAP",
            "n": "Trade Count"
        }, inplace=True)

        # Reorder columns in readable order
        df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume", "VWAP", "Trade Count"]]

        # Display first few rows
        print("\n📊 Sample data:")
        print(df.head())
        print(f"\n📈 Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data from API: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return pd.DataFrame()


def fetch_and_save_data(days_back=60):
    """
    Fetch data from Polygon.io and save to CSV

    Args:
        days_back: Number of days of historical data to fetch (default: 60)
    """
    print("=" * 60)
    print("🚀 S&P 500 Data Fetcher (CSV Version)")
    print("=" * 60)

    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✅ Created data directory: {DATA_DIR}")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Fetch data from Polygon.io
    print(f"\n📥 Fetching data from Polygon.io...")
    print(f"Symbol: {SYMBOL}")
    print(f"Date range: {start_str} to {end_str}")
    print(f"Interval: 5 minutes")

    df = fetch_polygon_data(SYMBOL, start_str, end_str, timespan="minute", multiplier="5")

    if df.empty:
        print("❌ No data to save. Exiting.")
        return

    # Check if CSV already exists
    if os.path.exists(CSV_FILE):
        print(f"\n📋 Existing data file found. Merging data...")
        existing_df = pd.read_csv(CSV_FILE)
        existing_df['Timestamp'] = pd.to_datetime(existing_df['Timestamp'])

        # Combine and remove duplicates
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Timestamp'], keep='last')
        combined_df = combined_df.sort_values('Timestamp')

        df = combined_df
        print(f"✅ Merged data. Total records: {len(df)}")

    # Save to CSV
    print(f"\n💾 Saving data to CSV...")
    df.to_csv(CSV_FILE, index=False)
    print(f"✅ Data saved to: {CSV_FILE}")

    # Display statistics
    print(f"\n📊 Summary:")
    print(f"  • Total Records: {len(df):,}")
    print(f"  • Date Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"  • Price Range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"  • Average Price: ${df['Close'].mean():.2f}")
    print(f"  • File Size: {os.path.getsize(CSV_FILE) / 1024:.2f} KB")

    print("\n" + "=" * 60)
    print("✅ Data fetch complete! You can now run the dashboard:")
    print("   streamlit run app.py")
    print("=" * 60)


def load_existing_data():
    """
    Load existing data from CSV file
    """
    if os.path.exists(CSV_FILE):
        print(f"📂 Loading data from: {CSV_FILE}")
        df = pd.read_csv(CSV_FILE)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        print(f"✅ Loaded {len(df):,} records")
        return df
    else:
        print(f"❌ No data file found at: {CSV_FILE}")
        print("Run this script first to fetch data!")
        return pd.DataFrame()


if __name__ == "__main__":
    # Fetch 60 days of historical data
    fetch_and_save_data(days_back=60)

    # Uncomment to fetch more data:
    # fetch_and_save_data(days_back=90)   # 3 months
    # fetch_and_save_data(days_back=180)  # 6 months
    # fetch_and_save_data(days_back=365)  # 1 year
