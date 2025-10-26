# 📈 Investment Analysis Dashboard with AI Assistant

A comprehensive investment analysis platform featuring real-time market data analysis, investment simulation, technical indicators, and an AI-powered chatbot for investment guidance.

![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![AI](https://img.shields.io/badge/AI-Gemini-blue)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)
![License](https://img.shields.io/badge/License-Educational-green)

---

## 🎯 Problem Statement

Individual investors face several challenges when analyzing market data and making investment decisions:

1. **Information Overload**: Overwhelming amount of financial data without clear insights
2. **Complex Analysis**: Technical indicators are difficult to understand and interpret
3. **Risk Assessment**: Lack of tools to simulate and evaluate investment scenarios
4. **Limited Guidance**: No accessible expert advice for investment questions
5. **Time Constraints**: Manual data analysis is time-consuming and error-prone

---

## 💡 Proposed Solution

Our **Investment Analysis Dashboard** provides an all-in-one platform that:

### Core Solutions:

1. **📊 Automated Market Analysis**
   - Real-time data processing from Polygon.io API
   - Automatic calculation of key metrics and indicators
   - Visual representation of complex data

2. **🎮 Risk-Free Investment Simulator**
   - Test investment strategies without risking real money
   - Evaluate historical performance with different entry/exit points
   - Calculate risk metrics (volatility, Sharpe ratio, max drawdown)

3. **📈 Advanced Technical Analysis**
   - Pre-configured technical indicators (MACD, RSI, Bollinger Bands)
   - Automated buy/sell signal generation
   - Support and resistance level identification

4. **🤖 AI Investment Assistant**
   - 24/7 expert investment guidance via chatbot
   - Instant answers to investment questions
   - Concise, actionable recommendations
   - Powered by Google Gemini 2.0

---

## ✨ Key Features

### 1. Market Overview Dashboard
- **Real-time Metrics**: Current price, change %, high/low, volume
- **Interactive Charts**: Candlestick and line charts with Plotly
- **Volume Analysis**: Trading volume trends and patterns
- **Quick Statistics**: At-a-glance market performance

### 2. Investment Simulator
- **Portfolio Simulation**: Test investment strategies with historical data
- **Custom Parameters**:
  - Investment amount (any value)
  - Stock symbol (SPY by default)
  - Entry and exit dates
- **Performance Metrics**:
  - Profit/Loss calculation
  - Return on Investment (ROI)
  - Maximum gain and loss
  - Portfolio volatility
- **Visual Analysis**:
  - Portfolio value over time
  - Daily returns chart
  - Performance breakdown

### 3. Technical Analysis Suite

#### Trend Analysis
- **Moving Averages** (SMA-20, SMA-50)
- **MACD** (Moving Average Convergence Divergence)
- **Buy/Sell Signals**: Automated trading signals
- **Trend Interpretation**: Clear trend direction indicators

#### Volatility Analysis
- **Bollinger Bands**: Price volatility channels
- **ATR** (Average True Range)
- **Risk Assessment**: Categorized risk levels (Low/Medium/High)
- **Volatility Metrics**: Standard deviation, percentage volatility

#### Support & Resistance
- **Key Price Levels**: Automatically identified
- **Entry/Exit Points**: Strategic price targets
- **Distance Analysis**: How far current price is from key levels

#### Risk-Return Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Cumulative Returns**: Total performance over time
- **Performance Charts**: Visual risk-return analysis

### 4. AI Investment Chatbot

**Powered by Google Gemini 2.0 Flash**

- **Instant Answers**: Get investment advice in seconds
- **Short & Concise**: 2-4 sentence responses, no information overload
- **Expert Knowledge**:
  - Stock market analysis
  - Technical indicators explanation
  - Portfolio diversification advice
  - Risk management strategies
- **Suggested Prompts**: Quick-start questions for common topics
- **Chat History**: Track conversation for context
- **Professional Tone**: Data-driven, risk-aware recommendations

**Example Interactions**:
- "What is a stop loss?"
- "Explain MACD indicators"
- "How do I diversify my portfolio?"
- "What does RSI mean in trading?"

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (free from [Google AI Studio](https://makersuite.google.com/app/apikey))
- Polygon.io API key (optional, for data fetching)

### Installation

1. **Clone or Download the Project**
```bash
cd GUTS
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API Keys**

Create `utils/.env` file:
```env
GEMINI_API_KEY=your-gemini-api-key-here
```

(Optional) Update Polygon API key in `fetch_data_csv.py`:
```python
API_KEY = "your_polygon_api_key"
```

4. **Fetch Market Data**
```bash
python fetch_data_csv.py
```

5. **Run the Dashboard**
```bash
streamlit run app.py
```

6. **Access the Application**
```
Open browser: http://localhost:8501
```

---

## 📁 Project Structure

```
GUTS/
├── app.py                          # Main application entry point
├── fetch_data_csv.py               # Data fetching from Polygon.io
├── analysis.py                     # Technical analysis functions
├── requirements.txt                # Python dependencies
│
├── pages/                          # Dashboard sections
│   ├── section1_overview.py        # Market overview dashboard
│   ├── section2_simulator.py       # Investment simulator
│   └── section3_analysis.py        # Technical analysis suite
│
├── utils/                          # Utility modules
│   ├── data_loader.py              # Data loading and caching
│   ├── simulator.py                # Investment simulation logic
│   ├── chatbot.py                  # AI chatbot integration
│   └── .env                        # Environment variables (API keys)
│
└── data/                           # Data storage
    └── sp500_data.csv              # Market data (auto-generated)
```

---

## 💻 Usage Guide

### Step 1: Navigate Sections

The dashboard has 3 main sections accessible via button navigation:

1. **📊 Section 1: Overview** - Market snapshot and current metrics
2. **💰 Section 2: Investment Simulator** - Test investment strategies
3. **📈 Section 3: S&P 500 Analysis** - Technical analysis tools

### Step 2: Use the Investment Simulator

1. Go to **Section 2**
2. Enter investment amount (e.g., $10,000)
3. Select stock symbol (SPY)
4. Choose entry date and exit date
5. Click **Run Simulation**
6. Analyze results:
   - View profit/loss
   - Check risk metrics
   - Examine portfolio charts

### Step 3: Explore Technical Analysis

1. Go to **Section 3**
2. Select analysis type from tabs:
   - Trend Analysis
   - Volatility Analysis
   - Support & Resistance
   - Risk-Return Metrics
3. Read auto-generated insights
4. Use "How to Interpret" guides
5. Export data if needed

### Step 4: Ask the AI Chatbot

1. Click **💬 AI Chat** button (top right)
2. Chat window opens with suggested questions
3. Type your question or click a suggestion
4. Get instant, concise answers
5. Continue conversation for follow-ups

---

## 🛠️ Technical Stack

### Backend
- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data processing and analysis
- **Streamlit**: Web application framework

### AI & APIs
- **Google Gemini 2.0 Flash**: AI chatbot engine
- **Polygon.io API**: Real-time market data
- **python-dotenv**: Environment variable management

### Visualization
- **Plotly**: Interactive charts and graphs
- **Streamlit Components**: UI elements

### Data Storage
- **CSV Files**: Lightweight data storage (no database required)
- **Streamlit Cache**: Performance optimization

---

## 🎨 Dashboard Features

### Data Management
- ✅ **Automatic Loading**: Data preloads from CSV on startup
- ✅ **Smart Caching**: Fast performance with Streamlit's cache
- ✅ **Date Filtering**: Custom date range selection
- ✅ **Data Refresh**: One-click data reload
- ✅ **CSV Export**: Download filtered data

### Analysis Tools
- ✅ **4 Technical Indicators**: SMA, MACD, Bollinger Bands, RSI
- ✅ **Automated Signals**: Buy/sell recommendations
- ✅ **Risk Metrics**: Volatility, Sharpe ratio, drawdown
- ✅ **Visual Charts**: Interactive Plotly graphs

### User Experience
- ✅ **Responsive Design**: Works on desktop and mobile
- ✅ **Intuitive Navigation**: Clear section buttons
- ✅ **Helpful Guides**: "How to Interpret" for each analysis
- ✅ **Color-Coded**: Visual indicators for trends and signals
- ✅ **No Setup Required**: Works out of the box

### AI Assistant
- ✅ **Instant Responses**: Sub-second answer time
- ✅ **Concise Answers**: 2-4 sentences maximum
- ✅ **Suggested Prompts**: 8 common investment questions
- ✅ **Chat History**: Maintains conversation context
- ✅ **Error Handling**: Clear error messages and solutions

---

## 📊 Data Flow

```
Polygon.io API
    ↓
fetch_data_csv.py (Fetches & Saves)
    ↓
data/sp500_data.csv
    ↓
data_loader.py (Loads & Caches)
    ↓
app.py (Distributes to Sections)
    ↓
├── Section 1: Overview
├── Section 2: Simulator
└── Section 3: Analysis
```

**Chatbot Flow**:
```
User Question
    ↓
chatbot.py
    ↓
Google Gemini API
    ↓
AI Response
    ↓
User Interface
```

---

## 🔧 Configuration

### API Keys

**Gemini API (Required for Chatbot)**:
- Get key: https://makersuite.google.com/app/apikey
- Add to `utils/.env`: `GEMINI_API_KEY=your-key-here`

**Polygon.io API (Optional for Data Fetching)**:
- Free tier: 5 calls/minute
- Update in `fetch_data_csv.py`

### Data Fetching Options

**Change Date Range**:
```python
# In fetch_data_csv.py
fetch_and_save_data(days_back=90)  # 3 months instead of 60
```

**Change Interval**:
```python
# In fetch_data_csv.py
df = fetch_polygon_data(
    SYMBOL, start_str, end_str,
    timespan="hour",    # hourly instead of 5-minute
    multiplier="1"
)
```

---

## ⚠️ Important Notes

### For Users
- 📌 **Educational Only**: Not financial advice, always DYOR (Do Your Own Research)
- 📌 **Historical Data**: Past performance doesn't guarantee future results
- 📌 **Risk Warning**: All investments carry risk
- 📌 **Simulation**: Test strategies before real trading

### Technical Notes
- 📌 **API Limits**: Polygon.io free tier has rate limits
- 📌 **Data Requirements**: CSV must have: Timestamp, Open, High, Low, Close, Volume
- 📌 **Chatbot Needs Key**: AI features require valid Gemini API key
- 📌 **Internet Required**: For API calls and data fetching

---

## 🎓 How It Helps Investors

### Problem: Information Overload
**Solution**: Clean, organized dashboard with only essential metrics

### Problem: Complex Analysis
**Solution**: Pre-calculated indicators with "How to Interpret" guides

### Problem: Risk Assessment
**Solution**: Investment simulator with comprehensive risk metrics

### Problem: Limited Guidance
**Solution**: AI chatbot available 24/7 for instant advice

### Problem: Time Constraints
**Solution**: Automated analysis, one-click simulations, fast responses

---

**Happy Investing! 📈**
