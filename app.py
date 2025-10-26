# app_main.py
import os
import sys
import importlib.util
from pathlib import Path

import streamlit as st

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="GUTS — Market & Portfolio Hub",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ---------------------------
# CUSTOM CSS STYLING
# ---------------------------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #0ea5e9;
        --secondary-color: #6366f1;
        --accent-color: #8b5cf6;
        --text-light: #94a3b8;
        --bg-card: #1e293b;
        --bg-hover: #334155;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .app-card {
        background: linear-gradient(145deg, #1e293b, #334155);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .app-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(14, 165, 233, 0.3);
        border-color: #0ea5e9;
    }
    
    .app-card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .app-card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0ea5e9;
        margin-bottom: 0.5rem;
    }
    
    .app-card-desc {
        color: #94a3b8;
        line-height: 1.6;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Feature list styling */
    .feature-list {
        background: rgba(14, 165, 233, 0.05);
        border-left: 4px solid #0ea5e9;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        color: #cbd5e1;
    }
    
    .feature-item::before {
        content: "✓";
        color: #0ea5e9;
        font-weight: bold;
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid #22c55e;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0ea5e9;
    }
    
    /* Loading animation */
    .loading-container {
        text-align: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        border: 4px solid rgba(14, 165, 233, 0.1);
        border-top: 4px solid #0ea5e9;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# CONFIG: update paths here
# ---------------------------
PROJECT_ROOT = Path("/Users/niteshranjansingh/market-wars-simulator/market-wars-simulator").resolve()
ALGO_PATH = str(PROJECT_ROOT / "app_algo.py")
PORT_PATH = str(Path("/Users/niteshranjansingh/market-wars-simulator/Portfolio_Management/app_port.py").resolve())

# Ensure project root(s) are on sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PORT_PARENT = Path(PORT_PATH).parent
if str(PORT_PARENT) not in sys.path:
    sys.path.insert(0, str(PORT_PARENT))

# ---------------------------
# Helper: load module by path
# ---------------------------
def load_module_from_path(path: str, module_name: str):
    """Dynamically import a module from file system path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Add to sys.modules to prevent re-imports
    spec.loader.exec_module(module)
    return module

# ---------------------------
# Initialize session state
# ---------------------------
if 'current_app' not in st.session_state:
    st.session_state.current_app = "Home"

# Check for query params (for button navigation)
query_params = st.query_params
if "app" in query_params:
    app_param = query_params["app"]
    if app_param == "market":
        st.session_state.current_app = "Market Wars (Strategy Arena)"
    elif app_param == "portfolio":
        st.session_state.current_app = "Investment Dashboard (Portfolio)"

# ---------------------------
# MAIN HEADER
# ---------------------------
st.markdown("""
<div class="main-header">
    <h1>📊 GUTS Hub — Market & Portfolio Intelligence</h1>
    <p>Your unified command center for market strategy simulation and portfolio management</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.markdown("### 🎯 Navigation")
st.sidebar.markdown("---")

choice = st.sidebar.radio(
    "Select Application",
    ["Home", "Market Wars (Strategy Arena)", "Investment Dashboard (Portfolio)"],
    index=["Home", "Market Wars (Strategy Arena)", "Investment Dashboard (Portfolio)"].index(st.session_state.current_app),
    key="nav_radio"
)

if choice != st.session_state.current_app:
    st.session_state.current_app = choice
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 1rem; background: rgba(14, 165, 233, 0.1); border-radius: 8px;">
    <h4 style="color: #0ea5e9; margin: 0 0 0.5rem 0;">Quick Info</h4>
    <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
        Navigate between apps using the menu above. Each app is fully modular and independent.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# HOME VIEW
# ---------------------------
if st.session_state.current_app == "Home":
    
    # Welcome message
    st.markdown("""
    <div class="info-box">
        <h2 style="color: #0ea5e9; margin-top: 0;">Welcome to GUTS Hub 🚀</h2>
        <p style="color: #cbd5e1; font-size: 1.1rem; line-height: 1.6;">
            Your comprehensive platform for quantitative trading strategy development and portfolio management.
            Choose an application below to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # App selection cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="app-card">
            <span class="app-card-icon">⚔️</span>
            <div class="app-card-title">Market Wars — Strategy Arena</div>
            <div class="app-card-desc">
                Advanced algorithmic trading simulator with genetic algorithms, 
                multi-regime analysis, and agent-based modeling.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Market Wars", key="launch_market", use_container_width=True):
            st.session_state.current_app = "Market Wars (Strategy Arena)"
            st.query_params.update({"app": "market"})
            st.rerun()
        
        with st.expander("📋 Features"):
            st.markdown("""
            <div class="feature-list">
                <div class="feature-item">Multi-regime market simulation</div>
                <div class="feature-item">Genetic algorithm optimization</div>
                <div class="feature-item">Agent-based strategy testing</div>
                <div class="feature-item">Real-time performance analytics</div>
                <div class="feature-item">Backtesting framework</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="app-card">
            <span class="app-card-icon">💼</span>
            <div class="app-card-title">Investment Dashboard</div>
            <div class="app-card-desc">
                Real-time portfolio analysis with AI-powered insights, 
                risk metrics, and intelligent investment recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Open Investment Dashboard", key="launch_portfolio", use_container_width=True):
            st.session_state.current_app = "Investment Dashboard (Portfolio)"
            st.query_params.update({"app": "portfolio"})
            st.rerun()
        
        with st.expander("📋 Features"):
            st.markdown("""
            <div class="feature-list">
                <div class="feature-item">Real-time market data integration</div>
                <div class="feature-item">AI-powered analysis assistant</div>
                <div class="feature-item">Portfolio risk metrics</div>
                <div class="feature-item">Performance tracking</div>
                <div class="feature-item">Interactive visualizations</div>
            </div>
            """, unsafe_allow_html=True)
    
    # System status
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_status = "✓ Ready" if os.path.exists(ALGO_PATH) else "✗ Not Found"
        status_color = "#22c55e" if os.path.exists(ALGO_PATH) else "#ef4444"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: rgba(30, 41, 59, 0.5); border-radius: 8px;">
            <div style="font-size: 0.9rem; color: #94a3b8;">Market Wars</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {status_color};">{market_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        port_status = "✓ Ready" if os.path.exists(PORT_PATH) else "✗ Not Found"
        status_color = "#22c55e" if os.path.exists(PORT_PATH) else "#ef4444"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: rgba(30, 41, 59, 0.5); border-radius: 8px;">
            <div style="font-size: 0.9rem; color: #94a3b8;">Portfolio App</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {status_color};">{port_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: rgba(30, 41, 59, 0.5); border-radius: 8px;">
            <div style="font-size: 0.9rem; color: #94a3b8;">System Status</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #22c55e;">✓ Online</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# MARKET WARS APP
# ---------------------------
elif st.session_state.current_app == "Market Wars (Strategy Arena)":
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <h3 style="color: #0ea5e9;">Loading Market Wars Strategy Arena...</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        mod = load_module_from_path(ALGO_PATH, "app_algo_module")
        
        # Clear the loading message
        st.empty()
        
        # Try to call render_app() if it exists
        if hasattr(mod, "render_app"):
            try:
                mod.render_app()
            except Exception as e:
                st.error("⚠️ Market Wars encountered an error while rendering.")
                with st.expander("🔍 View Error Details"):
                    st.exception(e)
        else:
            # If no render_app, the module's top-level code already executed
            st.success("✓ Market Wars loaded successfully")
            st.info("💡 Note: For better integration, add a `render_app()` function to app_algo.py")
            
    except FileNotFoundError as e:
        st.error(f"❌ Market Wars app file not found at: `{ALGO_PATH}`")
        st.info("Please verify the path in the configuration section of app_main.py")
    except Exception as e:
        st.error("❌ Failed to load Market Wars application")
        with st.expander("🔍 View Error Details"):
            st.exception(e)
            st.code(f"Path: {ALGO_PATH}\nProject Root: {PROJECT_ROOT}", language="text")

# ---------------------------
# PORTFOLIO APP
# ---------------------------
elif st.session_state.current_app == "Investment Dashboard (Portfolio)":
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <h3 style="color: #0ea5e9;">Loading Investment Dashboard...</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        mod = load_module_from_path(PORT_PATH, "app_port_module")
        
        # Clear the loading message
        st.empty()
        
        # Try to call render_app() if it exists
        if hasattr(mod, "render_app"):
            try:
                mod.render_app()
            except Exception as e:
                st.error("⚠️ Investment Dashboard encountered an error while rendering.")
                with st.expander("🔍 View Error Details"):
                    st.exception(e)
        else:
            # If no render_app, the module's top-level code already executed
            st.success("✓ Investment Dashboard loaded successfully")
            st.info("💡 Note: For better integration, add a `render_app()` function to app_port.py")
            
    except FileNotFoundError as e:
        st.error(f"❌ Portfolio app file not found at: `{PORT_PATH}`")
        st.info("Please verify the path in the configuration section of app_main.py")
    except Exception as e:
        st.error("❌ Failed to load Investment Dashboard")
        with st.expander("🔍 View Error Details"):
            st.exception(e)
            st.code(f"Path: {PORT_PATH}\nProject Root: {PROJECT_ROOT}", language="text")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0;">
    <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
        Built with ❤️ using Streamlit | GUTS Hub v2.0
    </p>
    <p style="color: #475569; font-size: 0.85rem; margin: 0.5rem 0 0 0;">
        💡 <em>Tip: Each sub-app should define a <code>render_app()</code> function and use unique widget keys</em>
    </p>
</div>
""", unsafe_allow_html=True)