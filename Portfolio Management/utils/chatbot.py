"""
Investment Chatbot - Google Gemini Integration
AI-powered chatbot for investment analysis and financial advice
"""
import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in utils folder
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try loading from project root
    root_env_path = Path(__file__).parent.parent / '.env'
    if root_env_path.exists():
        load_dotenv(root_env_path)


class InvestmentChatbot:
    """Investment-focused chatbot using Google Gemini API"""

    def __init__(self, api_key=None):
        """
        Initialize chatbot with Gemini API

        Args:
            api_key: Google Gemini API key (optional, can use env variable)
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in environment or pass as parameter.")

        # Strip any extra whitespace or quotes that might be present
        self.api_key = self.api_key.strip().strip('"').strip("'")

        # Configure Gemini with error handling
        try:
            genai.configure(api_key=self.api_key)
            # Use the latest stable Gemini model
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.initialized = True
        except Exception as e:
            self.initialized = False
            raise ValueError(f"Failed to initialize Gemini API: {str(e)}")

        # System context for investment advice
        self.system_context = """
You are an expert investment advisor and financial analyst AI assistant.

IMPORTANT: Keep all responses SHORT and CONCISE (2-4 sentences maximum).
- Give brief, direct answers
- Focus on key points only
- Use bullet points when listing multiple items
- Avoid lengthy explanations unless specifically asked

You specialize in:
- Stock market analysis and investment strategies
- Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands, etc.)
- Portfolio diversification and risk management
- Market trends and financial metrics

Be professional, concise, and data-driven. Always remind users to do their own research.
"""

    def get_response(self, user_message, chat_history=None):
        """
        Get response from Gemini API with enhanced error handling

        Args:
            user_message: User's question
            chat_history: List of previous messages (optional)

        Returns:
            AI response text
        """
        if not self.initialized:
            return "❌ Chatbot not initialized. Please check your API key."

        try:
            # Build conversation history
            if chat_history and len(chat_history) > 0:
                # Create chat with history
                formatted_history = self._format_history(chat_history)
                chat = self.model.start_chat(history=formatted_history)

                # Add system context to first message if history is empty
                if len(formatted_history) == 0:
                    message_with_context = f"{self.system_context}\n\nUser Question: {user_message}"
                else:
                    message_with_context = user_message

                response = chat.send_message(message_with_context)
            else:
                # Single message with context
                prompt = f"{self.system_context}\n\nUser Question: {user_message}\n\nAssistant:"
                response = self.model.generate_content(prompt)

            return response.text

        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                return "❌ Error: Invalid API key. Please check your GEMINI_API_KEY in the .env file.\n\nGet your API key from: https://makersuite.google.com/app/apikey"
            elif "quota" in error_msg.lower():
                return "❌ Error: API quota exceeded. Please check your API usage or try again later."
            elif "rate" in error_msg.lower():
                return "❌ Error: Rate limit exceeded. Please wait a moment and try again."
            else:
                return f"❌ Error: {error_msg}\n\nPlease try again or check your API configuration."

    def _format_history(self, chat_history):
        """
        Format chat history for Gemini API

        Args:
            chat_history: List of message dicts with 'role' and 'content'

        Returns:
            Formatted history for Gemini
        """
        formatted = []
        for msg in chat_history:
            formatted.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]]
            })
        return formatted

    def get_suggestion_prompts(self):
        """Get suggested questions for users"""
        return [
            "💡 What is the S&P 500 and how does it work?",
            "📊 Explain Moving Averages in technical analysis",
            "⚠️ What is stop loss and how should I use it?",
            "📈 How do I interpret MACD indicators?",
            "🎯 What's the difference between RSI and Bollinger Bands?",
            "💰 How should I diversify my investment portfolio?",
            "📉 What does volatility mean in investing?",
            "🔍 How do I analyze risk-return metrics?"
        ]


def initialize_chatbot():
    """Initialize chatbot in session state with robust error handling"""
    if 'chatbot' not in st.session_state:
        try:
            # Try to get API key from multiple sources
            api_key = None

            # 1. Try Streamlit secrets
            try:
                api_key = st.secrets.get("GEMINI_API_KEY")
            except:
                pass

            # 2. Try environment variable
            if not api_key:
                api_key = os.getenv("GEMINI_API_KEY")

            # 3. Initialize chatbot
            if api_key:
                st.session_state.chatbot = InvestmentChatbot(api_key)
                st.session_state.chat_history = []
                st.session_state.chatbot_initialized = True
            else:
                raise ValueError("API key not found in secrets or environment variables")

        except Exception as e:
            st.session_state.chatbot_initialized = False
            st.session_state.chatbot_error = str(e)


def render_chatbot_popup():
    """Render chatbot as a popup overlay"""

    # Initialize chatbot if needed
    if 'chatbot_initialized' not in st.session_state:
        initialize_chatbot()

    # Chatbot toggle button (fixed position)
    st.markdown("""
        <style>
        .chatbot-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 999;
            background-color: #1f77b4;
            color: white;
            padding: 15px 20px;
            border-radius: 50px;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            font-size: 18px;
            font-weight: bold;
        }
        .chatbot-toggle:hover {
            background-color: #155a8a;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize chat open state
    if 'chat_open' not in st.session_state:
        st.session_state.chat_open = False

    # Create columns for toggle button
    col1, col2, col3 = st.columns([8, 1, 1])

    with col3:
        if st.button("💬 AI Chat", key="chatbot_toggle", use_container_width=True):
            st.session_state.chat_open = not st.session_state.chat_open

    # Render chat window if open
    if st.session_state.chat_open:
        render_chat_window()


def render_chat_window():
    """Render the chat window interface"""

    # Chat window container
    st.markdown("""
        <style>
        .chat-window {
            position: fixed;
            bottom: 80px;
            right: 20px;
            width: 400px;
            max-height: 600px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            z-index: 998;
            display: flex;
            flex-direction: column;
        }
        .chat-header {
            background-color: #1f77b4;
            color: white;
            padding: 15px;
            border-radius: 10px 10px 0 0;
            font-weight: bold;
            font-size: 18px;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            max-height: 400px;
        }
        .chat-input {
            padding: 15px;
            border-top: 1px solid #ddd;
        }
        .user-message {
            background-color: #e3f2fd;
            color: #1a1a1a;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: right;
        }
        .bot-message {
            background-color: #f5f5f5;
            color: #1a1a1a;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Use expander for chat window
    with st.expander("💬 Investment Analysis AI Assistant", expanded=True):

        # Check if chatbot is initialized
        if not st.session_state.get('chatbot_initialized', False):
            error_msg = st.session_state.get('chatbot_error', 'Unknown error')
            st.error(f"❌ Chatbot Error: {error_msg}")

            # Check if it's an API key issue
            if "API_KEY_INVALID" in error_msg or "Invalid API key" in error_msg or "API key not valid" in error_msg:
                st.warning("""
                **⚠️ Invalid API Key Detected**

                Your API key appears to be invalid or expired. Please:
                """)
            else:
                st.warning("**To fix this issue:**")

            st.info("""
            **Setup Instructions:**
            1. Get a valid Google Gemini API key from: https://makersuite.google.com/app/apikey
            2. Update your API key in `utils/.env` file:
               ```
               GEMINI_API_KEY="your-new-api-key-here"
               ```
            3. Or add it to `.streamlit/secrets.toml`:
               ```
               GEMINI_API_KEY = "your-new-api-key-here"
               ```
            4. Restart the app (press Ctrl+C and run again)
            """)

            # Add retry button
            if st.button("🔄 Retry Connection", use_container_width=True):
                # Clear session state and retry
                if 'chatbot' in st.session_state:
                    del st.session_state['chatbot']
                if 'chatbot_initialized' in st.session_state:
                    del st.session_state['chatbot_initialized']
                st.rerun()
            return

        st.markdown("### 🤖 AI Investment Assistant")
        st.caption("Ask me anything about investment analysis, stocks, or technical indicators!")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bot-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.info("👋 Hi! I'm your investment analysis assistant. Ask me anything!")

        # Suggested prompts
        st.markdown("---")
        st.markdown("**💡 Suggested Questions:**")
        suggestions = st.session_state.chatbot.get_suggestion_prompts()

        cols = st.columns(2)
        for idx, suggestion in enumerate(suggestions[:4]):
            with cols[idx % 2]:
                if st.button(suggestion, key=f"suggestion_{idx}", use_container_width=True):
                    st.session_state.pending_message = suggestion.replace("💡 ", "").replace("📊 ", "").replace("⚠️ ", "").replace("📈 ", "")

        # Chat input
        st.markdown("---")
        user_input = st.text_input(
            "Type your question...",
            key="chat_input",
            placeholder="e.g., What is a stop loss?",
            label_visibility="collapsed"
        )

        col1, col2, col3 = st.columns([3, 1, 1])

        with col2:
            send_button = st.button("Send 📤", use_container_width=True)

        with col3:
            if st.button("Clear 🗑️", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Handle pending message from suggestion
        if 'pending_message' in st.session_state:
            user_input = st.session_state.pending_message
            del st.session_state.pending_message
            send_button = True

        # Process message
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })

            # Get bot response
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chatbot.get_response(
                    user_input,
                    st.session_state.chat_history[:-1]  # Exclude current message
                )

            # Add bot response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M")
            })

            # Rerun to update UI
            st.rerun()
