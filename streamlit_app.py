import uuid
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st

# Configuration
DEFAULT_BACKEND_URL = "http://localhost:8086/data-insight"

# Page setup
st.set_page_config(
    page_title="NL2SQL Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple, clean CSS
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 900px;
    }
    
    /* Header styling */
    .chat-header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }
    
    /* Chat messages */
    .chat-message {
        margin: 15px 0;
        padding: 0;
    }
    
    .user-message {
        text-align: right;
    }
    
    .user-bubble {
        display: inline-block;
        background: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        margin-left: 30%;
        word-wrap: break-word;
    }
    
    .assistant-message {
        text-align: left;
    }
    
    .assistant-bubble {
        display: inline-block;
        background: #f1f3f4;
        color: #333;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        margin-right: 30%;
        word-wrap: break-word;
    }
    
    /* Input area */
    .input-area {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 20px 0;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Hide form labels */
    .stTextInput label {
        display: none;
    }
    
    /* Style input box */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 12px 20px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff;
        box-shadow: none;
    }
    
    /* Send button */
    .stButton > button {
        border-radius: 25px;
        background: #007bff;
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: #0056b3;
    }
    
    /* Results styling */
    .result-container {
        margin: 10px 0;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Welcome message */
    .welcome {
        text-align: center;
        padding: 50px 20px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = check_backend_connection()

def check_backend_connection():
    """Check if backend is reachable"""
    try:
        test_url = DEFAULT_BACKEND_URL.replace('/data-insight', '/docs')
        response = requests.get(test_url, timeout=5)
        return response.status_code == 200
    except:
        return False

def send_query(query: str) -> Dict:
    """Send query to backend"""
    payload = {
        "session_id": st.session_state.session_id,
        "user_query": query,
        "chat_id": str(int(datetime.now().timestamp() * 1000))
    }
    
    try:
        # Increase timeout to 120 seconds for complex queries
        response = requests.post(DEFAULT_BACKEND_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "CanIAnswerThePrompt": False,
                "answer_text": f"Error {response.status_code}: {response.text}",
                "response_metadata": {}
            }
    except requests.exceptions.Timeout:
        return {
            "CanIAnswerThePrompt": False,
            "answer_text": "Query is taking longer than expected (2 minutes). The backend may still be processing. Please check the backend logs or try a simpler query.",
            "response_metadata": {}
        }
    except requests.exceptions.ConnectionError:
        return {
            "CanIAnswerThePrompt": False,
            "answer_text": f"Cannot connect to backend at {DEFAULT_BACKEND_URL}. Please ensure the backend server is running on port 8083.",
            "response_metadata": {}
        }
    except Exception as e:
        return {
            "CanIAnswerThePrompt": False,
            "answer_text": f"Connection error: {str(e)}",
            "response_metadata": {}
        }

def display_message(role: str, content: str, metadata: Optional[Dict] = None):
    """Display a chat message"""
    if role == "user":
        st.markdown(
            f"""
            <div class="chat-message user-message">
                <div class="user-bubble">{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat-message assistant-message">
                <div class="assistant-bubble">{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show results if available
        if metadata:
            # Show table
            if metadata.get("answer_table") and len(metadata["answer_table"]) > 1:
                try:
                    df = pd.DataFrame(metadata["answer_table"][1:], columns=metadata["answer_table"][0])
                    st.dataframe(df, use_container_width=True)
                except:
                    pass
            
            # Show SQL (collapsible)
            if metadata.get("generated_sql"):
                with st.expander("View SQL Query"):
                    st.code(metadata["generated_sql"], language="sql")

def main():
    init_session()
    
    # Header
    st.markdown(
        """
        <div class="chat-header">
            <h1>🤖 NL2SQL Agent</h1>
            <p>Ask questions about your data in natural language</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Backend status indicator
    if not st.session_state.backend_status:
        st.error("⚠️ Backend connection failed. Please ensure the FastAPI server is running on port 8083.")
        if st.button("Retry Connection"):
            st.session_state.backend_status = check_backend_connection()
            st.rerun()
    
    # Chat history
    if st.session_state.messages:
        for message in st.session_state.messages:
            display_message(
                message["role"], 
                message["content"], 
                message.get("metadata")
            )
    
    # Input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Type your question here...",
                key="user_query"
            )
        
        with col2:
            submit = st.form_submit_button("Send", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process input
    if submit and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get response
        with st.spinner("Processing your query... This may take up to 2 minutes for complex queries."):
            response = send_query(user_input)
        
        # Add assistant response
        answer_text = response.get("answer_text", "I couldn't process your request.")
        metadata = response.get("response_metadata", {})
        
        if response.get("generated_sql"):
            metadata["generated_sql"] = response["generated_sql"]
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        })
        
        st.rerun()

if __name__ == "__main__":
    main()
