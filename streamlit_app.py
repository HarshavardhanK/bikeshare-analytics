import streamlit as st
import requests
import time
import os
import json
import traceback

def env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    if not v:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)

# Initialize session state first (before any st.markdown calls)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_response" not in st.session_state:
    st.session_state.pending_response = False
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = None

# API Configuration
API_BASE_URL = (os.getenv("API_BASE_URL", "http://api:8000") or "http://api:8000").rstrip("/")
CONNECT_TIMEOUT = env_float("API_CONNECT_TIMEOUT", 5.0)
READ_TIMEOUT = env_float("API_READ_TIMEOUT", 60.0)

def call_api(question: str):
    url = f"{API_BASE_URL}/query"
    headers = {"Content-Type": "application/json"}
    payload = {"question": question or ""}

    t0 = time.time()
    try:
        # Separate connect/read timeouts; 5s connect, 60s read by default
        resp = requests.post(url, headers=headers, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except requests.exceptions.Timeout as e:
        st.error(f"API timeout after {READ_TIMEOUT}s. Details: {e}")
        st.code(traceback.format_exc())
        return None
    except requests.exceptions.RequestException as e:
        # Catches ConnectionError, SSLError, etc.
        st.error("Failed to reach API (RequestException).")
        st.code(f"{e}\n\n{traceback.format_exc()}")
        return None

    elapsed = time.time() - t0
    if resp.status_code != 200:
        st.error(f"API returned non-200 status: {resp.status_code}")
        return None

    try:
        data = resp.json()
    except Exception:
        st.error("API returned invalid JSON.")
        return None

    # Debug info removed for clean UI

    # Validate contract
    if not isinstance(data, dict) or not {"sql","result","error"} <= set(data.keys()):
        st.error("API contract mismatch. Expected keys: sql, result, error.")
        st.json(data)
        return None

    return data

# Custom dark blue theme and minimalistic CSS
st.markdown(
    """
    <style>
    body, .stApp { background-color: #0a2342 !important; }
    .main { background-color: #0a2342 !important; }
    .stChatMessage { background: none !important; }
    .user-bubble {
        background: #1e3a5c;
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        margin: 8px 0 8px 40px;
        max-width: 70%;
        float: right;
        font-size: 1.1em;
    }
    .assistant-bubble {
        background: #fff;
        color: #0a2342;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        margin: 8px 40px 8px 0;
        max-width: 70%;
        float: left;
        font-size: 1.1em;
        display: flex;
        align-items: center;
    }
    .bike-avatar {
        width: 32px;
        height: 32px;
        margin-right: 10px;
        display: inline-block;
        vertical-align: middle;
    }
    .logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 60px;
        margin-bottom: 10px;
    }
    .stTextInput > div > div > input {
        background: #1e3a5c !important;
        color: #fff !important;
        border-radius: 8px;
        border: 1px solid #274472;
    }
    .stTextInput > label { color: #fff !important; }
    .stChatInputContainer { background: #1e3a5c !important; }
    .stSpinner { color: #fff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Minimal bike SVG logo
bike_logo_svg = '''
<svg class="logo" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="16" cy="48" r="10" stroke="#1e90ff" stroke-width="3" fill="#fff"/>
<circle cx="48" cy="48" r="10" stroke="#1e90ff" stroke-width="3" fill="#fff"/>
<rect x="28" y="44" width="8" height="4" rx="2" fill="#1e90ff"/>
<path d="M16 48 L32 16 L48 48" stroke="#1e90ff" stroke-width="3" fill="none"/>
<circle cx="32" cy="16" r="3" fill="#1e90ff"/>
</svg>
'''

# Bike SVG for assistant avatar
bike_avatar_svg = '''
<svg class="bike-avatar" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="8" cy="24" r="6" stroke="#1e90ff" stroke-width="2" fill="#fff"/>
<circle cx="24" cy="24" r="6" stroke="#1e90ff" stroke-width="2" fill="#fff"/>
<rect x="13" y="21" width="6" height="2" rx="1" fill="#1e90ff"/>
<path d="M8 24 L16 8 L24 24" stroke="#1e90ff" stroke-width="2" fill="none"/>
<circle cx="16" cy="8" r="2" fill="#1e90ff"/>
</svg>
'''

st.markdown(bike_logo_svg, unsafe_allow_html=True)
st.markdown('<h2 style="text-align:center; color:#fff; font-family:sans-serif; margin-bottom: 0.5em;">Bikeshare Chatbot</h2>', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{bike_avatar_svg}<span>{msg["content"]}</span></div>', unsafe_allow_html=True)

user_input = st.chat_input("Type your question and hit Enter…")

if user_input and not st.session_state.pending_response:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.pending_response = True
    st.session_state.last_user_input = user_input
    st.rerun()

if st.session_state.pending_response and st.session_state.last_user_input:
    with st.spinner("Querying..."):
        try:
            data = call_api(st.session_state.last_user_input)
        except Exception as e:
            st.error(f"Exception in call_api: {e}")
            st.code(traceback.format_exc())
            data = None
    
    if data is None:
        st.error("Failed to get response from API")
        st.session_state.pending_response = False
        st.session_state.last_user_input = None
        st.stop()
    
    if data.get("error"):
        answer = data["error"]
    else:
        answer = data.get("result", "No result returned")
    
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.pending_response = False
    st.session_state.last_user_input = None
    st.rerun()
