import streamlit as st
from chain import get_response

st.set_page_config(page_title="YatriKosha", layout="wide")

# -----------------------------
# 🌟 PREMIUM CSS (ENHANCED)
# -----------------------------
st.markdown("""
<style>

/* 🌌 Background */
.stApp {
    background: radial-gradient(circle at top left, #1e3c72, #2a5298, #000000);
    color: #ffffff;
}

/* 🧭 Header */
.header {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff9966, #ff5e62);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.subtext {
    text-align: center;
    font-size: 16px;
    color: #dcdcdc;
    margin-bottom: 25px;
}

/* ✨ Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 22px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    margin-bottom: 18px;
    line-height: 1.6;
}

/* 💬 Chat Bubbles */
.user-msg {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 10px;
    color: white;
    max-width: 75%;
}

.bot-msg {
    background: rgba(255,255,255,0.1);
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 10px;
    max-width: 80%;
}

/* 📊 Summary Card */
.summary {
    background: linear-gradient(135deg, #ff7e5f, #feb47b);
    padding: 14px;
    border-radius: 14px;
    color: black;
    font-weight: 600;
    margin-top: 10px;
}

/* 📍 Map Container */
.map-box {
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Section Titles */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION
# -----------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {"Trip 1": []}
    st.session_state.current_chat = "Trip 1"

# -----------------------------
# MAP FUNCTIONS
# -----------------------------
def show_map(location):
    url = f"https://www.google.com/maps?q={location.replace(' ', '+')}&output=embed"
    st.markdown("<div class='map-box'>", unsafe_allow_html=True)
    st.components.v1.iframe(url, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

def show_route_map(places):
    if len(places) > 1:
        route = "/".join([p.replace(" ", "+") for p in places[:5]])
        st.markdown(f"[🗺️ Open Route Map](https://www.google.com/maps/dir/{route})")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("🌍 AI Planner")

    if st.button("➕ New Trip"):
        name = f"Trip {len(st.session_state.chats)+1}"
        st.session_state.chats[name] = []
        st.session_state.current_chat = name

    st.radio("Trips", list(st.session_state.chats.keys()), key="current_chat")

    st.markdown("### ⚙️ Settings")
    days = st.slider("Days", 1, 10, 3)
    budget = st.selectbox("Budget", ["Low", "Medium", "High"])
    travel_type = st.selectbox("Travel Type", ["Solo", "Family", "Friends", "Couple"])

    st.markdown("### 🎯 Quick Trips")
    if st.button("🏖️ Beach"):
        st.session_state.quick = "Goa"
    if st.button("🏔️ Hills"):
        st.session_state.quick = "Manali"
    if st.button("🏙️ City"):
        st.session_state.quick = "Paris"

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<div class='header'>YatriKosha Planner</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Smart trips powered by AI ✈️</div>", unsafe_allow_html=True)

st.success("🧠 LangSmith Tracing Enabled (V2)")

# -----------------------------
# CHAT DISPLAY
# -----------------------------
chat_history = st.session_state.chats[st.session_state.current_chat]

for msg in chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>👤 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

# -----------------------------
# INPUT
# -----------------------------
user_input = st.chat_input("Plan your dream trip...")

if "quick" in st.session_state:
    user_input = st.session_state.quick
    del st.session_state.quick

if user_input:
    query = f"{user_input}, {days} days, {budget} budget, {travel_type} trip"

    st.markdown(f"<div class='user-msg'>👤 {query}</div>", unsafe_allow_html=True)
    chat_history.append({"role": "user", "content": query})

    with st.spinner("✈️ Creating your travel plan..."):
        response, places = get_response(query, st.session_state.current_chat)

    # RESPONSE
    st.markdown(f"<div class='glass'>🤖 {response}</div>", unsafe_allow_html=True)

    # Hidden Gem Highlight
    if "hidden gem" in response.lower():
        st.success("🌟 Hidden gem included in your plan!")

    # SUMMARY
    st.markdown(
        f"<div class='summary'>📊 {days} Days • {budget} Budget • {travel_type} Trip</div>",
        unsafe_allow_html=True
    )

    # ROUTE MAP
    if places:
        st.markdown("<div class='section-title'>🗺️ Travel Route</div>", unsafe_allow_html=True)
        show_route_map(places)

    # LOCATION MAPS
    if places:
        st.markdown("<div class='section-title'>📍 Explore Locations</div>", unsafe_allow_html=True)

        for place in places[:5]:
            with st.expander(f"📍 {place}"):
                show_map(place)

    chat_history.append({"role": "assistant", "content": response})
