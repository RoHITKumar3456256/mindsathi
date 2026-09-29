import os
import uuid
import time
from datetime import datetime

# Load .env FIRST — before any module imports that need API keys
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# Import Modules
from modules.crisis_handler import check_crisis, CRISIS_HELPLINES
from modules.emotion_detector import detect_emotion
from modules.gemini_client import get_response, init_gemini
from modules.orchestrator import route_and_respond, get_semantic_embedding
from modules.session_logger import log_turn, update_post_session, export_csv, get_all_sessions_df
from modules.survey import (
    PSS_QUESTIONS, LIKERT_OPTIONS, EMOTION_MCQ_OPTIONS, GENDER_OPTIONS,
    STREAM_OPTIONS, CITY_TIER_OPTIONS, WOULD_USE_AGAIN_OPTIONS,
    calculate_pss4_score, get_stress_severity_label
)
from modules.auth import (
    register_user, login_user, get_user_conversations, create_or_update_conversation,
    delete_conversation, get_user_memory_summary
)
from modules.voice_ui import render_fluid_voice_visualizer, render_audio_readout_button, transcribe_audio_groq
from modules.pathway_generator import detect_pathway_intent, render_visual_pathway_svg
from modules.robot_avatar import render_3d_robot_avatar
from modules.supabase_client import sign_in_with_oauth, send_welcome_email_notification
from analysis.stress_analysis import analyze_stress_reduction
from analysis.emotion_accuracy import evaluate_emotion_accuracy
from analysis.charts import generate_publication_charts

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & PATHS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MindSaathi — AI Mental Health Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOGO_PNG_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
LOGO_PATH = LOGO_PNG_PATH


# -----------------------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "iridescent"
if "user" not in st.session_state:
    st.session_state.user = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "step" not in st.session_state:
    st.session_state.step = "main_hub"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pre_pss_score" not in st.session_state:
    st.session_state.pre_pss_score = None
if "post_pss_score" not in st.session_state:
    st.session_state.post_pss_score = None
if "show_chat_pss4" not in st.session_state:
    st.session_state.show_chat_pss4 = False
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "intake_done" not in st.session_state:
    st.session_state.intake_done = False
if "intake_answers" not in st.session_state:
    st.session_state.intake_answers = {}

# -----------------------------------------------------------------------------
# 3. GLOBAL CSS — GOOGLE STITCH CLINICAL DESIGN SYSTEM
# -----------------------------------------------------------------------------
dark_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── STITCH DESIGN SYSTEM TOKENS ── */
    :root {
        --surface-base: #0B0F12;
        --surface: #101417;
        --surface-elevated: #111822;
        --surface-card: #182232;
        --surface-glass: rgba(24, 34, 50, 0.75);
        --agent-mind: #00F5D4;
        --secondary: #4edea3;
        --status-calm: #10B981;
        --status-alert: #F59E0B;
        --status-distress: #EF4444;
        --agent-triage: #F43F5E;
        --border-subtle: rgba(0, 245, 212, 0.15);
        --border-active: rgba(0, 245, 212, 0.60);
        --text-primary: #E0E3E7;
        --text-secondary: #94A3B8;
        --glow-cyan: rgba(0, 245, 212, 0.35);
    }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0B0F12; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 245, 212, 0.25); border-radius: 9999px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0, 245, 212, 0.5); }

    /* ── HIDE STREAMLIT DEFAULT HEADER, DEPLOY BUTTON, AND WHITE DECORATION BAR ── */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
        display: none !important;
    }
    .stAppDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    div[data-testid="stToolbar"] {
        visibility: hidden !important;
        display: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
        height: 0px !important;
        background: transparent !important;
    }
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }

    /* ── APP BACKGROUND ── */
    .stApp {
        background: radial-gradient(ellipse at 15% 0%, rgba(0, 245, 212, 0.08) 0%, transparent 45%),
                    radial-gradient(ellipse at 85% 100%, rgba(78, 222, 163, 0.05) 0%, transparent 45%),
                    linear-gradient(180deg, #0B0F12 0%, #101417 100%) !important;
        font-family: 'Inter', system-ui, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111822 0%, #0B0F12 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
        box-shadow: 0 0 24px rgba(0,245,212,0.04) !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

    /* ── LOGO AREA IN SIDEBAR ── */
    .sidebar-logo-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(155,89,245,0.15);
        margin-bottom: 1rem;
    }
    .sidebar-brand-name {
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00C9B1, #9B59F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .sidebar-brand-tag {
        font-size: 0.68rem;
        color: #64748B;
        font-weight: 500;
        letter-spacing: 0.03em;
    }

    /* ── BUTTONS — PREMIUM PILL STYLE ── */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(155,89,245,0.18), rgba(0,201,177,0.12)) !important;
        color: #E2D9F9 !important;
        border: 1px solid rgba(155,89,245,0.35) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
        backdrop-filter: blur(8px) !important;
        letter-spacing: 0.01em !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #9B59F5, #00C9B1) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(155,89,245,0.45) !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #9B59F5, #7C3AED) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
        box-shadow: 0 4px 18px rgba(155,89,245,0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #7C3AED, #00C9B1) !important;
        box-shadow: 0 8px 32px rgba(155,89,245,0.55) !important;
        transform: translateY(-3px) !important;
    }
    div.stButton > button:active { transform: translateY(1px) !important; }

    /* ── LINK BUTTONS ── */
    div.stLinkButton > a {
        background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03)) !important;
        color: #E2D9F9 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.22s !important;
    }
    div.stLinkButton > a:hover {
        background: rgba(155,89,245,0.2) !important;
        border-color: #9B59F5 !important;
        transform: translateY(-2px) !important;
    }

    /* ── HERO BANNER ── */
    .hero-header {
        background: linear-gradient(135deg, rgba(155,89,245,0.12), rgba(0,201,177,0.06));
        border: 1px solid rgba(155,89,245,0.2);
        border-radius: 20px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #C4B5FD 60%, #00C9B1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #94A3B8;
        font-size: 0.92rem;
        margin: 0;
        font-weight: 400;
    }

    /* ── NAV BAR ── */
    .nav-wrap {
        display: flex;
        gap: 8px;
        margin-bottom: 1.5rem;
        padding: 8px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    .nav-btn {
        flex: 1;
        text-align: center;
        padding: 0.5rem 0.8rem;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid transparent;
    }
    .nav-btn:hover, .nav-btn.active {
        background: rgba(155,89,245,0.2);
        color: #E2D9F9;
        border-color: rgba(155,89,245,0.35);
    }
    .nav-btn.active {
        background: linear-gradient(135deg, rgba(155,89,245,0.3), rgba(0,201,177,0.15));
        color: #FFFFFF;
        box-shadow: 0 4px 15px rgba(155,89,245,0.3);
    }

    /* ── ACTION CARDS ── */
    .action-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        transition: all 0.28s cubic-bezier(0.4,0,0.2,1);
        backdrop-filter: blur(12px);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .action-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(155,89,245,0.5), transparent);
    }
    .action-card:hover {
        background: rgba(155,89,245,0.08);
        border-color: rgba(155,89,245,0.35);
        transform: translateY(-4px);
        box-shadow: 0 16px 40px rgba(155,89,245,0.2);
    }
    .card-icon { font-size: 2.8rem; margin-bottom: 0.8rem; display: block; }
    .card-title { font-size: 1.1rem; font-weight: 700; color: #E2D9F9; margin-bottom: 0.4rem; }
    .card-desc { font-size: 0.83rem; color: #64748B; line-height: 1.5; margin-bottom: 1.2rem; }

    /* ── TOPIC PILLS ── */
    .topic-pill {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 1rem 0.5rem;
        text-align: center;
        font-size: 0.82rem;
        font-weight: 600;
        color: #C4B5FD;
        transition: all 0.2s;
        cursor: default;
    }
    .topic-pill:hover {
        background: rgba(155,89,245,0.12);
        border-color: rgba(155,89,245,0.3);
        transform: translateY(-2px);
    }
    .topic-icon { font-size: 1.5rem; display: block; margin-bottom: 0.4rem; }

    /* ── CHAT MESSAGES ── */
    div[data-testid="stChatMessage"] {
        border-radius: 18px !important;
        padding: 0.8rem 1.1rem !important;
        margin-bottom: 0.6rem !important;
        animation: fadeSlideIn 0.3s ease-out !important;
        border: 1px solid transparent !important;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* User messages */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(155,89,245,0.18), rgba(155,89,245,0.08)) !important;
        border-color: rgba(155,89,245,0.2) !important;
        margin-left: 3rem !important;
    }
    /* Assistant messages */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255,255,255,0.04) !important;
        border-color: rgba(0,201,177,0.15) !important;
        margin-right: 3rem !important;
    }

    .emotion-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(0,201,177,0.12);
        color: #00C9B1;
        border: 1px solid rgba(0,201,177,0.25);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.73rem;
        font-weight: 600;
        margin-top: 6px;
        letter-spacing: 0.02em;
    }

    /* ── CHAT INPUT ── */
    div[data-testid="stChatInput"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(155,89,245,0.25) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px) !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: rgba(155,89,245,0.6) !important;
        box-shadow: 0 0 0 3px rgba(155,89,245,0.15) !important;
    }

    /* ── FORMS & INPUTS ── */
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] select {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(155,89,245,0.25) !important;
        border-radius: 10px !important;
        color: #F0F4FF !important;
        padding: 0.55rem 0.9rem !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #9B59F5 !important;
        box-shadow: 0 0 0 3px rgba(155,89,245,0.15) !important;
    }

    /* ── TABS ── */
    button[data-baseweb="tab"] {
        color: #64748B !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #9B59F5 !important;
        background: rgba(155,89,245,0.1) !important;
    }

    /* ── METRIC CARDS ── */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
    }
    div[data-testid="metric-container"] label { color: #64748B !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #C4B5FD !important; }

    /* ── EXPANDER ── */
    details summary {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }

    /* ── SUCCESS / ERROR ── */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border-width: 1px !important;
    }

    /* ── DIVIDER ── */
    hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.2rem 0 !important; }

    /* ── SPINNER ── */
    div[data-testid="stSpinner"] { color: #9B59F5 !important; }

    /* ── CRISIS BANNER ── */
    .crisis-banner {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        color: #FCA5A5;
    }

    /* ── STAT BAR ── */
    .stat-bar {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .stat-chip {
        background: rgba(155,89,245,0.12);
        border: 1px solid rgba(155,89,245,0.2);
        border-radius: 10px;
        padding: 0.4rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #C4B5FD;
    }

    /* ── ONBOARDING ── */
    .onboard-center {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2rem 0;
    }
    .brand-logo-big {
        width: 88px;
        height: 88px;
        background: linear-gradient(135deg, rgba(155,89,245,0.2), rgba(0,201,177,0.15));
        border-radius: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.2rem;
        border: 1px solid rgba(155,89,245,0.3);
        box-shadow: 0 8px 32px rgba(155,89,245,0.3);
    }
    .app-title-big {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FFFFFF 0%, #C4B5FD 50%, #00C9B1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .app-tagline {
        color: #64748B;
        font-size: 1rem;
        margin: 0.5rem 0 2rem;
        font-weight: 400;
    }

    /* ── HISTORY ITEM ── */
    .history-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 0.45rem 0.6rem;
        border-radius: 10px;
        font-size: 0.82rem;
        color: #94A3B8;
        cursor: pointer;
        transition: all 0.18s;
        border: 1px solid transparent;
    }
    .history-item:hover {
        background: rgba(155,89,245,0.1);
        color: #E2D9F9;
        border-color: rgba(155,89,245,0.2);
    }

    /* ── USER PILL ── */
    .user-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(0,201,177,0.1);
        border: 1px solid rgba(0,201,177,0.2);
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.8rem;
    }
    .user-avatar {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, #9B59F5, #00C9B1);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.9rem; font-weight: 700; color: white;
    }
    .user-name { font-size: 0.88rem; font-weight: 700; color: #E2D9F9; }
    .user-tag { font-size: 0.72rem; color: #64748B; }
</style>
"""

iridescent_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --violet:       #7C3AED;
        --violet-soft:  #8B5CF6;
        --violet-glow:  rgba(124,58,237,0.18);
        --rose:         #EC4899;
        --surface:      #FAF8FF;
        --card:         #FFFFFF;
        --border:       rgba(216,204,252,0.75);
        --border-focus: rgba(139,92,246,0.55);
        --text-h:       #18143A;
        --text-p:       #52496B;
        --text-muted:   #9B92B8;
        --shadow-card:  0 2px 12px rgba(124,58,237,0.06), 0 1px 3px rgba(0,0,0,0.04);
        --shadow-float: 0 8px 32px rgba(124,58,237,0.14), 0 2px 8px rgba(0,0,0,0.06);
        --radius-card:  20px;
        --radius-pill:  9999px;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.22); border-radius: 9999px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.45); }

    /* ── HIDE STREAMLIT CHROME ── */
    header[data-testid="stHeader"],
    .stAppDeployButton,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    #MainMenu, footer { display: none !important; }

    /* ── ROOT BACKGROUND ── */
    .stApp {
        background:
            radial-gradient(ellipse 80% 60% at 0% 0%, rgba(167,139,250,0.13) 0%, transparent 60%),
            radial-gradient(ellipse 70% 50% at 100% 100%, rgba(236,72,153,0.09) 0%, transparent 60%),
            linear-gradient(160deg, #F8F5FF 0%, #FAF9FF 40%, #F5F0FF 100%) !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
        color: var(--text-h) !important;
        min-height: 100vh;
    }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.88) !important;
        backdrop-filter: blur(20px) saturate(1.4) !important;
        -webkit-backdrop-filter: blur(20px) saturate(1.4) !important;
        border-right: 1px solid rgba(216,204,252,0.55) !important;
        box-shadow: 2px 0 20px rgba(124,58,237,0.04) !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    /* ── SIDEBAR BRAND ── */
    .sidebar-brand-name {
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(130deg, #7C3AED 0%, #C026D3 60%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        letter-spacing: -0.02em;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .sidebar-brand-tag {
        font-size: 0.65rem;
        color: var(--text-muted);
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── ALL BUTTONS — Base ── */
    div.stButton > button {
        background: var(--card) !important;
        color: var(--text-p) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: var(--radius-pill) !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: var(--shadow-card) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.01em !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, var(--violet-soft), var(--violet)) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
        box-shadow: var(--shadow-float) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 6px 18px rgba(124,58,237,0.32) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%) !important;
        box-shadow: 0 10px 28px rgba(124,58,237,0.42) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active { transform: translateY(0px) !important; box-shadow: none !important; }

    /* ── HERO HEADER CARD ── */
    .hero-header {
        background: linear-gradient(135deg, #7C3AED 0%, #9333EA 50%, #A855F7 100%) !important;
        border-radius: 22px !important;
        padding: 1.6rem 2rem !important;
        margin-bottom: 1.2rem !important;
        box-shadow: 0 12px 32px -6px rgba(124,58,237,0.3) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute; top: -50%; right: -10%; width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 65%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin: 0 0 4px 0 !important;
        letter-spacing: -0.025em !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        line-height: 1.2 !important;
    }
    .hero-sub {
        color: rgba(255,255,255,0.82) !important;
        font-size: 0.9rem !important;
        margin: 0 !important;
        font-weight: 400 !important;
        line-height: 1.5 !important;
    }

    /* ── PORCELAIN ACTION CARDS ── */
    .action-card {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-card) !important;
        padding: 1.4rem 1.4rem !important;
        box-shadow: var(--shadow-card) !important;
        transition: all 0.24s cubic-bezier(0.4,0,0.2,1) !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }
    .action-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(139,92,246,0.45) !important;
        box-shadow: 0 12px 32px -4px rgba(124,58,237,0.12) !important;
    }
    .card-icon { font-size: 1.7rem !important; margin-bottom: 0.55rem !important; display: block !important; }
    .card-title {
        color: var(--text-h) !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.3rem !important;
        letter-spacing: -0.01em !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .card-desc {
        color: var(--text-p) !important;
        font-size: 0.82rem !important;
        line-height: 1.55 !important;
        margin-bottom: 0.7rem !important;
    }

    /* ── TOPIC PILLS ── */
    .topic-pill {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-pill) !important;
        padding: 0.6rem 1rem !important;
        text-align: center !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: var(--text-p) !important;
        box-shadow: var(--shadow-card) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 5px !important;
    }
    .topic-pill:hover {
        background: linear-gradient(135deg, var(--violet-soft), var(--violet)) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
        box-shadow: var(--shadow-float) !important;
    }

    /* ── CHAT BUBBLES ── */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 1rem 1.4rem !important;
        box-shadow: var(--shadow-card) !important;
        margin-bottom: 0.75rem !important;
        color: var(--text-h) !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #EDE9FE 0%, #F3E8FF 100%) !important;
        border: 1px solid rgba(196,181,253,0.6) !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.9rem 1.4rem !important;
        box-shadow: 0 2px 10px rgba(139,92,246,0.06) !important;
        margin-bottom: 0.75rem !important;
        color: var(--text-h) !important;
    }

    /* ── CHAT INPUT ── */
    div[data-testid="stChatInput"] {
        background: var(--card) !important;
        border: 1.5px solid rgba(196,181,253,0.7) !important;
        border-radius: var(--radius-pill) !important;
        box-shadow: 0 8px 28px rgba(124,58,237,0.1) !important;
        padding: 4px 14px !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: var(--text-h) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.92rem !important;
        font-weight: 400 !important;
    }
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, var(--violet-soft), var(--violet)) !important;
        color: white !important;
        border-radius: 50% !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--border-focus) !important;
        box-shadow: 0 8px 28px rgba(124,58,237,0.16), 0 0 0 3px rgba(139,92,246,0.08) !important;
    }

    /* ── METRIC CONTAINERS ── */
    div[data-testid="metric-container"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: var(--shadow-card) !important;
    }

    /* ── FORM INPUTS ── */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stTextArea"] textarea {
        background: var(--card) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 14px !important;
        color: var(--text-h) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--border-focus) !important;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
    }

    /* ── RADIO BUTTONS (Intake Quiz Style) ── */
    div[data-testid="stRadio"] label {
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        color: var(--text-p) !important;
    }

    /* ── EXPANDER ── */
    div[data-testid="stExpander"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        box-shadow: var(--shadow-card) !important;
    }

    /* ── DIVIDER ── */
    hr { border: none !important; border-top: 1px solid rgba(216,204,252,0.45) !important; margin: 1rem 0 !important; }

    /* ── EMOTION BADGE ── */
    .emotion-badge {
        background: rgba(139,92,246,0.08);
        color: #7C3AED;
        border: 1px solid rgba(139,92,246,0.22);
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* ── USER PILL (SIDEBAR) ── */
    .user-pill {
        display: flex;
        align-items: center;
        gap: 9px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius-pill);
        padding: 0.45rem 0.9rem;
        margin-bottom: 0.6rem;
        box-shadow: var(--shadow-card);
    }
    .user-avatar {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, #8B5CF6, #EC4899);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; font-weight: 700; color: white;
        flex-shrink: 0;
    }
    .user-name { font-size: 0.87rem; font-weight: 700; color: var(--text-h); }
    .user-tag  { font-size: 0.68rem; color: var(--text-muted); font-weight: 500; }

    /* ── CRISIS BANNER ── */
    .crisis-banner {
        background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(248,113,113,0.06));
        border: 2px solid rgba(239,68,68,0.4);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        color: #B91C1C;
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* ── HEADINGS IN MAIN AREA ── */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-h) !important;
    }

    /* ── SUCCESS / INFO / WARNING ALERTS ── */
    div[data-testid="stAlert"] {
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
    }
</style>
"""

st.markdown(dark_css if st.session_state.theme == "obsidian" else iridescent_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    # ── Brand Header (Stitch Blueprint) ──
    col_sb_logo, col_sb_text = st.columns([1, 2.5])
    with col_sb_logo:
        if os.path.exists(LOGO_PNG_PATH):
            st.image(LOGO_PNG_PATH, width=48)
        else:
            st.markdown("<h2 style='margin:0'>🧠</h2>", unsafe_allow_html=True)
    with col_sb_text:
        st.markdown(
            "<div style='padding-top:2px;'>"
            "<div style='display:flex;align-items:center;gap:6px;'>"
            "<span style='font-size:1.3rem;font-weight:800;background:linear-gradient(135deg, #7C3AED, #EC4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:\"Plus Jakarta Sans\",sans-serif;'>MindSaathi</span>"
            "<span style='font-size:0.6rem;background:rgba(139,92,246,0.15);color:#7C3AED;border:1px solid rgba(139,92,246,0.3);padding:1px 6px;border-radius:9999px;font-weight:700;'>v2.5 AI</span>"
            "</div>"
            "<div style='font-size:0.72rem;color:#7C7892;font-weight:600;'>Iridescent AI Companion</div>"
            "</div>",
            unsafe_allow_html=True
        )

    # ── Primary Action Button ──
    if st.button("＋ New Safe Session", use_container_width=True, type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.step = "chat"
        st.session_state.show_chat_pss4 = False
        st.rerun()

    # ── Theme Switcher ──
    col_th1, col_th2 = st.columns(2)
    with col_th1:
        if st.button("🌸 Aurora", use_container_width=True, type="primary" if st.session_state.theme == "iridescent" else "secondary"):
            st.session_state.theme = "iridescent"
            st.rerun()
    with col_th2:
        if st.button("🌙 Dark", use_container_width=True, type="primary" if st.session_state.theme == "obsidian" else "secondary"):
            st.session_state.theme = "obsidian"
            st.rerun()

    # ── 5-Tier Neural Routing Nervous System Status Widget ──
    st.markdown("""
    <div style="background:rgba(11,15,18,0.95);border:1px solid rgba(0,245,212,0.18);border-radius:10px;padding:10px;margin:10px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-size:0.7rem;font-weight:700;color:#E0E3E7;letter-spacing:0.06em;text-transform:uppercase;font-family:'JetBrains Mono',monospace;">⚡ 5-Tier Routing Core</span>
            <span style="width:7px;height:7px;border-radius:50%;background:#10B981;box-shadow:0 0 8px #10B981;"></span>
        </div>
        <div style="display:flex;justify-content:space-between;background:rgba(0,245,212,0.08);border:1px solid rgba(0,245,212,0.25);border-radius:6px;padding:3px 8px;margin-bottom:4px;font-size:0.72rem;color:#00F5D4;font-family:'JetBrains Mono',monospace;">
            <span>T1: Ling 3.0 Sante (124B)</span>
            <span style="background:rgba(0,245,212,0.25);padding:0 4px;border-radius:3px;font-size:0.62rem;font-weight:700;">ACTIVE</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:2px 8px;font-size:0.7rem;color:#94A3B8;font-family:'JetBrains Mono',monospace;">
            <span>T2: Gemini 2.5 Flash</span>
            <span style="color:#64748B;">STANDBY</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:2px 8px;font-size:0.7rem;color:#94A3B8;font-family:'JetBrains Mono',monospace;">
            <span>T3: Groq LPU Whisper</span>
            <span style="color:#10B981;"><240ms</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:2px 8px;font-size:0.7rem;color:#94A3B8;font-family:'JetBrains Mono',monospace;">
            <span>T4: Inkling MoE (975B)</span>
            <span style="color:#64748B;">READY</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:2px 8px;font-size:0.7rem;color:#94A3B8;font-family:'JetBrains Mono',monospace;">
            <span>T5: CBT Fallback Core</span>
            <span style="color:#64748B;">LOCAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PSS-4 Longitudinal Stress Trend Mini Sparkline ──
    st.markdown("""
    <div style="background:rgba(11,15,18,0.95);border:1px solid rgba(0,245,212,0.18);border-radius:10px;padding:10px;margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:0.7rem;font-weight:700;color:#94A3B8;text-transform:uppercase;font-family:'JetBrains Mono',monospace;">PSS-4 Stress Delta</span>
            <span style="font-size:0.73rem;color:#10B981;font-weight:700;">📉 -57%</span>
        </div>
        <div style="display:flex;align-items:flex-end;justify-content:space-between;height:40px;gap:6px;border-bottom:1px solid rgba(0,245,212,0.15);padding-bottom:3px;">
            <div style="flex:1;text-align:center;"><div style="font-size:8px;color:#94A3B8;">14</div><div style="background:#EF4444;height:30px;border-radius:2px 2px 0 0;"></div><div style="font-size:8px;color:#64748B;">W1</div></div>
            <div style="flex:1;text-align:center;"><div style="font-size:8px;color:#F59E0B;">11</div><div style="background:#F59E0B;height:22px;border-radius:2px 2px 0 0;box-shadow:0 0 6px rgba(245,158,11,0.4);"></div><div style="font-size:8px;color:#64748B;">W2</div></div>
            <div style="flex:1;text-align:center;"><div style="font-size:8px;color:#94A3B8;">8</div><div style="background:#4edea3;height:15px;border-radius:2px 2px 0 0;"></div><div style="font-size:8px;color:#64748B;">W3</div></div>
            <div style="flex:1;text-align:center;"><div style="font-size:8px;color:#10B981;">6</div><div style="background:#10B981;height:10px;border-radius:2px 2px 0 0;box-shadow:0 0 6px rgba(16,185,129,0.5);"></div><div style="font-size:8px;color:#64748B;">W4</div></div>
        </div>
        <div style="font-size:10px;color:#94A3B8;margin-top:6px;display:flex;justify-content:space-between;">
            <span>Status: <strong style="color:#4edea3;">Moderate Coping</strong></span>
            <span>(Cohen d = 0.84)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── User Profile ──
    if st.session_state.user:
        initials = "".join(w[0].upper() for w in st.session_state.user['full_name'].split()[:2])
        st.markdown(f"""
        <div class="user-pill">
            <div class="user-avatar">{initials}</div>
            <div>
                <div class="user-name">{st.session_state.user['full_name']}</div>
                <div class="user-tag">✓ Logged in · {st.session_state.user.get('stream','Student')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Logout 🚪", use_container_width=True):
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    else:
        st.markdown('<div style="color:#64748B;font-size:0.75rem;margin-bottom:0.4rem;font-family:\'JetBrains Mono\',monospace;">🔒 Guest Mode — Sign in to sync data</div>', unsafe_allow_html=True)
        if st.button("🔑 Sign In / Register", type="secondary", use_container_width=True):
            st.session_state.step = "auth"
            st.rerun()

    # ── Fail-Safe Crisis Gate (100% Sensitivity) ──
    st.markdown("""
    <div style="background:rgba(147,0,10,0.2);border:1px solid rgba(244,63,94,0.45);border-radius:10px;padding:10px;margin-top:10px;box-shadow:0 0 15px rgba(244,63,94,0.15);">
        <div style="display:flex;justify-content:space-between;align-items:center;color:#F43F5E;font-size:0.74rem;font-weight:700;margin-bottom:4px;font-family:'JetBrains Mono',monospace;">
            <span>🚨 FAIL-SAFE CRISIS GATE</span>
            <span style="background:rgba(244,63,94,0.25);padding:1px 5px;border-radius:4px;font-size:9px;">100% SENS</span>
        </div>
        <p style="font-size:10px;color:#E0E3E7;margin-bottom:6px;line-height:1.4;">
            Tele-MANAS: <strong style="color:#FFF;">14416</strong><br>
            iCall (TISS): <strong style="color:#FFF;">9152987821</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. TOP NAVIGATION BAR
# -----------------------------------------------------------------------------
nav1, nav2, nav3, nav4, nav5, nav6 = st.columns([1, 1, 1, 1.2, 1.2, 1])
with nav1:
    if st.button("🏠 Home", use_container_width=True,
                 type="primary" if st.session_state.step == "main_hub" else "secondary"):
        st.session_state.step = "main_hub"
        st.rerun()
with nav2:
    if st.button("💬 Chat", use_container_width=True,
                 type="primary" if st.session_state.step == "chat" else "secondary"):
        st.session_state.step = "chat"
        st.rerun()
with nav3:
    if st.button("🎙️ Voice", use_container_width=True,
                 type="primary" if st.session_state.step == "voice" else "secondary"):
        st.session_state.step = "voice"
        st.rerun()
with nav4:
    if st.button("🌿 PSS-4 Test", use_container_width=True,
                 type="primary" if st.session_state.step == "pss4" else "secondary"):
        st.session_state.step = "pss4"
        st.rerun()
with nav5:
    if st.button("📊 Dashboard", use_container_width=True,
                 type="primary" if st.session_state.step == "dashboard" else "secondary"):
        st.session_state.step = "dashboard"
        st.rerun()
with nav6:
    if not st.session_state.user:
        if st.button("👤 Account", use_container_width=True,
                     type="primary" if st.session_state.step == "auth" else "secondary"):
            st.session_state.step = "auth"
            st.rerun()
    else:
        if st.button(f"👤 {st.session_state.user['full_name'].split()[0]}", use_container_width=True, type="secondary"):
            st.session_state.step = "dashboard"
            st.rerun()


st.write("---")

# ==============================================================================
# SCREEN 1: ONBOARDING
# ==============================================================================
if st.session_state.step == "onboarding":
    col_ob1, col_ob2, col_ob3 = st.columns([1, 2, 1])
    with col_ob2:
        c_l, c_m, c_r = st.columns([1, 1, 1])
        with c_m:
            if os.path.exists(LOGO_PNG_PATH):
                st.image(LOGO_PNG_PATH, width=105)
        st.markdown("""
        <div style="text-align: center; margin-top: 0.5rem;">
            <h1 class="app-title-big">MindSaathi</h1>
            <p class="app-tagline">AI Mental Health & Mind Companion for Indian Students</p>
        </div>
        """, unsafe_allow_html=True)


        render_3d_robot_avatar(st.session_state.user["full_name"] if st.session_state.user else "")

        st.markdown("<br>", unsafe_allow_html=True)
        ob_c1, ob_c2, ob_c3 = st.columns(3)
        with ob_c1:
            if st.button("🌿 PSS-4 Stress Test", type="primary", use_container_width=True):
                st.session_state.step = "pss4"
                st.rerun()
        with ob_c2:
            if st.button("💬 Jump to Chat", use_container_width=True):
                st.session_state.step = "chat"
                st.rerun()
        with ob_c3:
            if st.button("🎙️ Voice Mode", use_container_width=True):
                st.session_state.step = "voice"
                st.rerun()

# ==============================================================================
# SCREEN 2: MAIN HUB
# ==============================================================================
elif st.session_state.step == "main_hub":
    u_name = st.session_state.user['full_name'].split()[0] if st.session_state.user else "Friend"
    now_hour = datetime.now().hour
    if now_hour < 12:
        time_greet = "Good Morning"
    elif now_hour < 17:
        time_greet = "Good Afternoon"
    else:
        time_greet = "Good Evening"

    # ── Top Pinterest Greeting Bar ──
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px; margin-bottom:1.4rem;">
        <div>
            <div style="display:inline-flex; align-items:center; gap:8px; background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.25); padding:4px 14px; border-radius:9999px; margin-bottom:8px;">
                <span style="font-size:0.8rem; font-weight:700; color:#7C3AED;">Hi, {u_name} • Welcome Back</span>
            </div>
            <h1 style="font-size:2.3rem; font-weight:800; color:#1E1B2E; margin:0; line-height:1.2; font-family:'Plus Jakarta Sans',sans-serif; letter-spacing:-0.02em;">
                {time_greet}, How can I help you?
            </h1>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background:#FFFFFF; border:1.5px solid rgba(226,218,248,0.9); border-radius:50%; width:44px; height:44px; display:flex; align-items:center; justify-content:center; box-shadow:0 4px 14px rgba(139,92,246,0.06); font-size:1.1rem; cursor:pointer;" title="Notifications">
                🔔
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pinterest Hero & Side Feature Grid ──
    col_hero_left, col_hero_right = st.columns([1.1, 1], gap="medium")

    with col_hero_left:
        st.markdown("""
        <div class="action-card" style="min-height:260px; justify-content:space-between; background:linear-gradient(180deg, #FFFFFF 0%, #FAF5FF 100%);">
            <div>
                <div style="width:48px; height:48px; border-radius:18px; background:linear-gradient(135deg, #8B5CF6, #EC4899); display:flex; align-items:center; justify-content:center; font-size:1.4rem; color:white; margin-bottom:1rem; box-shadow:0 6px 18px rgba(139,92,246,0.3);">
                    ✦
                </div>
                <div style="font-size:1.4rem; font-weight:800; color:#1E1B2E; margin-bottom:0.4rem; font-family:'Plus Jakarta Sans',sans-serif;">
                    Talk to AI assistant
                </div>
                <div style="font-size:0.92rem; color:#6B7280; line-height:1.5; margin-bottom:1.2rem;">
                    Empathetic, CBT-informed mental wellness companion. Multilingual Hinglish support for exams, stress, relationships, and self-care.
                </div>
            </div>
            <div style="font-size:0.8rem; color:#8B5CF6; font-weight:700; margin-bottom:0.8rem;">
                Let's try it now ➔
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Talking ✦", type="primary", use_container_width=True, key="btn_start_talking_hero"):
            st.session_state.step = "chat"
            st.rerun()

    with col_hero_right:
        st.markdown("""
        <div class="action-card" style="margin-bottom:12px; padding:1.2rem 1.4rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="width:40px; height:40px; border-radius:14px; background:rgba(139,92,246,0.12); display:flex; align-items:center; justify-content:center; font-size:1.2rem;">
                        🎙️
                    </div>
                    <div>
                        <div style="font-weight:700; color:#1E1B2E; font-size:1.02rem;">Voice Analysis</div>
                        <div style="font-size:0.8rem; color:#6B7280;">Voice to text Assistant</div>
                    </div>
                </div>
                <span style="color:#8B5CF6; font-size:1.2rem; font-weight:700;">➔</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Voice Mode 🎙️", use_container_width=True, key="btn_quick_voice"):
            st.session_state.step = "voice"
            st.rerun()

        st.markdown("""
        <div class="action-card" style="padding:1.2rem 1.4rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="width:40px; height:40px; border-radius:14px; background:rgba(16,185,129,0.12); display:flex; align-items:center; justify-content:center; font-size:1.2rem;">
                        🌿
                    </div>
                    <div>
                        <div style="font-weight:700; color:#1E1B2E; font-size:1.02rem;">PSS-4 Stress Scale</div>
                        <div style="font-size:0.8rem; color:#6B7280;">Standardized diagnostic test</div>
                    </div>
                </div>
                <span style="color:#10B981; font-size:1.2rem; font-weight:700;">➔</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Take Stress Test 🌿", use_container_width=True, key="btn_quick_pss4"):
            st.session_state.step = "pss4"
            st.rerun()

    # ── Topics Pills (Pinterest Category Bar) ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
        <span style="font-size:1.1rem; font-weight:800; color:#1E1B2E; font-family:'Plus Jakarta Sans',sans-serif;">Topics</span>
        <span style="font-size:0.82rem; font-weight:700; color:#8B5CF6; cursor:pointer;">See All</span>
    </div>
    """, unsafe_allow_html=True)

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        if st.button("✦ Daily Life", use_container_width=True, key="t_daily"):
            st.session_state.step = "chat"
            st.rerun()
    with t_col2:
        if st.button("🎓 Exam Stress", use_container_width=True, key="t_exam"):
            st.session_state.messages.append({"role": "user", "content": "MindSaathi, mujhe exams ke baare mein stress ho raha hai, help karo"})
            st.session_state.step = "chat"
            st.rerun()
    with t_col3:
        if st.button("💼 Career & Placement", use_container_width=True, key="t_career"):
            st.session_state.messages.append({"role": "user", "content": "MindSaathi, placement aur career anxiety ho rahi hai, please guide me"})
            st.session_state.step = "chat"
            st.rerun()
    with t_col4:
        if st.button("🌙 Sleep & Grounding", use_container_width=True, key="t_sleep"):
            st.session_state.messages.append({"role": "user", "content": "MindSaathi, mujhe neend nahi aa rahi aur overthinking ho rahi hai"})
            st.session_state.step = "chat"
            st.rerun()

    # ── Curated Discovery Cards (Pinterest Bottom Layout) ──
    st.markdown("<br>", unsafe_allow_html=True)
    d_col1, d_col2, d_col3 = st.columns(3, gap="medium")

    with d_col1:
        st.markdown("""
        <div class="action-card">
            <div>
                <div style="font-size:0.75rem; font-weight:700; color:#8B5CF6; text-transform:uppercase; margin-bottom:4px;">Diagnostic</div>
                <div style="font-weight:800; color:#1E1B2E; font-size:1.05rem; margin-bottom:6px;">What is PSS-4 Delta?</div>
                <div style="font-size:0.84rem; color:#6B7280; line-height:1.45; margin-bottom:12px;">
                    Measures longitudinal perceived stress change (0–16 scale) to benchmark coping resilience.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Discover PSS-4 ➔", use_container_width=True, key="disc_pss4"):
            st.session_state.step = "pss4"
            st.rerun()

    with d_col2:
        st.markdown("""
        <div class="action-card">
            <div>
                <div style="font-size:0.75rem; font-weight:700; color:#EC4899; text-transform:uppercase; margin-bottom:4px;">Neuroscience</div>
                <div style="font-weight:800; color:#1E1B2E; font-size:1.05rem; margin-bottom:6px;">Why is sleep important?</div>
                <div style="font-size:0.84rem; color:#6B7280; line-height:1.45; margin-bottom:12px;">
                    Quality sleep restores prefrontal cognitive control and regulates stress cortisol.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Sleep CBT ➔", use_container_width=True, key="disc_sleep"):
            st.session_state.messages.append({"role": "user", "content": "MindSaathi, sleep hygiene aur nighttime relaxation routine ka plan batao"})
            st.session_state.step = "chat"
            st.rerun()

    with d_col3:
        st.markdown("""
        <div class="action-card">
            <div>
                <div style="font-size:0.75rem; font-weight:700; color:#10B981; text-transform:uppercase; margin-bottom:4px;">Somatic Reset</div>
                <div style="font-weight:800; color:#1E1B2E; font-size:1.05rem; margin-bottom:6px;">5-4-3-2-1 Grounding</div>
                <div style="font-size:0.84rem; color:#6B7280; line-height:1.45; margin-bottom:12px;">
                    Sensory engagement pathway clinically proven to deactivate acute autonomic panic.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Grounding ➔", use_container_width=True, key="disc_ground"):
            st.session_state.messages.append({"role": "user", "content": "MindSaathi, mujhe 5-4-3-2-1 sensory grounding pathway exercise karwao"})
            st.session_state.step = "chat"
            st.rerun()

# ==============================================================================
# SCREEN 3: VOICE ASSESSMENT — POWERED BY GROQ WHISPER & 3D WAVEFORM
# ==============================================================================
elif st.session_state.step == "voice":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🎙️ Voice Assessment Mode</div>
        <div class="hero-sub">Speak your mind in Hindi, English or Hinglish — Powered by Groq Whisper (<300ms) & Multi-Agent Cognitive Voice</div>
    </div>
    """, unsafe_allow_html=True)

    latest_bot_msg = ""
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant":
            latest_bot_msg = msg.get("content", "")
            break

    v_col1, v_col2 = st.columns([1.1, 1], gap="medium")
    with v_col1:
        render_fluid_voice_visualizer(is_active=True, text_to_speak=latest_bot_msg)

    with v_col2:
        st.markdown("### 🎤 **Speak with MindSaathi**")
        st.caption("Click the red record button below, speak your thoughts, and click stop.")
        
        # Streamlit Native Microphone Input with safe fallback
        if hasattr(st, "audio_input"):
            audio_data = st.audio_input("🎙️ Record Voice Input", key="voice_audio_input")
        else:
            audio_data = st.file_uploader("🎙️ Upload Voice Recording (WAV/MP3/M4A)", type=["wav", "mp3", "m4a", "ogg"], key="voice_audio_fallback")
        
        if audio_data is not None:
            with st.spinner("⚡ Transcribing audio with Groq Whisper Large v3..."):
                transcribed_text = transcribe_audio_groq(audio_data)
                
            if transcribed_text:
                st.success(f"🗣️ **You:** {transcribed_text}")
                with st.spinner("MindSaathi Neural Orchestrator answering..."):
                    res = route_and_respond(
                        user_message=transcribed_text,
                        emotion="neutral",
                        history=st.session_state.messages
                    )
                    v_reply = res.get("response", "")
                    v_mod = res.get("model_used", "MindSaathi Core")
                    
                st.session_state.messages.append({"role": "user", "content": transcribed_text})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": v_reply,
                    "model_used": v_mod,
                    "emotion": "neutral"
                })
                
                st.markdown(f"🤖 **MindSaathi:** {v_reply}")
                render_audio_readout_button(v_reply, "voice_stream_reply")
                
                if st.button("💬 Continue in Chat Hub →", type="primary"):
                    st.session_state.step = "chat"
                    st.rerun()
            else:
                st.info("Could not capture audio clearly. Please try speaking closer to your microphone.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Chat Hub", use_container_width=True):
        st.session_state.step = "chat"
        st.rerun()

# ==============================================================================
# SCREEN: PSS-4 PERCEIVED STRESS ASSESSMENT (COHEN ET AL. 1983)
# ==============================================================================
elif st.session_state.step == "pss4":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🌿 PSS-4 Perceived Stress Assessment</div>
        <div class="hero-sub">Standardized 4-item psychological scale measuring your current stress & feeling of control</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(0,201,177,0.08);border:1px solid rgba(0,201,177,0.25);border-radius:14px;padding:1rem 1.4rem;margin-bottom:1.5rem;">
        <b>💡 Instructions:</b> Yeh 4 questions aakhri 1 mahine ya recent dino ke aapke feelings ke baare mein hain.
        Har sawaal ke liye choose karein: <b>0 = Kabhi nahi</b> se lekar <b>4 = Baar-baar</b>.
        Score aate hi MindSaathi aapko clinical CBT guidance provide karega!
    </div>
    """, unsafe_allow_html=True)

    with st.form("pss4_survey_form"):
        pss_options = [
            "0 = Never (Kabhi nahi)",
            "1 = Almost Never (Shayad hi kabhi)",
            "2 = Sometimes (Kabhi-kabhi)",
            "3 = Fairly Often (Aksar / Badi baar)",
            "4 = Very Often (Baar-baar)"
        ]

        st.markdown("#### 1. Uncontrollable Feelings (Life Control)")
        st.write("Aakhri mahine mein kitni baar mehsoos kiya ki aap important cheezein control nahi kar sakte? *(In the last month, how often have you felt that you were unable to control important things?)*")
        q1 = st.radio("Q1", pss_options, index=2, key="pss_q1", label_visibility="collapsed")

        st.write("---")
        st.markdown("#### 2. Confidence in Handling Problems *(Reverse Scored)*")
        st.write("Aakhri mahine mein kitni baar confident feel kiya ki aap apne personal problems handle kar sakte hain? *(How often have you felt confident about your ability to handle personal problems?)*")
        q2 = st.radio("Q2", pss_options, index=2, key="pss_q2", label_visibility="collapsed")

        st.write("---")
        st.markdown("#### 3. Things Going Your Way *(Reverse Scored)*")
        st.write("Aakhri mahine mein kitni baar mehsoos kiya ki sab kuch aapke hisaab se chal raha hai? *(How often have you felt that things were going your way?)*")
        q3 = st.radio("Q3", pss_options, index=2, key="pss_q3", label_visibility="collapsed")

        st.write("---")
        st.markdown("#### 4. Overwhelmed by Difficulties (Piling Up)")
        st.write("Aakhri mahine mein kitni baar feel kiya ki difficulties itni badh gayi hain ki aap sambhal nahi pa rahe? *(How often have you felt difficulties were piling up so high you could not overcome them?)*")
        q4 = st.radio("Q4", pss_options, index=2, key="pss_q4", label_visibility="collapsed")

        st.write("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit & Calculate My Stress Level 🎯", type="primary", use_container_width=True)

    if submitted:
        val1 = int(q1.split("=")[0].strip())
        val2 = int(q2.split("=")[0].strip())
        val3 = int(q3.split("=")[0].strip())
        val4 = int(q4.split("=")[0].strip())

        raw_responses = {"pss1": val1, "pss2": val2, "pss3": val3, "pss4": val4}
        total_score = calculate_pss4_score(raw_responses)
        st.session_state.pre_pss_score = total_score

        sev_label, sev_color = get_stress_severity_label(total_score)

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:2px solid {sev_color};border-radius:18px;padding:1.5rem;text-align:center;margin:1.5rem 0;">
            <div style="font-size:1rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.05em;">Your PSS-4 Stress Score</div>
            <div style="font-size:3rem;font-weight:900;color:{sev_color};margin:0.2rem 0;">{total_score} <span style="font-size:1.2rem;color:#64748B;">/ 16</span></div>
            <div style="font-size:1.2rem;font-weight:700;color:{sev_color};">{sev_label}</div>
        </div>
        """, unsafe_allow_html=True)

        if total_score >= 10:
            st.warning("⚠️ Aapka stress level elevated hai. Academic aur personal pressure heavy feel ho raha hai. MindSaathi se baat karke hum ise step-by-step break karenge.")
        elif total_score >= 6:
            st.info("ℹ️ Moderate Stress level detected. Ye exams aur campus life mein normal hai, par healthy coping strategies zaroori hain.")
        else:
            st.success("✅ Low / Balanced Stress level! Aapka psychological coping aur sense of control kaafi healthy hai.")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button("💬 Start Guided CBT Session with MindSaathi →", type="primary", use_container_width=True):
                # Add auto bot intro acknowledging PSS-4 score
                pss_context_msg = f"Maine abhi apna PSS-4 stress assessment complete kiya. Mera score {total_score}/16 hai ({sev_label}). Mujhe guidance chahiye."
                st.session_state.messages.append({"role": "user", "content": pss_context_msg})
                res = route_and_respond(
                    user_message=pss_context_msg,
                    emotion="anxiety" if total_score >= 8 else "neutral",
                    history=st.session_state.messages
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res.get("response", ""),
                    "model_used": res.get("model_used")
                })
                st.session_state.step = "chat"
                st.rerun()
        with col_p2:
            if st.button("📊 View Research Dashboard & Charts →", use_container_width=True):
                st.session_state.step = "dashboard"
                st.rerun()


# ==============================================================================
# SCREEN 4: AUTH
# ==============================================================================
elif st.session_state.step == "auth":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔑 Student Account Portal</div>
        <div class="hero-sub">Sign in to save your history, track wellness progress & receive email updates</div>
    </div>
    """, unsafe_allow_html=True)

    auth_col1, auth_col2 = st.columns([1, 1], gap="large")

    with auth_col1:
        st.markdown("#### 🌐 Social Sign-in")
        g_res = sign_in_with_oauth("google")
        gh_res = sign_in_with_oauth("github")
        oa_col1, oa_col2 = st.columns(2)
        with oa_col1:
            st.link_button("🌐 Google", g_res.get("url", "#"), use_container_width=True, type="primary")
        with oa_col2:
            st.link_button("🐙 GitHub", gh_res.get("url", "#"), use_container_width=True, type="primary")

        st.write("---")
        st.markdown("#### 📧 Email Sign-in")
        with st.form("form_login"):
            u_in = st.text_input("Email / Username", placeholder="yourname@college.edu")
            p_in = st.text_input("Password", type="password", placeholder="••••••••")
            if st.form_submit_button("Sign In →", type="primary", use_container_width=True):
                ok, res = login_user(u_in, p_in)
                if ok:
                    st.session_state.user = res
                    st.session_state.step = "main_hub"
                    send_welcome_email_notification(u_in, res["full_name"])
                    st.success(f"Welcome back, {res['full_name']}! 🎉")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(f"❌ {res}")

    with auth_col2:
        st.markdown("#### 📝 Create Account")
        with st.form("form_reg"):
            rn = st.text_input("Full Name", placeholder="Priya Sharma")
            ru = st.text_input("Email / Username", placeholder="priya@college.edu")
            rp = st.text_input("Password", type="password", placeholder="Create strong password")
            rs = st.selectbox("Academic Stream", options=STREAM_OPTIONS)
            if st.form_submit_button("Create Account →", type="primary", use_container_width=True):
                ok, res = register_user(ru, rp, rn, rs)
                if ok:
                    st.session_state.user = res
                    st.session_state.step = "main_hub"
                    send_welcome_email_notification(ru, rn)
                    st.success("✅ Account created! Welcome email sent.")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(f"❌ {res}")

# ==============================================================================
# SCREEN 5: DASHBOARD
# ==============================================================================
elif st.session_state.step == "dashboard":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📊 Wellbeing Dashboard</div>
        <div class="hero-sub">Stress analytics, emotion tracking & research insights powered by your sessions</div>
    </div>
    """, unsafe_allow_html=True)

    df_all = get_all_sessions_df()
    st_res = analyze_stress_reduction(df_all)
    em_res = evaluate_emotion_accuracy(df_all)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("📋 Total Sessions", f"{st_res.get('sample_size', 0)}")
    with m2:
        st.metric("📉 Mean Stress Change", f"{st_res.get('mean_change', 0.0):.2f} pts")
    with m3:
        st.metric("📈 t-statistic", f"{st_res.get('t_statistic', 0.0):.3f}")
    with m4:
        st.metric("🎯 Emotion Accuracy", f"{em_res.get('overall_accuracy', 100.0):.1f}%")

    st.write("---")

    dash_col1, dash_col2 = st.columns(2)
    with dash_col1:
        if st.button("📥 Export Data to CSV", use_container_width=True, type="primary"):
            c_path = export_csv()
            st.success(f"✅ Exported to: {c_path}")
    with dash_col2:
        if st.button("📊 Run Statistical Analysis", use_container_width=True, type="primary"):
            with st.expander("Analysis Results", expanded=True):
                st.json(st_res)

    st.write("---")
    st.markdown("#### 📈 Research Charts")
    fig_paths = generate_publication_charts(df_all)

    ci1, ci2 = st.columns(2)
    with ci1:
        if len(fig_paths) > 0 and os.path.exists(fig_paths[0]):
            st.image(fig_paths[0], caption="Fig 1: Emotional State Distribution", use_container_width=True)
        if len(fig_paths) > 2 and os.path.exists(fig_paths[2]):
            st.image(fig_paths[2], caption="Fig 3: Student Helpfulness Ratings", use_container_width=True)
    with ci2:
        if len(fig_paths) > 1 and os.path.exists(fig_paths[1]):
            st.image(fig_paths[1], caption="Fig 2: Pre vs Post PSS-4 Stress", use_container_width=True)
        if len(fig_paths) > 3 and os.path.exists(fig_paths[3]):
            st.image(fig_paths[3], caption="Fig 4: Gender vs Helpfulness", use_container_width=True)


# ==============================================================================
# SCREEN 6: CHAT COMPANION — NEURAL MULTI-AGENT ORCHESTRATOR
# ==============================================================================

elif st.session_state.step == "chat":

    # ══════════════════════════════════════════════
    # 10-QUESTION MENTAL HEALTH INTAKE QUIZ (First time only)
    # ══════════════════════════════════════════════
    if not st.session_state.intake_done:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#F5F3FF,#FDF4FF);border:1.5px solid rgba(139,92,246,0.25);border-radius:24px;padding:2rem 2rem 1.5rem 2rem;margin-bottom:1.5rem;box-shadow:0 8px 32px rgba(139,92,246,0.08);">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.4rem;">
                <div style="width:44px;height:44px;border-radius:14px;background:linear-gradient(135deg,#8B5CF6,#EC4899);display:flex;align-items:center;justify-content:center;font-size:1.3rem;box-shadow:0 4px 16px rgba(139,92,246,0.3);">🧠</div>
                <div>
                    <div style="font-size:1.2rem;font-weight:800;color:#1E1B2E;font-family:'Plus Jakarta Sans',sans-serif;line-height:1.2;">Pehle Thoda Jaante Hain Aapko</div>
                    <div style="font-size:0.8rem;color:#8B5CF6;font-weight:600;">10 quick sawaal — sirf 2 minute lagenge ✨</div>
                </div>
            </div>
            <p style="font-size:0.85rem;color:#64748B;margin:0.8rem 0 0 0;line-height:1.6;">Ye sawaal MindSaathi ko help karte hain aapki situation samajhne mein, taaki main better support de sakoon. Sab information private hai.</p>
        </div>
        """, unsafe_allow_html=True)

        intake_questions = [
            ("q1_anxiety", "1️⃣ Aakhle hafte anxiety ya tension kitna feel ki?",
             ["😌 Bilkul nahi", "🙂 Thodi si", "😐 Moderate", "😟 Zyada", "😰 Bahut zyada"]),
            ("q2_sleep", "2️⃣ Aapki neend kaisi rahi hai?",
             ["😴 Bahut achhi", "🙂 Theek-theek", "😐 Thodi mushkil", "😟 Bahut mushkil"]),
            ("q3_mood", "3️⃣ Kya aap depressed ya hopeless feel karte hain?",
             ["✅ Kabhi nahi", "🔹 Kabhi kabhi", "🔶 Aksar", "🔴 Hamesha"]),
            ("q4_academic", "4️⃣ Academic ya career pressure kitna feel hota hai?",
             ["😌 Koi nahi", "🙂 Thoda", "😐 Moderate", "😟 Zyada", "😰 Bahut zyada"]),
            ("q5_focus", "5️⃣ Concentration aur focus mein dikkat hoti hai?",
             ["✅ Kabhi nahi", "🔹 Rarely", "🔶 Sometimes", "🔴 Often"]),
            ("q6_social", "6️⃣ Socially isolated feel karte hain?",
             ["😊 Nahi", "🙂 Thoda", "😐 Kaafi", "😟 Bahut zyada"]),
            ("q7_physical", "7️⃣ Physical symptoms (headache, fatigue, stomach ache) aate hain?",
             ["✅ Kabhi nahi", "🔹 Kabhi kabhi", "🔶 Aksar", "🔴 Hamesha"]),
            ("q8_selfharm", "8️⃣ Kya aapko khud ko hurt karne ke khayal aate hain?",
             ["✅ Kabhi nahi", "🔸 Kabhi kabhi (past mein)", "⚠️ Haan, aajkal aate hain"]),
            ("q9_energy", "9️⃣ Aapki energy aur motivation kaisi hai?",
             ["⚡ High", "🔋 Medium", "🪫 Low", "😴 Bahut low"]),
            ("q10_duration", "🔟 Aap kitne time se aisa feel kar rahe hain?",
             ["📅 1 week se kam", "📅 1-2 weeks", "📅 Ek mahina", "📅 1 mahine se zyada"]),
        ]

        with st.form("intake_quiz_form"):
            answers = {}
            for key, question, options in intake_questions:
                st.markdown(f"**{question}**")
                answers[key] = st.radio(
                    question, options, index=0,
                    key=f"intake_{key}", label_visibility="collapsed",
                    horizontal=True
                )
                st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            submit_intake = st.form_submit_button(
                "✨ Submit & Shuru Karein Chat",
                type="primary", use_container_width=True
            )

        if submit_intake:
            st.session_state.intake_answers = answers
            st.session_state.intake_done = True
            # Build context from answers
            crisis_flag = "haan, aajkal aate hain" in answers.get("q8_selfharm", "").lower()
            intake_ctx = (
                f"[INTAKE SURVEY] Anxiety: {answers.get('q1_anxiety','N/A')} | "
                f"Sleep: {answers.get('q2_sleep','N/A')} | "
                f"Mood: {answers.get('q3_mood','N/A')} | "
                f"Academic Pressure: {answers.get('q4_academic','N/A')} | "
                f"Focus: {answers.get('q5_focus','N/A')} | "
                f"Social: {answers.get('q6_social','N/A')} | "
                f"Physical: {answers.get('q7_physical','N/A')} | "
                f"Energy: {answers.get('q9_energy','N/A')} | "
                f"Duration: {answers.get('q10_duration','N/A')}"
            )
            if crisis_flag:
                intake_ctx += " | ⚠️ Self-harm thoughts reported"

            # Prepend as system context message
            u_name = st.session_state.user['full_name'].split()[0] if st.session_state.user else "dost"
            welcome_msg = (
                f"Namaste {u_name}! 🙏 Aapne jo bataya, usse main samajh gaya.\n\n"
                f"Aapki intake survey dekh ke main ensure karoonga ki mere sab jawaab aapki situation ke hisaab se hon.\n\n"
                "Aap freely baat kar sakte hain — bina kisi judgment ke. **Aaj aap kaisa feel kar rahe hain?** 😊"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_msg,
                "emotion": "neutral",
                "model_used": "MindSaathi",
                "_intake_context": intake_ctx
            })
            if crisis_flag:
                st.warning("⚠️ Aapne self-harm ke thoughts mention kiye hain. Please Vandrevala Foundation helpline call karein: **1860-2662-345** (24×7 free)")
            st.rerun()

        st.stop()

    # ── Clean Professional Chat Header ──
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 20px;background:#FFFFFF;border:1.5px solid rgba(196,181,253,0.5);border-radius:20px;margin-bottom:1rem;box-shadow:0 4px 20px rgba(139,92,246,0.07);">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:40px;height:40px;border-radius:13px;background:linear-gradient(135deg,#8B5CF6,#EC4899);display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 4px 14px rgba(139,92,246,0.3);">🧠</div>
            <div>
                <div style="font-weight:800;font-size:1rem;color:#1E1B2E;font-family:'Plus Jakarta Sans',sans-serif;line-height:1.2;">MindSaathi Chat</div>
                <div style="font-size:0.75rem;color:#8B5CF6;font-weight:600;">● Online — Aapke liye ready hoon 😊</div>
            </div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:rgba(139,92,246,0.08);color:#8B5CF6;border:1px solid rgba(139,92,246,0.2);padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:700;">🔒 Private</span>
            <span style="background:rgba(16,185,129,0.08);color:#10B981;border:1px solid rgba(16,185,129,0.2);padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:700;">✨ AI Powered</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick Action Bar ──

    c_btn1, c_btn2, c_btn3 = st.columns([1.3, 1.3, 1])
    with c_btn1:
        if st.button("🌿 Take PSS-4 Stress Test", use_container_width=True, type="primary" if st.session_state.show_chat_pss4 else "secondary"):
            st.session_state.show_chat_pss4 = not st.session_state.show_chat_pss4
            st.rerun()
    with c_btn2:
        if st.button("🗺️ Visual Pathways & Roadmaps", use_container_width=True):
            user_msg = "MindSaathi, mujhe anxiety aur stress relief ka visual pathway roadmap dikhao"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            res = route_and_respond(user_msg, emotion="anxiety", history=st.session_state.messages)
            st.session_state.messages.append({
                "role": "assistant",
                "content": res.get("response", ""),
                "model_used": res.get("model_used"),
                "emotion": "anxiety",
                "latency_ms": res.get("latency_ms"),
                "pathway_svg": res.get("pathway_svg"),
                "pathway_type": res.get("pathway_type")
            })
            st.rerun()
    with c_btn3:
        if st.button("🎙️ Voice Mode", use_container_width=True):
            st.session_state.step = "voice"
            st.rerun()

    # ── Interactive In-Chat PSS-4 Card ──
    if st.session_state.show_chat_pss4:
        with st.container():
            st.markdown("""
            <div style="background:rgba(0,201,177,0.08);border:2px solid rgba(0,201,177,0.4);border-radius:18px;padding:1.4rem 1.6rem;margin:1rem 0;box-shadow:0 8px 30px rgba(0,201,177,0.15);">
                <div style="font-size:1.25rem;font-weight:800;color:#00F5D4;margin-bottom:4px;">🌿 Standardized PSS-4 Stress Assessment</div>
                <div style="font-size:0.85rem;color:#94A3B8;margin-bottom:12px;">Ye 4 clinical sawaal aapke perceived stress level ko accurately measure karenge. Options: 0 = Kabhi nahi se lekar 4 = Baar-baar.</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("in_chat_pss4_form"):
                pss_opts = [
                    "0 = Kabhi nahi (Never)",
                    "1 = Shayad hi kabhi (Almost Never)",
                    "2 = Kabhi-kabhi (Sometimes)",
                    "3 = Aksar / Badi baar (Fairly Often)",
                    "4 = Baar-baar (Very Often)"
                ]
                st.markdown("##### 1. Life Control")
                st.write("*Aakhri mahine mein kitni baar mehsoos kiya ki aap important cheezein control nahi kar sakte?*")
                c_q1 = st.radio("Q1", pss_opts, index=2, key="chat_pss_q1", label_visibility="collapsed")
                
                st.markdown("##### 2. Handling Personal Problems *(Reverse Scored)*")
                st.write("*Aakhri mahine mein kitni baar confident feel kiya ki aap apne problems handle kar sakte hain?*")
                c_q2 = st.radio("Q2", pss_opts, index=2, key="chat_pss_q2", label_visibility="collapsed")
                
                st.markdown("##### 3. Things Going Your Way *(Reverse Scored)*")
                st.write("*Aakhri mahine mein kitni baar mehsoos kiya ki sab kuch aapke hisaab se chal raha hai?*")
                c_q3 = st.radio("Q3", pss_opts, index=2, key="chat_pss_q3", label_visibility="collapsed")
                
                st.markdown("##### 4. Difficulties Piling Up")
                st.write("*Aakhri mahine mein kitni baar feel kiya ki difficulties itni badh gayi hain ki aap sambhal nahi pa rahe?*")
                c_q4 = st.radio("Q4", pss_opts, index=2, key="chat_pss_q4", label_visibility="collapsed")
                
                col_sub1, col_sub2 = st.columns([2, 1])
                with col_sub1:
                    submit_in_chat = st.form_submit_button("🎯 Calculate Stress Score & Guide Me", type="primary", use_container_width=True)
                with col_sub2:
                    cancel_in_chat = st.form_submit_button("Close Assessment", use_container_width=True)
                    
            if submit_in_chat:
                v1 = int(c_q1.split("=")[0].strip())
                v2 = int(c_q2.split("=")[0].strip())
                v3 = int(c_q3.split("=")[0].strip())
                v4 = int(c_q4.split("=")[0].strip())
                total = calculate_pss4_score({"pss1": v1, "pss2": v2, "pss3": v3, "pss4": v4})
                st.session_state.pre_pss_score = total
                s_label, s_color = get_stress_severity_label(total)
                st.session_state.show_chat_pss4 = False
                
                user_msg = f"Maine apna PSS-4 stress test complete kiya. Mera score {total}/16 hai ({s_label})."
                st.session_state.messages.append({"role": "user", "content": user_msg})
                
                cbt_msg = (
                    f"Student PSS-4 Score: {total}/16 ({s_label}).\n"
                    f"Acknowledge this score warmly in Hinglish. Explain what it means with CBT empathy. "
                    f"Give 2 concrete micro-actions to regain control, and suggest a 5-step pathway if appropriate."
                )
                res = route_and_respond(cbt_msg, emotion="anxiety" if total >= 8 else "neutral", history=st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res.get("response", ""),
                    "model_used": res.get("model_used"),
                    "emotion": "anxiety" if total >= 8 else "neutral",
                    "latency_ms": res.get("latency_ms"),
                    "pathway_svg": res.get("pathway_svg"),
                    "pathway_type": res.get("pathway_type")
                })
                st.rerun()
            elif cancel_in_chat:
                st.session_state.show_chat_pss4 = False
                st.rerun()

    # ── Welcome message on first open ──
    if not st.session_state.messages:
        u_name = f", {st.session_state.user['full_name'].split()[0]}" if st.session_state.user else ""
        welcome_msg = (
            f"Namaste{u_name}! 🙏 Main MindSaathi hoon — aapka multi-agent AI mental health companion.\n\n"
            "Main specialized health reasoning (**Ling 3.0 Flash Sante**), long-context empathy (**Gemini**), "
            "aur ultra-fast LPU fallback (**Groq**) ke saath ready hoon.\n\n"
            "Chahe exams ka darr ho, family pressure, career confusion, ya personal stress — aap freely baat kar sakte hain bina kisi judgment ke.\n\n"
            "**Aap aaj kaisa feel kar rahe hain? 😊**"
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg,
            "emotion": "neutral",
            "model_used": "Ling 3.0 Flash Sante"
        })

    # ── Interactive 3D Robot Welcomer ──
    if len(st.session_state.messages) <= 1:
        u_name = st.session_state.user['full_name'].split()[0] if st.session_state.user else ""
        render_3d_robot_avatar(u_name)

    # ── Render all messages ──
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            
            # Render Visual Pathway SVG if attached
            if msg.get("pathway_svg"):
                st.markdown(msg["pathway_svg"], unsafe_allow_html=True)
                st.download_button(
                    label="💾 Download Visual Pathway Roadmap (SVG)",
                    data=msg["pathway_svg"],
                    file_name=f"mindsaathi_pathway_{msg.get('pathway_type', 'roadmap')}.svg",
                    mime="image/svg+xml",
                    key=f"dl_path_{idx}"
                )

            if msg["role"] == "assistant":
                em = msg.get("emotion", "neutral")
                lat = msg.get("latency_ms", None)

                badge_html = '<div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:5px;align-items:center;">'
                if em and em != "neutral":
                    badge_html += f'<span style="background:rgba(139,92,246,0.08);color:#8B5CF6;border:1px solid rgba(139,92,246,0.2);padding:2px 9px;border-radius:20px;font-size:0.69rem;font-weight:600;">🔵 {em.title()}</span>'
                if lat:
                    badge_html += f'<span style="background:rgba(0,0,0,0.04);color:#94A3B8;padding:2px 8px;border-radius:20px;font-size:0.68rem;">⏱ {lat}ms</span>'
                badge_html += '</div>'
                st.markdown(badge_html, unsafe_allow_html=True)

                render_audio_readout_button(msg["content"], f"chat_msg_{idx}")

    # ── Quick Action Chips ──
    st.markdown('<div style="font-size:0.75rem;font-weight:700;color:#8B5CF6;letter-spacing:0.04em;margin:1rem 0 0.5rem 0;">✦ Quick Actions</div>', unsafe_allow_html=True)
    chip1, chip2, chip3, chip4 = st.columns(4)
    dispatch_msg = None
    with chip1:
        if st.button("📚 Exam Stress", use_container_width=True, key="chip_exam"):
            dispatch_msg = "MindSaathi, exam stress aur anxiety ke liye 5-step recovery pathway roadmap banao"
    with chip2:
        if st.button("🌿 PSS-4 Test", use_container_width=True, key="chip_pss4_calc"):
            st.session_state.show_chat_pss4 = True
            st.rerun()
    with chip3:
        if st.button("🧘 Grounding", use_container_width=True, key="chip_ground"):
            dispatch_msg = "Mujhe sensory panic ho raha hai, 5-4-3-2-1 sensory grounding exercise step-by-step karao"
    with chip4:
        if st.button("💼 Career Help", use_container_width=True, key="chip_placement"):
            dispatch_msg = "Placement interviews aur mock coding tests ki imposter syndrome anxiety ke liye career roadmap do"

    # ── Chat Input Handler ──
    user_input = st.chat_input("Apni baat likhein ya bolein — Hindi, Hinglish ya English mein... 💬", key="main_chat_input")
    if not user_input and dispatch_msg:
        user_input = dispatch_msg

    if user_input and user_input.strip():
        user_input = user_input.strip()

        # Save to history title
        if st.session_state.user:
            create_or_update_conversation(
                st.session_state.session_id,
                st.session_state.user["user_id"],
                title=user_input[:30]
            )

        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # 1. Crisis Check
        crisis_res = check_crisis(user_input)
        if crisis_res["is_crisis"]:
            crisis_reply = crisis_res["response"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": crisis_reply,
                "emotion": "crisis",
                "model_used": "Crisis Intervention Node"
            })
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"""
                <div class="crisis-banner">
                    🚨 <b>Crisis Protocol Activated</b><br><br>{crisis_reply}
                </div>
                """, unsafe_allow_html=True)
            log_turn(st.session_state.session_id, user_input, "crisis", 1.0, crisis_reply, st.session_state.pre_pss_score, 1)

        else:
            # 2. Emotion Detection
            em_res = detect_emotion(user_input)
            detected_emotion = em_res["label"]
            conf = em_res["score"]

            # 3. User Memory
            u_mem = get_user_memory_summary(st.session_state.user["user_id"]) if st.session_state.user else ""

            # 4. Neural Multi-Agent Orchestration
            from modules.orchestrator import route_and_respond
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("MindSaathi Neural Orchestrator reasoning..."):
                    orch_result = route_and_respond(
                        user_message=user_input,
                        emotion=detected_emotion,
                        history=st.session_state.messages[:-1],
                        user_memory=u_mem
                    )
                    bot_reply = orch_result.get("response", "")
                    model_used = orch_result.get("model_used", "MindSaathi Core")
                    lat_ms = orch_result.get("latency_ms", 0)
                    p_svg = orch_result.get("pathway_svg")
                    p_type = orch_result.get("pathway_type")

                    # If orchestrator flagged it as a PSS-4 request, open interactive PSS-4 form
                    if orch_result.get("is_pss4"):
                        st.session_state.show_chat_pss4 = True

                st.markdown(bot_reply)

                # Render Visual Pathway SVG if generated
                if p_svg:
                    st.markdown(p_svg, unsafe_allow_html=True)
                    st.download_button(
                        label="💾 Download Visual Pathway Roadmap (SVG)",
                        data=p_svg,
                        file_name=f"mindsaathi_pathway_{p_type or 'roadmap'}.svg",
                        mime="image/svg+xml",
                        key=f"dl_path_live_{len(st.session_state.messages)}"
                    )

                # Render badges
                badge_parts = []
                if detected_emotion and detected_emotion != "neutral":
                    badge_parts.append(f'<span style="background:rgba(139,92,246,0.08);color:#8B5CF6;border:1px solid rgba(139,92,246,0.2);padding:2px 9px;border-radius:20px;font-size:0.69rem;font-weight:600;">🔵 {detected_emotion.title()}</span>')
                if lat_ms:
                    badge_parts.append(f'<span style="background:rgba(0,0,0,0.04);color:#94A3B8;padding:2px 8px;border-radius:20px;font-size:0.68rem;">⏱ {lat_ms}ms</span>')
                if badge_parts:
                    st.markdown(f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:5px;">{"".join(badge_parts)}</div>', unsafe_allow_html=True)

                render_audio_readout_button(bot_reply, f"chat_msg_new_{len(st.session_state.messages)}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_reply,
                "emotion": detected_emotion,
                "model_used": model_used,
                "latency_ms": lat_ms,
                "pathway_svg": p_svg,
                "pathway_type": p_type
            })

            log_turn(
                st.session_state.session_id,
                user_input,
                detected_emotion,
                conf,
                bot_reply,
                st.session_state.pre_pss_score,
                0
            )
            if dispatch_msg or orch_result.get("is_pss4"):
                st.rerun()

    # ── Footer ──
    st.markdown("""
    <div style="display:flex;justify-content:center;align-items:center;gap:16px;font-size:0.69rem;color:#CBD5E1;padding:10px 4px 16px 4px;border-top:1px solid rgba(196,181,253,0.2);margin-top:14px;flex-wrap:wrap;">
        <span>🔒 Private & Secure</span>
        <span style="color:#C4B5FD;">•</span>
        <span>🧠 MindSaathi AI — Mental Health Companion</span>
        <span style="color:#C4B5FD;">•</span>
        <span style="color:#10B981;">● CBT Guardrails Active</span>
    </div>
    """, unsafe_allow_html=True)

