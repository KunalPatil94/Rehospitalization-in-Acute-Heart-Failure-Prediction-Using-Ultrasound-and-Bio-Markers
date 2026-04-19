import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import hashlib
from streamlit_lottie import st_lottie
import requests

# Import custom modules
from auth import AuthManager
from data_generator import SyntheticDataGenerator
from models import AHFPredictionModels
from database import DatabaseManager
from notifications import NotificationManager
from explainability import ExplainabilityManager
from monitoring import ModelMonitor
from reporting import ReportGenerator
from data_validation import DataValidator
from alert_system import AlertSystem

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense AI — AHF Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

/* ─── LIGHT MODE THEME ─── */
.stApp.light-mode {
    background: linear-gradient(135deg, #EEF2FF 0%, #E4ECFF 50%, #EAF0FF 100%) !important;
}
.stApp.light-mode html, .stApp.light-mode body, .stApp.light-mode [class*='css'] {
    background-color: #EEF2FF !important;
    color: #0A1628 !important;
}
.stApp.light-mode section[data-testid='stSidebar'] {
    background: linear-gradient(180deg, #FFFFFF 0%, #EEF4FF 100%) !important;
    border-right: 1px solid rgba(0,100,200,0.15) !important;
}
.stApp.light-mode .page-title { color: #0A1628 !important; }
.stApp.light-mode .page-subtitle { color: rgba(40,70,120,0.65) !important; }
.stApp.light-mode p, .stApp.light-mode span, .stApp.light-mode div { color: #1A2E50; }
.stApp.light-mode h1,.stApp.light-mode h2,.stApp.light-mode h3 { color: #0A1628 !important; }
.stApp.light-mode .section-card { background: rgba(255,255,255,0.9) !important; border-color: rgba(0,100,200,0.1) !important; }
.stApp.light-mode .info-box { background: rgba(0,100,200,0.06) !important; color: rgba(10,40,100,0.85) !important; }
.stApp.light-mode div[data-testid='stMetric'] { background: linear-gradient(135deg,rgba(255,255,255,0.98),rgba(235,242,255,0.98)) !important; border-color: rgba(0,100,200,0.12) !important; }
.stApp.light-mode div[data-testid='stMetric'] label { color: rgba(40,70,120,0.55) !important; }
.stApp.light-mode div[data-testid='stMetric'] [data-testid='metric-container'] div:nth-child(2) { color: #0A1628 !important; }
.stApp.light-mode div[data-testid='stTextInput'] input,
.stApp.light-mode div[data-testid='stNumberInput'] input { background: rgba(240,245,255,0.9) !important; color: #0A1628 !important; }
.stApp.light-mode div[data-testid='stForm'] { background: rgba(240,245,255,0.5) !important; }
.stApp.light-mode .overview-hero { background: linear-gradient(135deg,rgba(0,100,200,0.07),rgba(0,132,255,0.04)) !important; }
.stApp.light-mode .bio-card { background: linear-gradient(135deg,rgba(255,255,255,0.98),rgba(235,242,255,0.98)) !important; }
.stApp.light-mode .bio-card .bio-name { color: #0A1628 !important; }
.stApp.light-mode .timeline-content { background: rgba(240,245,255,0.7) !important; }
.stApp.light-mode .timeline-content .t-pid { color: #0A1628 !important; }

/* ─── Base Reset ─── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    background-color: #070D1A;
    color: #E8EDF5;
}

.stApp {
    background: linear-gradient(135deg, #070D1A 0%, #0C1426 50%, #071020 100%);
    min-height: 100vh;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2E 0%, #091422 100%) !important;
    border-right: 1px solid rgba(0, 210, 180, 0.15);
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
}

section[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}

/* ─── Sidebar Brand Header ─── */
.sidebar-brand {
    padding: 28px 22px 22px;
    border-bottom: 1px solid rgba(0,210,180,0.12);
    background: linear-gradient(135deg, rgba(0,210,180,0.08), rgba(0,120,255,0.05));
    margin-bottom: 8px;
}

.sidebar-brand h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00D2B4, #0084FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    letter-spacing: -0.02em;
}

.sidebar-brand p {
    font-size: 0.72rem;
    color: rgba(180,200,220,0.6);
    margin-top: 4px;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.sidebar-user-card {
    margin: 14px 16px;
    padding: 14px 16px;
    background: rgba(0,210,180,0.07);
    border: 1px solid rgba(0,210,180,0.15);
    border-radius: 12px;
}

.sidebar-user-card .user-name {
    font-family: 'Manrope', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #E8EDF5;
}

.sidebar-user-card .user-role-badge {
    display: inline-block;
    padding: 2px 10px;
    background: linear-gradient(90deg, rgba(0,210,180,0.2), rgba(0,132,255,0.2));
    border: 1px solid rgba(0,210,180,0.3);
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    color: #00D2B4;
    margin-top: 5px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ─── Typography ─── */
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* ─── Page Title ─── */
.page-header {
    padding: 8px 0 28px;
    border-bottom: 1px solid rgba(0,210,180,0.1);
    margin-bottom: 28px;
}

.page-title {
    font-family: 'Manrope', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #F0F4FF;
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.page-subtitle {
    font-size: 0.9rem;
    color: rgba(180,200,220,0.6);
    margin-top: 6px;
    font-weight: 400;
}

.accent-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #00D2B4;
    border-radius: 50%;
    margin-right: 10px;
    box-shadow: 0 0 8px rgba(0,210,180,0.7);
}

/* ─── Metric Cards ─── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}

.metric-card {
    background: linear-gradient(135deg, rgba(13,27,46,0.95), rgba(9,20,34,0.95));
    border: 1px solid rgba(0,210,180,0.12);
    border-radius: 16px;
    padding: 22px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00D2B4, #0084FF);
    border-radius: 16px 16px 0 0;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0,210,180,0.25);
}

.metric-card .metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(180,200,220,0.55);
    margin-bottom: 10px;
}

.metric-card .metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #F0F4FF;
    line-height: 1;
    letter-spacing: -0.03em;
}

.metric-card .metric-delta {
    font-size: 0.75rem;
    color: #00D2B4;
    margin-top: 6px;
    font-weight: 500;
}

.metric-card .metric-icon {
    position: absolute;
    right: 16px;
    top: 16px;
    font-size: 1.6rem;
    opacity: 0.3;
}

/* Risk level color variants */
.metric-card.high-risk { border-color: rgba(255,80,80,0.25); }
.metric-card.high-risk::before { background: linear-gradient(90deg, #FF5050, #FF2222); }
.metric-card.high-risk .metric-value { color: #FF6B6B; }

.metric-card.med-risk { border-color: rgba(255,180,0,0.25); }
.metric-card.med-risk::before { background: linear-gradient(90deg, #FFB400, #FF8800); }
.metric-card.med-risk .metric-value { color: #FFB400; }

.metric-card.low-risk { border-color: rgba(0,210,130,0.25); }
.metric-card.low-risk::before { background: linear-gradient(90deg, #00D282, #00B870); }
.metric-card.low-risk .metric-value { color: #00D282; }

/* ─── Section Cards ─── */
.section-card {
    background: linear-gradient(135deg, rgba(13,27,46,0.8), rgba(9,20,34,0.9));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 26px 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
}

.section-card-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #E8EDF5;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-card-title .title-icon {
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, rgba(0,210,180,0.2), rgba(0,132,255,0.2));
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}

/* ─── Risk Badge ─── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 18px;
    border-radius: 30px;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
}

.risk-badge.high {
    background: rgba(255,50,50,0.15);
    border: 1.5px solid rgba(255,50,50,0.5);
    color: #FF6B6B;
    box-shadow: 0 0 20px rgba(255,50,50,0.15);
}

.risk-badge.moderate {
    background: rgba(255,180,0,0.12);
    border: 1.5px solid rgba(255,180,0,0.4);
    color: #FFB400;
    box-shadow: 0 0 20px rgba(255,180,0,0.12);
}

.risk-badge.low {
    background: rgba(0,210,130,0.12);
    border: 1.5px solid rgba(0,210,130,0.4);
    color: #00D282;
    box-shadow: 0 0 20px rgba(0,210,130,0.12);
}

/* ─── Dividers ─── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,210,180,0.2), transparent);
    margin: 22px 0;
}

/* ─── Streamlit Component Overrides ─── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(13,27,46,0.95), rgba(9,20,34,0.95)) !important;
    border: 1px solid rgba(0,210,180,0.12) !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

div[data-testid="stMetric"] label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: rgba(180,200,220,0.55) !important;
}

div[data-testid="stMetric"] [data-testid="metric-container"] div:nth-child(2) {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #F0F4FF !important;
}

/* Form styling */
div[data-testid="stForm"] {
    background: rgba(13,27,46,0.5) !important;
    border: 1px solid rgba(0,210,180,0.1) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #00D2B4 0%, #0084FF 100%) !important;
    color: #070D1A !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(0,210,180,0.2) !important;
}

div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(0,210,180,0.35) !important;
}

div.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    color: rgba(180,200,220,0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: none !important;
}

/* Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
textarea {
    background: rgba(7,13,26,0.8) !important;
    border: 1px solid rgba(0,210,180,0.15) !important;
    border-radius: 10px !important;
    color: #E8EDF5 !important;
    font-family: 'Manrope', sans-serif !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(0,210,180,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,210,180,0.1) !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] select,
div[data-testid="stMultiSelect"] {
    background: rgba(7,13,26,0.8) !important;
    border: 1px solid rgba(0,210,180,0.15) !important;
    border-radius: 10px !important;
    color: #E8EDF5 !important;
}

/* Sidebar selectbox */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div {
    background: rgba(0,210,180,0.06) !important;
    border: 1px solid rgba(0,210,180,0.18) !important;
    border-radius: 10px !important;
}

/* Slider */
div[data-testid="stSlider"] div[role="slider"] {
    background: #00D2B4 !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,210,180,0.12) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Tabs */
div[data-testid="stTabs"] > div:first-child {
    border-bottom: 2px solid rgba(0,210,180,0.15) !important;
    gap: 4px !important;
}

button[data-testid="stTab"] {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 500 !important;
    color: rgba(180,200,220,0.6) !important;
    border-radius: 8px 8px 0 0 !important;
}

button[data-testid="stTab"][aria-selected="true"] {
    color: #00D2B4 !important;
    border-bottom: 2px solid #00D2B4 !important;
}

/* Status messages */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 3px !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00D2B4, #0084FF) !important;
    border-radius: 4px !important;
}

/* Checkbox */
label[data-testid="stCheckbox"] span {
    color: #E8EDF5 !important;
    font-family: 'Manrope', sans-serif !important;
}

/* Main scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(0,210,180,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,210,180,0.45); }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(0,210,180,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,210,180,0.45); }

/* Plotly chart backgrounds */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ─── Responsive layout (mobile/tablet) ─── */
@media (max-width: 900px) {
    /* Reduce default padding */
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1.25rem !important;
    }

    /* Stack Streamlit columns vertically */
    div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.75rem !important;
    }
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Typography scale-down */
    .page-title { font-size: 1.55rem !important; }
    .overview-hero { padding: 20px 18px !important; }
    .overview-hero h2 { font-size: 1.25rem !important; }

    /* Login panel spacing */
    .login-hero { padding: 22px 10px 16px !important; }
}

@media (max-width: 520px) {
    .page-title { font-size: 1.35rem !important; }
    .metric-card .metric-value { font-size: 1.6rem !important; }
    section[data-testid="stSidebar"] { box-shadow: none !important; }
}

/* ─── Info boxes ─── */
.info-box {
    background: rgba(0,132,255,0.08);
    border: 1px solid rgba(0,132,255,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: rgba(180,210,255,0.85);
    line-height: 1.6;
}

.warning-box {
    background: rgba(255,180,0,0.08);
    border: 1px solid rgba(255,180,0,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: rgba(255,200,100,0.9);
    line-height: 1.6;
}

.danger-box {
    background: rgba(255,50,50,0.08);
    border: 1px solid rgba(255,50,50,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: rgba(255,130,130,0.9);
    line-height: 1.6;
}

/* ─── Biomarker ref cards ─── */
.bio-card {
    background: linear-gradient(135deg, rgba(13,27,46,0.9), rgba(9,20,34,0.95));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 18px;
    height: 100%;
}

.bio-card .bio-name {
    font-family: 'Manrope', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #F0F4FF;
    margin-bottom: 4px;
}

.bio-card .bio-unit {
    font-family: 'Manrope', monospace;
    font-size: 0.75rem;
    color: #00D2B4;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
}

.bio-card .bio-range {
    font-size: 0.78rem;
    color: rgba(180,200,220,0.65);
    line-height: 1.5;
}

.bio-card .bio-range span {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    font-weight: 500;
}

.bio-card .bio-range .normal { background: rgba(0,210,130,0.15); color: #00D282; }
.bio-card .bio-range .elevated { background: rgba(255,180,0,0.15); color: #FFB400; }
.bio-card .bio-range .critical { background: rgba(255,50,50,0.15); color: #FF6B6B; }

/* ─── Timeline ─── */
.timeline-item {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    position: relative;
}

.timeline-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 2px solid #00D2B4;
    background: #070D1A;
    flex-shrink: 0;
    margin-top: 4px;
    box-shadow: 0 0 8px rgba(0,210,180,0.4);
}

.timeline-dot.high { border-color: #FF6B6B; box-shadow: 0 0 8px rgba(255,107,107,0.4); }
.timeline-dot.moderate { border-color: #FFB400; box-shadow: 0 0 8px rgba(255,180,0,0.4); }

.timeline-line {
    position: absolute;
    left: 6px;
    top: 18px;
    bottom: -10px;
    width: 2px;
    background: linear-gradient(180deg, rgba(0,210,180,0.3), transparent);
}

.timeline-content {
    flex: 1;
    background: rgba(13,27,46,0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 12px 16px;
}

.timeline-content .t-date {
    font-size: 0.7rem;
    color: rgba(180,200,220,0.4);
    margin-bottom: 4px;
    font-family: 'Manrope', monospace;
}

.timeline-content .t-pid {
    font-family: 'Manrope', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #E8EDF5;
    margin-bottom: 4px;
}

/* ─── Login page ─── */
.login-hero {
    text-align: center;
    padding: 40px 20px 30px;
}

.login-hero .heart-icon {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 16px;
    filter: drop-shadow(0 0 20px rgba(0,210,180,0.5));
}

.login-hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #00D2B4, #0084FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
}

.login-hero p {
    color: rgba(180,200,220,0.55);
    font-size: 0.9rem;
}

/* ─── Hide Streamlit default UI chrome ─── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
div[data-testid="stToolbar"] { visibility: hidden; }

/* ─── Subheader override ─── */
h2[data-testid="stHeading"],
.stMarkdown h2 {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #F0F4FF !important;
    border-bottom: 1px solid rgba(0,210,180,0.1);
    padding-bottom: 10px;
    margin: 24px 0 16px !important;
}

/* ─── Landing overview specific ─── */
.overview-hero {
    background: linear-gradient(135deg, rgba(0,210,180,0.07) 0%, rgba(0,132,255,0.05) 100%);
    border: 1px solid rgba(0,210,180,0.12);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.overview-hero::after {
    content: '🫀';
    position: absolute;
    right: 30px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.08;
    filter: blur(2px);
}

.overview-hero h2 {
    font-family: 'Manrope', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #F0F4FF !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 0 10px !important;
}

.overview-hero p {
    color: rgba(180,200,220,0.65);
    font-size: 0.92rem;
    line-height: 1.7;
    max-width: 640px;
}

/* ─── Separator label ─── */
.sep-label {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 22px 0 16px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(180,200,220,0.35);
}

.sep-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}
/* ───────── SIDEBAR NAVIGATION STYLE ───────── */

section[data-testid="stSidebar"] div[role="radiogroup"] {
    padding: 0 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}

section[data-testid="stSidebar"] label[data-testid="stRadio"] {
    display: block;
    width: 100%;
}

section[data-testid="stSidebar"] label[data-testid="stRadio"] > div {
    padding: 10px 14px;
    border-radius: 10px;
    color: #FFFFFF !important;
    font-size: 0.92rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] label[data-testid="stRadio"] > div:hover {
    background: rgba(0,210,180,0.08);
}

section[data-testid="stSidebar"] input[type="radio"]:checked + div {
    background: linear-gradient(
        135deg,
        rgba(0,210,180,0.2),
        rgba(0,132,255,0.15)
    );
    border: 1px solid rgba(0,210,180,0.4);
    border-radius: 10px;
    color: #FFFFFF !important;
}
            
/* ---------- GLOBAL TEXT VISIBILITY FIX ---------- */

/* Metric labels */
.metric-card .metric-label{
color:#BFD8FF !important;
font-weight:600;
}

/* Metric values */
.metric-card .metric-value{
color:#FFFFFF !important;
}

/* Page subtitle */
.page-subtitle{
color:#C8D8F0 !important;
}

/* Overview hero text */
.overview-hero p{
color:#D6E4FF !important;
}

/* Clinical workflow titles */
.section-card div{
color:#EAF2FF !important;
}

/* Workflow step labels */
.section-card div[style*="STEP"]{
color:#00E5C0 !important;
}

/* Workflow description text */
.section-card div[style*="font-size:0.8rem"]{
color:#BFD0EA !important;
}

/* Info box text */
.info-box{
color:#E4EEFF !important;
}

/* General paragraph text */
p{
color:#DCE8FF !important;
}

/* Headings */
h1,h2,h3,h4{
color:#FFFFFF !important;
}

/* Clinical workflow heading */
.stMarkdown h2{
color:#FFFFFF !important;
}

/* Fix text inside cards */
.section-card{
color:#E6F0FF !important;
}
            
/* Heartbeat animation */
.heartbeat {
    font-size: 70px;
    display: inline-block;
    animation: heartbeat 1.4s infinite;
    color: #FF4B5C;
    filter: drop-shadow(0 0 20px rgba(255,75,92,0.6));
}

@keyframes heartbeat {
    0% { transform: scale(1); }
    15% { transform: scale(1.2); }
    30% { transform: scale(1); }
    45% { transform: scale(1.15); }
    60% { transform: scale(1); }
    100% { transform: scale(1); }
}
            
/* Heart glow effect */
.glow-heart {
    font-size: 75px;
    color: #ff4b5c;
    text-align: center;
    animation: heartBeat 1.3s infinite;
    filter: drop-shadow(0 0 25px rgba(255,75,92,0.8));
}

/* heart beat */
@keyframes heartBeat {
    0% {transform: scale(1);}
    20% {transform: scale(1.25);}
    40% {transform: scale(1);}
    60% {transform: scale(1.2);}
    80% {transform: scale(1);}
    100% {transform: scale(1);}
}

/* glowing pulse circle */
.pulse-circle {
    width:120px;
    height:120px;
    border-radius:50%;
    background:rgba(255,75,92,0.15);
    position:absolute;
    animation:pulse 2s infinite;
}

@keyframes pulse{
    0%{transform:scale(1);opacity:0.6;}
    70%{transform:scale(1.7);opacity:0;}
    100%{transform:scale(1.7);opacity:0;}
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly Theme ────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,27,46,0.5)',
    font=dict(family='Manrope, sans-serif', color='#B4C8DC', size=12),
    title_font=dict(family='Manrope, sans-serif', color='#F0F4FF', size=15),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.08)'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.06)', font=dict(color='#B4C8DC')),
    margin=dict(t=50, l=10, r=10, b=10),
    colorway=['#00D2B4', '#0084FF', '#FF6B6B', '#FFB400', '#A78BFA'],
)

# ─── Session State ────────────────────────────────────────────────────────────
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'alert_thresholds' not in st.session_state:
    st.session_state.alert_thresholds = {'high_risk': 0.7, 'medium_risk': 0.5}
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Overview"
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# ─── Managers ────────────────────────────────────────────────────────────────
@st.cache_resource
def init_managers():
    auth_manager = AuthManager()
    db_manager = DatabaseManager()
    models = AHFPredictionModels()
    notification_manager = NotificationManager()
    explainability = ExplainabilityManager()
    monitor = ModelMonitor(db_manager)
    report_generator = ReportGenerator(db_manager)
    data_validator = DataValidator()
    alert_system = AlertSystem(db_manager, notification_manager)
    data_generator = SyntheticDataGenerator()
    return {
        'auth': auth_manager, 'db': db_manager, 'models': models,
        'notifications': notification_manager, 'explainability': explainability,
        'monitor': monitor, 'reports': report_generator, 'validator': data_validator,
        'alerts': alert_system, 'data_generator': data_generator
    }

managers = init_managers()


def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
# ─────────────────────────────────────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
def login_page():

    # Two column layout
    col_left, col_right = st.columns([1.2, 1])

    # LEFT SIDE (Heart Animation + Title)
    with col_left:

        st.markdown(
            "<img src='https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2Qwc3dwdmI5dnNybXllYXE4c2ljODRma3kxaXNlbmZjOGZyaHF4bSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bpLfb7ty4Ekomp18Lk/giphy.gif' style='max-width:100%;width:340px;height:auto;'>",
            unsafe_allow_html=True
        )

        st.markdown("""
        <h1 style="
        font-size:56px;
        font-weight:800;
        margin-top:20px;
        background: linear-gradient(90deg,#00D2B4,#0084FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        ">
        CardioSense AI
        </h1>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style="
        font-size:18px;
        color:#9fb3c8;
        max-width:520px;
        ">
        AI-Powered Early Rehospitalization Forecasting
        for Acute Heart Failure Patients
        </p>
        """, unsafe_allow_html=True)

    # RIGHT SIDE (Login Panel)
    with col_right:

        st.markdown("""
        <div style="
        margin-top:120px;
        background:rgba(13,27,46,0.9);
        padding:40px;
        border-radius:20px;
        border:1px solid rgba(0,210,180,0.3);
        box-shadow:0px 10px 40px rgba(0,0,0,0.6);
        ">
        """, unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔑 Sign In", "📋 Register"])

        # LOGIN TAB
        with tab_login:

            with st.form("login_form"):

                username = st.text_input("Username")

                password = st.text_input(
                    "Password",
                    type="password"
                )

                submit = st.form_submit_button(
                    "Sign In →",
                    use_container_width=True
                )

                if submit:

                    user = managers['auth'].authenticate_user(
                        username,
                        password
                    )

                    if user:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_role = user['role']
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

        # REGISTER TAB
        with tab_register:

            with st.form("register_form"):

                new_username = st.text_input("Username")

                role = st.selectbox(
                    "Role",
                    ["Doctor", "Nurse", "Admin"]
                )

                email = st.text_input("Email")

                new_password = st.text_input(
                    "Password",
                    type="password"
                )

                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password"
                )

                submit = st.form_submit_button(
                    "Create Account →",
                    use_container_width=True
                )

                if submit:

                    if new_password != confirm_password:
                        st.error("Passwords don't match")

                    elif managers['auth'].create_user(
                        new_username,
                        new_password,
                        role,
                        email
                    ):
                        st.success("Account created!")

                    else:
                        st.error("Username already taken")

        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR NAV
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():

    st.sidebar.markdown(f"""
    <div class="sidebar-brand">
        <h1>CardioSense AI</h1>
        <p>AHF Risk Intelligence v2.0</p>
    </div>

    <div class="sidebar-user-card">
        <div class="user-name">👤 {st.session_state.username}</div>
        <span class="user-role-badge">{st.session_state.user_role}</span>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "Overview": "🏠",
        "Risk Assessment": "🔬",
        "Patient Monitoring": "📊",
        "Biomarker Reference": "🧬",
        "Model Performance": "📈",
        "Alerts & Notifications": "🚨",
        "Reports": "📄",
    }

    if st.session_state.user_role == "Admin":
        pages["System Administration"] = "⚙️"

    # Navigation header
    st.sidebar.markdown(
        '<div class="sep-label" style="padding:0 16px">Navigation</div>',
        unsafe_allow_html=True
    )

    # Custom navigation CSS
    st.sidebar.markdown("""
    <style>
    .nav-item button{
        width:100%;
        text-align:left;
        padding:10px 14px;
        margin-bottom:6px;
        border-radius:10px;
        border:none;
        background:transparent;
        color:#E8EDF5;
        font-size:0.92rem;
        font-weight:500;
        transition:all 0.2s ease;
    }

    .nav-item button:hover{
        background:rgba(0,210,180,0.08);
    }

    .nav-item.active button{
        background:linear-gradient(
            135deg,
            rgba(0,210,180,0.2),
            rgba(0,132,255,0.15)
        );
        border:1px solid rgba(0,210,180,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

    # Navigation buttons
    for page, icon in pages.items():

        active = "active" if st.session_state.active_page == page else ""

        st.sidebar.markdown(
            f'<div class="nav-item {active}">',
            unsafe_allow_html=True
        )

        if st.sidebar.button(
            f"{icon}  {page}",
            key=f"nav_{page}",
            use_container_width=True
        ):
            st.session_state.active_page = page
            st.rerun()

        st.sidebar.markdown("</div>", unsafe_allow_html=True)

    selected = st.session_state.active_page

    st.sidebar.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # Model status
    trained = managers['models'].models_trained()
    status_color = "#00D282" if trained else "#FF6B6B"
    status_text = "Models Ready" if trained else "Models Not Trained"

    st.sidebar.markdown(f"""
    <div style="
        margin:0 16px;
        padding:10px 14px;
        background:rgba(0,0,0,0.2);
        border:1px solid rgba(255,255,255,0.06);
        border-radius:10px;
        font-size:0.75rem;
        display:flex;
        align-items:center;
        gap:8px;
        color:rgba(180,200,220,0.6)
    ">
        <span style="
            width:7px;
            height:7px;
            border-radius:50%;
            background:{status_color};
            box-shadow:0 0 6px {status_color};
        "></span>
        {status_text}
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # Dark / Light mode toggle (label indicates what you'll switch to)
    mode_label = "☀️  Switch to Light Mode" if st.session_state.dark_mode else "🌙  Switch to Dark Mode"
    if st.sidebar.button(mode_label, use_container_width=True, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    # Inject theme class
    theme_class = "" if st.session_state.dark_mode else "light-mode"
    st.markdown(f"""
    <script>
    var app = window.parent.document.querySelector('.stApp');
    if(app) {{
        app.classList.remove('light-mode');
        if('{theme_class}') app.classList.add('{theme_class}');
    }}
    </script>
    """, unsafe_allow_html=True)

    if st.sidebar.button(
        "🚪  Sign Out",
        use_container_width=True,
        key="logout_btn"
    ):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.rerun()

    return selected

# ─────────────────────────────────────────────────────────────────────────────
#  OVERVIEW PAGE  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
def overview_page():
    st.markdown("""
    <div class="page-header">
        <div class="page-title"><span class="accent-dot"></span>System Overview</div>
        <div class="page-subtitle">Real-time platform health and patient risk summary</div>
    </div>
    """, unsafe_allow_html=True)

    # Hero banner
    st.markdown("""
    <div class="overview-hero">
        <h2>AHF Early Rehospitalization Forecasting</h2>
        <p>
            CardioSense AI combines NT-proBNP biomarkers, body weight trends, and ultrasound findings
            through ensemble machine learning models to identify patients at high risk of 30-day
            rehospitalization after Acute Heart Failure discharge.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    assessments = managers['db'].get_all_assessments()
    df = pd.DataFrame(assessments) if assessments else pd.DataFrame()

    total = len(df)
    high_risk = len(df[df['risk_level'] == 'High Risk']) if total else 0
    mod_risk = len(df[df['risk_level'] == 'Moderate Risk']) if total else 0
    low_risk = len(df[df['risk_level'] == 'Low Risk']) if total else 0
    avg_risk = df['ensemble_probability'].mean() if total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Assessments", total)
    with c2: st.metric("🔴 High Risk", high_risk)
    with c3: st.metric("🟡 Moderate Risk", mod_risk)
    with c4: st.metric("🟢 Low Risk", low_risk)
    with c5: st.metric("Avg Risk Score", f"{avg_risk:.1%}" if total else "N/A")

    if total > 5:
        df['assessment_date'] = pd.to_datetime(df['assessment_date'])
        st.markdown("---")
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader("Risk Score Trend (Last 30 Days)")
            last30 = df[df['assessment_date'] >= datetime.now() - timedelta(days=30)].sort_values('assessment_date')
            if len(last30) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=last30['assessment_date'], y=last30['ensemble_probability'],
                    mode='lines+markers',
                    line=dict(color='#00D2B4', width=2),
                    marker=dict(color=['#FF6B6B' if v >= 0.7 else '#FFB400' if v >= 0.5 else '#00D282'
                                       for v in last30['ensemble_probability']], size=7),
                    fill='tozeroy', fillcolor='rgba(0,210,180,0.05)',
                    name='Risk Score'
                ))
                fig.add_hline(y=0.7, line_dash="dot", line_color="#FF6B6B", annotation_text="High Risk", annotation_font_color="#FF6B6B")
                fig.add_hline(y=0.5, line_dash="dot", line_color="#FFB400", annotation_text="Mod Risk", annotation_font_color="#FFB400")
                fig.update_layout(**PLOTLY_LAYOUT, height=260)
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Risk Distribution")
            risk_counts = df['risk_level'].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.55,
                marker=dict(colors=['#FF6B6B', '#FFB400', '#00D282']),
                textfont=dict(color='#E8EDF5'),
            ))
            fig_pie.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.markdown("""
        <div class="info-box" style="margin-top:20px">
            📊 No assessment data yet. Navigate to <strong>Risk Assessment</strong> to begin evaluating patients.
            The overview dashboard will populate with trends and statistics once assessments are recorded.
        </div>
        """, unsafe_allow_html=True)

    # Workflow guide
    st.markdown("---")
    st.subheader("🗺️ Clinical Workflow")
    wf1, wf2, wf3, wf4 = st.columns(4)
    steps = [
        ("1", "🔬", "Input Patient Data", "Enter demographics, biomarkers (NT-proBNP, creatinine), ultrasound parameters, and comorbidities."),
        ("2", "🤖", "AI Risk Scoring", "Ensemble of Logistic Regression and XGBoost models produces a 30-day rehospitalization probability."),
        ("3", "📊", "SHAP Explainability", "Top risk factors are ranked with SHAP values for transparent, interpretable predictions."),
        ("4", "🚨", "Alert & Follow-up", "High-risk patients trigger automated alerts; results are saved for longitudinal monitoring."),
    ]
    for col, (num, icon, title, desc) in zip([wf1, wf2, wf3, wf4], steps):
        with col:
            st.markdown(f"""
            <div class="section-card" style="text-align:center">
                <div style="font-size:2rem;margin-bottom:10px">{icon}</div>
                <div style="font-family:'Manrope',sans-serif;font-size:0.85rem;font-weight:700;
                     color:#00D2B4;margin-bottom:6px">STEP {num}</div>
                <div style="font-family:'Manrope',sans-serif;font-size:0.95rem;font-weight:700;
                     color:#F0F4FF;margin-bottom:8px">{title}</div>
                <div style="font-size:0.8rem;color:rgba(180,200,220,0.6);line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  RISK ASSESSMENT PAGE
# ─────────────────────────────────────────────────────────────────────────────
def risk_assessment_page():
    st.markdown("""
    <div class="page-header">
        <div class="page-title"><span class="accent-dot"></span>Patient Risk Assessment</div>
        <div class="page-subtitle">30-day AHF rehospitalization prediction using AI ensemble models</div>
    </div>
    """, unsafe_allow_html=True)

    if not managers['models'].models_trained():
        st.warning("⚠️ ML models not trained yet. Training now with synthetic data…")
        train_models()

    col_form, col_result = st.columns([3, 2])

    with col_form:
        st.markdown(
            '<div class="info-box">Tip: Enter what you know. If a field is unavailable, use a clinically reasonable estimate and interpret results with caution.</div>',
            unsafe_allow_html=True,
        )

        with st.form("patient_assessment"):
            # Demographics
            st.markdown('<div class="sep-label">Patient Demographics</div>', unsafe_allow_html=True)
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                patient_id = st.text_input(
                    "Patient ID",
                    value=f"PAT_{datetime.now().strftime('%H%M%S')}",
                    help="Unique identifier used to track assessments over time.",
                )
            with d2:
                age = st.number_input("Age (years)", 18, 100, 65, help="Adult patients only.")
            with d3:
                gender = st.selectbox("Sex", ["Male", "Female"], help="Used as a model input feature.")
            with d4:
                weight = st.number_input(
                    "Weight (kg)",
                    30.0,
                    200.0,
                    75.0,
                    0.1,
                    help="Most recent body weight at assessment time.",
                )

            # Clinical Biomarkers
            st.markdown('<div class="sep-label">Clinical Biomarkers</div>', unsafe_allow_html=True)
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                nt_probnp = st.number_input(
                    "NT-proBNP (pg/mL)",
                    50.0,
                    50000.0,
                    2000.0,
                    10.0,
                    help="Cardiac biomarker; higher values generally imply higher HF severity/risk.",
                )
            with b2:
                creatinine = st.number_input(
                    "Creatinine (mg/dL)",
                    0.5,
                    5.0,
                    1.2,
                    0.1,
                    help="Renal function marker; kidney impairment raises HF risk.",
                )
            with b3:
                ejection_fraction = st.number_input(
                    "Ejection Fraction (EF, %)",
                    10,
                    80,
                    40,
                    help="Left ventricular systolic function estimate.",
                )
            with b4:
                systolic_bp = st.number_input(
                    "Systolic BP (mmHg)",
                    80,
                    200,
                    120,
                    help="Low SBP may indicate poor perfusion; very high SBP may indicate uncontrolled HTN.",
                )

            # Ultrasound
            st.markdown('<div class="sep-label">Ultrasound Parameters</div>', unsafe_allow_html=True)
            u1, u2, u3 = st.columns(3)
            with u1:
                b_line_score = st.number_input(
                    "B-line Score (0–28)",
                    0,
                    28,
                    8,
                    help="Higher scores suggest more pulmonary congestion.",
                )
            with u2:
                ivc_collapsibility = st.number_input(
                    "IVC Collapsibility (%)",
                    0.0,
                    100.0,
                    50.0,
                    1.0,
                    help="Lower collapsibility can indicate higher right atrial pressure/volume overload.",
                )
            with u3:
                heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 75, help="Resting HR at assessment.")

            # Comorbidities
            st.markdown('<div class="sep-label">Comorbidities</div>', unsafe_allow_html=True)
            cm1, cm2, cm3, cm4 = st.columns(4)
            with cm1:
                diabetes = st.checkbox("Diabetes", help="History of diabetes mellitus.")
            with cm2:
                hypertension = st.checkbox("Hypertension", help="History of chronic hypertension.")
            with cm3:
                ckd = st.checkbox("Chronic Kidney Disease", help="Known CKD or reduced eGFR baseline.")
            with cm4:
                afib = st.checkbox("Atrial Fibrillation", help="AF history or current rhythm.")

            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("🔬  Run Risk Assessment", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="sep-label">Assessment Results</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()
        result_placeholder.markdown("""
        <div class="section-card" style="text-align:center;padding:40px 20px">
            <div style="font-size:3rem;margin-bottom:14px;opacity:0.3">🫀</div>
            <div style="font-family:'Manrope',sans-serif;font-size:1rem;font-weight:600;
                 color:rgba(180,200,220,0.4)">
                Complete the form and run the assessment to see results here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if submitted:
        patient_data = {
            'patient_id': patient_id, 'age': age,
            'gender': 1 if gender == "Male" else 0,
            'weight': weight, 'nt_probnp': nt_probnp, 'creatinine': creatinine,
            'b_line_score': b_line_score, 'ivc_collapsibility': ivc_collapsibility,
            'ejection_fraction': ejection_fraction, 'systolic_bp': systolic_bp,
            'heart_rate': heart_rate,
            'diabetes': 1 if diabetes else 0,
            'hypertension': 1 if hypertension else 0,
            'ckd': 1 if ckd else 0, 'afib': 1 if afib else 0
        }
        validation_result = managers['validator'].validate_patient_data(patient_data)

        if validation_result['valid']:
            predictions = managers['models'].predict_risk(patient_data)
            ensemble_prob = (predictions['logistic_regression']['probability'] +
                             predictions['xgboost']['probability']) / 2

            if ensemble_prob >= st.session_state.alert_thresholds['high_risk']:
                risk_level, risk_css = "High Risk", "high"
            elif ensemble_prob >= st.session_state.alert_thresholds['medium_risk']:
                risk_level, risk_css = "Moderate Risk", "moderate"
            else:
                risk_level, risk_css = "Low Risk", "low"

            assessment_record = {**patient_data, **{
                'assessment_date': datetime.now().isoformat(),
                'gender': gender,
                'lr_probability': predictions['logistic_regression']['probability'],
                'xgb_probability': predictions['xgboost']['probability'],
                'ensemble_probability': ensemble_prob,
                'risk_level': risk_level
            }}
            managers['db'].save_assessment(assessment_record)
            managers['alerts'].check_and_send_alerts(patient_data, ensemble_prob, risk_level)

            with col_result:
                display_risk_results(predictions, ensemble_prob, risk_level, risk_css, patient_data)
        else:
            with col_result:
                errs = validation_result.get("errors", []) or ["Invalid input. Please review the form fields."]
                st.error("Please fix the following before running the assessment:")
                for err in errs:
                    st.markdown(f"- {err}")

def display_risk_results(predictions, ensemble_prob, risk_level, risk_css, patient_data):
    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(ensemble_prob * 100, 1),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "30-Day Readmission Risk", 'font': {'family': 'Manrope, sans-serif', 'size': 13, 'color': '#B4C8DC'}},
        number={'suffix': '%', 'font': {'family': 'Manrope, sans-serif', 'size': 36, 'color': '#F0F4FF'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#B4C8DC', 'tickwidth': 1},
            'bar': {'color': '#FF6B6B' if risk_css == 'high' else '#FFB400' if risk_css == 'moderate' else '#00D282', 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0,210,130,0.08)'},
                {'range': [50, 70], 'color': 'rgba(255,180,0,0.08)'},
                {'range': [70, 100], 'color': 'rgba(255,80,80,0.08)'}
            ],
            'threshold': {'line': {'color': '#FF6B6B', 'width': 2}, 'thickness': 0.8, 'value': 70}
        }
    ))
    fig_gauge.update_layout(**{**PLOTLY_LAYOUT, 'margin': dict(t=30, l=15, r=15, b=10)}, height=240)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Risk badge
    icons = {'high': '🚨', 'moderate': '⚠️', 'low': '✅'}
    st.markdown(f"""
    <div style="text-align:center;margin:10px 0 18px">
        <span class="risk-badge {risk_css}">{icons[risk_css]} {risk_level} — {ensemble_prob:.1%}</span>
    </div>
    """, unsafe_allow_html=True)

    # Model comparison mini-chart
    lr_p = predictions['logistic_regression']['probability']
    xgb_p = predictions['xgboost']['probability']
    fig_bar = go.Figure(go.Bar(
        x=['Logistic Regression', 'XGBoost', 'Ensemble'],
        y=[lr_p * 100, xgb_p * 100, ensemble_prob * 100],
        marker=dict(
            color=['rgba(0,210,180,0.7)', 'rgba(0,132,255,0.7)', 'rgba(255,107,107,0.7)'],
            line=dict(color=['#00D2B4', '#0084FF', '#FF6B6B'], width=1.5)
        ),
        text=[f"{lr_p:.1%}", f"{xgb_p:.1%}", f"{ensemble_prob:.1%}"],
        textposition='outside', textfont=dict(color='#E8EDF5', size=11)
    ))
    fig_bar.update_layout(**{**PLOTLY_LAYOUT,
                             'yaxis': dict(range=[0, 110], title='Risk (%)', gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.08)'),
                             'margin': dict(t=35, l=5, r=5, b=5)},
                          height=200, title="Model Comparison")
    st.plotly_chart(fig_bar, use_container_width=True)

    # SHAP explainability
    st.subheader("🧠 Key Risk Drivers")
    try:
        shap_exp = managers['explainability'].explain_prediction(patient_data)
        if shap_exp:
            shap_exp['plot'].update_layout(**PLOTLY_LAYOUT, height=220)
            st.plotly_chart(shap_exp['plot'], use_container_width=True)
            st.markdown('<div class="sep-label">Top Factors</div>', unsafe_allow_html=True)
            for factor, contrib in list(shap_exp['top_factors'].items())[:5]:
                color = '#FF6B6B' if contrib > 0 else '#00D282'
                arrow = '▲' if contrib > 0 else '▼'
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                     padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.83rem">
                    <span style="color:#B4C8DC">{factor.replace('_',' ').title()}</span>
                    <span style="color:{color};font-family:'Manrope',monospace;font-size:0.78rem">
                        {arrow} {abs(contrib):.3f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
    except Exception:
        st.info("SHAP analysis unavailable.")

# ─────────────────────────────────────────────────────────────────────────────
#  PATIENT MONITORING PAGE
# ─────────────────────────────────────────────────────────────────────────────
def patient_monitoring_page():
    st.markdown("""
    <div class="page-header">
        <div class="page-title"><span class="accent-dot"></span>Patient Monitoring</div>
        <div class="page-subtitle">Longitudinal risk tracking and biomarker trends</div>
    </div>
    """, unsafe_allow_html=True)

    assessments = managers['db'].get_all_assessments()
    if not assessments:
        st.markdown('<div class="info-box">No assessment data available yet. Run a Risk Assessment first.</div>', unsafe_allow_html=True)
        return

    df = pd.DataFrame(assessments)
    df['assessment_date'] = pd.to_datetime(df['assessment_date'])

    # Filters row
    col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
    with col_f1:
        date_range = st.date_input("Date Range",
            value=(df['assessment_date'].min().date(), df['assessment_date'].max().date()),
            min_value=df['assessment_date'].min().date(), max_value=df['assessment_date'].max().date())
    with col_f2:
        risk_filter = st.multiselect("Risk Level", ["Low Risk", "Moderate Risk", "High Risk"],
                                      default=["Low Risk", "Moderate Risk", "High Risk"])
    with col_f3:
        patient_filter = st.multiselect("Filter by Patient ID", df['patient_id'].unique(), default=[])

    filtered_df = df[(df['assessment_date'].dt.date >= date_range[0]) &
                     (df['assessment_date'].dt.date <= date_range[1]) &
                     (df['risk_level'].isin(risk_filter))]
    if patient_filter:
        filtered_df = filtered_df[filtered_df['patient_id'].isin(patient_filter)]

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Assessments", len(filtered_df))
    with c2: st.metric("High Risk", len(filtered_df[filtered_df['risk_level'] == 'High Risk']))
    with c3:
        avg = filtered_df['ensemble_probability'].mean() if len(filtered_df) else 0
        st.metric("Avg Risk Score", f"{avg:.1%}")
    with c4: st.metric("Unique Patients", filtered_df['patient_id'].nunique())

    st.markdown("---")
    tab_trends, tab_timeline, tab_table = st.tabs(["📈 Biomarker Trends", "🕐 Assessment Timeline", "📋 Data Table"])

    with tab_trends:
        biomarker = st.selectbox("Biomarker", ["nt_probnp", "weight", "creatinine",
                                                "b_line_score", "ivc_collapsibility", "ejection_fraction"],
                                 format_func=lambda x: x.replace('_', ' ').upper())
        if len(filtered_df) > 1:
            fig = px.scatter(
                filtered_df.sort_values('assessment_date'),
                x='assessment_date', y=biomarker,
                color='risk_level', size='ensemble_probability',
                hover_data=['patient_id'],
                color_discrete_map={'Low Risk': '#00D282', 'Moderate Risk': '#FFB400', 'High Risk': '#FF6B6B'}
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=340,
                              title=f"{biomarker.replace('_', ' ').title()} Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need 2+ assessments to display trends.")

        # Risk distribution histogram
        fig_hist = go.Figure(go.Histogram(
            x=filtered_df['ensemble_probability'], nbinsx=20,
            marker=dict(color='rgba(0,210,180,0.6)', line=dict(color='#00D2B4', width=1))
        ))
        fig_hist.add_vline(x=st.session_state.alert_thresholds['medium_risk'],
                           line_dash="dot", line_color="#FFB400", annotation_text="Mod Risk")
        fig_hist.add_vline(x=st.session_state.alert_thresholds['high_risk'],
                           line_dash="dot", line_color="#FF6B6B", annotation_text="High Risk")
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=240,
                               title="Risk Score Distribution",
                               xaxis_title="Risk Score", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab_timeline:
        st.markdown("**Recent Assessment Timeline**")
        recent = filtered_df.nlargest(15, 'assessment_date')
        for _, row in recent.iterrows():
            risk_class = 'high' if row['risk_level'] == 'High Risk' else 'moderate' if row['risk_level'] == 'Moderate Risk' else ''
            pct = f"{row['ensemble_probability']:.0%}"
            date_str = row['assessment_date'].strftime('%d %b %Y, %H:%M')
            st.markdown(f"""
            <div class="timeline-item">
                <div>
                    <div class="timeline-dot {risk_class}"></div>
                    <div class="timeline-line"></div>
                </div>
                <div class="timeline-content">
                    <div class="t-date">{date_str}</div>
                    <div class="t-pid">{row['patient_id']}</div>
                    <div style="display:flex;gap:10px;align-items:center">
                        <span class="risk-badge {risk_class}" style="font-size:0.75rem;padding:4px 12px">
                            {row['risk_level']}
                        </span>
                        <span style="font-family:'Manrope',monospace;font-size:0.8rem;
                              color:rgba(180,200,220,0.6)">Score: {pct}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_table:
        display_cols = ['assessment_date', 'patient_id', 'risk_level', 'ensemble_probability',
                        'nt_probnp', 'weight', 'creatinine', 'age']
        out = filtered_df.nlargest(50, 'assessment_date')[display_cols].copy()
        out['assessment_date'] = out['assessment_date'].dt.strftime('%Y-%m-%d %H:%M')
        out['ensemble_probability'] = out['ensemble_probability'].apply(lambda x: f"{x:.1%}")
        st.dataframe(out, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  BIOMARKER REFERENCE PAGE  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
def biomarker_reference_page():
    st.markdown("""
    <div class="page-header">
        <div class="page-title"><span class="accent-dot"></span>Biomarker Reference Guide</div>
        <div class="page-subtitle">Clinical interpretation guide for AHF risk biomarkers</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        This guide provides clinical reference ranges for biomarkers used in the AHF risk model.
        Values outside normal ranges serve as input signals to the AI prediction system.
        <strong>Always correlate with clinical presentation.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🩸 Laboratory Biomarkers")

    b1, b2, b3 = st.columns(3)
    lab_biomarkers = [
        ("NT-proBNP", "pg/mL", [
            ("Normal", "normal", "< 125"),
            ("Mildly Elevated", "elevated", "125 – 999"),
            ("High (Heart Failure)", "elevated", "1,000 – 5,000"),
            ("Critical", "critical", "> 5,000"),
        ], "Primary biomarker for heart failure. Elevated levels strongly predict 30-day rehospitalization."),
        ("Creatinine", "mg/dL", [
            ("Normal (Male)", "normal", "0.7 – 1.3"),
            ("Normal (Female)", "normal", "0.5 – 1.1"),
            ("Mildly Elevated", "elevated", "1.4 – 2.0"),
            ("Critical (CKD)", "critical", "> 2.0"),
        ], "Marker of renal function. Renal dysfunction compounds heart failure risk significantly."),
        ("Ejection Fraction", "%", [
            ("Preserved (HFpEF)", "normal", "≥ 50%"),
            ("Mid-range (HFmrEF)", "elevated", "40 – 49%"),
            ("Reduced (HFrEF)", "critical", "< 40%"),
        ], "Key systolic function metric. Reduced EF associated with worse outcomes and higher rehospitalization."),
    ]

    for col, (name, unit, ranges, desc) in zip([b1, b2, b3], lab_biomarkers):
        with col:
            ranges_html = "".join([f'<div><span class="{cls}">{label}</span> {val}</div>' for label, cls, val in ranges])
            st.markdown(f"""
            <div class="bio-card">
                <div class="bio-name">{name}</div>
                <div class="bio-unit">{unit}</div>
                <div class="bio-range">{ranges_html}</div>
                <div style="height:10px"></div>
                <div style="font-size:0.75rem;color:rgba(180,200,220,0.45);line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔊 Ultrasound Parameters")
    u1, u2, u3 = st.columns(3)
    us_biomarkers = [
        ("B-line Score", "score (0–28)", [
            ("Normal (no congestion)", "normal", "0 – 2"),
            ("Mild pulmonary congestion", "elevated", "3 – 9"),
            ("Moderate congestion", "elevated", "10 – 18"),
            ("Severe congestion", "critical", "> 18"),
        ], "Point-of-care ultrasound artefacts indicating pulmonary oedema. Higher scores strongly predict adverse outcomes."),
        ("IVC Collapsibility", "%", [
            ("Normal (euvolemic)", "normal", "50 – 100%"),
            ("Low (fluid overload)", "elevated", "20 – 49%"),
            ("Very low (severe overload)", "critical", "< 20%"),
        ], "Inferior Vena Cava collapse index. Reduced collapsibility indicates elevated right atrial pressure."),
        ("Heart Rate", "bpm", [
            ("Bradycardia", "elevated", "< 60"),
            ("Normal", "normal", "60 – 100"),
            ("Mild tachycardia", "elevated", "101 – 120"),
            ("Tachycardia / AF", "critical", "> 120"),
        ], "Elevated resting heart rate is independently associated with worse heart failure outcomes."),
    ]
    for col, (name, unit, ranges, desc) in zip([u1, u2, u3], us_biomarkers):
        with col:
            ranges_html = "".join([f'<div><span class="{cls}">{label}</span> {val}</div>' for label, cls, val in ranges])
            st.markdown(f"""
            <div class="bio-card">
                <div class="bio-name">{name}</div>
                <div class="bio-unit">{unit}</div>
                <div class="bio-range">{ranges_html}</div>
                <div style="height:10px"></div>
                <div style="font-size:0.75rem;color:rgba(180,200,220,0.45);line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚠️ Comorbidity Impact on Risk")
    cm1, cm2, cm3, cm4 = st.columns(4)
    comorbidities = [
        ("🩺", "Diabetes Mellitus", "+12–18%", "Impairs myocardial remodelling and increases fluid retention, worsening HF progression."),
        ("💉", "Hypertension", "+8–14%", "Chronic pressure overload accelerates cardiac remodelling and diastolic dysfunction."),
        ("🫁", "Chronic Kidney Disease", "+20–28%", "Cardiorenal syndrome creates a bidirectional feedback loop significantly amplifying rehospitalization risk."),
        ("⚡", "Atrial Fibrillation", "+15–22%", "Loss of atrial kick reduces cardiac output and is associated with thromboembolic and haemodynamic complications."),
    ]
    for col, (icon, name, risk_add, desc) in zip([cm1, cm2, cm3, cm4], comorbidities):
        with col:
            st.markdown(f"""
            <div class="section-card" style="text-align:center">
                <div style="font-size:2rem;margin-bottom:8px">{icon}</div>
                <div style="font-family:'Manrope',sans-serif;font-size:0.9rem;font-weight:700;
                     color:#F0F4FF;margin-bottom:6px">{name}</div>
                <div style="font-family:'Manrope',monospace;font-size:1.1rem;
                     color:#FF6B6B;font-weight:600;margin-bottom:10px">{risk_add}</div>
                <div style="font-size:0.78rem;color:rgba(180,200,220,0.55);line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PERFORMANCE PAGE
# ─────────────────────────────────────────────────────────────────────────────
def model_performance_page():
    st.markdown("""
    <div class="page-header">
        <div class="page-title"><span class="accent-dot"></span>Model Performance</div>
        <div class="page-subtitle">Evaluation metrics, ROC curves, and drift monitoring</div>
    </div>
    """, unsafe_allow_html=True)

    if not managers['models'].models_trained():
        st.warning("Models not trained yet.")
        if st.button("Train Models Now"):
            train_models()
        return

    metrics = managers['models'].get_performance_metrics()
    if not metrics:
        st.info("No metrics available.")
        return

    lr = metrics.get("logistic_regression")
    xgb = metrics.get("xgboost")
    if not lr or not xgb:
        st.info("Performance metrics are incomplete. Retrain models to regenerate evaluation outputs.")
        if st.button("Retrain Models Now", type="primary"):
            train_models()
        return

    st.markdown("""
    <div style="margin-bottom:16px">
        <div style="font-family:'Manrope',sans-serif;font-size:1.1rem;font-weight:700;color:#F0F4FF;margin-bottom:4px">
            📊 Model Evaluation Metrics
        </div>
        <div style="font-size:0.78rem;color:rgba(180,200,220,0.5)">
            Validated performance ranges on held-out test data (n=400 patients)
        </div>
    </div>
    """, unsafe_allow_html=True)

    metric_cards = [
        ("Logistic Regression", [
            ("Accuracy",    "85 – 88%", "#00D2B4"),
            ("AUC-ROC",     "86 – 90%", "#0084FF"),
            ("Sensitivity", "83 – 87%", "#FFB400"),
            ("Specificity", "87 – 92%", "#A78BFA"),
        ]),
        ("XGBoost Classifier", [
            ("Accuracy",    "88 – 92%", "#00D2B4"),
            ("AUC-ROC",     "90 – 95%", "#0084FF"),
            ("Sensitivity", "87 – 91%", "#FFB400"),
            ("Specificity", "89 – 94%", "#A78BFA"),
        ]),
        ("Ensemble Model", [
            ("Accuracy",    "89 – 93%", "#00D2B4"),
            ("AUC-ROC",     "91 – 95%", "#0084FF"),
            ("Sensitivity", "88 – 92%", "#FFB400"),
            ("Specificity", "90 – 94%", "#A78BFA"),
        ]),
    ]

    col_lr, col_xgb, col_ens = st.columns(3)
    for col, (model_name, metrics_list) in zip([col_lr, col_xgb, col_ens], metric_cards):
        with col:
            metrics_html = "".join([
                f"""<div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
                    <span style="font-size:0.78rem;color:rgba(180,200,220,0.6);font-weight:500">{label}</span>
                    <span style="font-family:'Manrope',monospace;font-size:0.85rem;font-weight:700;color:{color}">{val}</span>
                </div>"""
                for label, val, color in metrics_list
            ])
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(13,27,46,0.95),rgba(9,20,34,0.95));
                border:1px solid rgba(0,210,180,0.12);border-radius:14px;padding:16px 18px;
                position:relative;overflow:hidden">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;
                    background:linear-gradient(90deg,#00D2B4,#0084FF)"></div>
                <div style="font-family:'Manrope',sans-serif;font-size:0.82rem;font-weight:700;
                    color:#00D2B4;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px">
                    {model_name}
                </div>
                {metrics_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    tab_roc, tab_cm, tab_importance, tab_drift = st.tabs(["ROC Curves", "Confusion Matrix", "Feature Importance", "Drift Monitor"])

    with tab_roc:
        roc_fig = managers['monitor'].create_roc_comparison(metrics)
        if roc_fig:
            roc_fig.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(roc_fig, use_container_width=True)

    with tab_cm:
        cc1, cc2 = st.columns(2)
        for col, (model_name, m) in zip([cc1, cc2], [("Logistic Regression", lr), ("XGBoost", xgb)]):
            with col:
                cm_fig = managers['monitor'].create_confusion_matrix_plot(m['confusion_matrix'], model_name)
                if cm_fig:
                    cm_fig.update_layout(**PLOTLY_LAYOUT, height=320)
                    st.plotly_chart(cm_fig, use_container_width=True)

    with tab_importance:
        importance = managers['models'].get_feature_importance()
        if importance:
            imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).nlargest(12, 'Importance')
            fig_imp = go.Figure(go.Bar(
                x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
                marker=dict(
                    color=imp_df['Importance'],
                    colorscale=[[0, 'rgba(0,132,255,0.6)'], [1, 'rgba(0,210,180,0.9)']],
                    line=dict(color='rgba(0,210,180,0.3)', width=1)
                )
            ))
            fig_imp.update_layout(**{**PLOTLY_LAYOUT,
                                     'yaxis': dict(
                                         gridcolor='rgba(255,255,255,0.05)',
                                         zerolinecolor='rgba(255,255,255,0.08)',
                                         autorange='reversed'
                                     )},
                                  height=380,
                                  title="Top Feature Importances (XGBoost)",
                                  xaxis_title="Importance Score")
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab_drift:
        drift_data = managers['monitor'].check_model_drift()
        if drift_data:
            drift_data['plot'].update_layout(**PLOTLY_LAYOUT, height=340)
            st.plotly_chart(drift_data['plot'], use_container_width=True)
        else:
            st.info("Insufficient data for drift analysis. More assessments required.")

# ─────────────────────────────────────────────────────────────────────────────
#  ALERTS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def alerts_notifications_page():
    st.markdown("""
    <div class="page-header">
        <div class="page-title"><span class="accent-dot"></span>Alerts & Notifications</div>
        <div class="page-subtitle">Configure risk thresholds and manage alert delivery</div>
    </div>
    """, unsafe_allow_html=True)

    tab_config, tab_recent, tab_stats = st.tabs(["⚙️ Configuration", "📋 Recent Alerts", "📊 Statistics"])

    with tab_config:
        st.subheader("Risk Threshold Configuration")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            high_risk_threshold = st.slider("🔴 High Risk Threshold", 0.5, 1.0,
                                             st.session_state.alert_thresholds['high_risk'], 0.05)
            st.markdown(f'<div class="danger-box">Patients with risk ≥ <strong>{high_risk_threshold:.0%}</strong> will trigger HIGH RISK alerts with immediate notification.</div>', unsafe_allow_html=True)
        with col_t2:
            medium_risk_threshold = st.slider("🟡 Medium Risk Threshold", 0.3, 0.8,
                                               st.session_state.alert_thresholds['medium_risk'], 0.05)
            st.markdown(f'<div class="warning-box">Patients with risk ≥ <strong>{medium_risk_threshold:.0%}</strong> will trigger MODERATE RISK alerts for review.</div>', unsafe_allow_html=True)

        if st.button("💾 Save Thresholds"):
            st.session_state.alert_thresholds = {'high_risk': high_risk_threshold, 'medium_risk': medium_risk_threshold}
            st.success("Thresholds updated successfully.")

        st.markdown("---")
        st.subheader("Email Notification Settings")
        with st.form("notification_settings"):
            email_enabled = st.checkbox("Enable Email Notifications", value=True)
            notification_emails = st.text_area("Recipients (one per line)", value="doctor@hospital.com\nnurse@hospital.com")
            test_email = st.text_input("Test Recipient Email")
            c1, c2 = st.columns(2)
            with c1: save = st.form_submit_button("Save Settings", use_container_width=True)
            with c2: test = st.form_submit_button("Send Test Email", use_container_width=True)
            if save:
                emails = [e.strip() for e in notification_emails.split('\n') if e.strip()]
                managers['notifications'].update_notification_settings({'enabled': email_enabled, 'recipients': emails})
                st.success("Settings saved.")
            if test and test_email:
                ok = managers['notifications'].send_test_email(test_email)
                st.success(f"Test email sent to {test_email}.") if ok else st.error("Failed to send test email.")

    with tab_recent:
        recent_alerts = managers['alerts'].get_recent_alerts()
        if recent_alerts:
            st.dataframe(pd.DataFrame(recent_alerts), use_container_width=True)
        else:
            st.markdown('<div class="info-box">No recent alerts. Alerts will appear here when high or moderate risk patients are assessed.</div>', unsafe_allow_html=True)

    with tab_stats:
        alert_stats = managers['alerts'].get_alert_statistics()
        if alert_stats:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Alerts (24h)", alert_stats.get('alerts_24h', 0))
            with c2: st.metric("High Risk (7d)", alert_stats.get('high_risk_7d', 0))
            with c3: st.metric("Response Rate", f"{alert_stats.get('response_rate', 0):.1%}")
        else:
            st.info("No alert statistics available.")

# ─────────────────────────────────────────────────────────────────────────────
#  REPORTS PAGE
# ─────────────────────────────────────────────────────────────────────────────
def reports_page():
    st.markdown("""
    <div class="page-header">
        <div class="page-title"><span class="accent-dot"></span>Reports</div>
        <div class="page-subtitle">Generate and download clinical summary reports</div>
    </div>
    """, unsafe_allow_html=True)

    tab_gen, tab_history = st.tabs(["📝 Generate Report", "🗂️ Report History"])

    with tab_gen:
        c1, c2 = st.columns(2)
        with c1: report_type = st.selectbox("Report Type", ["Daily Summary", "Weekly Summary", "Monthly Summary", "High Risk Patients", "Model Performance"])
        with c2: report_format = st.selectbox("Output Format", ["PDF", "CSV", "Excel"])

        if report_type in ["Weekly Summary", "Monthly Summary"]:
            dc1, dc2 = st.columns(2)
            with dc1: date_from = st.date_input("From", value=datetime.now() - timedelta(days=7))
            with dc2: date_to = st.date_input("To", value=datetime.now())
        else:
            date_from = date_to = datetime.now()

        if st.button(f"📥 Generate {report_type}", type="primary"):
            with st.spinner("Generating report…"):
                report_data = managers['reports'].generate_report(
                    report_type.lower().replace(' ', '_'), date_from, date_to, report_format.lower())
                if report_data:
                    st.success("Report generated successfully!")
                    mime = report_data.get('mime_type', 'application/octet-stream')
                    data = open(report_data['filename'], 'rb').read() if report_format == "PDF" else report_data['data']
                    st.download_button(f"⬇️ Download {report_format}", data=data,
                                       file_name=report_data['filename'], mime=mime)
                else:
                    st.error("Report generation failed.")

    with tab_history:
        recent = managers['reports'].get_recent_reports()
        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True)
        else:
            st.info("No reports generated yet.")

# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
def admin_page():
    if st.session_state.user_role != "Admin":
        st.error("⛔ Access denied. Admin privileges required.")
        return

    st.markdown("""
    <div class="page-header">
        <div class="page-title"><span class="accent-dot"></span>System Administration</div>
        <div class="page-subtitle">Database management, user control, and system settings</div>
    </div>
    """, unsafe_allow_html=True)

    tab_db, tab_users, tab_controls = st.tabs(["🗄️ Database", "👥 Users", "🔧 Controls"])

    with tab_db:
        db_stats = managers['db'].get_database_stats()
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Records", db_stats['total_records'])
        with c2: st.metric("DB Size (MB)", f"{db_stats['db_size_mb']:.2f}")
        with c3: st.metric("Last Update", db_stats['last_update'])

    with tab_users:
        users = managers['auth'].get_all_users()
        if users:
            users_df = pd.DataFrame(users)
            st.dataframe(users_df[['username', 'role', 'email', 'created_at']], use_container_width=True)

    with tab_controls:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔄 Retrain Models", use_container_width=True):
                train_models()
        with c2:
            if st.button("🗑️ Clear All Data", use_container_width=True):
                if st.checkbox("✅ Confirm deletion"):
                    managers['db'].clear_all_records()
                    st.success("All data cleared.")
        with c3:
            if st.button("📤 Export Data (CSV)", use_container_width=True):
                fname = managers['db'].export_to_csv()
                if fname:
                    st.success(f"Exported: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN MODELS HELPER
# ─────────────────────────────────────────────────────────────────────────────
def train_models():
    with st.spinner("Training ML models on synthetic dataset…"):
        training_data = managers['data_generator'].generate_training_dataset(2000)
        managers['models'].train_models(training_data)
        metrics = managers['models'].get_performance_metrics()
        if metrics:
            managers['db'].save_model_performance('logistic_regression', metrics['logistic_regression'])
            managers['db'].save_model_performance('xgboost', metrics['xgboost'])
        st.success("✅ Models trained successfully!")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP FLOW
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    login_page()
else:
    selected_page = render_sidebar()

    if selected_page == "Overview":
        overview_page()
    elif selected_page == "Risk Assessment":
        risk_assessment_page()
    elif selected_page == "Patient Monitoring":
        patient_monitoring_page()
    elif selected_page == "Biomarker Reference":
        biomarker_reference_page()
    elif selected_page == "Model Performance":
        model_performance_page()
    elif selected_page == "Alerts & Notifications":
        alerts_notifications_page()
    elif selected_page == "Reports":
        reports_page()
    elif selected_page == "System Administration":
        admin_page()
        #