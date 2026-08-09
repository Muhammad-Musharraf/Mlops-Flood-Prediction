# =============================================================================
# streamlit_app.py  —  Flood Prediction  |  Beautiful Streamlit UI
# =============================================================================
# Run:  streamlit run streamlit_app.py
# Requires FastAPI running at http://localhost:8000
# =============================================================================

import streamlit as st
import requests
import json
import time
import random

# =============================================================================
# PAGE CONFIG  —  must be FIRST Streamlit call
# =============================================================================

st.set_page_config(
    page_title = "FloodSense — Flood Risk Predictor",
    page_icon  = "🌊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# THEME & CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root palette ── */
:root {
    --navy:      #0a1628;
    --navy-mid:  #0f2040;
    --blue-dark: #0d3b6e;
    --blue:      #1a5fa8;
    --cyan:      #00c2e0;
    --cyan-soft: #00e5ff22;
    --white:     #f0f6ff;
    --muted:     #7a9bbf;
    --card-bg:   #0d1e35;
    --border:    #1a3a5c;
    --green:     #00e676;
    --amber:     #ffb300;
    --red:       #ff5252;
    --font:      'Space Grotesk', sans-serif;
    --mono:      'JetBrains Mono', monospace;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
    background-color: var(--navy) !important;
    color: var(--white) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2.5rem 2rem !important;
    max-width: 1300px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy-mid) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--white) !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d3b6e 0%, #0a1628 60%, #00c2e022 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #00c2e033 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(90deg, #f0f6ff 0%, #00c2e0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
}
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--cyan-soft);
    border: 1px solid var(--cyan);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-family: var(--mono);
    color: var(--cyan);
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem;
    font-family: var(--mono);
    letter-spacing: 0.12em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Cards ── */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-hover {
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card-hover:hover {
    border-color: var(--cyan);
    box-shadow: 0 0 20px #00c2e015;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #0d3b6e, #0a1628);
    border: 2px solid var(--cyan);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    box-shadow: 0 0 40px #00c2e020;
}
.result-value {
    font-size: 4.5rem;
    font-weight: 700;
    font-family: var(--mono);
    line-height: 1;
    margin: 0.5rem 0;
}
.result-label {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    font-family: var(--mono);
}

/* ── Risk gauge bar ── */
.gauge-track {
    background: #1a3a5c;
    border-radius: 99px;
    height: 10px;
    margin: 1rem 0 0.4rem 0;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    font-family: var(--mono);
    color: var(--muted);
}

/* ── Stat pills ── */
.stat-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.stat-pill {
    background: #0f2040;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    text-align: center;
    flex: 1;
    min-width: 90px;
}
.stat-pill-value {
    font-size: 1.3rem;
    font-weight: 600;
    font-family: var(--mono);
    color: var(--cyan);
}
.stat-pill-label {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Slider overrides ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--green) 0%, var(--amber) 55%, var(--red) 100%) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--white) !important;
    border: 3px solid var(--cyan) !important;
    box-shadow: 0 0 8px #00c2e070 !important;
}
[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] label {
    font-size: 0.8rem !important;
    color: var(--muted) !important;
}
input[type="number"] {
    background: #0f2040 !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
}
input[type="number"]:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px #00c2e020 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a5fa8, #00c2e0) !important;
    color: #0a1628 !important;
    font-family: var(--font) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Rate limit banner ── */
.rate-limit-banner {
    background: linear-gradient(135deg, #2a0a0a, #1a0000);
    border: 2px solid #ff5252;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-top: 1rem;
    text-align: center;
    box-shadow: 0 0 30px #ff525230;
    animation: glow-red 2s ease-in-out infinite;
}
@keyframes glow-red {
    0%, 100% { box-shadow: 0 0 20px #ff525220; }
    50%       { box-shadow: 0 0 40px #ff525250; }
}
.rate-limit-icon   { font-size: 2.5rem; margin-bottom: 0.4rem; }
.rate-limit-title  {
    font-size: 1.1rem; font-weight: 700; color: #ff5252;
    letter-spacing: 0.06em; margin-bottom: 0.3rem;
}
.rate-limit-sub    { font-size: 0.8rem; color: #7a9bbf; margin-bottom: 1rem; }
.rate-limit-timer  {
    font-size: 2rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #ff5252;
}
.rate-limit-note   {
    font-size: 0.7rem; color: #7a9bbf;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.5rem;
}

/* ── API status dot ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
.dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); }
.dot-red   { background: var(--red);   box-shadow: 0 0 6px var(--red);   }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Feature tag cloud ── */
.tag {
    display: inline-block;
    background: #0f2040;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.7rem;
    font-family: var(--mono);
    color: var(--muted);
    margin: 2px;
}

/* ── Toast-style info box ── */
.info-box {
    background: #00c2e010;
    border-left: 3px solid var(--cyan);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── History table ── */
.history-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
}
.history-row:last-child { border-bottom: none; }

/* ── Streamlit selectbox ── */
[data-testid="stSelectbox"] label { color: var(--muted) !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIG
# =============================================================================

API_BASE         = "http://localhost:8000"
RATE_LIMIT_SECS  = 60   # seconds to lock the button after a 429

FEATURE_META = {
    "MonsoonIntensity":                {"icon": "🌧️",  "group": "Climate",        "min": 1, "max": 10, "help": "Intensity of monsoon rainfall season"},
    "TopographyDrainage":              {"icon": "⛰️",  "group": "Terrain",        "min": 1, "max": 10, "help": "Natural drainage capacity of terrain"},
    "RiverManagement":                 {"icon": "🏞️",  "group": "Infrastructure", "min": 1, "max": 10, "help": "Quality of river management systems"},
    "Deforestation":                   {"icon": "🌲",  "group": "Environment",    "min": 1, "max": 10, "help": "Level of deforestation in the region"},
    "Urbanization":                    {"icon": "🏙️",  "group": "Development",    "min": 1, "max": 10, "help": "Rate of urban expansion"},
    "ClimateChange":                   {"icon": "🌡️",  "group": "Climate",        "min": 1, "max": 10, "help": "Impact of long-term climate change"},
    "DamsQuality":                     {"icon": "🏗️",  "group": "Infrastructure", "min": 1, "max": 10, "help": "Structural quality of dams"},
    "Siltation":                       {"icon": "💧",  "group": "Terrain",        "min": 1, "max": 10, "help": "Silt buildup in rivers and channels"},
    "AgriculturalPractices":           {"icon": "🌾",  "group": "Environment",    "min": 1, "max": 10, "help": "Flood-risk impact of farming methods"},
    "Encroachments":                   {"icon": "🏠",  "group": "Development",    "min": 1, "max": 10, "help": "Flood plain encroachment level"},
    "IneffectiveDisasterPreparedness": {"icon": "🚨",  "group": "Governance",     "min": 1, "max": 10, "help": "Weakness of disaster response systems"},
    "DrainageSystems":                 {"icon": "🔩",  "group": "Infrastructure", "min": 1, "max": 10, "help": "Quality of drainage network"},
    "CoastalVulnerability":            {"icon": "🌊",  "group": "Terrain",        "min": 1, "max": 10, "help": "Exposure to coastal flood events"},
    "Landslides":                      {"icon": "⛏️",  "group": "Terrain",        "min": 1, "max": 10, "help": "Landslide risk in the region"},
    "Watersheds":                      {"icon": "🗺️",  "group": "Environment",    "min": 1, "max": 10, "help": "Watershed health and management"},
    "DeterioratingInfrastructure":     {"icon": "🧱",  "group": "Infrastructure", "min": 1, "max": 10, "help": "State of existing flood defences"},
    "WetlandLoss":                     {"icon": "🌿",  "group": "Environment",    "min": 1, "max": 10, "help": "Loss of natural wetland buffers"},
}

SAMPLE_VALUES = {
    "MonsoonIntensity": 1, "TopographyDrainage": 3, "RiverManagement": 4,
    "Deforestation": 5, "Urbanization": 3, "ClimateChange": 2,
    "DamsQuality": 8, "Siltation": 9, "AgriculturalPractices": 3,
    "Encroachments": 2, "IneffectiveDisasterPreparedness": 9,
    "DrainageSystems": 8, "CoastalVulnerability": 6, "Landslides": 2,
    "Watersheds": 1, "DeterioratingInfrastructure": 1, "WetlandLoss": 4,
}

GROUPS = ["Climate", "Terrain", "Infrastructure", "Environment", "Development", "Governance"]
GROUP_ICONS = {
    "Climate": "🌧️", "Terrain": "⛰️", "Infrastructure": "🏗️",
    "Environment": "🌿", "Development": "🏙️", "Governance": "🚨",
}

def get_risk_level(prob: float):
    if prob < 0.3:   return ("LOW",      "#00e676", "🟢", "Minimal flood risk. Normal monitoring advised.")
    if prob < 0.5:   return ("MODERATE", "#ffb300", "🟡", "Moderate risk. Precautionary measures recommended.")
    if prob < 0.7:   return ("HIGH",     "#ff7043", "🟠", "High risk. Activate flood preparedness protocols.")
    return               ("CRITICAL",  "#ff5252", "🔴", "Critical risk. Immediate evacuation may be required.")

# =============================================================================
# API HELPERS
# ─ Health check is cached for 30 s so it doesn't fire on every Streamlit
#   re-render (slider move, button press, etc.).  Cache is keyed on the URL
#   so a real outage is still detected within half a minute.
# =============================================================================

@st.cache_data(ttl=30, show_spinner=False)
def check_api_health() -> tuple[bool, dict]:
    """Cached health check — re-fetches at most once every 30 seconds."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}


def call_predict(payload: dict) -> dict:
    """POST /predict and return the JSON body.

    Raises:
        requests.exceptions.HTTPError  – includes .response for status inspection
        requests.exceptions.ConnectionError
    """
    r = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# =============================================================================
# SESSION STATE
# =============================================================================

if "history"           not in st.session_state: st.session_state.history           = []
if "last_result"       not in st.session_state: st.session_state.last_result       = None
if "input_values"      not in st.session_state: st.session_state.input_values      = SAMPLE_VALUES.copy()
if "rate_limited_at"   not in st.session_state: st.session_state.rate_limited_at   = None  # epoch seconds
if "rate_limit_msg"    not in st.session_state: st.session_state.rate_limit_msg    = ""

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div style="font-size:1.5rem; font-weight:700; letter-spacing:-0.02em;">🌊 FloodSense</div>
        <div style="font-size:0.72rem; color:#7a9bbf; font-family:'JetBrains Mono',monospace; margin-top:2px;">
            Flood Risk Predictor
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Health check — uses the 30-second cache, no extra network call per render
    api_ok, health_data = check_api_health()

    # ── Presets ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Quick Presets</div>', unsafe_allow_html=True)

    preset = st.selectbox(
        "Load a scenario",
        ["Custom", "Low Risk Area", "Urban Flood Zone", "Coastal High Risk", "Rural Monsoon Zone"],
        label_visibility="collapsed"
    )

    PRESETS = {
        "Low Risk Area":       {k: 2 for k in SAMPLE_VALUES},
        "Urban Flood Zone":    {**SAMPLE_VALUES, "Urbanization": 9, "DrainageSystems": 2, "Encroachments": 8},
        "Coastal High Risk":   {**SAMPLE_VALUES, "CoastalVulnerability": 10, "MonsoonIntensity": 8, "Landslides": 7},
        "Rural Monsoon Zone":  {**SAMPLE_VALUES, "MonsoonIntensity": 9, "Deforestation": 8, "WetlandLoss": 8},
    }

    if preset != "Custom" and preset in PRESETS:
        st.session_state.input_values = PRESETS[preset].copy()
        st.rerun()

    if st.button("🎲 Random Scenario"):
        st.session_state.input_values = {k: random.randint(1, 10) for k in SAMPLE_VALUES}
        st.rerun()

    if st.button("↺ Reset to Sample"):
        st.session_state.input_values = SAMPLE_VALUES.copy()
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Prediction History ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            lvl, col, _, _ = get_risk_level(h["prediction"])
            st.markdown(
                f'<div class="history-row">'
                f'<span style="color:#7a9bbf; font-family:monospace; font-size:0.7rem;">#{len(st.session_state.history)-i}</span>'
                f'<span style="font-family:monospace; font-size:0.8rem;">{h["prediction"]:.4f}</span>'
                f'<span style="color:{col}; font-size:0.7rem; font-weight:600;">{lvl}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown('<div style="font-size:0.78rem; color:#7a9bbf;">No predictions yet.</div>', unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

# =============================================================================
# MAIN CONTENT
# =============================================================================

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Flood Risk Intelligence</div>
    <p class="hero-sub">
        Estimate flood likelihood for a region by weighing key climate, terrain,
        and infrastructure conditions.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Two column layout ─────────────────────────────────────────────────────────
col_inputs, col_result = st.columns([1.6, 1], gap="large")

# =============================================================================
# LEFT — FEATURE INPUTS
# =============================================================================

with col_inputs:
    st.markdown('<div class="section-label">Input Parameters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Adjust each indicator on a scale of <b style="color:#00c2e0">1 (low)</b> to '
        '<b style="color:#ff5252">10 (high)</b>. Higher values generally indicate greater flood risk.</div>',
        unsafe_allow_html=True
    )

    for group in GROUPS:
        features_in_group = [f for f, m in FEATURE_META.items() if m["group"] == group]
        if not features_in_group:
            continue

        st.markdown(
            f'<div class="section-label">{GROUP_ICONS[group]} {group}</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(2)
        for idx, feat in enumerate(features_in_group):
            meta    = FEATURE_META[feat]
            current = st.session_state.input_values.get(feat, 5)
            with cols[idx % 2]:
                val = st.slider(
                    label    = f"{meta['icon']} {feat}",
                    min_value= meta["min"],
                    max_value= meta["max"],
                    value    = int(current),
                    step     = 1,
                    help     = meta["help"],
                    key      = f"slider_{feat}",
                )
                st.session_state.input_values[feat] = val

# =============================================================================
# RIGHT — RESULT PANEL
# =============================================================================

with col_result:
    st.markdown('<div class="section-label">Risk Assessment</div>', unsafe_allow_html=True)

    # ── Current input summary ─────────────────────────────────────────────
    avg_val  = sum(st.session_state.input_values.values()) / len(st.session_state.input_values)
    max_feat = max(st.session_state.input_values, key=st.session_state.input_values.get)
    min_feat = min(st.session_state.input_values, key=st.session_state.input_values.get)

    st.markdown(f"""
    <div class="card">
        <div style="font-size:0.72rem; color:#7a9bbf; font-family:monospace; margin-bottom:0.75rem;">
            CURRENT INPUT SUMMARY
        </div>
        <div class="stat-row">
            <div class="stat-pill">
                <div class="stat-pill-value">{avg_val:.1f}</div>
                <div class="stat-pill-label">Avg Score</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-value">{st.session_state.input_values[max_feat]}</div>
                <div class="stat-pill-label">Max Value</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-value">{st.session_state.input_values[min_feat]}</div>
                <div class="stat-pill-label">Min Value</div>
            </div>
        </div>
        <div style="margin-top:0.75rem; font-size:0.72rem; color:#7a9bbf; font-family:monospace;">
            ▲ Highest: {FEATURE_META[max_feat]['icon']} {max_feat}<br>
            ▼ Lowest:  {FEATURE_META[min_feat]['icon']} {min_feat}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Rate-limit countdown helpers ──────────────────────────────────────
    def seconds_until_reset() -> int:
        """Remaining cool-down seconds; 0 when the window has expired."""
        if st.session_state.rate_limited_at is None:
            return 0
        elapsed = time.time() - st.session_state.rate_limited_at
        remaining = int(RATE_LIMIT_SECS - elapsed)
        return max(remaining, 0)

    secs_left = seconds_until_reset()

    # ── Predict button (disabled while rate-limited) ──────────────────────
    predict_clicked = st.button(
        "🌊 Assess Flood Risk",
        use_container_width=True,
        disabled=(secs_left > 0),
    )

    if predict_clicked:
        if not api_ok:
            st.error("❌ Prediction service is offline. Please try again shortly.")
        else:
            with st.spinner("Running prediction..."):
                try:
                    payload = {k: float(v) for k, v in st.session_state.input_values.items()}
                    result  = call_predict(payload)
                    pred    = result["prediction"]

                    # Clear any stale rate-limit state on success
                    st.session_state.rate_limited_at = None
                    st.session_state.rate_limit_msg  = ""

                    st.session_state.last_result = {
                        "prediction": pred,
                        "features"  : payload,
                    }
                    st.session_state.history.append({"prediction": pred})

                except requests.exceptions.HTTPError as e:
                    # ── 429 — Rate limit hit ──────────────────────────────
                    if e.response is not None and e.response.status_code == 429:
                        st.session_state.rate_limited_at = time.time()
                        try:
                            msg = e.response.json().get("detail", "Rate limit exceeded.")
                        except Exception:
                            msg = "Rate limit exceeded: 10 requests per minute."
                        st.session_state.rate_limit_msg = msg
                        st.rerun()          # re-render immediately to show the banner
                    else:
                        code = e.response.status_code if e.response is not None else "?"
                        st.error(f"❌ Error {code}: {e.response.text[:200]}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the prediction service. Please try again shortly.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

    # ── Rate-limit banner (shown instead of / alongside the result) ───────
    if secs_left > 0:
        st.markdown(f"""
        <div class="rate-limit-banner">
            <div class="rate-limit-icon">🚫</div>
            <div class="rate-limit-title">RATE LIMIT REACHED</div>
            <div class="rate-limit-sub">{st.session_state.rate_limit_msg}</div>
            <div class="rate-limit-timer">{secs_left}s</div>
            <div class="rate-limit-note">Cooldown resets automatically · limit is 10 req / min</div>
        </div>
        """, unsafe_allow_html=True)

        # Auto-refresh every second so the countdown ticks down visually
        time.sleep(1)
        st.rerun()

    # ── Result display ────────────────────────────────────────────────────
    elif st.session_state.last_result:
        pred   = st.session_state.last_result["prediction"]
        pct    = min(pred * 100, 100)
        lvl, color, emoji, advice = get_risk_level(pred)

        st.markdown(f"""
        <div class="result-card" style="margin-top:1rem;">
            <div class="result-label">FLOOD PROBABILITY</div>
            <div class="result-value" style="color:{color};">{pred:.4f}</div>
            <div style="font-size:0.72rem; color:#7a9bbf; font-family:monospace; margin-bottom:1rem;">
                {pct:.1f}% probability
            </div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{pct}%;
                    background: linear-gradient(90deg, #00e676, {color});"></div>
            </div>
            <div class="gauge-labels">
                <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1rem; padding:1rem 1.25rem;
            background:{color}18; border:1px solid {color}55;
            border-radius:10px; text-align:center;">
            <div style="font-size:1.5rem; margin-bottom:4px;">{emoji}</div>
            <div style="font-size:1.1rem; font-weight:700; color:{color};
                letter-spacing:0.05em;">{lvl} RISK</div>
            <div style="font-size:0.8rem; color:#7a9bbf; margin-top:4px;">{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem;">Top Factors</div>', unsafe_allow_html=True)
        sorted_features = sorted(
            st.session_state.last_result["features"].items(),
            key=lambda x: x[1], reverse=True
        )[:5]

        for feat, val in sorted_features:
            bar_pct = val / 10 * 100
            icon    = FEATURE_META.get(feat, {}).get("icon", "•")
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between;
                    font-size:0.75rem; margin-bottom:3px;">
                    <span style="color:#f0f6ff;">{icon} {feat}</span>
                    <span style="font-family:monospace; color:#00c2e0;">{val}/10</span>
                </div>
                <div class="gauge-track" style="height:5px;">
                    <div class="gauge-fill" style="width:{bar_pct}%;
                        background:linear-gradient(90deg,#1a5fa8,#00c2e0);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:2.5rem 1rem; color:#7a9bbf;">
            <div style="font-size:3rem; margin-bottom:0.5rem; opacity:0.4;">🌊</div>
            <div style="font-size:0.85rem;">Set the parameters and click<br>
            <b style="color:#00c2e0;">Assess Flood Risk</b> to run the model.</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding:1.5rem 0 0.5rem;
    font-size:0.72rem; color:#7a9bbf; font-family:monospace;">
    FloodSense v2.0
</div>
""", unsafe_allow_html=True)