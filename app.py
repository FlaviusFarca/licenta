"""
Detecția textului generat de AI — Licență Farca Flavius
Versiunea 3.0 — Temă albastru deschis, fără emoji
"""

import os
import io
import html as html_lib
import datetime
import torch
import pandas as pd
import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ── Configurare pagină ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TextScan",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constante ─────────────────────────────────────────────────────────────────
MAX_TOKENS          = 128           # Max tokens per chunk (RoBERTa limit)
DEFAULT_THRESHOLD   = 0.5          # Default decision threshold (50%)
DEFAULT_THRESHOLD_PCT = 50         # Default threshold as percentage
MIN_WORDS_RECOMMENDED = 80         # Minimum recommended word count for reliable results
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
MODEL_V1_DIR        = os.path.join(BASE_DIR, "..", "roberta2")
MODEL_V2_DIR        = os.path.join(BASE_DIR, "..", "roberta3")
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PLACEHOLDER_TEXT = (
    f"Paste or write the text you want to analyze.\n\n"
    f"For the most accurate results, we recommend submitting a text of at least {MIN_WORDS_RECOMMENDED} words "
    "written in English. Very short texts or isolated fragments may produce less reliable "
    f"results. If the text exceeds {MAX_TOKENS} tokens, it will be automatically split into "
    "multiple fragments and analyzed sequentially."
)

# ── CSS — temă albastru deschis ───────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
/* ── Reset ───────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
font-family: 'DM Sans', system-ui, sans-serif !important;
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
/* ── Fundal pagina: culori prietenoase pentru proiector ─────────────────────── */
.stApp {
background-color: #ffffff !important; /* soft bluish-gray for better visibility on projectors */
}
.block-container {
padding-top: 1.5rem !important;
padding-bottom: 2rem !important;
max-width: 1200px !important;
background-color: transparent !important;
}
/* ── Hide Anchor Links in Headers ────────────────────────────────────────── */
.stMarkdown h1 a,
.stMarkdown h2 a,
.stMarkdown h3 a,
.stMarkdown h4 a,
.stMarkdown h5 a,
.stMarkdown h6 a {
    display: none !important;
}
/* ── UI Element Cursors ──────────────────────────────────────────── */
.stTextArea [data-testid="stInputInstructions"] {
    display: none !important;
}
div[data-baseweb="select"],
div[data-baseweb="select"] * {
    cursor: pointer !important;
}
/* ── Header ──────────────────────────────────────────────── */
.app-header {
background: #ffffff;
border-radius: 12px;
padding: 24px 32px;
margin-bottom: 20px;
border: 1px solid #d1d5db; /* lighter gray border */
box-shadow: 0 1px 4px rgba(30, 58, 138, 0.1); /* subtle dark blue shadow */
border-top: 3px solid #1e3a8a; /* darker blue accent for better contrast */
}
.app-header h1 {
font-family: 'DM Sans', system-ui, sans-serif !important;
font-size: 1.55rem !important;
font-weight: bold !important;
color: #1a1a1a !important;
margin: 0 0 5px 0 !important;
letter-spacing: -0.02em !important;
}
.app-header .tagline {
font-size: 0.83rem;
color: #4b5563;
margin: 0;
line-height: 1.55;
}
.live-dot {
display: inline-block;
width: 7px;
height: 7px;
background: #16a34a;
border-radius: 50%;
margin-right: 6px;
vertical-align: middle;
animation: live-pulse 2.5s ease infinite;
}
@keyframes live-pulse {
0%, 100% { opacity: 1; }
50%       { opacity: 0.35; }
}
/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
background: #eff6ff !important;
border-right: 1px solid #d1d5db !important;
}
section[data-testid="stSidebar"] > div {
background: #eff6ff !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label {
color: #1a1a1a !important;
}
.sidebar-logo {
padding: 16px 0 12px;
border-bottom: 1px solid #d1d5db;
margin-bottom: 14px;
}
.sidebar-logo h2 {
font-family: 'Syne', sans-serif !important;
font-size: 1.1rem !important;
font-weight: 800 !important;
color: #1a1a1a !important;
margin: 0 !important;
}
.sidebar-section {
font-size: 0.8rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.1em;
color: #4b5563;
margin: 18px 0 8px;
padding-bottom: 5px;
border-bottom: 1px solid #dbeafe;
}
/* Selectbox în sidebar */
section[data-testid="stSidebar"] .stSelectbox > div > div {
background: #ffffff !important;
border: 1px solid #d1d5db !important;
border-radius: 8px !important;
color: #1a1a1a !important;
}
section[data-testid="stSidebar"] .stFileUploader {
background: #ffffff !important;
border: 1px dashed #d1d5db !important;
border-radius: 8px !important;
}
/* ── Text area ───────────────────────────────────────────── */
.stTextArea textarea {
background: #ffffff !important;
border: 1px solid #d1d5db !important;
border-radius: 10px !important;
color: #1a1a1a !important;
font-family: 'DM Sans', sans-serif !important;
font-size: 0.92rem !important;
line-height: 1.75 !important;
padding: 14px 16px !important;
resize: vertical !important;
transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stTextArea textarea:focus {
border-color: #1d4ed8 !important;
box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.1) !important;
outline: none !important;
}
.stTextArea textarea::placeholder {
color: #4b5563 !important;
font-size: 0.87rem !important;
}
/* ── Contor tokens ───────────────────────────────────────── */
.token-bar {
display: flex;
align-items: center;
gap: 10px;
margin: 6px 0 14px;
}
.token-label {
font-size: 0.8rem;
font-family: 'DM Mono', monospace;
color: #4b5563;
flex-shrink: 0;
width: 52px;
}
.token-track {
flex: 1;
height: 4px;
background: #bfdbfe;
border-radius: 999px;
overflow: hidden;
}
.token-fill-ok   { height: 100%; background: #16a34a; border-radius: 999px; transition: width 0.3s ease; }
.token-fill-warn { height: 100%; background: #ef4444; border-radius: 999px; transition: width 0.3s ease; }
.token-count-ok   { color: #16a34a; font-family: 'DM Mono', monospace; font-size: 0.8rem; width: 56px; text-align: right; }
.token-count-warn { color: #dc2626; font-family: 'DM Mono', monospace; font-size: 0.8rem; font-weight: 600; width: 56px; text-align: right; }
/* ── Butoane exemple ─────────────────────────────────────── */
.stButton button {
font-family: 'DM Sans', sans-serif !important;
font-weight: 500 !important;
transition: all 0.15s ease !important;
}
/* Butoane secundare (exemple) */
div[data-testid="column"] .stButton button:not([kind="primary"]) {
background: #ffffff !important;
border: 1px solid #d1d5db !important;
color: #1a1a1a !important;
border-radius: 8px !important;
font-size: 0.83rem !important;
}
div[data-testid="column"] .stButton button:not([kind="primary"]):hover {
background: #eff6ff !important;
border-color: #1d4ed8 !important;
color: #1a1a1a !important;
}
/* ── Buton primar — Detecteaza ───────────────────────────── */
.stButton button[kind="primary"] {
background: #1d4ed8 !important;
color: #ffffff !important;
border: none !important;
border-radius: 10px !important;
font-weight: 700 !important;
font-size: 0.95rem !important;
letter-spacing: -0.01em !important;
box-shadow: 0 2px 8px rgba(29, 78, 216, 0.3) !important;
transition: all 0.15s ease !important;
}
.stButton button[kind="primary"]:hover {
background: #1d4ed8 !important;
box-shadow: 0 4px 16px rgba(29, 78, 216, 0.4) !important;
transform: translateY(-1px) !important;
}
.stButton button[kind="primary"]:active {
transform: translateY(0) !important;
}
.stButton button[kind="primary"]:disabled {
background: #bfdbfe !important;
color: #93c5fd !important;
box-shadow: none !important;
transform: none !important;
}
/* ── Aviz trunchiere ─────────────────────────────────────── */
.trunc-notice {
display: flex;
align-items: flex-start;
gap: 10px;
background: #fefce8;
border: 1px solid #fde047;
border-radius: 8px;
padding: 10px 14px;
margin: 8px 0 12px;
}
.trunc-notice-text {
font-size: 0.81rem;
color: #92400e;
line-height: 1.5;
margin: 0;
}
/* ── Card rezultat ───────────────────────────────────────── */
.result-panel {
border-radius: 12px;
padding: 24px;
margin-bottom: 16px;
position: relative;
overflow: hidden;
background: #ffffff;
border: 1px solid #d1d5db;
box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.result-panel-ai::before {
content: '';
position: absolute;
top: 0; left: 0; right: 0; height: 3px;
background: #ef4444;
}
.result-panel-human::before {
content: '';
position: absolute;
top: 0; left: 0; right: 0; height: 3px;
background: #16a34a;
}
.result-verdict {
font-family: 'Syne', sans-serif !important;
font-size: 1.35rem !important;
font-weight: 800 !important;
letter-spacing: -0.02em !important;
margin: 0 0 8px 0 !important;
}
.result-verdict-ai    { color: #dc2626 !important; }
.result-verdict-human { color: #16a34a !important; }
.score-big {
font-family: 'Syne', sans-serif;
font-size: 3.2rem;
font-weight: 800;
line-height: 1;
letter-spacing: -0.04em;
margin: 10px 0 4px;
}
.score-big-ai    { color: #ef4444; }
.score-big-human { color: #16a34a; }
.score-sublabel {
font-size: 0.8rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.08em;
color: #4b5563;
margin: 0;
}
.result-confidence {
font-family: 'DM Mono', monospace;
font-size: 0.77rem;
color: #4b5563;
margin: 8px 0 0;
}
.result-confidence strong { color: #4b5563; }
/* ── Bare probabilitate ──────────────────────────────────── */
.prob-section { margin-top: 18px; }
.prob-row {
display: flex;
align-items: center;
gap: 10px;
margin: 8px 0;
}
.prob-row-label {
font-size: 0.75rem;
font-weight: 700;
color: #4b5563;
width: 52px;
flex-shrink: 0;
text-transform: uppercase;
letter-spacing: 0.05em;
}
.prob-track {
flex: 1;
background: #f1f5f9;
border-radius: 999px;
height: 6px;
overflow: hidden;
}
.prob-fill-human { height: 100%; background: #16a34a; border-radius: 999px; }
.prob-fill-ai    { height: 100%; background: #ef4444; border-radius: 999px; }
.prob-pct {
font-family: 'DM Mono', monospace;
font-size: 0.8rem;
color: #4b5563;
width: 40px;
text-align: right;
flex-shrink: 0;
}
/* ── Model chip ──────────────────────────────────────────── */
.model-chip {
display: inline-flex;
align-items: center;
gap: 6px;
background: #f1f5f9;
border: 1px solid #d1d5db;
border-radius: 6px;
padding: 4px 10px;
font-family: 'DM Mono', monospace;
font-size: 0.75rem;
color: #4b5563;
margin-top: 14px;
}
.model-chip strong { color: #334155; }
/* ── Stare goala ─────────────────────────────────────────── */
.empty-state {
background: #ffffff;
border: 1px dashed #d1d5db;
border-radius: 12px;
padding: 52px 24px;
text-align: center;
box-shadow: 0 1px 4px rgba(29,78,216,0.06);
}
.empty-state-title {
font-size: 0.9rem;
font-weight: 600;
color: #4b5563;
margin: 0 0 4px;
}
.empty-state-sub {
font-size: 0.78rem;
color: #bfdbfe;
margin: 0;
}
/* ── Sectiune export ─────────────────────────────────────── */
.export-section {
background: #ffffff;
border: 1px solid #d1d5db;
border-radius: 12px;
padding: 20px;
margin-top: 20px;
box-shadow: 0 1px 4px rgba(29,78,216,0.06);
}
.export-header {
font-size: 0.8rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.1em;
color: #4b5563;
margin: 0 0 12px;
}
.export-code {
background: #f8fafc;
border: 1px solid #d1d5db;
border-radius: 8px;
padding: 14px 16px;
font-family: 'DM Mono', monospace;
font-size: 0.8rem;
color: #4b5563;
line-height: 1.8;
white-space: pre;
overflow-x: auto;
}
.export-code .hl-green { color: #16a34a; font-weight: 600; }
.export-code .hl-red   { color: #dc2626; font-weight: 600; }
/* ── Download button ─────────────────────────────────────── */
.stDownloadButton button {
background: #ffffff !important;
border: 1px solid #d1d5db !important;
color: #1a1a1a !important;
border-radius: 8px !important;
font-family: 'DM Sans', sans-serif !important;
font-size: 0.82rem !important;
font-weight: 500 !important;
transition: all 0.15s ease !important;
}
.stDownloadButton button:hover {
background: #eff6ff !important;
border-color: #1d4ed8 !important;
color: #1a1a1a !important;
}
/* ── Spinner ─────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #1d4ed8 !important; }
/* ── Eticheta sectiune ───────────────────────────────────── */
.section-label {
font-size: 0.8rem;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.1em;
color: #4b5563;
margin: 0 0 8px;
}
/* ── Info/Alert boxes ────────────────────────────────────── */
.stAlert {
background: #ffffff !important;
border-color: #d1d5db !important;
border-radius: 8px !important;
}
/* ── Tooltip Animatie ────────────────────────────────────── */
.stTooltipIcon {
    transition: color 0.2s ease, transform 0.2s ease !important;
}
.stTooltipIcon:hover {
    color: #1d4ed8 !important;
    transform: scale(1.15) !important;
}
div[data-baseweb="tooltip"] {
    animation: tooltipFadeIn 0.2s ease-in forwards !important;
}
div[data-baseweb="tooltip"] > div {
    animation: tooltipSlideUp 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    transform-origin: bottom center !important;
}
@keyframes tooltipFadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
@keyframes tooltipSlideUp {
    0% { transform: scale(0.85) translateY(4px); }
    100% { transform: scale(1) translateY(0); }
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Incarcare modele (cache) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Se initializeaza modelele RoBERTa...")
def load_models():
    """
    Incarca tokenizer-ul si ambele modele RoBERTa o singura data la pornire.
    Returneaza dict cu tokenizer si modelele v1 / v2.
    """
    v1_path = os.path.normpath(MODEL_V1_DIR)
    v2_path = os.path.normpath(MODEL_V2_DIR)

    tokenizer = RobertaTokenizer.from_pretrained(v1_path)

    model_v1 = RobertaForSequenceClassification.from_pretrained(v1_path)
    model_v1.to(DEVICE)
    model_v1.eval()

    model_v2 = RobertaForSequenceClassification.from_pretrained(v2_path)
    model_v2.to(DEVICE)
    model_v2.eval()

    return {"tokenizer": tokenizer, "v1": model_v1, "v2": model_v2}


# ── Logica predictie ──────────────────────────────────────────────────────────
def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def predict(model, tokenizer, text: str):
    """
    Împarte textul în fragments de max MAX_TOKENS și face media probabilităților.
    Returneaza (avg_human, avg_ai, num_chunks, high_ai_segments, chunk_details, ai_percentage).
    ai_percentage = % din fragmente care depășesc threshold-ul setat.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if not tokens:
        return 0.5, 0.5, 0, [], [], 0.0

    chunk_size = MAX_TOKENS - 2
    chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    # Merge the last chunk into the previous one if it's smaller than 80 tokens
    if len(chunks) > 1 and len(chunks[-1]) < 80:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)

    human_probs = []
    ai_probs = []
    chunk_details = []

    for chunk in chunks:
        input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        pad_len = max(0, MAX_TOKENS - len(input_ids))
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        inputs = {
            "input_ids": torch.tensor([input_ids]).to(DEVICE),
            "attention_mask": torch.tensor([attention_mask]).to(DEVICE),
        }

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
            human_probs.append(probs[0])
            ai_probs.append(probs[1])

            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            chunk_details.append({"text": chunk_text, "ai_prob": probs[1] * 100})

    avg_human = sum(human_probs) / len(human_probs)
    avg_ai = sum(ai_probs) / len(ai_probs)

    # Threshold aplicat per-fragment
    threshold_pct = st.session_state.get('threshold', DEFAULT_THRESHOLD) * 100
    high_ai_segments = [c for c in chunk_details if c["ai_prob"] >= threshold_pct]
    high_ai_segments.sort(key=lambda x: x["ai_prob"], reverse=True)

    # Procentul de text detectat ca AI = fragmente flagged / total fragmente
    ai_percentage = (len(high_ai_segments) / len(chunks)) * 100 if chunks else 0.0

    return float(avg_human), float(avg_ai), len(chunks), high_ai_segments, chunk_details, ai_percentage


# ── Functie reutilizabila: afisare rezultat ────────────────────────────────────


def render_evolution_chart(result_dict):
    all_segments = result_dict.get("all_segments", [])
    if not all_segments or len(all_segments) <= 1:
        return
    
    st.markdown('<p style="font-weight:600; font-size: 1.1rem; margin-top: 10px;">Fragment Evolution Chart</p>', unsafe_allow_html=True)
    
    chart_data = pd.DataFrame(
        {"AI Probability (%)": [seg["ai_prob"] for seg in all_segments]},
        index=[f"Chunk {i+1}" for i in range(len(all_segments))]
    )
    
    st.line_chart(chart_data)

def render_result(result_dict: dict):
    """Afiseaza cardul de rezultat bazat pe procentul de fragmente AI."""

    ai_pct = result_dict.get("ai_percentage", 0.0)
    num_chunks = result_dict.get("num_chunks", 1)
    high_ai = result_dict.get("high_ai_segments", [])
    model_name = result_dict.get("model", "")
    threshold_pct = st.session_state.get('threshold', DEFAULT_THRESHOLD) * 100
    flagged = len(high_ai)

    # Culoare graduală: verde (0%) → galben (1-30%) → roşu (>30%)
    if ai_pct == 0:
        color = "#16a34a"
        bg_color = "#f0fdf4"
        border_color = "#86efac"
        verdict_text = "No AI fragments detected"
    elif ai_pct <= 30:
        color = "#d97706"
        bg_color = "#fffbeb"
        border_color = "#fcd34d"
        verdict_text = "Minimal AI content"
    else:
        color = "#dc2626"
        bg_color = "#fef2f2"
        border_color = "#fca5a5"
        verdict_text = "Majority AI content"

    verdict_sub = f"{flagged} din {num_chunks} fragment(s) exceeded the threshold of {threshold_pct:.0f}%"

    st.markdown(
        f"""
<div style="background:{bg_color}; border:2px solid {border_color}; border-radius:16px; padding:24px 20px; text-align:center; margin-bottom:16px;">
  <p style="font-size:0.85rem; font-weight:700; color:#6b7280; letter-spacing:0.08em; text-transform:uppercase; margin:0 0 8px 0;">{verdict_text}</p>
  <div style="font-size:4rem; font-weight:900; color:{color}; line-height:1;">{ai_pct:.0f}<span style="font-size:1.8rem;">%</span></div>
  <p style="font-size:0.9rem; color:{color}; font-weight:600; margin:6px 0 0 0;">din text este generat de AI</p>
  <p style="font-size:0.78rem; color:#6b7280; margin:4px 0 0 0;">{verdict_sub}</p>
</div>
<div style="background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:12px 16px; margin-bottom:8px;">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
    <span style="font-size:0.8rem; color:#6b7280; min-width:90px;">AI detectat</span>
    <div style="flex:1; background:#f3f4f6; border-radius:999px; height:10px; overflow:hidden;">
      <div style="width:{ai_pct:.1f}%; background:{color}; height:100%; border-radius:999px;"></div>
    </div>
    <span style="font-size:0.8rem; font-weight:700; color:{color}; min-width:42px; text-align:right;">{ai_pct:.1f}%</span>
  </div>
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="font-size:0.8rem; color:#6b7280; min-width:90px;">Uman detectat</span>
    <div style="flex:1; background:#f3f4f6; border-radius:999px; height:10px; overflow:hidden;">
      <div style="width:{100 - ai_pct:.1f}%; background:#16a34a; height:100%; border-radius:999px;"></div>
    </div>
    <span style="font-size:0.8rem; font-weight:700; color:#16a34a; min-width:42px; text-align:right;">{100 - ai_pct:.1f}%</span>
  </div>
</div>
<div style="font-size:0.72rem; color:#9ca3af; text-align:center; margin-top:2px;">
  Average model score: Human <strong>{result_dict.get('prob_human', 0):.1f}%</strong> &nbsp;·&nbsp; AI <strong>{result_dict.get('prob_ai', 0):.1f}%</strong>
  &nbsp;·&nbsp; Model: <strong>{model_name}</strong> &nbsp;·&nbsp; Prag: <strong>{threshold_pct:.0f}%</strong>
</div>
""",
        unsafe_allow_html=True,
    )


def render_highlighted_text(result_dict: dict):
    """
    Renders the analyzed text with per-fragment color highlights based on AI probability.
    Green = Human, Orange = borderline, Red = AI.
    Each fragment has a tooltip showing the exact AI score.
    """
    all_segs = result_dict.get("all_segments", [])
    if not all_segs:
        return

    ai_threshold = st.session_state.get('threshold', DEFAULT_THRESHOLD) * 100
    human_threshold = ai_threshold * 0.6  # Borderline starts at 60% of the AI threshold

    st.markdown(
        '<p style="font-weight:600; font-size:1.05rem; margin-top:18px; margin-bottom:6px;">'
        'Highlighted Text Analysis</p>',
        unsafe_allow_html=True,
    )

    # ── Legend ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
<div style="display:flex; gap:16px; margin-bottom:10px; flex-wrap:wrap;">
  <span style="font-size:0.78rem; color:#374151;">
    <span style="background:#bbf7d0; border-radius:4px; padding:1px 7px; margin-right:4px;">&nbsp;</span>
    Human (&lt;&nbsp;{human_threshold:.0f}%)
  </span>
  <span style="font-size:0.78rem; color:#374151;">
    <span style="background:#fde68a; border-radius:4px; padding:1px 7px; margin-right:4px;">&nbsp;</span>
    Borderline ({human_threshold:.0f}–{ai_threshold:.0f}%)
  </span>
  <span style="font-size:0.78rem; color:#374151;">
    <span style="background:#fca5a5; border-radius:4px; padding:1px 7px; margin-right:4px;">&nbsp;</span>
    AI (&ge;&nbsp;{ai_threshold:.0f}%)
  </span>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Hover effect CSS (injected once per render) ─────────────────────────────
    st.markdown(
        """
<style>
.hl-span {
    display: inline;
    transition: all 0.15s ease;
}
.hl-span:hover {
    -webkit-text-stroke: 0.3px currentColor;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    filter: brightness(0.92);
    position: relative;
    z-index: 2;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # ── Build highlighted HTML ───────────────────────────────────────────────────
    html_parts = [
        '<div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; '
        'padding:20px 22px; line-height:1.8; font-size:0.92rem; color:#1f2937; '
        'font-family:\'DM Sans\', sans-serif; word-wrap:break-word;">'
    ]

    for seg in all_segs:
        raw_text = seg["text"]
        score    = seg["ai_prob"]

        # Replace newlines with <br> so span background covers across line breaks
        text = html_lib.escape(raw_text).replace("\n", "<br>")

        if score < human_threshold:
            bg      = "#bbf7d0"
            border  = "#86efac"
            verdict = f"Human — AI probability: {score:.1f}%"
        elif score < ai_threshold:
            bg      = "#fde68a"
            border  = "#fbbf24"
            verdict = f"Borderline — AI probability: {score:.1f}%"
        else:
            bg      = "#fca5a5"
            border  = "#f87171"
            verdict = f"AI detected — AI probability: {score:.1f}%"

        html_parts.append(
            f'<span class="hl-span" title="{verdict}" style="'
            f'background:{bg}; border-bottom:2px solid {border}; border-radius:3px; '
            f'padding:4px 3px; margin:0; cursor:help;">'
            f'{text}</span>'
        )

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:0.72rem; color:#9ca3af; margin-top:4px;">'
        'Hover over any highlighted fragment to see its exact AI probability score.</p>',
        unsafe_allow_html=True,
    )





def generate_pdf_report(r, original_pdf_bytes=None):
    """

    Daca exista PDF original: il deschide cu PyMuPDF si aplica highlight rosu

    pe secventele detectate ca AI, pastrand layout-ul 1:1.

    Fallback: genereaza un PDF nou cu fpdf2 daca nu exista original.

    """

    # ── Varianta 1: PDF original disponibil → 1:1 cu highlight ──────────────

    if original_pdf_bytes is not None:

        import fitz  # PyMuPDF

        doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
        for seg in r.get("high_ai_segments", []):
            segment_text = seg["text"].strip()
            if not segment_text:
                continue
            # Impartim segmentul in fraze mai scurte pentru match mai bun
            # (PyMuPDF cauta string-uri exacte pe fiecare pagina)
            words = segment_text.split()
            # Incercam mai intai textul complet, apoi bucati de 10 cuvinte
            search_phrases = [segment_text]
            if len(words) > 10:
                for i in range(0, len(words), 8):
                    phrase = " ".join(words[i : i + 8])
                    if len(phrase) > 10:
                        search_phrases.append(phrase)

            for page in doc:
                for phrase in search_phrases:
                    instances = page.search_for(phrase, quads=False)
                    for rect in instances:
                        # Adaugam adnotare highlight cu culoare rosie
                        highlight = page.add_highlight_annot(rect)
                        highlight.set_colors(stroke=[1, 0.2, 0.2])  # RGB rosu
                        highlight.update()

        return doc.tobytes()

    # ── Varianta 2: Fallback fpdf2 daca nu exista PDF original ──────────────
    from fpdf import FPDF
    import html as html_lib

    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 15)
            self.cell(
                0,
                10,
                "TEXTSCAN - ANALYSIS REPORT",
                align="C",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            self.ln(5)

    pdf = PDF()
    pdf.add_font("Arial", "", "C:\\Windows\\Fonts\\arial.ttf")
    pdf.add_font("Arial", "B", "C:\\Windows\\Fonts\\arialbd.ttf")

    pdf.add_page()

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Timestamp: {r['timestamp']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0,
        8,
        f"Global result: {r['label']} (AI: {r['prob_ai']:.2f}% | Human: {r['prob_human']:.2f}%)",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(0, 8, f"Model used: {r['model']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0,
        8,
        f"Analyzed fragments: {r.get('num_chunks', 1)}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(
        0,
        10,
        "ANALYZED TEXT (AI fragments are highlighted in red):",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)

    html_text = ""
    for seg in r.get("all_segments", []):
        safe_text = html_lib.escape(seg["text"]).replace("\n", "<br>")
        threshold_pct = st.session_state.get('threshold', DEFAULT_THRESHOLD) * 100
        if seg["ai_prob"] >= threshold_pct:
            # fpdf2 write_html doesn't support CSS background colors or <mark>.
            # We use bold red font to highlight AI text in the PDF.
            html_text += f'<b><font color="#dc2626">{safe_text}</font></b> '
        else:
            html_text += f"{safe_text} "

    pdf.set_font("Arial", size=11)
    pdf.write_html(html_text)

    return bytes(pdf.output())


# ── Initializare session state ────────────────────────────────────────────────

if "text_input" not in st.session_state:

    st.session_state.text_input = ""

if "last_result" not in st.session_state:

    st.session_state.last_result = None

if "pdf_bytes" not in st.session_state:

    st.session_state.pdf_bytes = None

if "batch_results" not in st.session_state:

    st.session_state.batch_results = []


# ── Incarcare modele ──────────────────────────────────────────────────────────

resources = load_models()

tokenizer = resources["tokenizer"]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-logo">
<h2>TextScan</h2>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sidebar-section">Active Model</p>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        label="model_selector",
        options=[
            "RoBERTa v1 — ChatGPT only",
            "RoBERTa v2 — 5 AI models",
            "Compare Models (v1 & v2)",
        ],
        index=0,
        label_visibility="collapsed",
    )
    


    if "v1" in model_choice and "v2" not in model_choice:
        selected_key = "v1"
    elif "v2" in model_choice and "v1" not in model_choice:
        selected_key = "v2"
    else:
        selected_key = "compare" 

    active_model = resources[selected_key] if selected_key != "compare" else None

    if selected_key == "v1":

        st.markdown(
            """
<div style="background:#ffffff;border:1px solid #d1d5db;border-radius:8px;
padding:10px 12px;margin:6px 0;">
<div style="font-size:0.85rem;color:#4b5563;line-height:1.6;">
Trained on ChatGPT-3.5 and ChatGPT-4 texts.<br>
Reduced performance on other AI models.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    elif selected_key == "v2":

        st.markdown(
            """
<div style="background:#ffffff;border:1px solid #d1d5db;border-radius:8px;
padding:10px 12px;margin:6px 0;">
<div style="font-size:0.85rem;color:#4b5563;line-height:1.6;">
Trained on ChatGPT, Llama, Claude, Gemini, Mistral.<br>
Superior robustness across multiple AI models.
</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div style="background:#ffffff;border:1px solid #d1d5db;border-radius:8px;
padding:10px 12px;margin:6px 0;">
<div style="font-size:0.85rem;color:#4b5563;line-height:1.6;">
Executes parallel inference across both architectures, producing a comprehensive comparative analysis.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<p class="sidebar-section">System</p>', unsafe_allow_html=True)

    device_str = "GPU - CUDA" if DEVICE.type == "cuda" else "CPU"

    st.markdown(
f"<div style=\"font-family:'DM Mono',monospace;font-size:0.85rem;"
f'color:#4b5563;line-height:1.9;">'
f'Device: <span style="color:#4b5563;">{device_str}</span><br>'
f'Max tokens: <span style="color:#4b5563;">{MAX_TOKENS}</span>'
f"</div>",
        unsafe_allow_html=True,
    )




# ── HEADER PRINCIPAL ──────────────────────────────────────────────────────────

st.markdown(
    """
<div class="app-header">
<h1>AI Generated Text Detection</h1>
<p class="tagline">
<span class="live-dot"></span>
Fine-tuned RoBERTa models &nbsp;·&nbsp; Binary classification Human / AI

</p>
</div>
""",
    unsafe_allow_html=True,
)




# ── ZONA PRINCIPALA ───────────────────────────────────────────────────────────

tab_text, tab_file, tab_guide, tab_metrics = st.tabs(["Write / Paste text", "Upload file", "Model Guide", "Performance Metrics"])

# ─── TAB 1: Text manual ───────────────────────────────────────────────────

with tab_text:
    col_input, col_output = st.columns([1.1, 1], gap="large")
    
    with col_input:
        text_value = st.text_area(
            label="input_text",
            value=st.session_state.text_input,
            placeholder=PLACEHOLDER_TEXT,
            height=250,
            label_visibility="collapsed",
            key="text_area_widget",
        )

        st.session_state.text_input = text_value

        # Contor tokens

        n_tok = count_tokens(tokenizer, text_value.strip()) if text_value.strip() else 0

        num_chunks_ui = (n_tok + MAX_TOKENS - 3) // (MAX_TOKENS - 2) if n_tok > 0 else 0

        st.markdown(
            f"""
<div class="token-bar" style="margin-bottom:10px;">
<span class="token-label">Tokens</span>
<div style="font-size:0.8rem;color:#4b5563;font-family:'DM Mono',monospace;">
Total: <strong>{n_tok}</strong> &nbsp;·&nbsp; <strong>{num_chunks_ui}</strong> fragments
</div>
</div>
""",
            unsafe_allow_html=True,
        )
        # Exemple rapide scoase conform cerintei

        threshold_val_text = st.slider(
            "Decision Threshold (%)", 
            min_value=1, 
            max_value=99, 
            value=st.session_state.get('threshold_pct', DEFAULT_THRESHOLD_PCT), 
            step=1, 
            help="The decision threshold sets the sensitivity of the detection. Any text fragment with an AI probability greater than or equal to this threshold will be flagged as artificially generated.",
            key="slider_text"
        )
        st.session_state.threshold = threshold_val_text / 100.0
        st.session_state.threshold_pct = threshold_val_text
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            detect_btn = st.button(
                "Detect text",
                type="primary",
                use_container_width=True,
                disabled=not text_value.strip(),
                key="detect_btn",
            )
        with btn_col2:
            clear_btn = st.button(
                "Clear text",
                type="secondary",
                use_container_width=True,
                key="clear_btn",
            )
            if clear_btn:
                st.session_state.text_input = ""
                st.session_state.last_result = None
                st.session_state.last_compare = None
                st.rerun()

        file_detect_btn = False

        file_text_clean = None

    # ─── TAB 2: Incarca fisier ────────────────────────────────────────────────

with tab_file:

    uploaded_files = st.file_uploader(
        label="Upload one or more files (.csv, .pdf, .docx)",
        type=["csv", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="visible",
        key="file_uploader",
    )

    # Structura: lista de (filename, text, pdf_raw|None)

    loaded_files = []

    if uploaded_files:

        import pypdf

        for uf in uploaded_files:

            try:

                if uf.name.endswith(".txt"):

                    content = uf.read().decode("utf-8", errors="replace").strip()

                    if content:

                        loaded_files.append((uf.name, content, None))

                    else:

                        st.warning(f"{uf.name}: empty file.")

                elif uf.name.endswith(".csv"):

                    df = pd.read_csv(uf)

                    if not df.empty:

                        cols = list(df.columns)

                        default = "text" if "text" in cols else cols[0]

                        sel_col = st.selectbox(
                            f"{uf.name} - Text column:",
                            cols,
                            index=cols.index(default),
                            key=f"csv_col_{uf.name}",
                        )

                        csv_text_parts = []
                        for idx, val in enumerate(df[sel_col]):
                            if pd.notna(val) and str(val).strip():
                                csv_text_parts.append(f"Row {idx+1}: {str(val).strip()}")
                        
                        csv_content = "\n\n".join(csv_text_parts)
                        
                        if csv_content:
                            loaded_files.append(
                                (uf.name, csv_content, None)
                            )
                        else:
                            st.warning(f"{uf.name}: no valid text found in column '{sel_col}'.")

                elif uf.name.endswith(".docx"):
                    import docx
                    doc = docx.Document(uf)
                    full_text = []
                    for para in doc.paragraphs:
                        if para.text.strip():
                            full_text.append(para.text)
                    content = '\n'.join(full_text).strip()
                    if content:
                        loaded_files.append((uf.name, content, None))
                    else:
                        st.warning(f"{uf.name}: empty file.")

                elif uf.name.endswith(".pdf"):

                    pdf_raw = uf.read()

                    reader = pypdf.PdfReader(io.BytesIO(pdf_raw))

                    pages_text = []

                    for page in reader.pages:

                        t = page.extract_text()

                        if t:

                            pages_text.append(t)

                    full_text = "\n".join(pages_text).strip()

                    if full_text:

                        loaded_files.append((uf.name, full_text, pdf_raw))

                    else:

                        st.warning(
                            f"{uf.name}: could not extract text (possibly scanned)."
                        )

            except Exception as e:

                st.error(f"Error {uf.name}: {e}")

    # Afisam lista de fisiere incarcate
    if loaded_files:
        st.session_state.loaded_files_count = len(loaded_files)

        def toggle_select_all():
            val = st.session_state.select_all_files
            count = st.session_state.get("loaded_files_count", 0)
            for i in range(count):
                st.session_state[f"chk_file_{i}"] = val

        st.markdown(
            f'<div style="font-size:0.85rem;color:#4b5563;margin:8px 0;">'
            f"{len(loaded_files)} file(s) loaded. Check the files you want to analyze:</div>",
            unsafe_allow_html=True,
        )

        if "select_all_files" not in st.session_state:
            st.session_state.select_all_files = True

        st.checkbox("Select / Deselect all", key="select_all_files", on_change=toggle_select_all)

        selected_files_to_run = []

        for idx, (fname, ftext, pdf_raw) in enumerate(loaded_files):
            n_tok_f = count_tokens(tokenizer, ftext)
            n_ch_f = (
                (n_tok_f + MAX_TOKENS - 3) // (MAX_TOKENS - 2) if n_tok_f > 0 else 0
            )

            c_chk, c_info = st.columns([0.5, 9.5])
            with c_chk:
                st.markdown('<div style="margin-top:10px;">', unsafe_allow_html=True)
                if f"chk_file_{idx}" not in st.session_state:
                    st.session_state[f"chk_file_{idx}"] = True
                is_selected = st.checkbox("", key=f"chk_file_{idx}", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with c_info:
                st.markdown(
                    f'<div style="background:#f8fafc;border:1px solid #d1d5db;border-radius:8px;'
                    f'padding:8px 14px;margin:2px 0 6px 0;font-size:0.85rem;color:#1a1a1a;">'
                    f"<strong>{fname}</strong> &nbsp;·&nbsp; {len(ftext):,} caractere"
                    f" &nbsp;·&nbsp; {n_tok_f:,} tokens &nbsp;·&nbsp; {n_ch_f} fragments"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            
            if is_selected:
                selected_files_to_run.append((fname, ftext, pdf_raw))

        threshold_val_file = st.slider(
            "Decision Threshold (%)", 
            min_value=1, 
            max_value=99, 
            value=st.session_state.get('threshold_pct', DEFAULT_THRESHOLD_PCT), 
            step=1, 
            help="The decision threshold sets the sensitivity of the detection. Any text fragment with an AI probability greater than or equal to this threshold will be flagged as artificially generated.",
            key="slider_file"
        )
        st.session_state.threshold = threshold_val_file / 100.0
        st.session_state.threshold_pct = threshold_val_file
        
        batch_btn = st.button(
            f"Detect ({len(selected_files_to_run)} files selected)",
            type="primary",
            use_container_width=True,
            key="batch_detect_btn",
            disabled=(len(selected_files_to_run) == 0)
        )

        if batch_btn and selected_files_to_run:
            batch_results_new = []
            prog = st.progress(0, text="Analysis in progress...")

            for idx, (fname, ftext, fpdf_raw) in enumerate(selected_files_to_run):
                prog.progress(
                    int((idx / len(selected_files_to_run)) * 100),
                    text=f"Analyzing: {fname}",
                )

                if selected_key == "compare":
                    # In compare mode, run both models and add two entries
                    for m_key, m_obj, m_name in [
                        ("v1", resources["v1"], "RoBERTa v1 — ChatGPT only"),
                        ("v2", resources["v2"], "RoBERTa v2 — 5 AI models"),
                    ]:
                        ph, pai, nch, high_segs, all_segs, ai_pct = predict(m_obj, tokenizer, ftext)
                        ph_pct = ph * 100
                        pai_pct = pai * 100
                        threshold_pct_val = st.session_state.get('threshold', DEFAULT_THRESHOLD) * 100
                        br = {
                            "source_filename": f"{fname} [{m_key.upper()}]",
                            "label": f"{ai_pct:.0f}% AI",
                            "prob_ai": pai_pct, "prob_human": ph_pct,
                            "ai_percentage": ai_pct,
                            "model": m_name, "model_key": m_key,
                            "num_chunks": nch, "high_ai_segments": high_segs,
                            "all_segments": all_segs,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        try:
                            br["pdf_report_bytes"] = generate_pdf_report(br, fpdf_raw)
                        except Exception as _e:
                            br["pdf_report_bytes"] = None
                            br["pdf_error"] = str(_e)
                        ai_seg_txt = "".join(
                            f"Fragment {si} (AI Score: {seg['ai_prob']:.1f}%):\n\"{seg['text']}\"\n\n"
                            for si, seg in enumerate(high_segs, 1)
                        )
                        br["report_plain"] = (
                            f"TEXTSCAN AI — RAPORT: {fname} [{m_key.upper()}]\n"
                            f"================================\n"
                            f"Timestamp : {br['timestamp']}\nResult: {br['label']}\n"
                            f"Scor AI: {pai_pct:.2f}%\nHuman Score: {ph_pct:.2f}%\n"
                            f"Model: {m_name}\nFragments: {nch}\n"
                            f"{ai_seg_txt}================================\n"
                        )
                        batch_results_new.append(br)
                    continue  # skip the single-model block below

                ph, pai, nch, high_segs, all_segs, ai_pct = predict(
                    active_model, tokenizer, ftext
                )

                ph_pct = ph * 100

                pai_pct = pai * 100

                br = {
                    "source_filename": fname,
                    "label": f"{ai_pct:.0f}% AI",
                    "prob_ai": pai_pct,
                    "prob_human": ph_pct,
                    "ai_percentage": ai_pct,
                    "model": model_choice,
                    "model_key": selected_key,
                    "num_chunks": nch,
                    "high_ai_segments": high_segs,
                    "all_segments": all_segs,
                    "timestamp": datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }

                # Generam raportul PDF pentru fiecare fisier

                try:

                    br["pdf_report_bytes"] = generate_pdf_report(br, fpdf_raw)

                except Exception as _e:

                    br["pdf_report_bytes"] = None

                    br["pdf_error"] = str(_e)

                # Generam raportul text

                ai_seg_txt = ""

                for si, seg in enumerate(high_segs, 1):

                    ai_seg_txt += f"Fragment {si} (AI Score: {seg['ai_prob']:.1f}%):\n\"{seg['text']}\"\n\n"

                br["report_plain"] = (
                    f"TEXTSCAN AI — RAPORT: {fname}\n"
                    f"================================\n"
                    f"Timestamp : {br['timestamp']}\n"
                    f"Result  : {br['label']}\n"
                    f"Scor AI   : {br['prob_ai']:.2f}%\n"
                    f"Human Score : {br['prob_human']:.2f}%\n"
                    f"Model     : {br['model']}\n"
                    f"Fragments : {nch}\n"
                    f"{('\nSECTIUNI SUSPECTE AI:\n' + '-'*40 + '\n' + ai_seg_txt) if ai_seg_txt else ''}\n"
                    f"================================\n"
                )

                batch_results_new.append(br)

            prog.progress(100, text="Analysis complete!")

            st.session_state.batch_results = batch_results_new

            # Nu setam last_result pentru a nu interfera cu Tab 1

            st.session_state.last_result = None

            st.rerun()

    else:

        st.markdown(
'<div style="color:#9ca3af;font-size:0.9rem;margin-top:20px;text-align:center;">'
"Upload one or more files to analyze them.</div>",
            unsafe_allow_html=True,
        )

        file_detect_btn = False

        file_text_clean = None

# Detectia se poate declansa din ambele tab-uri

with tab_guide:
    st.markdown(
        """
### How to choose the right model?

**RoBERTa v1 (ChatGPT only)**  
Use this model if you specifically suspect the text was generated by **ChatGPT (versions 3.5 or 4)**. Because it was fine-tuned exclusively on OpenAI's specific writing style, it achieves near-perfect accuracy (over 99%) for ChatGPT texts. However, it may struggle to detect text from other AI generators.

**RoBERTa v2 (Multi-model)**  
Use this model as your **default choice** or when the source of the AI text is unknown. It was trained on a highly diverse dataset (including Llama, Claude, Gemini, Mistral, and ChatGPT). It offers significantly better cross-model generalization and robustness across the entire AI landscape.
        """
    )

with tab_metrics:
    st.markdown(
        """
### Transparent Performance Metrics

Unlike many commercial AI detection solutions that act as "black boxes" and do not disclose exact accuracy figures, this application is built on rigorous academic research and provides full transparency regarding its performance capabilities.

#### Model Evolution & Generalization

**RoBERTa v1 (Trained exclusively on ChatGPT)**
On the in-domain test set, the RoBERTa v1 model achieved an accuracy of 99.91%. Only 5 out of 5,368 examples were misclassified (2 AI texts labeled as human and 3 human texts labeled as AI). However, the accuracy dropped significantly to **25.96%** in the cross-model scenario on unseen text generators, proving that transformer architectures trained on a single source suffer from severe degradation in generalization capability.

**RoBERTa v2 (Trained on Multi-model Dataset)**
By extending the training dataset with texts from 5 different AI models, RoBERTa v2 maintained a near-perfect 99.62% in-domain accuracy while dramatically improving cross-model accuracy to **88.49%** (+62.53 percentage points compared to v1). This significant improvement supports the theory that cross-model generalization is driven by the diversity of the training sources.
        """
    )
    
    evolution_data = {
        "Model": ["RoBERTa v1 (ChatGPT only)", "RoBERTa v2 (Multi-model)"],
        "In-domain Accuracy": ["99.91%", "99.62%"],
        "Cross-model Accuracy": ["25.96%", "88.49%"]
    }
    
    st.dataframe(pd.DataFrame(evolution_data), use_container_width=True, hide_index=True)

    st.markdown(
        """
<br>

#### RoBERTa v2: RAID Benchmark Validation
The **RoBERTa v2 (Multi-model)** has been externally validated on the **RAID Benchmark**, testing its robustness against 11 different AI generators. Below are the exact, peer-reviewable performance figures:
        """,
        unsafe_allow_html=True
    )

    metrics_data = {
        "AI Model (Generator)": ["ChatGPT", "Llama-chat", "GPT-3", "Mistral-chat", "Cohere-chat", "GPT-4", "MPT-chat", "Cohere", "GPT-2", "MPT", "Mistral"],
        "Accuracy": ["99.20%", "99.00%", "98.70%", "96.90%", "96.80%", "96.10%", "93.70%", "90.00%", "86.40%", "62.00%", "61.30%"],
        "Precision": ["98.62%", "98.61%", "98.60%", "98.55%", "98.55%", "98.53%", "98.45%", "98.31%", "98.15%", "94.78%", "94.49%"],
        "Recall": ["99.80%", "99.40%", "98.80%", "95.20%", "95.00%", "93.60%", "88.80%", "81.40%", "74.20%", "25.40%", "24.40%"],
        "F1-Score": ["99.20%", "99.00%", "98.70%", "96.85%", "96.74%", "96.00%", "93.38%", "89.06%", "84.51%", "40.06%", "38.28%"]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    st.dataframe(
        df_metrics,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown(
        """
**Key Takeaway:** The model demonstrates exceptional generalization, maintaining an average accuracy of **88.09%** across completely unseen cross-model generators, proving its high reliability in real-world scenarios.
        """
    )

should_detect = (
    detect_btn
    or st.session_state.get("auto_detect", False)
    or bool(file_text_clean)
)

if should_detect:

    st.session_state.auto_detect = False

# Textul de analizat: fie din fisier (tab 2), fie din text area (tab 1)

active_text = file_text_clean if file_text_clean else text_value


# ── COLOANA REZULTATE ─────────────────────────────────────────────────────────

def run_single_model(model_k, mod, m_choice, active_text):
    text_clean = active_text.strip()
    n_tok_check = count_tokens(tokenizer, text_clean)
    num_chunks_check = ((n_tok_check + MAX_TOKENS - 3) // (MAX_TOKENS - 2) if n_tok_check > 0 else 0)

    with st.spinner(f"Analyzing {num_chunks_check} text fragments cu {model_k}..."):
        ph, pa, nc, h_segs, a_segs, ai_pct = predict(mod, tokenizer, text_clean)

    p_h_pct = ph * 100
    p_a_pct = pa * 100
    threshold_pct = st.session_state.get('threshold', DEFAULT_THRESHOLD) * 100

    res = {
        "label": f"{ai_pct:.0f}% AI",
        "prob_ai": p_a_pct,
        "prob_human": p_h_pct,
        "ai_percentage": ai_pct,
        "model": m_choice,
        "model_key": model_k,
        "num_chunks": nc,
        "text_preview": text_clean[:120],
        "high_ai_segments": h_segs,
        "all_segments": a_segs,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with st.spinner(f"Generating PDF report for {model_k}..."):
        try:
            res["pdf_report_bytes"] = generate_pdf_report(res, st.session_state.get("pdf_bytes"))
        except Exception as _pdf_err:
            res["pdf_report_bytes"] = None
            res["pdf_report_error"] = str(_pdf_err)

    return res

def render_result_summary(r):
    render_result(r)
    if r.get("pdf_report_bytes"):
        st.download_button(
            label=f"Download PDF ({r['model_key']})",
            data=r["pdf_report_bytes"],
            file_name=f"TextScan_Report_{r['model_key']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

def render_result_details(r):
    render_highlighted_text(r)
    render_evolution_chart(r)

with tab_text:
    with col_output:
        # 1. Run detection logic first to update session state
        if should_detect and active_text and active_text.strip():
            if selected_key == "compare":
                st.session_state.last_result = None
                with st.spinner("Running both models..."):
                    st.session_state.last_compare = [
                        run_single_model("v1", resources["v1"], "RoBERTa v1 — ChatGPT only", active_text),
                        run_single_model("v2", resources["v2"], "RoBERTa v2 — 5 AI models", active_text)
                    ]
            else:
                st.session_state.last_compare = None
                st.session_state.last_result = run_single_model(selected_key, active_model, model_choice, active_text)
    
        elif should_detect:
            st.warning("Please enter some text to analyze.", icon=None)
    
        # 2. Render UI based on updated state
        if st.session_state.get("last_compare") is None:
            st.markdown('<p class="section-label">Result</p>', unsafe_allow_html=True)
            if st.session_state.get("last_result") is not None:
                render_result_summary(st.session_state.last_result)
            else:
                st.markdown(
                    """
    <div class="empty-state">
    <p class="empty-state-title">No text analyzed</p>
    <p class="empty-state-sub">Write or paste text, or upload a file, then press Detect</p>
    </div>
    """,
                    unsafe_allow_html=True,
                )

with tab_text:
    # ── SINGLE RESULT DETAILS (full width, below main columns) ─────────────────
    if st.session_state.get("last_compare") is None and st.session_state.get("last_result") is not None:
        st.markdown("---")
        render_result_details(st.session_state.last_result)

    # ── COMPARE RESULTS (full width, below main columns) ──────────────────────────
    if st.session_state.get("last_compare") is not None:
        r1, r2 = st.session_state.last_compare
        st.markdown("---")
        st.markdown("""
        <p style="font-size:1.1rem; font-weight:700; margin-bottom:16px;">
        Model Comparison Results
        </p>
        """, unsafe_allow_html=True)
        cmp1, cmp2 = st.columns(2, gap="large")

        def _cmp_colors(ai_pct):
            if ai_pct == 0:
                return "#16a34a", "#f0fdf4", "#86efac"
            elif ai_pct <= 30:
                return "#d97706", "#fffbeb", "#fcd34d"
            else:
                return "#dc2626", "#fef2f2", "#fca5a5"

        with cmp1:
            ai1 = r1.get("ai_percentage", 0.0)
            color1, bg1, border1 = _cmp_colors(ai1)
            st.markdown(f"""
            <div style="background:{bg1}; border:2px solid {border1}; border-radius:14px; padding:20px; text-align:center;">
                <div style="font-size:1rem; font-weight:700; color:#6b7280; margin-bottom:6px;">RoBERTa v1 — ChatGPT only</div>
                <div style="font-size:2rem; font-weight:900; color:{color1};">{r1['label']}</div>
                <div style="margin-top:12px; display:flex; justify-content:space-around;">
                    <div><div style="font-size:0.8rem; color:#6b7280;">Human</div><div style="font-size:1.4rem; font-weight:800;">{r1['prob_human']:.1f}%</div></div>
                    <div><div style="font-size:0.8rem; color:#6b7280;">AI</div><div style="font-size:1.4rem; font-weight:800;">{r1['prob_ai']:.1f}%</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if r1.get("pdf_report_bytes"):
                st.download_button("Download PDF (v1)", r1["pdf_report_bytes"],
                    file_name=f"Report_v1_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf", use_container_width=True, key="dl_v1")
        with cmp2:
            ai2 = r2.get("ai_percentage", 0.0)
            color2, bg2, border2 = _cmp_colors(ai2)
            st.markdown(f"""
            <div style="background:{bg2}; border:2px solid {border2}; border-radius:14px; padding:20px; text-align:center;">
                <div style="font-size:1rem; font-weight:700; color:#6b7280; margin-bottom:6px;">RoBERTa v2 — 5 AI models</div>
                <div style="font-size:2rem; font-weight:900; color:{color2};">{r2['label']}</div>
                <div style="margin-top:12px; display:flex; justify-content:space-around;">
                    <div>
                    <div style="font-size:0.8rem; color:#6b7280;">Human</div>
                    <div style="font-size:1.4rem; font-weight:800;">{r2['prob_human']:.1f}%</div>
                    </div>
                    <div>
                    <div style="font-size:0.8rem; color:#6b7280;">AI</div>
                    <div style="font-size:1.4rem; font-weight:800;">{r2['prob_ai']:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if r2.get("pdf_report_bytes"):
                st.download_button("Download PDF (v2)", r2["pdf_report_bytes"],
                    file_name=f"Report_v2_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf", use_container_width=True, key="dl_v2")

        # ── Highlighted Text (full width, one expander per model) ──────────────
        st.markdown("---")
        st.markdown(
            '<p style="font-weight:700; font-size:1rem; margin-bottom:8px;">Highlighted Text — Per Model</p>',
            unsafe_allow_html=True,
        )
        with st.expander("RoBERTa v1 — ChatGPT only", expanded=True):
            render_highlighted_text(r1)
        with st.expander("RoBERTa v2 — 5 AI models", expanded=True):
            render_highlighted_text(r2)


with tab_file:
    # ── BATCH RESULTS (fisiere multiple) ──────────────────────────────────────────
    batch_res = st.session_state.get("batch_results", [])
    if batch_res:
        st.markdown("---")
        st.markdown(
            f'<p style="font-size:1.1rem; font-weight:700; margin-bottom:16px;">File analysis results ({len(batch_res)} reports)</p>',
            unsafe_allow_html=True,
        )

        # Butoane export global
        all_text = "\n\n".join(r.get("report_plain", "") for r in batch_res)
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            st.download_button(
                "Download all reports (TXT)",
                data=all_text.encode("utf-8"),
                file_name=f"TextScan_BatchReport_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_batch_txt",
            )
        with ecol2:
            import csv
            import io
            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer)
            csv_writer.writerow(["Filename", "AI Probability (%)", "Human Probability (%)", "Result Label", "Model Used", "Analyzed Fragments"])
            for br in batch_res:
                raw_filename = br.get("source_filename", "unknown")
                clean_filename = raw_filename.replace(" [V1]", "").replace(" [V2]", "")
                csv_writer.writerow([
                    clean_filename,
                    f"{br.get('prob_ai', 0):.2f}",
                    f"{br.get('prob_human', 0):.2f}",
                    br.get("label", ""),
                    br.get("model", ""),
                    br.get("num_chunks", 0)
                ])
            st.download_button(
                "Download all reports (CSV)",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name=f"TextScan_BatchReport_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_batch_csv",
            )
        with ecol3:
            import zipfile
            import io
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for i, br in enumerate(batch_res):
                    original_name = br.get("source_filename", f"file_{i}").lower()
                    is_pdf_or_docx = ".pdf" in original_name or ".docx" in original_name
                
                    fname = br.get("source_filename", f"file_{i}").rsplit(".", 1)[0]
                    pdf_bytes = br.get("pdf_report_bytes")
                    if pdf_bytes and is_pdf_or_docx:
                        zip_file.writestr(f"Report_{fname}.pdf", pdf_bytes)
        
            st.download_button(
                "Download all reports (PDF)",
                data=zip_buffer.getvalue(),
                file_name=f"TextScan_BatchReports_PDF_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_batch_pdf",
            )

        # Carduri per fisier
        for i, br in enumerate(batch_res):
            ai_pct = br.get("ai_percentage", 0.0)
            num_ch = br.get("num_chunks", 1)
            flagged = round(ai_pct * num_ch / 100)
            threshold_pct = st.session_state.get('threshold', DEFAULT_THRESHOLD) * 100

            # Culoare bazata pe procentul de fragmente flagged
            if ai_pct == 0:
                color = "#16a34a"; bg = "#f0fdf4"; border = "#86efac"
                icon = ""; verdict = "No AI fragments detected"
            elif ai_pct < 30:
                color = "#d97706"; bg = "#fffbeb"; border = "#fcd34d"
                icon = ""; verdict = "Minimal AI content"
            else:
                color = "#dc2626"; bg = "#fef2f2"; border = "#fca5a5"
                icon = ""; verdict = "Majority AI content"

            expander_title = f"{icon} {br['source_filename']} — {flagged} din {num_ch} fragmente detectate ca AI"
            with st.expander(expander_title, expanded=(i == 0)):
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown(f"""
                    <div style="background:{bg}; border:2px solid {border}; border-radius:12px; padding:20px; text-align:center;">
                        <div style="font-size:0.8rem; color:#6b7280; margin-bottom:6px; font-weight:600;">{br['model']}</div>
                        <div style="font-size:0.8rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">{verdict}</div>
                        <div style="font-size:3rem; font-weight:900; color:{color}; line-height:1;">{ai_pct:.0f}<span style="font-size:1.4rem;">%</span></div>
                        <div style="font-size:0.85rem; color:{color}; font-weight:600; margin:4px 0;">din text este generat de AI</div>
                        <div style="font-size:0.78rem; color:#6b7280; margin-top:6px;">
                            {flagged} din {num_ch} fragment(s) exceeded the threshold of {threshold_pct:.0f}%
                        </div>
                        <div style="font-size:0.72rem; color:#9ca3af; margin-top:4px;">
                            Average model score: Human {br['prob_human']:.1f}% · AI {br['prob_ai']:.1f}%
                        </div>
                        <div style="font-size:0.72rem; color:#9ca3af;">{br['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_b:
                    original_name = br.get("source_filename", "").lower()
                    is_pdf_or_docx = ".pdf" in original_name or ".docx" in original_name
                
                    if br.get("pdf_report_bytes") and is_pdf_or_docx:
                        st.download_button(
                            "Download PDF",
                            data=br["pdf_report_bytes"],
                            file_name=f"Report_{br['source_filename'].replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"dl_batch_pdf_{i}",
                        )
                    if br.get("report_plain"):
                        st.download_button(
                            "Download TXT",
                            data=br["report_plain"].encode("utf-8"),
                            file_name=f"Report_{br['source_filename'].replace(' ','_')}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key=f"dl_batch_txt_{i}",
                        )

