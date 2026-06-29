"""
Detecția textului generat de AI — Licență Farca Flavius
Versiunea 3.0 — Temă albastru deschis, fără emoji
"""

import os
import io
import datetime
import torch
import pandas as pd
import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ── Configurare pagină ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TextScan AI — Licență Farca Flavius",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constante ─────────────────────────────────────────────────────────────────
MAX_TOKENS = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_V1_DIR = os.path.join(BASE_DIR, "..", "roberta2")
MODEL_V2_DIR = os.path.join(BASE_DIR, "..", "roberta3")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Texte exemple ─────────────────────────────────────────────────────────────
EXAMPLE_HUMAN = (
    "Yesterday I tried to fix the leaky tap in the bathroom again. Third time this month. "
    "I thought I finally had it — tightened everything, watched a YouTube tutorial twice — "
    "but no, water still dripping when I woke up this morning. My landlord hasn't replied "
    "in six days. The towels are basically permanently damp at this point. On the bright side, "
    "I made a pretty decent pasta last night with whatever was left in the fridge: some old "
    "cherry tomatoes, half a block of feta, and way too much garlic. Ate it standing at the "
    "counter because I couldn't be bothered to set the table. Sometimes that's the best kind."
)

EXAMPLE_AI = (
    "The proliferation of artificial intelligence technologies has fundamentally transformed "
    "the landscape of modern education, offering unprecedented opportunities for personalized "
    "learning experiences. By leveraging advanced machine learning algorithms and natural "
    "language processing capabilities, educational institutions can now provide adaptive "
    "curricula that dynamically respond to individual student needs and learning trajectories. "
    "Furthermore, AI-powered assessment tools enable educators to gain deeper insights into "
    "student performance metrics, facilitating evidence-based pedagogical decision-making "
    "and ultimately enhancing overall educational outcomes across diverse learning environments."
)

PLACEHOLDER_TEXT = (
    "Lipeste sau scrie textul pe care vrei sa il analizezi.\n\n"
    "Aplicatia analizeaza stilul de scriere si structura lingvistica a textului "
    "pentru a determina daca a fost redactat de un om sau generat de un model AI. "
    "Rezultatele sunt calculate pe baza probabilitatilor returnate de modelul RoBERTa "
    "antrenat pe mii de exemple de text uman si text generat artificial.\n\n"
    "Pentru rezultate cat mai precise, recomandam un text de cel putin 50 de cuvinte, "
    "redactat in limba engleza. Textele foarte scurte sau fragmentele izolate pot produce "
    "rezultate mai putin fiabile. Daca textul depaseste 128 de tokeni, acesta va fi "
    "trunchiat automat la primele 128 de tokeni inainte de analiza."
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
    font-family: 'Syne', sans-serif !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
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

/* ── Contor tokeni ───────────────────────────────────────── */
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
    Împarte textul în fragmente de max MAX_TOKENS și face media probabilităților.
    Returneaza (avg_human, avg_ai, num_chunks, high_ai_segments).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if not tokens:
        return 0.5, 0.5, 0, []

    chunk_size = MAX_TOKENS - 2
    chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    human_probs = []
    ai_probs = []
    chunk_details = []

    for chunk in chunks:
        input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        pad_len = MAX_TOKENS - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        inputs = {
            "input_ids": torch.tensor([input_ids]).to(DEVICE),
            "attention_mask": torch.tensor([attention_mask]).to(DEVICE),
        }

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
            # Depending on batch size and PyTorch version, squeeze(0) ensures 1D list
            human_probs.append(probs[0])
            ai_probs.append(probs[1])

            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            chunk_details.append({"text": chunk_text, "ai_prob": probs[1] * 100})

    avg_human = sum(human_probs) / len(human_probs)
    avg_ai = sum(ai_probs) / len(ai_probs)

    high_ai_segments = [c for c in chunk_details if c["ai_prob"] >= 50.0]

    high_ai_segments.sort(key=lambda x: x["ai_prob"], reverse=True)

    return float(avg_human), float(avg_ai), len(chunks), high_ai_segments, chunk_details


# ── Functie reutilizabila: afisare rezultat ────────────────────────────────────


def render_result(
    prob_ai_pct: float, prob_human_pct: float, model_name: str, num_chunks: int = 1
):
    """Afiseaza cardul de rezultat, barele si chip-ul de model."""

    is_ai = prob_ai_pct > 50.0

    if is_ai:

        st.markdown(
            f"""

            <div class="result-panel result-panel-ai">

              <p class="result-verdict result-verdict-ai">Text generat de AI</p>

              <div class="score-big score-big-ai">{prob_ai_pct:.1f}<span style="font-size:1.4rem;">%</span></div>

              <p class="score-sublabel">Probabilitate AI</p>

              <p class="result-confidence">

                Scor uman: <strong>{prob_human_pct:.2f}%</strong>

                &nbsp;&nbsp;|&nbsp;&nbsp;

                Scor AI: <strong>{prob_ai_pct:.2f}%</strong>

              </p>

            </div>

            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""

            <div class="result-panel result-panel-human">

              <p class="result-verdict result-verdict-human">Text scris de om</p>

              <div class="score-big score-big-human">{prob_human_pct:.1f}<span style="font-size:1.4rem;">%</span></div>

              <p class="score-sublabel">Probabilitate Uman</p>

              <p class="result-confidence">

                Scor uman: <strong>{prob_human_pct:.2f}%</strong>

                &nbsp;&nbsp;|&nbsp;&nbsp;

                Scor AI: <strong>{prob_ai_pct:.2f}%</strong>

              </p>

            </div>

            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""

        <div class="prob-section">

          <div class="prob-row">

            <span class="prob-row-label">Uman</span>

            <div class="prob-track"><div class="prob-fill-human" style="width:{prob_human_pct:.1f}%;"></div></div>

            <span class="prob-pct">{prob_human_pct:.1f}%</span>

          </div>

          <div class="prob-row">

            <span class="prob-row-label">AI</span>

            <div class="prob-track"><div class="prob-fill-ai" style="width:{prob_ai_pct:.1f}%;"></div></div>

            <span class="prob-pct">{prob_ai_pct:.1f}%</span>

          </div>

        </div>

        <div class="model-chip">

          Model activ: <strong>{model_name}</strong>

        </div>

        <div style="font-size:0.75rem; color:#64748b; margin-top:10px;">

          Notă: Acest scor este media a <strong>{num_chunks}</strong> fragmente de text analizate.

        </div>

        """,
        unsafe_allow_html=True,
    )


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

      <h2>TextScan AI</h2>

    </div>

    """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sidebar-section">Model activ</p>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        label="model_selector",
        options=[
            "RoBERTa v1 — ChatGPT only",
            "RoBERTa v2 — 5 modele AI",
        ],
        index=0,
        label_visibility="collapsed",
    )

    selected_key = "v1" if "v1" in model_choice else "v2"

    active_model = resources[selected_key]

    if selected_key == "v1":

        st.markdown(
            """

        <div style="background:#ffffff;border:1px solid #d1d5db;border-radius:8px;

                    padding:10px 12px;margin:6px 0;">

          <div style="font-size:0.85rem;color:#4b5563;line-height:1.6;">

            Antrenat pe texte ChatGPT-3.5 si ChatGPT-4.<br>

            Performanta redusa pe alte modele AI.

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

            Antrenat pe ChatGPT, Llama, Claude, Gemini, Mistral.<br>

            Robustete superioara pe mai multe modele AI.

          </div>

        </div>

        """,
            unsafe_allow_html=True,
        )

    st.markdown('<p class="sidebar-section">Sistem</p>', unsafe_allow_html=True)

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

  <h1>Detectia textului generat de AI</h1>

  <p class="tagline">

    <span class="live-dot"></span>

    Modele RoBERTa fine-tuned &nbsp;·&nbsp; Clasificare binara Uman / AI

    &nbsp;·&nbsp; Licenta Farca Flavius

  </p>

</div>

""",
    unsafe_allow_html=True,
)


# ── ZONA PRINCIPALA ───────────────────────────────────────────────────────────

col_input, col_output = st.columns([1.1, 1], gap="large")


with col_input:

    tab_text, tab_file = st.tabs(["Scrie / Lipeste text", "Incarca fisier"])

    # ─── TAB 1: Text manual ───────────────────────────────────────────────────

    with tab_text:

        text_value = st.text_area(
            label="input_text",
            value=st.session_state.text_input,
            placeholder=PLACEHOLDER_TEXT,
            height=250,
            label_visibility="collapsed",
            key="text_area_widget",
        )

        st.session_state.text_input = text_value

        # Contor tokeni

        n_tok = count_tokens(tokenizer, text_value.strip()) if text_value.strip() else 0

        num_chunks_ui = (n_tok + MAX_TOKENS - 3) // (MAX_TOKENS - 2) if n_tok > 0 else 0

        st.markdown(
            f"""

            <div class="token-bar" style="margin-bottom:10px;">

              <span class="token-label">Tokeni</span>

              <div style="font-size:0.8rem;color:#4b5563;font-family:'DM Mono',monospace;">

                Total: <strong>{n_tok}</strong> &nbsp;·&nbsp; <strong>{num_chunks_ui}</strong> fragmente

              </div>

            </div>

            """,
            unsafe_allow_html=True,
        )

        # Exemple rapide

        ex_c1, ex_c2 = st.columns(2)

        with ex_c1:

            if st.button(
                "Exemplu text uman", use_container_width=True, key="btn_human"
            ):

                st.session_state.text_input = EXAMPLE_HUMAN

                st.session_state.last_result = None

                st.rerun()

        with ex_c2:

            if st.button("Exemplu text AI", use_container_width=True, key="btn_ai"):

                st.session_state.text_input = EXAMPLE_AI

                st.session_state.last_result = None

                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        detect_btn = st.button(
            "Detecteaza",
            type="primary",
            use_container_width=True,
            disabled=not bool(text_value.strip()),
            key="detect_btn",
        )

        file_detect_btn = False

        file_text_clean = None

    # ─── TAB 2: Incarca fisier ────────────────────────────────────────────────

    with tab_file:

        uploaded_files = st.file_uploader(
            label="Incarca unul sau mai multe fisiere (.txt, .csv, .pdf)",
            type=["txt", "csv", "pdf"],
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

                            st.warning(f"{uf.name}: fisier gol.")

                    elif uf.name.endswith(".csv"):

                        df = pd.read_csv(uf)

                        if not df.empty:

                            cols = list(df.columns)

                            default = "text" if "text" in cols else cols[0]

                            sel_col = st.selectbox(
                                f"{uf.name} - Coloana text:",
                                cols,
                                index=cols.index(default),
                                key=f"csv_col_{uf.name}",
                            )

                            sel_row = st.number_input(
                                f"{uf.name} - Randul:",
                                min_value=0,
                                max_value=len(df) - 1,
                                value=0,
                                step=1,
                                key=f"csv_row_{uf.name}",
                            )

                            loaded_files.append(
                                (uf.name, str(df[sel_col].iloc[sel_row]), None)
                            )

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
                                f"{uf.name}: nu s-a putut extrage text (posibil scanat)."
                            )

                except Exception as e:

                    st.error(f"Eroare {uf.name}: {e}")

        # Afisam lista de fisiere incarcate

        if loaded_files:

            st.markdown(
                f'<div style="font-size:0.85rem;color:#4b5563;margin:8px 0;">'
                f"{len(loaded_files)} fisier(e) gata pentru analiză:</div>",
                unsafe_allow_html=True,
            )

            for fname, ftext, _ in loaded_files:

                n_tok_f = count_tokens(tokenizer, ftext)

                n_ch_f = (
                    (n_tok_f + MAX_TOKENS - 3) // (MAX_TOKENS - 2) if n_tok_f > 0 else 0
                )

                st.markdown(
                    f'<div style="background:#f8fafc;border:1px solid #d1d5db;border-radius:8px;'
                    f'padding:8px 14px;margin:4px 0;font-size:0.85rem;color:#1a1a1a;">'
                    f"<strong>{fname}</strong> &nbsp;·&nbsp; {len(ftext):,} caractere"
                    f" &nbsp;·&nbsp; {n_tok_f:,} tokeni &nbsp;·&nbsp; {n_ch_f} fragmente"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            batch_btn = st.button(
                f"Detectează toate ({len(loaded_files)} fisiere)",
                type="primary",
                use_container_width=True,
                key="batch_detect_btn",
            )

            if batch_btn:

                batch_results_new = []

                prog = st.progress(0, text="Analiză în curs...")

                for idx, (fname, ftext, fpdf_raw) in enumerate(loaded_files):

                    prog.progress(
                        int((idx / len(loaded_files)) * 100),
                        text=f"Se analizează: {fname}",
                    )

                    ph, pai, nch, high_segs, all_segs = predict(
                        active_model, tokenizer, ftext
                    )

                    ph_pct = ph * 100

                    pai_pct = pai * 100

                    br = {
                        "source_filename": fname,
                        "label": "AI" if pai_pct > 50 else "Uman",
                        "prob_ai": pai_pct,
                        "prob_human": ph_pct,
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

                        ai_seg_txt += f"Fragment {si} (Scor AI: {seg['ai_prob']:.1f}%):\n\"{seg['text']}\"\n\n"

                    br["report_plain"] = (
                        f"TEXTSCAN AI — RAPORT: {fname}\n"
                        f"================================\n"
                        f"Timestamp : {br['timestamp']}\n"
                        f"Rezultat  : {br['label']}\n"
                        f"Scor AI   : {br['prob_ai']:.2f}%\n"
                        f"Scor Uman : {br['prob_human']:.2f}%\n"
                        f"Model     : {br['model']}\n"
                        f"Fragmente : {nch}\n"
                        f"{('\nSECTIUNI SUSPECTE AI:\n' + '-'*40 + '\n' + ai_seg_txt) if ai_seg_txt else ''}\n"
                        f"================================\n"
                    )

                    batch_results_new.append(br)

                prog.progress(100, text="Analiză finalizată!")

                st.session_state.batch_results = batch_results_new

                # Nu setam last_result pentru a nu interfera cu Tab 1

                st.session_state.last_result = None

                st.rerun()

        else:

            st.markdown(
                '<div style="color:#9ca3af;font-size:0.9rem;margin-top:20px;text-align:center;">'
                "Incarca unul sau mai multe fisiere pentru a le analiza.</div>",
                unsafe_allow_html=True,
            )

            file_detect_btn = False

            file_text_clean = None

    # Detectia se poate declansa din ambele tab-uri

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

with col_output:

    st.markdown('<p class="section-label">Rezultat</p>', unsafe_allow_html=True)

    if should_detect and active_text and active_text.strip():

        text_clean = active_text.strip()

        n_tok_check = count_tokens(tokenizer, text_clean)

        num_chunks_check = (
            (n_tok_check + MAX_TOKENS - 3) // (MAX_TOKENS - 2) if n_tok_check > 0 else 0
        )

        with st.spinner(f"Se analizează {num_chunks_check} fragmente de text..."):

            prob_human, prob_ai, num_chunks, high_ai_segments, all_segments = predict(
                active_model, tokenizer, text_clean
            )

        prob_human_pct = prob_human * 100

        prob_ai_pct = prob_ai * 100

        result_dict = {
            "label": "AI" if prob_ai_pct > 50 else "Uman",
            "prob_ai": prob_ai_pct,
            "prob_human": prob_human_pct,
            "model": model_choice,
            "model_key": selected_key,
            "num_chunks": num_chunks,
            "text_preview": text_clean[:120],
            "high_ai_segments": high_ai_segments,
            "all_segments": all_segments,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Generam PDF-ul imediat, cu spinner, si il stocam in session_state

        with st.spinner("Se generează raportul PDF..."):

            try:

                result_dict["pdf_report_bytes"] = generate_pdf_report(
                    result_dict, st.session_state.get("pdf_bytes")
                )

            except Exception as _pdf_err:

                result_dict["pdf_report_bytes"] = None

                result_dict["pdf_report_error"] = str(_pdf_err)

        st.session_state.last_result = result_dict

        render_result(prob_ai_pct, prob_human_pct, model_choice, num_chunks)

    elif st.session_state.last_result is not None:

        r = st.session_state.last_result

        render_result(r["prob_ai"], r["prob_human"], r["model"], r.get("num_chunks", 1))

    else:

        st.markdown(
            """

        <div class="empty-state">

          <p class="empty-state-title">Niciun text analizat</p>

          <p class="empty-state-sub">Scrie sau lipeste text, sau incarca un fisier, apoi apasa Detecteaza</p>

        </div>

        """,
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
                "TEXTSCAN AI - RAPORT DE ANALIZA",
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
        f"Rezultat global: {r['label']} (AI: {r['prob_ai']:.2f}% | Uman: {r['prob_human']:.2f}%)",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(0, 8, f"Model folosit: {r['model']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0,
        8,
        f"Fragmente analizate: {r.get('num_chunks', 1)}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(
        0,
        10,
        "TEXT ANALIZAT (fragmentele AI sunt evidentiate cu rosu):",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)

    html_text = ""
    for seg in r.get("all_segments", []):
        safe_text = html_lib.escape(seg["text"]).replace("\n", "<br>")
        if seg["ai_prob"] > 50.0:
            html_text += f'<mark style="background-color: #ffcccc;">{safe_text}</mark> '
        else:
            html_text += f"{safe_text} "

    pdf.set_font("Arial", size=11)
    pdf.write_html(html_text)

    return bytes(pdf.output())


# ── EXPORT (rezultat unic din Tab 1 sau fisier singular) ────────────────────────
if st.session_state.last_result is not None:
    r = st.session_state.last_result
    label_str = r["label"]
    ci_color = "hl-red" if label_str == "AI" else "hl-green"

    ai_segments_text = ""
    if r.get("high_ai_segments"):
        ai_segments_text = "\n\nSECȚIUNI SUSPECTE DE A FI GENERATE DE AI (Peste 50%):\n"
        ai_segments_text += "-" * 40 + "\n"
        for idx, seg in enumerate(r["high_ai_segments"], 1):
            ai_segments_text += f"Fragment {idx} (Scor AI: {seg['ai_prob']:.1f}%):\n"
            ai_segments_text += f"\"{seg['text']}\"\n\n"

    report_plain = (
        f"TEXTSCAN AI — RAPORT DE ANALIZA\n"
        f"================================\n"
        f"Timestamp : {r['timestamp']}\n"
        f"Rezultat  : {label_str}\n"
        f"Scor AI   : {r['prob_ai']:.2f}%\n"
        f"Scor Uman : {r['prob_human']:.2f}%\n"
        f"Model     : {r['model']}\n"
        f"Fragmente : {r.get('num_chunks', 1)}\n"
        f"\n"
        f"Text analizat (primele 120 caractere):\n"
        f"  {r.get('text_preview', '...')}...\n"
        f"{ai_segments_text}"
        f"================================\n"
        f"Licenta Farca Flavius · RoBERTa fine-tuned\n"
        f"Rezultatele sunt probabilistice si nu inlocuiesc judecata unui expert.\n"
    )

    report_html = report_plain.replace(
        label_str,
        f'<span class="{ci_color}">{label_str}</span>',
        1,
    )

    st.markdown(
        """
    <div class="export-section">
      <p class="export-header">Export raport</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    exp_c1, exp_c2 = st.columns([2, 1])

    with exp_c1:
        st.markdown(
            f'<div class="export-code">{report_html.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

    with exp_c2:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_name = r.get("source_filename", "")
        base_name = (
            f"TextScan_{orig_name}" if orig_name else f"textscan_{r['model_key']}_{ts}"
        )
        fname_txt = base_name + ".txt"
        fname_pdf = base_name + ".pdf"

        st.download_button(
            label="Descarca raport text",
            data=report_plain.encode("utf-8"),
            file_name=fname_txt,
            mime="text/plain",
            use_container_width=True,
        )

        # PDF-ul este deja generat si stocat in session_state — buton instant
        cached_pdf = r.get("pdf_report_bytes")
        if cached_pdf:
            st.download_button(
                label="Descarca raport PDF",
                data=cached_pdf,
                file_name=fname_pdf,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        elif r.get("pdf_report_error"):
            st.error(f"PDF indisponibil: {r['pdf_report_error']}")


# ── EXPORT BATCH (multiple PDF-uri din Tab 2) ───────────────────────────────────
if st.session_state.get("batch_results"):
    st.markdown(
        """
    <div class="export-section">
      <p class="export-header">Rezultate batch (multiple fisiere)</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    for br in st.session_state.batch_results:
        label_c = "hl-red" if br["label"] == "AI" else "hl-green"
        with st.expander(
            f"{br['source_filename']}  —  {br['label']} ({br['prob_ai']:.1f}% AI)",
            expanded=False,
        ):
            render_result(
                br["prob_ai"], br["prob_human"], br["model"], br.get("num_chunks", 1)
            )
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.download_button(
                    label="Descarca raport text",
                    data=br["report_plain"].encode("utf-8"),
                    file_name=f"TextScan_{br['source_filename']}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"dl_txt_{br['source_filename']}",
                )
            with dl_c2:
                if br.get("pdf_report_bytes"):
                    st.download_button(
                        label="Descarca raport PDF",
                        data=br["pdf_report_bytes"],
                        file_name=f"TextScan_{br['source_filename']}",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                        key=f"dl_pdf_{br['source_filename']}",
                    )
                elif br.get("pdf_error"):
                    st.error(f"Eroare PDF: {br['pdf_error']}")
