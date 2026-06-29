    avg_ai = sum(ai_probs) / len(ai_probs)
    
    high_ai_segments = [c for c in chunk_details if c["ai_prob"] >= 50.0]
    high_ai_segments.sort(key=lambda x: x["ai_prob"], reverse=True)
    
    return float(avg_human), float(avg_ai), len(chunks), high_ai_segments, chunk_details


# ── Functie reutilizabila: afisare rezultat ────────────────────────────────────
def render_result(prob_ai_pct: float, prob_human_pct: float, model_name: str, num_chunks: int = 1):
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
    st.markdown("""
    <div class="sidebar-logo">
      <h2>TextScan AI</h2>
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #d1d5db;border-radius:8px;
                    padding:10px 12px;margin:6px 0;">
          <div style="font-size:0.85rem;color:#4b5563;line-height:1.6;">
            Antrenat pe texte ChatGPT-3.5 si ChatGPT-4.<br>
            Performanta redusa pe alte modele AI.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #d1d5db;border-radius:8px;
                    padding:10px 12px;margin:6px 0;">
          <div style="font-size:0.85rem;color:#4b5563;line-height:1.6;">
            Antrenat pe ChatGPT, Llama, Claude, Gemini, Mistral.<br>
            Robustete superioara pe mai multe modele AI.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">Sistem</p>', unsafe_allow_html=True)
    device_str = "GPU - CUDA" if DEVICE.type == "cuda" else "CPU"
    st.markdown(
        f'<div style="font-family:\'DM Mono\',monospace;font-size:0.85rem;'
        f'color:#4b5563;line-height:1.9;">'
        f'Device: <span style="color:#4b5563;">{device_str}</span><br>'
        f'Max tokens: <span style="color:#4b5563;">{MAX_TOKENS}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── HEADER PRINCIPAL ──────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>Detectia textului generat de AI</h1>
  <p class="tagline">
    <span class="live-dot"></span>
    Modele RoBERTa fine-tuned &nbsp;·&nbsp; Clasificare binara Uman / AI
    &nbsp;·&nbsp; Licenta Farca Flavius
  </p>
</div>
""", unsafe_allow_html=True)


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
            if st.button("Exemplu text uman", use_container_width=True, key="btn_human"):
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
                            sel_col = st.selectbox(f"{uf.name} - Coloana text:", cols,
                                                   index=cols.index(default),
                                                   key=f"csv_col_{uf.name}")
                            sel_row = st.number_input(f"{uf.name} - Randul:", min_value=0,
                                                      max_value=len(df)-1, value=0,
                                                      step=1, key=f"csv_row_{uf.name}")
                            loaded_files.append((uf.name, str(df[sel_col].iloc[sel_row]), None))

                    elif uf.name.endswith(".pdf"):
                        pdf_raw = uf.read()
                        reader  = pypdf.PdfReader(io.BytesIO(pdf_raw))
                        pages_text = []
                        for page in reader.pages:
                            t = page.extract_text()
                            if t:
                                pages_text.append(t)
                        full_text = "\n".join(pages_text).strip()
                        if full_text:
                            loaded_files.append((uf.name, full_text, pdf_raw))
                        else:
                            st.warning(f"{uf.name}: nu s-a putut extrage text (posibil scanat).")

                except Exception as e:
                    st.error(f"Eroare {uf.name}: {e}")

        # Afisam lista de fisiere incarcate
        if loaded_files:
            st.markdown(
                f'<div style="font-size:0.85rem;color:#4b5563;margin:8px 0;">'
                f'{len(loaded_files)} fisier(e) gata pentru analiză:</div>',
                unsafe_allow_html=True,
            )
            for fname, ftext, _ in loaded_files:
                n_tok_f = count_tokens(tokenizer, ftext)
                n_ch_f  = (n_tok_f + MAX_TOKENS - 3) // (MAX_TOKENS - 2) if n_tok_f > 0 else 0
                st.markdown(
                    f'<div style="background:#f8fafc;border:1px solid #d1d5db;border-radius:8px;'
                    f'padding:8px 14px;margin:4px 0;font-size:0.85rem;color:#1a1a1a;">'
                    f'<strong>{fname}</strong> &nbsp;·&nbsp; {len(ftext):,} caractere'
                    f' &nbsp;·&nbsp; {n_tok_f:,} tokeni &nbsp;·&nbsp; {n_ch_f} fragmente'
                    f'</div>',
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
                        text=f"Se analizează: {fname}"
                    )
                    ph, pai, nch, high_segs, all_segs = predict(active_model, tokenizer, ftext)
                    ph_pct  = ph  * 100
                    pai_pct = pai * 100
                    br = {
                        "source_filename": fname,
                        "label":    "AI" if pai_pct > 50 else "Uman",
                        "prob_ai":  pai_pct,
                        "prob_human": ph_pct,
                        "model":    model_choice,
                        "model_key": selected_key,
                        "num_chunks": nch,
                        "high_ai_segments": high_segs,
                        "all_segments": all_segs,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                'Incarca unul sau mai multe fisiere pentru a le analiza.</div>',
                unsafe_allow_html=True,
            )
            file_detect_btn = False
            file_text_clean = None

    # Detectia se poate declansa din ambele tab-uri
    should_detect = detect_btn or st.session_state.get("auto_detect", False) or bool(file_text_clean)
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
        num_chunks_check = (n_tok_check + MAX_TOKENS - 3) // (MAX_TOKENS - 2) if n_tok_check > 0 else 0

        with st.spinner(f"Se analizează {num_chunks_check} fragmente de text..."):
            prob_human, prob_ai, num_chunks, high_ai_segments, all_segments = predict(active_model, tokenizer, text_clean)

        prob_human_pct = prob_human * 100
        prob_ai_pct    = prob_ai    * 100

        result_dict = {
            "label":        "AI" if prob_ai_pct > 50 else "Uman",
            "prob_ai":      prob_ai_pct,
            "prob_human":   prob_human_pct,
            "model":        model_choice,
            "model_key":    selected_key,
            "num_chunks":   num_chunks,
            "text_preview": text_clean[:120],
            "high_ai_segments": high_ai_segments,
            "all_segments": all_segments,
            "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        st.markdown("""
        <div class="empty-state">
          <p class="empty-state-title">Niciun text analizat</p>
          <p class="empty-state-sub">Scrie sau lipeste text, sau incarca un fisier, apoi apasa Detecteaza</p>
        </div>
        """, unsafe_allow_html=True)


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