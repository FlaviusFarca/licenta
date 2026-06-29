import sys

file_path = "d:/licenta_practica/AI_Detector_App/app.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Hide the Ctrl+Enter instructions
css_addition = """/* ── UI Element Cursors ──────────────────────────────────────────── */
.stTextArea [data-testid="stInputInstructions"] {
    display: none !important;
}"""
content = content.replace("/* ── UI Element Cursors ──────────────────────────────────────────── */", css_addition)

# 2. Update the button
target_button = """        detect_btn = st.button(
            "Detecteaza",
            type="primary",
            use_container_width=True,
            disabled=not bool(text_value.strip()),
            key="detect_btn",
        )"""

replacement_button = """        detect_btn = st.button(
            "Detect text",
            type="primary",
            use_container_width=True,
            key="detect_btn",
        )"""

if target_button in content:
    content = content.replace(target_button, replacement_button)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Button updated and CSS instructions hidden.")
else:
    print("Could not find the target button to replace.")
