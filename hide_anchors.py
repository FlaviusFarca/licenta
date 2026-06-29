import sys

file_path = "d:/licenta_practica/AI_Detector_App/app.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

css_addition = """/* ── Hide Anchor Links in Headers ────────────────────────────────────────── */
.stMarkdown h1 a,
.stMarkdown h2 a,
.stMarkdown h3 a,
.stMarkdown h4 a,
.stMarkdown h5 a,
.stMarkdown h6 a {
    display: none !important;
}
/* ── UI Element Cursors ──────────────────────────────────────────── */"""

if "/* ── UI Element Cursors ──────────────────────────────────────────── */" in content:
    content = content.replace("/* ── UI Element Cursors ──────────────────────────────────────────── */", css_addition)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Added CSS to hide anchor links.")
else:
    print("Could not find the target CSS to replace.")
