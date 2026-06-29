import io
import fitz
from fpdf import FPDF
import html as html_lib

def generate_pdf_report(r, original_pdf_bytes=None):
    if original_pdf_bytes is not None:
        doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
        for seg in r.get("high_ai_segments", []):
            segment_text = seg["text"].strip()
            if not segment_text:
                continue
            words = segment_text.split()
            search_phrases = [segment_text]
            if len(words) > 10:
                for i in range(0, len(words), 8):
                    phrase = " ".join(words[i:i+8])
                    if len(phrase) > 10:
                        search_phrases.append(phrase)
            
            for page in doc:
                for phrase in search_phrases:
                    instances = page.search_for(phrase, quads=False)
                    for rect in instances:
                        highlight = page.add_highlight_annot(rect)
                        highlight.set_colors(stroke=[1, 0.2, 0.2])
                        highlight.update()
        return doc.tobytes()
    return None

doc_init = fitz.open()
page = doc_init.new_page()
page.insert_text((50, 50), "Test PDF that is Human")
raw_bytes = doc_init.tobytes()

r = {
    "high_ai_segments": [],
    "label": "Uman",
    "timestamp": "2026",
    "prob_ai": 10.0,
    "prob_human": 90.0,
    "model": "model",
    "num_chunks": 1,
    "all_segments": []
}

try:
    res = generate_pdf_report(r, raw_bytes)
    print("Success. length:", len(res) if res else "None")
except Exception as e:
    print("Error:", e)
