with open("d:/licenta_practica/AI_Detector_App/app.py", "r", encoding="utf-8") as f:
    app_lines = f.readlines()

with open("d:/licenta_practica/AI_Detector_App/recovered_block.py", "r", encoding="utf-8") as f:
    rec_lines = f.readlines()

new_app = app_lines[:562] + rec_lines + app_lines[563:]

# But wait, in recovered_block.py, the function `generate_pdf_report` was placed at the end!
# Actually, I wanted to move `generate_pdf_report` up.
# Right now, `recovered_block.py` ends at `doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")`
# And then `app_lines[563:]` is `for seg in r.get("high_ai_segments", []):`
# So the merging is PERFECT to restore the state before the bad multi_replace!
# Let's restore the state first.

with open("d:/licenta_practica/AI_Detector_App/app_restored.py", "w", encoding="utf-8") as f:
    f.writelines(new_app)
