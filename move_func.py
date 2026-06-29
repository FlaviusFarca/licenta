with open("d:/licenta_practica/AI_Detector_App/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

def find_line(lines, search_str):
    for i, l in enumerate(lines):
        if search_str in l:
            return i
    return -1

start_idx = find_line(lines, "def generate_pdf_report(r, original_pdf_bytes=None):")
# end of generate_pdf_report is right before `# ── EXPORT` or similar. Let's find `# ── EXPORT`
end_idx = find_line(lines, "# ── EXPORT (rezultat unic din Tab 1")

# Extract the function
gen_func = lines[start_idx:end_idx]

# Remove the function from its original place
new_lines = lines[:start_idx] + lines[end_idx:]

# Insert the function before `# ── Initializare session state ────────────────────────────────────────────────`
insert_idx = find_line(new_lines, "# ── Initializare session state ────────────────────────────────────────────────")

new_lines = new_lines[:insert_idx] + gen_func + new_lines[insert_idx:]

with open("d:/licenta_practica/AI_Detector_App/app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Moved generate_pdf_report. Start:", start_idx, "End:", end_idx, "Insert:", insert_idx)
