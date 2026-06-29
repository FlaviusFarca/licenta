lines = open("d:/licenta_practica/AI_Detector_App/app.py", "r", encoding="utf-8").readlines()

start_idx = -1
for i, l in enumerate(lines):
    if "def generate_pdf_report(r, original_pdf_bytes=None):" in l:
        start_idx = i
        break

end_idx = -1
for i, l in enumerate(lines):
    if i > start_idx and "EXPORT (rezultat unic din Tab 1 sau fisier singular)" in l:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    gen_func = lines[start_idx:end_idx]
    new_lines = lines[:start_idx] + lines[end_idx:]
    
    insert_idx = -1
    for i, l in enumerate(new_lines):
        if "Initializare session state" in l:
            insert_idx = i
            break
            
    if insert_idx != -1:
        new_lines = new_lines[:insert_idx] + gen_func + new_lines[insert_idx:]
        with open("d:/licenta_practica/AI_Detector_App/app.py", "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print("Moved generate_pdf_report successfully to app.py.")
    else:
        print("Could not find insert_idx")
else:
    print(f"Could not find start_idx ({start_idx}) or end_idx ({end_idx})")
