import sys

file_path = "d:/licenta_practica/AI_Detector_App/app.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Update file_uploader label and type
content = content.replace(
    'label="Upload one or more files (.txt, .csv, .pdf, .docx)",',
    'label="Upload one or more files (.csv, .pdf, .docx)",'
)
content = content.replace(
    'type=["txt", "csv", "pdf", "docx"],',
    'type=["csv", "pdf", "docx"],'
)

# Remove the `.txt` handling logic
target_txt_logic = """                    if uf.name.endswith(".txt"):
                        content = uf.read().decode("utf-8", errors="replace").strip()
                        if content:
                            loaded_files.append((uf.name, content, None))
                        else:
                            st.warning(f"{uf.name}: empty file.")

                    elif uf.name.endswith(".csv"):"""

if target_txt_logic in content:
    content = content.replace(target_txt_logic, '                    if uf.name.endswith(".csv"):')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Removed TXT upload support.")
