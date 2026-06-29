import sys

file_path = "d:/licenta_practica/AI_Detector_App/app.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Update file_uploader label and type
content = content.replace(
    'label="Upload one or more files (.txt, .csv, .pdf)",',
    'label="Upload one or more files (.txt, .csv, .pdf, .docx)",'
)
content = content.replace(
    'type=["txt", "csv", "pdf"],',
    'type=["txt", "csv", "pdf", "docx"],'
)

# Add docx handling logic
docx_logic = """                    elif uf.name.endswith(".docx"):
                        import docx
                        doc = docx.Document(uf)
                        full_text = []
                        for para in doc.paragraphs:
                            if para.text.strip():
                                full_text.append(para.text)
                        content = '\\n'.join(full_text).strip()
                        if content:
                            loaded_files.append((uf.name, content, None))
                        else:
                            st.warning(f"{uf.name}: empty file.")

                    elif uf.name.endswith(".pdf"):"""

content = content.replace('                    elif uf.name.endswith(".pdf"):', docx_logic)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Added DOCX support.")
