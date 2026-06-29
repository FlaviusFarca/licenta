import sys

file_path = "d:/licenta_practica/AI_Detector_App/app.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the translation
content = content.replace("Rezultate batch (multiple files)", "Batch results (multiple files)")

# Insert the ZIP download logic
target_text = """    for br in st.session_state.batch_results:
        label_c = "hl-red" if br["label"] == "AI" else "hl-green\""""

replacement_text = """    import zipfile
    import io
    
    zip_buffer = io.BytesIO()
    has_pdfs = False
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for br in st.session_state.batch_results:
            if br.get("pdf_report_bytes"):
                has_pdfs = True
                zip_file.writestr(f"TextScan_{br['source_filename']}.pdf", br["pdf_report_bytes"])
                
    if has_pdfs:
        st.download_button(
            label="Download ALL PDF reports (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="TextScan_All_Reports.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    for br in st.session_state.batch_results:
        label_c = "hl-red" if br["label"] == "AI" else "hl-green\""""

if target_text in content:
    content = content.replace(target_text, replacement_text)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully added ZIP download button.")
else:
    print("Could not find the target text to replace.")
