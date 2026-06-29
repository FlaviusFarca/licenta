import sys

file_path = "d:/licenta_practica/AI_Detector_App/app.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

target_logic = """                    elif uf.name.endswith(".csv"):
                        df = pd.read_csv(uf)
                        if not df.empty:
                            cols = list(df.columns)
                            default = "text" if "text" in cols else cols[0]
                            sel_col = st.selectbox(
                                f"{uf.name} - Text column:",
                                cols,
                                index=cols.index(default),
                                key=f"csv_col_{uf.name}",
                            )
                            sel_row = st.number_input(
                                f"{uf.name} - Row:",
                                min_value=0,
                                max_value=len(df) - 1,
                                value=0,
                                step=1,
                                key=f"csv_row_{uf.name}",
                            )
                            loaded_files.append(
                                (uf.name, str(df[sel_col].iloc[sel_row]), None)
                            )"""

replacement_logic = """                    elif uf.name.endswith(".csv"):
                        df = pd.read_csv(uf)
                        if not df.empty:
                            cols = list(df.columns)
                            default = "text" if "text" in cols else cols[0]
                            sel_col = st.selectbox(
                                f"{uf.name} - Text column:",
                                cols,
                                index=cols.index(default),
                                key=f"csv_col_{uf.name}",
                            )
                            
                            csv_text_parts = []
                            for idx, val in enumerate(df[sel_col]):
                                if pd.notna(val) and str(val).strip():
                                    csv_text_parts.append(f"Row {idx+1}: {str(val).strip()}")
                            
                            csv_content = "\\n\\n".join(csv_text_parts)
                            
                            if csv_content:
                                loaded_files.append(
                                    (uf.name, csv_content, None)
                                )
                            else:
                                st.warning(f"{uf.name}: no valid text found in column '{sel_col}'.")"""

if target_logic in content:
    content = content.replace(target_logic, replacement_logic)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("CSV logic updated successfully.")
else:
    print("Could not find target logic to replace. Checking if it matches spacing...")
    # fallback if spacing is slightly different
