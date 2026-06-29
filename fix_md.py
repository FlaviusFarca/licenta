with open('d:/licenta_practica/AI_Detector_App/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_md = False
new_lines = []
for line in lines:
    stripped = line.strip()
    if 'st.markdown(' in line:
        in_md = True
        new_lines.append(line)
        # Check if unsafe is on same line
        if 'unsafe_allow_html=True' in line:
            in_md = False
        continue
    
    if in_md:
        if 'unsafe_allow_html=True' in line:
            in_md = False
            new_lines.append(line)
            continue
        
        if not stripped:
            continue
            
        if stripped in ('"""', "'''", '"""+', "'''+", 'f"""', "f'''", '",', "',", '"""', '""', "f\"\"\"", '")'):
            new_lines.append(line)
            continue
            
        new_lines.append(line.lstrip())
    else:
        new_lines.append(line)

with open('d:/licenta_practica/AI_Detector_App/app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
