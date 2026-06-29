import sys

with open('d:/licenta_practica/AI_Detector_App/app_restored.py', encoding='utf-8') as f:
    lines = f.read().split('\n')

in_markdown = False
for i in range(len(lines)):
    line = lines[i]
    stripped = line.strip()
    
    # We also want to dedent specific lines
    s = line.lstrip()
    if s.startswith('<div') or s.startswith('</div') or s.startswith('<p') or s.startswith('</p') or s.startswith('<span') or s.startswith('</span') or s.startswith('<h1') or s.startswith('</h1') or s.startswith('Modele RoBERTa') or s.startswith('Total:') or s.startswith('Antrenat pe') or s.startswith('Robustete superioara'):
        lines[i] = s

# To avoid double empty lines, we can remove empty lines that are inside st.markdown blocks.
# But it's easier to just remove all consecutive empty lines!
new_lines = []
for line in lines:
    if not line.strip() and new_lines and not new_lines[-1].strip():
        continue
    new_lines.append(line)

with open('d:/licenta_practica/AI_Detector_App/app.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))
