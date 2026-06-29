replacements = {
    'fisier gol.': 'empty file.',
    'Coloana text:': 'Text column:',
    'Randul:': 'Row:'
}

file_path = "d:/licenta_practica/AI_Detector_App/app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

for ro, en in replacements.items():
    content = content.replace(ro, en)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("More translations completed.")
