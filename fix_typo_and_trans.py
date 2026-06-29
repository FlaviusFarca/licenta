replacements = {
    'tokenszer': 'tokenizer',
    'Antrenat pe texte ChatGPT-3.5 si ChatGPT-4.<br>': 'Trained on ChatGPT-3.5 and ChatGPT-4 texts.<br>',
    'Performanta redusa pe alte modele AI.': 'Reduced performance on other AI models.'
}

file_path = "d:/licenta_practica/AI_Detector_App/app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

for ro, en in replacements.items():
    content = content.replace(ro, en)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed typo and remaining translations.")
