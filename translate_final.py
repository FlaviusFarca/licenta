replacements = {
    'Text generat de AI': 'AI generated text',
    'Scor uman:': 'Human score:',
    'Text scris de om': 'Human written text',
    'Nota: Acest scor este media a': 'Note: This score is the average of',
    'fragments de text analizate.': 'analyzed text fragments.',
    'Eroare': 'Error',
    'fisier(e) gata pentru analiza:': 'file(s) ready for analysis:',
    ' tokeni': ' tokens',
    'Detecteaza toate': 'Detect all',
    'fisiere)': 'files)',
    'Scor AI:': 'AI Score:',
    'Rezultat  :': 'Result  :',
    'Scor Human :': 'Human Score :',
    'Fragmente :': 'Fragments :',
    'Rezultat</p>': 'Result</p>',
    'Se analizeaza': 'Analyzing',
    'fragments de text...': 'text fragments...',
    'SEC?IUNI SUSPECTE DE A FI GENERATE DE AI (Peste 50%):': 'SUSPECTED AI-GENERATED SECTIONS (Over 50%):',
    'SECȚIUNI SUSPECTE DE A FI GENERATE DE AI (Peste 50%):': 'SUSPECTED AI-GENERATED SECTIONS (Over 50%):',
    'Text analizat (primele 120 caractere):': 'Analyzed text (first 120 characters):',
    'Rezultatele sunt probabilistice si nu inlocuiesc judecata unui expert.': 'Results are probabilistic and do not replace expert judgment.',
    'Descarca raport text': 'Download text report',
    'Rezultate batch (multiple fisiere)': 'Batch results (multiple files)',
    'Eroare PDF:': 'PDF Error:',
    'Fragment': 'Fragment' # Capital F
}

file_path = "d:/licenta_practica/AI_Detector_App/app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

for ro, en in replacements.items():
    content = content.replace(ro, en)

# Fix double replacements if any (like Fragment -> Fragment, it's the same, but let's avoid issues)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Remaining translations completed.")
