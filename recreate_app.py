import json

log_path = r"C:\Users\farca\.gemini\antigravity\brain\5480664d-bf9f-4315-a20e-396617f933c9\.system_generated\logs\transcript_full.jsonl"

original_app_content = None

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Step 332 was my planner response saying "Gata! Am redesenat complet fluxul"
        # Before step 339, app.py was perfectly working.
        if "The following code has been modified to include a line number before every line" in line:
            obj = json.loads(line)
            content = obj.get("output", "")
            if "Total Lines: 1173" in content:
                # We can't easily extract the whole file from a view_file unless we viewed the whole thing.
                pass

# Let's extract the deleted lines from the bad replace tool call again.
deleted_lines = []
inserted_lines = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        if "@@ -560,431 +560,6 @@" in line:
            obj = json.loads(line)
            content = obj.get("output", "")
            if not content and "content" in obj:
                content = obj["content"]
            
            diff_start = content.find("@@ -560,431 +560,6 @@")
            if diff_start != -1:
                diff_text = content[diff_start:]
                for part in diff_text.split("\n"):
                    if part.startswith("-") and not part.startswith("---"):
                        deleted_lines.append(part[1:])
                    elif part.startswith("+") and not part.startswith("+++"):
                        inserted_lines.append(part[1:])

with open("d:/licenta_practica/AI_Detector_App/app_restored.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# We know lines[:562] is correct.
# And we know deleted_lines is the correct middle block.
# Where is the rest of the file? It starts at `for seg in r.get("high_ai_segments", []):`
idx = -1
for i, l in enumerate(lines):
    if 'for seg in r.get("high_ai_segments", []):' in l:
        idx = i
        break

if idx != -1:
    final_lines = lines[:562] + [dl + "\n" for dl in deleted_lines] + lines[idx:]
    with open("d:/licenta_practica/AI_Detector_App/app_perfect.py", "w", encoding="utf-8") as f:
        f.writelines(final_lines)
    print("app_perfect.py written. Length:", len(final_lines))
else:
    print("Could not find rest of the file")
