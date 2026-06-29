import json

log_path = r"C:\Users\farca\.gemini\antigravity\brain\5480664d-bf9f-4315-a20e-396617f933c9\.system_generated\logs\transcript_full.jsonl"

deleted_lines = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        if "@@ -560,431 +560,6 @@" in line:
            data = json.loads(line)
            content = data.get("content", "")
            # If it's the tool response, it might be in output
            if "output" in data:
                content = data["output"]
            
            diff_start = content.find("@@ -560,431 +560,6 @@")
            if diff_start != -1:
                diff_text = content[diff_start:]
                for part in diff_text.split("\n"):
                    if part.startswith("-") and not part.startswith("---"):
                        deleted_lines.append(part[1:])

with open("d:/licenta_practica/AI_Detector_App/recovered_block.py", "w", encoding="utf-8") as f:
    f.write("\n".join(deleted_lines))
print("Recovered", len(deleted_lines), "lines")
