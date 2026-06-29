import fitz
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), "Test PDF")
try:
    b = doc.tobytes()
    print("Success. length:", len(b))
except Exception as e:
    print("Error:", str(e))
