from fpdf import FPDF

try:
    pdf = FPDF()
    pdf.add_font("Arial", "", "C:\\Windows\\Fonts\\arial.ttf")
    pdf.add_font("Arial", "B", "C:\\Windows\\Fonts\\arialbd.ttf")
    pdf.set_font("Arial", size=12)
    pdf.add_page()
    pdf.cell(0, 10, "Text cu diacritice: ăâîșț ĂÂÎȘȚ și linioară —", new_x="LMARGIN", new_y="NEXT")
    pdf.output("test_diacritice.pdf")
    print("Success")
except Exception as e:
    print(f"Error: {e}")
