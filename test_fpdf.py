from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("helvetica", size=12)
html = 'This is normal text. <mark style="background-color: #ffcccc;">This should be red background.</mark> <span style="background-color: #ffcccc;">Or this</span>. <font bgcolor="#ffcccc">Or this</font>.'
try:
    pdf.write_html(html)
    pdf.output("test_report.pdf")
    print("Success")
except Exception as e:
    print("Error:", e)
