from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pandas as pd

def remove_cell_borders(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for border_name in ['top','left','bottom','right','insideH','insideV']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'nil')
        tcBorders.append(border)
    tcPr.append(tcBorders)

doc = Document()
df = pd.read_csv("../Desktop/r17/summary_metrics.csv")

cols = ["Station", "Timescale", "Std Ref", "Model","Std Model","RMSE","Corr","CRMSE","MAE","MAPE"]
table = doc.add_table(rows=1, cols=len(cols))
table.style = None  # حذف استایل پیشفرض

# هدر جدول
for i, col in enumerate(cols):
    table.rows[0].cells[i].text = col
    remove_cell_borders(table.rows[0].cells[i])  # حذف مرزهای هدر

for (station, spi), group in df.groupby(["station", "spi"]):
    std_ref = group["std_ref"].iloc[0]
    for _, r in group.iterrows():
        row = table.add_row().cells
        row[0].text = str(station)
        row[1].text = str(spi)
        row[2].text = str(std_ref)
        for i, col in enumerate(["model","std_model","rmse","corr","crmse","mae","mape"], start=3):
            row[i].text = str(r[col])
        for cell in row:
            remove_cell_borders(cell)  # حذف مرزهای داخلی

    # بعد از پایان هر ایستگاه یک ردیف جداکننده با یک خط افقی
    sep_row = table.add_row().cells
    sep_row[0].merge(sep_row[-1])  # سلول یکپارچه
    sep_row[0].text = ""  # فقط برای ظاهر
    # اضافه کردن خط افقی به sep_row[0]
    tc = sep_row[0]._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')  # خط افقی
    bottom.set(qn('w:sz'), '6')  # ضخامت
    bottom.set(qn('w:space'), '0')
    bottom.set(qn('w:color'), '000000')
    tcBorders.append(bottom)
    tcPr.append(tcBorders)

doc.save("Results/SPI_results.docx")
