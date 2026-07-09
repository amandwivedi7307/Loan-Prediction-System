from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


def create_pdf(data, prediction, probability, filename="Loan_Report.pdf"):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("<b>Loan Prediction Report</b>", styles["Title"])
    )

    elements.append(
        Paragraph("<br/>", styles["Normal"])
    )

    table_data = [["Field", "Value"]]

    for key, value in data.items():
        table_data.append([key, str(value)])

    table_data.append(["Prediction", prediction])
    table_data.append(["Probability", f"{probability:.2f}%"])

    table = Table(table_data, colWidths=[2.5 * inch, 3.5 * inch])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
    ]))

    elements.append(table)

    doc.build(elements)

    return filename