from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet

# Global styles variable
styles = getSampleStyleSheet()

def create_pdf(name):
    doc = SimpleDocTemplate(f"{name}.pdf", pagesize=letter)
    story = []
    return doc, story

def add_text(story, text, style):
  story.append(Paragraph(text, styles[style]))

def add_title(story, text):
  add_text(story, text, 'Title')

def add_subtitle(story, text):
  add_text(story, text, 'Heading2')

def add_paragraph(story, text):
  add_text(story, text, 'Normal')

def add_spacer(story, i, j):
  story.append(Spacer(i, j))

def add_list(story, items):
  list = ListFlowable([ListItem(Paragraph(i, styles['Normal'])) for i in items],
    bulletType='1',start='circle')
  story.append(list)

def add_image(story, file, x, y):
  story.append(Image(file, width=x, height=y))

def add_table(story, df):
  table_data = [df.columns.to_list()] + df.values.tolist()
  table = Table(table_data)
  table.setStyle(TableStyle([
      ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
      ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
      ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
      ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
      ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
      ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])
  ]))
  story.append(table)

def build_pdf(doc, story):
    doc.build(story)