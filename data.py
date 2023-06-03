import os
import arxiv
from PyPDF2 import PdfReader

pdf_dir = "papers_pdf"
if not os.path.exists(pdf_dir):
    os.mkdir(pdf_dir)

search = arxiv.Search(
  query="climate changes",
  max_results=10,
  sort_by=arxiv.SortCriterion.Relevance
)


for result in search.results():
    paper = result
    paper_filename = os.path.join(pdf_dir, paper.title+".pdf")
    paper.download_pdf(filename=paper_filename)

# # creating a pdf reader object
# paper_filename = "papers_pdf/Trend and Thoughts: Understanding Climate Change Concern using Machine Learning and Social Media Data.pdf"
reader = PdfReader(paper_filename)
#
# getting a specific page from the pdf file
for page in reader.pages:
    # extracting text from page
    text = page.extract_text()
    print(text)
