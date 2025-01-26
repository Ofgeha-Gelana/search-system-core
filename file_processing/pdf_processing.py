import PyPDF2


def extract_text_from_pdf(file_path):
    print(f"Extracting text from: {file_path}")
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
            print(f"Extracted text from page {reader.pages.index(page)}")
    return text
