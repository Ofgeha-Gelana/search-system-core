# Document Search System

The **Document Search System** is a versatile web application built with **Streamlit** that allows users to upload PDF or Word documents, or enter a website URL to scrape and search for specific queries within the document. The system utilizes advanced text processing techniques, including **TF-IDF vectorization** and **lemmatization**, to retrieve relevant sentences based on cosine similarity.

## Features

- **File Upload**: Upload PDF or Word documents for text extraction.
- **Website Scraping**: Extract text from websites by providing a URL.
- **Text Preprocessing**: Clean and preprocess text (remove punctuation, lemmatize, etc.).
- **Search**: Perform query-based searches to find the most similar sentences in the document.
- **Similarity Scoring**: Results are ranked based on cosine similarity.
- **Customizable Results**: Adjust the number of search results shown using a slider.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.8 or higher
- `streamlit`
- `nltk`
- `spacy`
- `pandas`
- `numpy`
- `scikit-learn`
- `fitz` (PyMuPDF)
- `python-docx`
- `requests`
- `beautifulsoup4`
- `io`

To install all the required libraries, run:

```bash
pip install -r requirements.txt
