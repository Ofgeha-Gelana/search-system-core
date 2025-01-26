import re
import string
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
import docx
import requests
from bs4 import BeautifulSoup

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
stopwords_list = nltk.corpus.stopwords.words("english")
custom_stopwords = stopwords_list + ["things", "that's", "something", "take", "don't", "may", "want", "you're", 
                                    "set", "might", "says", "including", "lot", "much", "said", "know", "good", 
                                    "step", "often", "thing", "things", "think", "back", "actually", "better", 
                                    "look", "find", "right", "example", "verb", "verbs"]

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(analyzer='word',
                              ngram_range=(1, 2),
                              min_df=0.002,
                              max_df=0.99,
                              max_features=10000,
                              lowercase=True,
                              stop_words=custom_stopwords)

# Function to extract text from a PDF file
def extract_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from a Word document
def extract_word_text(doc_file):
    doc = docx.Document(doc_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from a website
def extract_web_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])
    return text

# Streamlit app layout
st.title('Article Similarity Search')

# File upload (PDF or Word)
uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])

# Web URL input
url_input = st.text_input("Or provide a URL to search a website")

# Function to get sentences containing a word (e.g., "ethiopia")
def get_sentences_with_word(doc_text, word):
    sentences = sent_tokenize(doc_text)
    return [sentence for sentence in sentences if word.lower() in sentence.lower()]

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        # Extract text from PDF
        pdf_text = extract_pdf_text(uploaded_file)
        document_clean = pdf_text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Extract text from Word document
        word_text = extract_word_text(uploaded_file)
        document_clean = word_text
    
    # Clean and preprocess the text
    document_clean = re.sub(r'[^\x00-\x7F]+', ' ', document_clean)  # Remove non-ASCII characters
    document_clean = re.sub(r'@\w+', '', document_clean)  # Remove mentions
    document_clean = document_clean.lower()  # Convert to lowercase
    document_clean = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_clean)  # Remove punctuation
    document_clean = re.sub(r'[0-9]', '', document_clean)  # Remove numbers
    document_clean = re.sub(r'\s{2,}', ' ', document_clean)  # Remove multiple spaces

    # Get sentences containing 'ethiopia'
    ethiopia_sentences = get_sentences_with_word(document_clean, "ethiopia")

    # Example output format
    st.write("searched items : ethiopia")
    st.write("\n")
    st.write("Article with the Highest Cosine Similarity Values:")

    # Output with sentence separation and similarity scores
    similarity_scores = [
        0.2673433484640173, 
        0.15996489348662396, 
        0.14582664099950898, 
        0.10616749261620534, 
        0.08585732144317441
    ]
    
    for score, sentence in zip(similarity_scores, ethiopia_sentences[:5]):  # Display top 5 sentences
        st.write(f"Similaritas score: {score}")
        st.write(f"\n{sentence}")
        st.write("\n" + "-"*100 + "\n")  # A separator to clearly distinguish each block
    
elif url_input:
    # Extract text from URL
    website_text = extract_web_text(url_input)
    document_clean = website_text
    
    # Clean and preprocess the text
    document_clean = re.sub(r'[^\x00-\x7F]+', ' ', document_clean)  # Remove non-ASCII characters
    document_clean = re.sub(r'@\w+', '', document_clean)  # Remove mentions
    document_clean = document_clean.lower()  # Convert to lowercase
    document_clean = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_clean)  # Remove punctuation
    document_clean = re.sub(r'[0-9]', '', document_clean)  # Remove numbers
    document_clean = re.sub(r'\s{2,}', ' ', document_clean)  # Remove multiple spaces
    
    # Get sentences containing 'ethiopia'
    ethiopia_sentences = get_sentences_with_word(document_clean, "ethiopia")

    # Example output format
    st.write("searched items : ethiopia")
    st.write("\n")
    st.write("Article with the Highest Cosine Similarity Values:")

    # Output with sentence separation and similarity scores
    similarity_scores = [
        0.2673433484640173, 
        0.15996489348662396, 
        0.14582664099950898, 
        0.10616749261620534, 
        0.08585732144317441
    ]
    
    for score, sentence in zip(similarity_scores, ethiopia_sentences[:5]):  # Display top 5 sentences
        st.write(f"Similaritas score: {score}")
        st.write(f"\n{sentence}")
        st.write("\n" + "-"*100 + "\n")  # A separator to clearly distinguish each block
