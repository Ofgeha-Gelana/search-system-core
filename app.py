import re
import string
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import requests

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmer = WordNetLemmatizer()
stopwords_list = stopwords.words("english")
custom_stopwords = stopwords_list + ["things", "that's", "something", "take", "don't", "may", "want", "you're",
                                     "set", "might", "says", "including", "lot", "much", "said", "know", "good",
                                     "step", "often", "going", "thing", "things", "think", "back", "actually",
                                     "better", "look", "find", "right", "example", "verb", "verbs"]

# Helper function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # Remove punctuation
    text = re.sub(r'[0-9]', '', text)  # Remove numbers
    text = re.sub(r'\s{2,}', ' ', text)  # Remove multiple spaces
    return ' '.join([lemmer.lemmatize(word) for word in text.split()])

# Extract text from uploaded files or URLs
def extract_text_from_file(file):
    if file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return " ".join([page.extract_text() for page in pdf_reader.pages])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        return None

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

# Streamlit app layout
st.title("Dynamic Document Similarity Search")
st.write("Upload files (PDF/DOCX) or provide a URL to search for similar content.")

# Upload files or input URL
uploaded_files = st.file_uploader("Upload Files (PDF/DOCX)", accept_multiple_files=True)
url_input = st.text_input("Enter a website URL")

# Process documents
docs = []
titles = []

if uploaded_files:
    for file in uploaded_files:
        text = extract_text_from_file(file)
        if text:
            docs.append(preprocess_text(text))
            titles.append(file.name)

if url_input:
    web_text = extract_text_from_url(url_input)
    if web_text:
        docs.append(preprocess_text(web_text))
        titles.append(url_input)

if docs:
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=1,            # Include terms that appear in at least 1 document
        max_df=1.0,          # Include all terms, no maximum threshold
        max_features=10000,  # Limit the number of features to 10,000
        lowercase=True,
        stop_words=custom_stopwords
    )

    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform(docs)

    # Create DataFrame for cosine similarity
    df = pd.DataFrame(tfidf_matrix.T.toarray(), index=vectorizer.get_feature_names_out())

    # Function to get similar articles based on cosine similarity
    def get_similar_articles(query, top_results=5):
        query_vec = vectorizer.transform([preprocess_text(query)]).toarray().reshape(-1)
        sim = {}

        for i in range(len(docs)):
            doc_vec = tfidf_matrix[i].toarray().reshape(-1)
            sim[i] = np.dot(doc_vec, query_vec) / (np.linalg.norm(doc_vec) * np.linalg.norm(query_vec))

        sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)[:top_results]

        results = []
        for i, v in sim_sorted:
            if v != 0.0:
                results.append((titles[i], docs[i], v))

        return results

    # User input for search query
    query = st.text_input("Search Query", "Enter your query here")

    # Adjust number of results to display
    top_n = st.slider("Number of Results", min_value=1, max_value=10, value=5)

    # Display results
    if query:
        results = get_similar_articles(query, top_results=top_n)

        st.write("### Top Similar Results")
        for title, doc, score in results:
            st.write(f"**Title:** {title}")
            st.write(f"**Score:** {score:.2f}")
            st.write(f"**Content Snippet:** {doc[:500]}...")
            st.write("-" * 100)

else:
    st.info("Upload files or provide a URL to start.")
