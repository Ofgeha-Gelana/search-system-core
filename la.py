# import re
# import string
# import numpy as np
# import pandas as pd
# import nltk
# import spacy
# import streamlit as st
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # NLTK setup
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")

# # Sample documents
# docs = ['i loved you ethiopian, stored elements in Compress find Sparse Ethiopia is the greatest country in the world of nation at universe',
#         'also, sometimes, the same words can have multiple different ‘lemma’s. So, based on the context it’s used, you should identify the \
#         part-of-speech (POS) tag for the word in that specific context and extract the appropriate lemma. Examples of implementing this comes \
#         in the following sections countries.ethiopia With a planned.The name that the Blue Nile river loved took in Ethiopia is derived from the \
#         Geez word for great to imply its being the river of rivers The word Abay still exists in ethiopia major languages',
#         'With more than million people, ethiopia is the second most populous nation in Africa after Nigeria, and the fastest growing \
#          economy in the region. However, it is also one of the poorest, with a per capita income',
#         'The primary purpose of the dam ethiopia is electricity production to relieve Ethiopia’s acute energy shortage and for electricity export to neighboring\
#          countries.ethiopia With a planned.',
#         'The name that the Blue Nile river loved takes in Ethiopia "abay" is derived from the Geez blue loved word for great to imply its being the river of rivers The \
#          word Abay still exists in Ethiopia major languages to refer to anything or anyone considered to be superior.',
#         'Two non-upgraded loved turbine-generators with MW each are the first loveto go into operation with loved MW delivered to the national power grid. This early power\
#          generation will start well before the completion']

# title = ['Two upgraded', 'Loved Turbine-Generators', 'Operation With Loved', 'National', 'Power Grid', 'Generator']

# # Lemmatization and Preprocessing
# lemmer = WordNetLemmatizer()
# documents_clean = []

# for d in docs:
#     document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)  # Remove non-ASCII characters
#     document_test = re.sub(r'@\w+', '', document_test)  # Remove Mentions
#     document_test = document_test.lower()  # Convert to lowercase
#     document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)  # Clean punctuation
#     document_test = re.sub(r'[0-9]', '', document_test)  # Remove numbers
#     document_test = re.sub(r'\s{2,}', ' ', document_test)  # Remove extra spaces
#     documents_clean.append(document_test)

# new_docs = [' '.join([lemmer.lemmatize(word) for word in text.split()]) for text in documents_clean]

# # Vectorizing the documents
# vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.002, max_df=0.99, max_features=10000, lowercase=True, stop_words='english')
# X = vectorizer.fit_transform(new_docs)
# df = pd.DataFrame(X.T.toarray())

# # Function to search for similar articles
# def get_similar_articles(query, df):
#     query = [query]
#     query_vec = vectorizer.transform(query).toarray().reshape(df.shape[0],)
    
#     sim = {}
#     for i in range(len(new_docs)):
#         sim[i] = np.dot(df.loc[:, i].values, query_vec) / (np.linalg.norm(df.loc[:, i]) * np.linalg.norm(query_vec))
    
#     sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)[:5]
    
#     result = []
#     for i, v in sim_sorted:
#         if v != 0.0:
#             result.append((title[i], new_docs[i], v))  # Use 'title' instead of 'titles'
#     return result

# # Streamlit UI
# st.title("Document Search System")
# query = st.text_input("Enter search query:")
# if query:
#     results = get_similar_articles(query, df)
#     if results:
#         st.subheader("Top Results:")
#         for title, doc, score in results:
#             st.write(f"**Title:** {title}")
#             st.write(f"**Similarity Score:** {score}")
#             st.write(f"**Document:** {doc}")
#             st.write("-" * 100)
#     else:
#         st.write("No results found.")



import re
import string
import numpy as np
import pandas as pd
import nltk
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import fitz  # PyMuPDF for PDF text extraction
from docx import Document  # python-docx for Word documents
import requests
from bs4 import BeautifulSoup
import requests
# import io
import io  # To handle file-like object for PDF


# NLTK setup
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Lemmatization and Preprocessing
lemmer = WordNetLemmatizer()

# Function to extract text from PDF (handling file-like object)
def extract_pdf_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")  # Directly use the file buffer with PyMuPDF
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from Word document
def extract_word_text(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from URL (Website)
def extract_website_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    text = ""
    for para in paragraphs:
        text += para.get_text() + "\n"
    return text

# Preprocessing function to clean and lemmatize text
def preprocess_text(docs):
    cleaned_docs = []
    for doc in docs:
        doc = re.sub(r'[^\x00-\x7F]+', ' ', doc)  # Remove non-ASCII characters
        doc = re.sub(r'@\w+', '', doc)  # Remove Mentions
        doc = doc.lower()  # Convert to lowercase
        doc = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', doc)  # Clean punctuation
        doc = re.sub(r'[0-9]', '', doc)  # Remove numbers
        doc = re.sub(r'\s{2,}', ' ', doc)  # Remove extra spaces
        cleaned_docs.append(' '.join([lemmer.lemmatize(word) for word in doc.split()]))
    return cleaned_docs

# Split text into sentences
def split_into_sentences(text):
    return nltk.sent_tokenize(text)

# Streamlit UI
st.title("Document Search System")

# File Upload for PDF, Word, or Text
uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])

# URL Input for Website
url_input = st.text_input("Enter website URL (for scraping)")

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        document_text = extract_pdf_text(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        document_text = extract_word_text(uploaded_file)
    else:
        st.warning("Unsupported file format")
else:
    document_text = ""
    
if url_input:
    document_text = extract_website_text(url_input)

if document_text:
    # Split text into sentences and preprocess
    sentences = split_into_sentences(document_text)
    new_docs = preprocess_text(sentences)

    # Vectorizing the document with adjusted min_df and max_df
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        ngram_range=(1, 2),  # Using unigrams and bigrams
        min_df=0.0,  # Lowered min_df to allow more terms
        max_df=1.0,  # Raised max_df to avoid excluding too many common words
        max_features=10000, 
        lowercase=True, 
        stop_words='english'
    )
    X = vectorizer.fit_transform(new_docs)
    
    # Ensure that terms are present after vectorization
    if X.shape[1] == 0:
        st.write("No terms were extracted from the document. Try using a different document or adjusting the min_df/max_df parameters.")
    else:
        df = pd.DataFrame(X.T.toarray())
        
        # Function to search for similar sentences
        def get_similar_sentences(query, df):
            query = [query]
            query_vec = vectorizer.transform(query).toarray().reshape(df.shape[0],)
            
            sim = {}
            for i in range(len(new_docs)):
                sim[i] = np.dot(df.loc[:, i].values, query_vec) / (np.linalg.norm(df.loc[:, i]) * np.linalg.norm(query_vec))
            
            sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)[:5]
            
            result = []
            for i, v in sim_sorted:
                if v != 0.0:
                    result.append((f"Sentence {i + 1}", new_docs[i], v))  # For simplicity, just use sentence index
            return result

        # Display the input search query
        query = st.text_input("Enter search query:")
        if query:
            results = get_similar_sentences(query, df)
            if results:
                st.subheader("Top Results:")
                for title, doc, score in results:
                    st.write(f"**Title:** {title}")
                    st.write(f"**Similarity Score:** {score}")
                    st.write(f"**Sentence:** {doc}")
                    st.write("-" * 100)
            else:
                st.write("No results found.")
else:
    st.write("Upload a document or enter a website URL to begin.")
