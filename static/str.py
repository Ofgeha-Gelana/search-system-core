import re
import string
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmer = WordNetLemmatizer()
# stopwords_list = stopwords.words("english")

# # Extended list of stopwords
# english_stopset = set(stopwords.words('english')).union(
#     {"things", "that's", "something", "take", "don't", "may", "want", "you're", "set", "might", "says", "including", "lot", 
#      "much", "said", "know", "good", "step", "often", "going", "thing", "things", "think", "back", "actually", "better", 
#      "look", "find", "right", "example", "verb", "verbs"}
# )

# Updated stopwords to be a list instead of a set
stopwords_list = stopwords.words("english")
custom_stopwords = stopwords_list + ["things", "that's", "something", "take", "don't", "may", "want", "you're", 
                                    "set", "might", "says", "including", "lot", "much", "said", "know", "good", 
                                    "step", "often", "going", "thing", "things", "think", "back", "actually", 
                                    "better", "look", "find", "right", "example", "verb", "verbs"]

# Initialize the TF-IDF vectorizer


# Rest of the code remains the same


# Sample documents and titles
docs = ['i loved you ethiopian, stored elements in Compress find Sparse Ethiopia is the greatest country in the world of nation at universe',
        'also, sometimes, the same words can have multiple different ‘lemma’s...',
        'With more than  million people, ethiopia is the second most populous nation...',
        'The primary purpose of the dam ethiopia is electricity production...',
        'The name that the Blue Nile river loved takes in Ethiopia "abay"...',
        'Two non-upgraded loved turbine-generators with MW each are the first to go into operation...']

titles = ['Two upgraded', 'Loved Turbine-Generators', 'Operation With Loved', 'National', 'Power Grid', 'Generator']

# Preprocess documents
documents_clean = []
for d in docs:
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)  # Remove non-ASCII characters
    document_test = re.sub(r'@\w+', '', document_test)  # Remove mentions
    document_test = document_test.lower()  # Convert to lowercase
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)  # Remove punctuation
    document_test = re.sub(r'[0-9]', '', document_test)  # Remove numbers
    document_test = re.sub(r'\s{2,}', ' ', document_test)  # Remove multiple spaces
    documents_clean.append(document_test)

# Lemmatize the documents and titles
new_docs = [' '.join([lemmer.lemmatize(word) for word in doc.split()]) for doc in docs]
titles = [' '.join([lemmer.lemmatize(word) for word in title.split()]) for title in titles]

vectorizer = TfidfVectorizer(analyzer='word',
                              ngram_range=(1, 2),
                              min_df=0.002,
                              max_df=0.99,
                              max_features=10000,
                              lowercase=True,
                              stop_words=custom_stopwords)  # Use the list of stopwords instead of a set

# Transform the documents
X = vectorizer.fit_transform(new_docs)

# Create DataFrame for cosine similarity
df = pd.DataFrame(X.T.toarray())

# Define function to get similar articles based on cosine similarity
def get_similar_articles(query, top_results=5):
    query_vec = vectorizer.transform([query]).toarray().reshape(df.shape[0],)
    sim = {}
    
    for i in range(len(new_docs)):
        sim[i] = np.dot(df.loc[:, i].values, query_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(query_vec)
    
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)[:top_results]
    
    results = []
    for i, v in sim_sorted:
        if v != 0.0:
            results.append((titles[i], new_docs[i], v))
    
    return results

# Streamlit app layout
st.title('Article Similarity Search')
st.write("Enter a keyword to search for similar articles:")

# User input for query
query = st.text_input("Search Query", "ethiopia")

# Fetch similar articles based on user input
if query:
    results = get_similar_articles(query)
    
    st.write("### Top Similar Articles")
    for title, doc, score in results:
        st.write(f"**Title:** {title}")
        st.write(f"**Score:** {score:.2f}")
        st.write(f"**Article Content:** {doc}")
        st.write("-" * 100)

# Running the app
if __name__ == "__main__":
    st.write("Search similar articles by entering a query!")
