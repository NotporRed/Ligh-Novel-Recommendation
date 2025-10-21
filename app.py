import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df = pd.read_csv("output.csv")

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['description'].apply(clean_text)


vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(book_title, top_n=5):
    if book_title not in df['title'].values:
        return []
    idx = df[df['title'] == book_title].index[0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_books = [df['title'][i] for i, _ in sim_scores]
    return recommended_books

st.title("Book Recommendation System")

book_input = st.text_input("Enter a book title:")

if st.button("Recommend"):
    if book_input not in df['title'].values:
        st.error("Book not found in dataset.")
    else:
        recommendations = recommend(book_input, top_n=5)
        st.subheader("Top 5 recommended books:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # --- t-SNE visualization ---
        idx = df[df['title'] == book_input].index[0]
        rec_idx = df[df['title'].isin(recommendations)].index.tolist()
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(tfidf_matrix.toarray())

        plt.figure(figsize=(8,6))
        # all books in grey
        plt.scatter(tsne_results[:,0], tsne_results[:,1], color='lightgrey', alpha=0.5)
        # recommended books in red
        plt.scatter(tsne_results[rec_idx,0], tsne_results[rec_idx,1], color='red', label='Recommended', s=100)
        # target book in blue
        plt.scatter(tsne_results[idx,0], tsne_results[idx,1], color='blue', label='Target', s=120)
        plt.legend()
        plt.title("t-SNE projection of book similarities")
        st.pyplot(plt)