import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nltk
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("output.csv")
df_original = df.copy()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    return ' '.join(words)

for col in df.columns:
    df[col] = df[col].apply(clean_text)

df['combined_text'] = df.apply(lambda row: ' '.join(row), axis=1)

vectorizer = TfidfVectorizer(max_features=5000)  # you can adjust max_features
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

feature_names = vectorizer.get_feature_names_out()

print(f"Matrix shape: {tfidf_matrix.shape}")

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(book_title, df=df, top_n=5):
    book_title = clean_text(book_title)
    if book_title not in df['title'].values:
        return f"Book '{book_title}' not found in dataset."
    
    idx = df[df['title'] == book_title].index[0]
    
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommended_books = [df_original['title'][i] for i, score in sim_scores]
    #recommended_books = [book.title() for book in recommended_books]


    return recommended_books

st.title("Book Recommendation System")

st.write("""
Enter a book title and get similar book recommendations!
""")

book_input = st.text_input("Enter Book Title:")

if st.button("Recommend"):
    if book_input:
        results = recommend(book_input)
        st.subheader("Recommended Books:")
        
        for rec in results:
            book_info = df_original[df_original['title'] == rec].iloc[0]
            with st.expander(rec):
                st.write(f"**Author:** {book_info.get('authors', 'Unknown')}")
                st.write(f"**Description:** {book_info.get('description', 'No description available.')}")
                
    else:
        st.warning("Please enter a book title to get recommendations.") 

if st.button("Show 2D Map"):
    if book_input:
        target_book = clean_text(book_input)
        target_idx = df[df['title'] == target_book].index[0]

        top_similar = recommend(book_input)
        similar_idx = df[df_original['title'].isin(top_similar)].index.tolist()

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(tfidf_matrix.toarray())

        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], color='lightgrey', alpha=0.5)
        plt.scatter(tsne_results[similar_idx, 0], tsne_results[similar_idx, 1], color='red', label='Similar Books', s=25)
        plt.scatter(tsne_results[target_idx, 0], tsne_results[target_idx, 1], color='blue', label=target_book, s=50)

        plt.legend()
        plt.title(f"t-SNE 2D Projection: '{target_book}' and Similar Books")
        st.pyplot(plt)
                
    else:
        st.warning("Please enter a book title to get recommendations.")