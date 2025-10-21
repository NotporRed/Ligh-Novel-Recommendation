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

def clean_text(text):
    if not isinstance(text, str):
        return ''
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)

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
    
    recommended_books = [df['title'][i] for i, score in sim_scores]
    
    return recommended_books

print(recommend("Sword Art Online Alternative – Gun Gale Online"))
