import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sentence_transformers import SentenceTransformer
import os

# Load dataset directly from the provided CSV
csv_path = 'beauty-services-data (1).csv'

# Check if the file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Ensure the file is in the same directory as the script.")

# Load the CSV
df = pd.read_csv(csv_path)

# Debug: Check loaded DataFrame
print("DataFrame after loading CSV:")
print(df.head())
print("Columns before fixing:", df.columns.tolist())

# Fix column names by stripping whitespace
df.columns = [col.strip() for col in df.columns]

# Debug: Check columns after fixing
print("Columns after fixing:", df.columns.tolist())

# Verify required columns exist
required_columns = ['service', 'vendor', 'rating']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

# Combine service and vendor for a full description
df['description'] = df['service'] + df['vendor']

# ---- NLP Preprocessing ----
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Preprocess descriptions
df['description_processed'] = df['description'].apply(preprocess_text)

# ---- TF-IDF Vectorization ----
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['description_processed'])

# ---- Sentence Embeddings ----
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight pre-trained model
description_embeddings = model.encode(df['description_processed'].tolist(), convert_to_tensor=False)

# ---- Recommendation Function ----
def recommend_services(query, top_n=2, method='tfidf'):
    """
    Recommend services based on user query.
    Args:
        query (str): User's search query (e.g., "haircut")
        top_n (int): Number of recommendations
        method (str): 'tfidf' or 'embeddings'
    Returns:
        DataFrame with top recommendations
    """
    query_processed = preprocess_text(query)

    if method == 'tfidf':
        query_vector = vectorizer.transform([query_processed])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    elif method == 'embeddings':
        query_embedding = model.encode([query_processed], convert_to_tensor=False)
        similarities = cosine_similarity(query_embedding, description_embeddings).flatten()
    else:
        raise ValueError("Method must be 'tfidf' or 'embeddings'")

    df['similarity'] = similarities
    recommendations = df.sort_values(by=['similarity', 'rating'], ascending=[False, False]).head(top_n)
    return recommendations[['service', 'vendor', 'rating', 'similarity']]

# ---- User Input Loop ----
def get_user_recommendations():
    while True:
        # Get user query
        query = input("\nEnter a service to search for (e.g., 'haircut', 'facial') or 'quit' to exit: ").strip()
        
        # Check if user wants to exit
        if query.lower() == 'quit':
            print("Exiting recommendation system.")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue

        # Get recommendations
        print("\nRecommendations using TF-IDF:")
        tfidf_recommendations = recommend_services(query, top_n=2, method='tfidf')
        print(tfidf_recommendations)

        print("\nRecommendations using Embeddings:")
        embedding_recommendations = recommend_services(query, top_n=10, method='embeddings')
        print(embedding_recommendations)

# ---- Main Execution ----
if __name__ == "__main__":
    # Run the user input loop
    get_user_recommendations()

    # Optional: Save models for deployment (uncomment if needed)
    # import pickle
    # pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
    # pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))
    # pickle.dump(model, open('embedding_model.pkl', 'wb'))
    # np.save('description_embeddings.npy', description_embeddings)

    # ---- Data Pipeline Simulation (uncomment if you want to add new data) ----
    # def update_database(new_data, vectorizer, model):
    #     new_df = pd.DataFrame(new_data)
    #     new_df['description'] = new_df['service'] + ' ' + new_df['vendor']
    #     new_df['description_processed'] = new_df['description'].apply(preprocess_text)
    #     global tfidf_matrix, df, description_embeddings
    #     df = pd.concat([df, new_df], ignore_index=True)
    #     tfidf_matrix = vectorizer.fit_transform(df['description_processed'])
    #     description_embeddings = model.encode(df['description_processed'].tolist(), convert_to_tensor=False)
    
    # new_data = {'service': ['Nail Salon'], 'vendor': ['Glow'], 'rating': [4.5]}
    # update_database(new_data, vectorizer, model)
    # print("\nUpdated Recommendations after adding new data:")
    # print(recommend_services("nail salon", top_n=1, method='embeddings'))