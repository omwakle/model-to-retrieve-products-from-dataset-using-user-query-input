import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
import google.generativeai as genai
import json
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# Configure Google Generative AI API
genai.configure(api_key='AIzaSyDLZa70GOCW-aqmzFUDgX90kkQBTl0EuE0')

# Load CSV
df = pd.read_csv('products.csv')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Initial combined text
df['combined_text'] = (df['name'] + ' ' + df['name'] + ' ' + df['name'] + ' ' +
                       df['category'] + ' ' + df['category'] + ' ' +
                       df['description'] + ' ' +
                       df['features'])
df['combined_text'] = df['combined_text'].apply(preprocess_text)

# Gemini query optimization
def optimize_query_with_gemini(query):
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = f"""
        Process this user query for a product search system:
        - Correct any spelling errors.
        - Extract key product-related keywords (e.g., product type, features).
        - Detect the user's intent (specific product, category, or general suggestion).
        - Suggest synonyms for key terms to broaden the search.
        - Return the result as a JSON object with:
          - 'corrected_query': space-separated keywords
          - 'intent': 'specific', 'category', or 'general'
          - 'synonyms': list of synonym lists for each key term
        Query: '{query}'
        """
        response = model.generate_content(prompt)
        raw_response = response.text.strip()
        print(f"Raw Gemini response: {raw_response}")

        # Extract JSON content between first '{' and last '}'
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No valid JSON found in response")
        json_str = raw_response[start_idx:end_idx]

        parsed_response = json.loads(json_str)
        print(f"Parsed Gemini response: {parsed_response}")
        return parsed_response
    except Exception as e:
        print(f"Gemini error: {e}")
        return {'corrected_query': preprocess_text(query), 'intent': 'general', 'synonyms': []}

# Recommendation function
def get_recommendations(query, top_n=10):
    gemini_result = optimize_query_with_gemini(query)
    corrected_query = gemini_result['corrected_query']
    intent = gemini_result['intent']
    synonyms = gemini_result['synonyms']

    if intent == 'specific':
        search_text = (df['name'] + ' ' + df['name'] + ' ' + df['name'] + ' ' +
                       df['features'] + ' ' + df['description'] + ' ' + df['category'])
    elif intent == 'category':
        search_text = (df['category'] + ' ' + df['category'] + ' ' + df['category'] + ' ' +
                       df['name'] + ' ' + df['description'] + ' ' + df['features'])
    else:
        search_text = df['combined_text']

    search_text = search_text.apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix_adjusted = vectorizer.fit_transform(search_text)

    expanded_query = corrected_query
    if synonyms:
        for syn_list in synonyms:
            expanded_query += ' ' + ' '.join(syn_list)

    print(f"Expanded query: {expanded_query}")
    query_vec = vectorizer.transform([preprocess_text(expanded_query)])
    similarities = cosine_similarity(query_vec, tfidf_matrix_adjusted).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    top_score = similarities[top_indices[0]]
    if top_score < 0.1:
        print(f"Did you mean something like '{corrected_query}'? Try refining your query.")
    
    results = df.iloc[top_indices][['name', 'description', 'price']]
    return results

# User input loop
print("Welcome to the Product Search System!")
print("Enter a query to search for products, or type 'exit' to quit.")

while True:
    query = input("Enter your query (or 'exit' to quit): ").strip()
    
    if query.lower() == 'exit':
        print("Thank you for using the Product Search System. Goodbye!")
        break
    
    if not query:
        print("Please enter a query.")
        continue
    
    print("Processing your query...")
    try:
        recommendations = get_recommendations(query)
        if recommendations.empty:
            print("No products found. Try a different query.")
        else:
            print(recommendations)
    except Exception as e:
        print(f"An error occurred: {e}. Please try again.")
