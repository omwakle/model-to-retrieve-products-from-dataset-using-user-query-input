import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
import google.generativeai as genai
import json
import nltk
from fuzzywuzzy import process

# Download NLTK stopwords
nltk.download('stopwords')

# Configure Google Generative AI API
genai.configure(api_key='your_api_key')

# Spa-specific stopwords to add
spa_stopwords = {'spa', 'salon', 'service', 'services', 'treatment', 'treatments'}

# Load CSV (assuming one of the CSV files like 'spa_services_part10.csv')
df = pd.read_csv('newproduct.csv')  # Replace with your actual CSV file path

# Add explicit gender tags based on Category
def add_gender_tag(category):
    if any(gender in category.lower() for gender in ['women', 'ladies', 'female']):
        return 'female'
    elif any(gender in category.lower() for gender in ['men', 'male']):
        return 'male'
    else:
        return 'unisex'

df['gender'] = df['Category'].apply(add_gender_tag)

# Preprocessing function with improved handling
def preprocess_text(text, remove_stopwords=True):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english')).union(spa_stopwords)
        text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Create better combined text with balanced field weights
df['processed_name'] = df['Name'].apply(lambda x: preprocess_text(x, remove_stopwords=False))
df['processed_category'] = df['Category'].apply(preprocess_text)

# Add description if available
if 'Description' in df.columns:
    df['processed_description'] = df['Description'].apply(preprocess_text)
    df['combined_text'] = (
        df['processed_name'] + ' ' + df['processed_name'] + ' ' + 
        df['processed_category'] + ' ' + 
        df['gender'] + ' ' + df['gender'] + ' ' +
        df.get('processed_description', '')
    )
else:
    df['combined_text'] = (
        df['processed_name'] + ' ' + df['processed_name'] + ' ' + 
        df['processed_category'] + ' ' + 
        df['gender'] + ' ' + df['gender']
    )

# Improved Gemini query optimization
def optimize_query_with_gemini(query):
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = f"""
        Process this user query for a spa service search system:
        - Correct any spelling errors.
        - Extract key service-related keywords (e.g., service type, category).
        - Detect gender preference (men/male, women/female/ladies, or unspecified).
        - Detect the user's intent (specific service, category, or general suggestion).
        - Suggest synonyms for key terms to broaden the search.
        - Extract any special requirements or preferences.
        - Return the result as a JSON object with:
          - 'corrected_query': space-separated keywords
          - 'intent': 'specific', 'category', or 'general'
          - 'gender': 'male', 'female', or 'unspecified'
          - 'synonyms': list of synonym lists for each key term
          - 'special_requirements': any special requirements mentioned
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
        # Fallback to simple processing if Gemini fails
        gender = 'unspecified'
        if any(term in query.lower() for term in ['women', 'woman', 'female', 'ladies', 'lady']):
            gender = 'female'
        elif any(term in query.lower() for term in ['men', 'man', 'male']):
            gender = 'male'
            
        return {
            'corrected_query': preprocess_text(query, remove_stopwords=False), 
            'intent': 'general', 
            'gender': gender,
            'synonyms': [],
            'special_requirements': []
        }

# Fuzzy matching function to handle typos and variations
def fuzzy_match_terms(query_terms, service_terms, threshold=80):
    matched_terms = []
    for q_term in query_terms:
        if len(q_term) < 3:  # Skip very short terms
            continue
        matches = process.extractBests(q_term, service_terms, score_cutoff=threshold)
        matched_terms.extend([match[0] for match in matches])
    return matched_terms

# Improved recommendation function
def get_recommendations(query, top_n=10):
    # Get enhanced query information
    gemini_result = optimize_query_with_gemini(query)
    corrected_query = gemini_result['corrected_query']
    intent = gemini_result['intent']
    gender_preference = gemini_result.get('gender', 'unspecified')
    synonyms = gemini_result.get('synonyms', [])
    special_requirements = gemini_result.get('special_requirements', [])
    
    # Pre-filter by gender if specified
    if gender_preference != 'unspecified':
        df_filtered = df[df['gender'] == gender_preference]
        if len(df_filtered) < 5:  # If too few results, fall back to all services
            print(f"Few {gender_preference} services found, showing all relevant services")
            df_filtered = df
    else:
        df_filtered = df
    
    # Select which fields to prioritize based on intent
    if intent == 'specific':
        # For specific service searches, name is more important
        search_text = df_filtered['processed_name'] + ' ' + df_filtered['processed_name'] + ' ' + df_filtered['processed_category']
    elif intent == 'category':
        # For category searches, category is more important
        search_text = df_filtered['processed_category'] + ' ' + df_filtered['processed_category'] + ' ' + df_filtered['processed_name']
    else:
        # For general searches, use balanced weights
        search_text = df_filtered['combined_text']
    
    # Build expanded query with synonyms
    expanded_query = corrected_query
    if synonyms:
        for syn_list in synonyms:
            expanded_query += ' ' + ' '.join(syn_list)
    
    # Add gender terms to query if specified
    if gender_preference == 'female':
        expanded_query += ' women ladies female'
    elif gender_preference == 'male':
        expanded_query += ' men male'
    
    print(f"Expanded query: {expanded_query}")
    
    # Vectorize the content
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(search_text)
    
    # Convert expanded query to vector
    query_vec = vectorizer.transform([expanded_query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Boost similarities for fuzzy matches
    query_terms = expanded_query.split()
    for idx, service_text in enumerate(search_text):
        service_terms = service_text.split()
        matches = fuzzy_match_terms(query_terms, service_terms)
        if matches:
            # Boost the similarity score based on number of fuzzy matches
            boost = min(0.2, len(matches) * 0.05)  # Cap the boost at 0.2
            similarities[idx] += boost
    
    # Get top results
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_indices_filtered = [idx for idx in top_indices if similarities[idx] > 0.1]
    
    if not top_indices_filtered:
        print(f"No strong matches found. Did you mean something like '{corrected_query}'? Try refining your query.")
        return pd.DataFrame()
    
    # Map back to original dataframe indices
    original_indices = df_filtered.index[top_indices_filtered].tolist()
    results = df.iloc[original_indices][['Name', 'Category', 'gender']]
    
    # Add similarity scores for debugging
    results['relevance_score'] = [similarities[idx] for idx in top_indices_filtered]
    
    return results

# User input loop
def main():
    print("Welcome to the Improved Spa Service Search System!")
    print("Enter a query to search for spa services, or type 'exit' to quit.")

    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        
        if query.lower() == 'exit':
            print("Thank you for using the Spa Service Search System. Goodbye!")
            break
        
        if not query:
            print("Please enter a query.")
            continue
        
        print("Processing your query...")
        try:
            recommendations = get_recommendations(query)
            if recommendations.empty:
                print("No services found. Try a different query.")
            else:
                # Format the output nicely
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                print("\nTop recommendations for your query:")
                print(recommendations.to_string(index=False))
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")

if __name__ == "__main__":
    main()
