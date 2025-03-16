import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import google.generativeai as genai
import json
import functools
from fuzzywuzzy import process
import concurrent.futures

# Configure Google Generative AI API
genai.configure(api_key='AIzaSyBJ3yrDXPB2OTX26guc2Jx_9vGlpl64iyw')

# Load and preprocess the data only once at startup
@functools.lru_cache(maxsize=1)
def load_and_process_data():
    # Spa-specific stopwords - predefined to avoid NLTK download
    # REMOVED 'treatment' and 'treatments' to avoid filtering important terms
    spa_stopwords = {'spa', 'salon', 'service', 'services',
                    'the', 'and', 'a', 'an', 'of', 'for', 'to', 'in', 'on', 'with', 'by'}

    # Load CSV
    df = pd.read_csv('newproduct.csv')
    
    # Add explicit gender tags based on Category - vectorized operation
    df['gender'] = 'unisex'
    df.loc[df['Category'].str.lower().str.contains('women|ladies|female'), 'gender'] = 'female'
    df.loc[df['Category'].str.lower().str.contains('men|male'), 'gender'] = 'male'
    
    # Preprocess function optimized
    def fast_preprocess(text, remove_stopwords=True):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'\W+', ' ', text)
        
        if remove_stopwords:
            text = ' '.join([word for word in text.split() if word not in spa_stopwords])
        
        return text
    
    # Create processed columns - use vectorized operations where possible
    df['processed_name'] = df['Name'].apply(lambda x: fast_preprocess(x, False))
    df['processed_category'] = df['Category'].apply(fast_preprocess)
    
    # Create optimized combined text
    if 'Description' in df.columns:
        df['processed_description'] = df['Description'].apply(fast_preprocess)
        df['combined_text'] = (
            df['processed_name'] + ' ' + df['processed_name'] + ' ' + 
            df['processed_category'] + ' ' + df['processed_category'] + ' ' +  # Added more weight to category
            df['gender'] + ' ' + df['gender'] + ' ' +
            df.get('processed_description', '')
        )
    else:
        df['combined_text'] = (
            df['processed_name'] + ' ' + df['processed_name'] + ' ' + 
            df['processed_category'] + ' ' + df['processed_category'] + ' ' +  # Added more weight to category
            df['gender'] + ' ' + df['gender']
        )
    
    # Precompute TF-IDF matrices for different intents
    # Modified vectorizer to include unigrams and bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    
    # Specific intent (name focused)
    specific_text = df['processed_name'] + ' ' + df['processed_name'] + ' ' + df['processed_category']
    specific_matrix = vectorizer.fit_transform(specific_text)
    
    # Category intent (category focused)
    category_text = df['processed_category'] + ' ' + df['processed_category'] + ' ' + df['processed_name']
    category_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    category_matrix = category_vectorizer.fit_transform(category_text)
    
    # General intent (combined)
    general_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    general_matrix = general_vectorizer.fit_transform(df['combined_text'])
    
    return {
        'df': df,
        'vectorizers': {
            'specific': vectorizer,
            'category': category_vectorizer,
            'general': general_vectorizer
        },
        'matrices': {
            'specific': specific_matrix,
            'category': category_matrix,
            'general': general_matrix
        }
    }

# Lightweight query optimization with optional Gemini
def optimize_query(query, use_gemini=True):
    # Simple gender detection for fallback
    gender = 'unspecified'
    if any(term in query.lower() for term in ['women', 'woman', 'female', 'ladies', 'lady']):
        gender = 'female'
    elif any(term in query.lower() for term in ['men', 'man', 'male']):
        gender = 'male'
    
    # Simple intent detection for fallback
    intent = 'general'
    category_keywords = ['category', 'type', 'kind', 'service', 'treatment']
    
    # Add special handling for Mehendi
    if 'mehendi' in query.lower() or 'mehndi' in query.lower() or 'henna' in query.lower():
        intent = 'category'
        
    elif any(word in query.lower() for word in category_keywords):
        intent = 'category'
    
    # Use Gemini only when necessary to save API calls
    if use_gemini:
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            prompt = f"""
            Process this spa service query and return only a JSON object with:
            - 'corrected_query': space-separated keywords with spelling fixed
            - 'intent': 'specific', 'category', or 'general'
            - 'gender': 'male', 'female', or 'unspecified'
            - 'synonyms': list of synonym lists for key terms
            Query: '{query}'
            """
            response = model.generate_content(prompt)
            raw_response = response.text.strip()
            
            # Extract JSON content
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = raw_response[start_idx:end_idx]
                parsed_response = json.loads(json_str)
                
                # Add special handling for Mehendi-related terms in Gemini response
                if any(term in parsed_response['corrected_query'].lower() for term in ['mehendi', 'mehndi', 'henna']):
                    parsed_response['intent'] = 'category'
                    # Add synonyms if not already present
                    mehendi_synonyms = ['mehendi', 'mehndi', 'henna', 'body art']
                    has_mehendi_synonyms = False
                    for syn_list in parsed_response.get('synonyms', []):
                        if any(term in syn_list for term in mehendi_synonyms):
                            has_mehendi_synonyms = True
                            break
                    if not has_mehendi_synonyms:
                        parsed_response.setdefault('synonyms', []).append(mehendi_synonyms)
                
                return parsed_response
        except Exception as e:
            print(f"Gemini fallback: {e}")
    
    # Improve fallback with specific handling for Mehendi
    query_lower = query.lower()
    synonyms = []
    
    # Handle mehendi/mehndi variations
    if any(term in query_lower for term in ['mehendi', 'mehndi', 'henna']):
        synonyms.append(['mehendi', 'mehndi', 'henna', 'body art'])
    
    # Fallback to simple processing
    return {
        'corrected_query': query.lower(), 
        'intent': intent, 
        'gender': gender,
        'synonyms': synonyms
    }

# Fuzzy matching with performance optimization
def fuzzy_match_terms(query_terms, service_terms, threshold=80):
    # Only process longer terms to reduce computation
    query_terms = [term for term in query_terms if len(term) >= 3]
    
    # Limit the number of terms to process for performance
    if len(query_terms) > 5:
        query_terms = query_terms[:5]
    
    # Use process.extract with limit for better performance
    matches = []
    for q_term in query_terms:
        term_matches = process.extract(q_term, service_terms, limit=3)
        matches.extend([match[0] for match in term_matches if match[1] >= threshold])
    
    return matches

# Fast recommendation function
def get_recommendations(query, top_n=10, use_gemini=True):
    # Load preprocessed data (cached)
    data = load_and_process_data()
    df = data['df']
    vectorizers = data['vectorizers']
    matrices = data['matrices']
    
    # Process query with optional Gemini
    result = optimize_query(query, use_gemini)
    corrected_query = result['corrected_query']
    intent = result['intent']
    gender_preference = result.get('gender', 'unspecified')
    synonyms = result.get('synonyms', [])
    
    # Special case for Mehendi queries
    if any(term in corrected_query.lower() for term in ['mehendi', 'mehndi', 'henna']):
        intent = 'category'  # Force category intent for mehendi queries
        
        # Direct filtering for mehendi if it's a simple query
        if len(corrected_query.split()) <= 2:
            mehendi_results = df[df['Category'].str.lower().str.contains('mehendi|mehndi|henna')]
            if not mehendi_results.empty:
                # Filter by gender if specified
                if gender_preference != 'unspecified':
                    gender_mask = mehendi_results['gender'] == gender_preference
                    if gender_mask.sum() >= 1:
                        mehendi_results = mehendi_results[gender_mask]
                
                # Return top results sorted by name
                mehendi_results = mehendi_results.sort_values('Name')
                mehendi_results = mehendi_results.head(top_n)
                mehendi_results['relevance_score'] = 1.0  # Perfect match
                return mehendi_results[['Name', 'Category', 'gender', 'relevance_score']]
    
    # Pre-filter by gender if specified - use boolean indexing for speed
    if gender_preference != 'unspecified':
        mask = df['gender'] == gender_preference
        if mask.sum() < 5:  # If too few results, fall back to all
            gender_filtered_indices = list(range(len(df)))
        else:
            gender_filtered_indices = mask[mask].index.tolist()
    else:
        gender_filtered_indices = list(range(len(df)))
    
    # Select appropriate vectorizer and matrix based on intent
    vectorizer = vectorizers[intent]
    tfidf_matrix = matrices[intent]
    
    # Build expanded query
    expanded_query = corrected_query
    # Add synonyms
    if synonyms:
        for syn_list in synonyms:
            expanded_query += ' ' + ' '.join(syn_list)
    
    # Add gender terms
    if gender_preference == 'female':
        expanded_query += ' women ladies female'
    elif gender_preference == 'male':
        expanded_query += ' men male'
    
    # Transform query to vector space
    query_vec = vectorizer.transform([expanded_query])
    
    # Calculate similarities only for gender-filtered indices for better performance
    if len(gender_filtered_indices) < len(df):
        filtered_matrix = tfidf_matrix[gender_filtered_indices]
        similarities = cosine_similarity(query_vec, filtered_matrix).flatten()
    else:
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top results without sorting the entire array
    if len(similarities) <= top_n:
        top_indices = list(range(len(similarities)))
    else:
        # Partial sort is faster than full sort for large arrays
        top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Filter by similarity threshold - lowered for mehendi queries
    threshold = 0.05 if any(term in corrected_query.lower() for term in ['mehendi', 'mehndi', 'henna']) else 0.1
    top_indices_filtered = [idx for idx in top_indices if similarities[idx] > threshold]
    
    if not top_indices_filtered:
        print(f"No strong matches found for '{corrected_query}'. Try refining your query.")
        return pd.DataFrame()
    
    # Map back to original dataframe indices
    if len(gender_filtered_indices) < len(df):
        original_indices = [gender_filtered_indices[idx] for idx in top_indices_filtered]
    else:
        original_indices = top_indices_filtered
    
    # Get results
    results = df.iloc[original_indices][['Name', 'Category', 'gender']]
    results['relevance_score'] = [similarities[idx] for idx in top_indices_filtered]
    
    return results

# Fast user interface
def main():
    print("Welcome to the High-Performance Spa Service Search System!")
    print("Enter a query to search for spa services, or type 'exit' to quit.")
    print("Loading data and preparing search engine...")
    
    # Eagerly load data to reduce latency on first search
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(load_and_process_data)
        future.result()  # Wait for completion
    
    print("Search engine ready!")
    
    # Toggle for Gemini API usage - can turn off to avoid API calls
    use_gemini = True
    
    while True:
        query = input("Enter your query (or 'exit' to quit, 'toggle' to toggle AI): ").strip()
        
        if query.lower() == 'exit':
            print("Thank you for using the Spa Service Search System. Goodbye!")
            break
            
        if query.lower() == 'toggle':
            use_gemini = not use_gemini
            print(f"AI query enhancement is now {'ON' if use_gemini else 'OFF'}")
            continue
        
        if not query:
            print("Please enter a query.")
            continue
        
        print("Processing your query...")
        try:
            start_time = pd.Timestamp.now()
            recommendations = get_recommendations(query, use_gemini=use_gemini)
            end_time = pd.Timestamp.now()
            search_time = (end_time - start_time).total_seconds()
            
            if recommendations.empty:
                print("No services found. Try a different query.")
            else:
                # Format the output
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                print(f"\nTop recommendations (found in {search_time:.3f} seconds):")
                print(recommendations.to_string(index=False))
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")

if __name__ == "__main__":
    main()
