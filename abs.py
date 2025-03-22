from flask import Flask, request, jsonify
import pandas as pd
from rapidfuzz import process, fuzz
from spellchecker import SpellChecker
import os
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ServiceRetriever:
    def __init__(self, csv_path="newproduct.csv", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the ServiceRetriever with data from the CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing service data
            model_name (str): Name of the sentence transformer model to use
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
                
            # Load the CSV file into a pandas DataFrame
            self.df = pd.read_csv(csv_path)
            
            logger.info(f"DataFrame loaded with shape: {self.df.shape}")
            
            # Verify 'Name' column exists
            if 'Name' not in self.df.columns:
                raise ValueError(f"CSV file must contain a 'Name' column. Available columns: {self.df.columns.tolist()}")
            
            # Create a spell checker instance
            self.spell_checker = SpellChecker()
            
            # Create a list of all service names for fuzzy matching
            self.service_names = self.df['Name'].tolist()
            
            # Remove any NaN or empty values
            self.service_names = [str(name) for name in self.service_names if name and pd.notna(name)]
            
            # Pre-compute gender indicators for services
            self.gender_indicators = {
                'female': ['women', 'woman', 'female', 'ladies', 'lady'],
                'male': ['men', 'man', 'male', 'gentlemen', 'gentleman']
            }
            
            # Generate gender mappings for services
            self._generate_gender_mappings()
            
            logger.info(f"Loaded {len(self.service_names)} services from {csv_path}")
            
            # Create a backup list with all words from service names for extreme cases
            self.all_words = []
            for name in self.service_names:
                self.all_words.extend(name.lower().split())
            self.all_words = list(set(self.all_words))  # Remove duplicates
            logger.info(f"Extracted {len(self.all_words)} unique words from all service names")
            
            # Initialize the sentence transformer model for semantic search
            try:
                logger.info(f"Loading sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("Sentence transformer model loaded successfully")
                
                # Generate embeddings for all services
                self._generate_embeddings()
                logger.info(f"Generated embeddings for {len(self.service_names)} services")
                
                self.semantic_search_enabled = True
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer model: {e}")
                logger.warning("Semantic search will be disabled")
                self.semantic_search_enabled = False
                
        except Exception as e:
            logger.error(f"Error initializing ServiceRetriever: {e}")
            raise
    
    def _generate_gender_mappings(self):
        """Generate gender mappings for services based on their names."""
        self.service_gender = {}
        
        for service in self.service_names:
            service_lower = service.lower()
            
            # Check for female indicators
            if any(indicator in service_lower for indicator in self.gender_indicators['female']):
                self.service_gender[service] = 'female'
            # Check for male indicators
            elif any(indicator in service_lower for indicator in self.gender_indicators['male']):
                self.service_gender[service] = 'male'
            else:
                self.service_gender[service] = 'neutral'
    
    def _generate_embeddings(self):
        """Generate embeddings for all services using the sentence transformer."""
        if not hasattr(self, 'model') or self.model is None:
            logger.warning("Sentence transformer model not available, skipping embedding generation")
            return
            
        self.service_embeddings = self.model.encode(self.service_names, show_progress_bar=False)
        logger.info(f"Generated embeddings with shape: {self.service_embeddings.shape}")
    
    def _correct_spelling(self, query):
        """
        Correct spelling errors in the query.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: The corrected query
        """
        # Split the query into words
        words = query.split()
        
        # Correct each word
        corrected_words = []
        for word in words:
            # Don't correct short words or words with numbers
            if len(word) <= 2 or any(char.isdigit() for char in word):
                corrected_words.append(word)
            else:
                # Check if the word is misspelled
                if word.lower() in self.spell_checker or word.lower() in self.all_words:
                    corrected_words.append(word)
                else:
                    # Get the correction
                    corrected_word = self.spell_checker.correction(word)
                    corrected_words.append(corrected_word if corrected_word else word)
        
        # Join the corrected words back into a query
        corrected_query = ' '.join(corrected_words)
        
        # Log the correction if it's different from the original
        if corrected_query != query:
            logger.info(f"Corrected query: '{query}' -> '{corrected_query}'")
        
        return corrected_query
    
    def _detect_gender_preference(self, query):
        """
        Detect if the query has a gender preference.
        
        Args:
            query (str): The user's query
            
        Returns:
            str or None: 'male', 'female', or None if no preference
        """
        query_lower = query.lower()
        
        # Check for female indicators
        if any(indicator in query_lower for indicator in self.gender_indicators['female']):
            logger.info(f"Detected gender preference: female")
            return 'female'
        
        # Check for male indicators
        if any(indicator in query_lower for indicator in self.gender_indicators['male']):
            logger.info(f"Detected gender preference: male")
            return 'male'
        
        # No gender preference
        return None
    
    def _semantic_search(self, query, top_n=5):
        """
        Perform semantic search using sentence embeddings.
        
        Args:
            query (str): The user's query
            top_n (int): Number of results to return
            
        Returns:
            list: List of tuples (service_name, similarity_score)
        """
        if not self.semantic_search_enabled:
            logger.warning("Semantic search is disabled")
            return []
            
        # Generate embedding for the query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity between query and all services
        similarities = cosine_similarity([query_embedding], self.service_embeddings)[0]
        
        # Get top N results
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Convert similarity scores to a 0-100 scale
        results = [(self.service_names[i], float(similarities[i] * 100)) for i in top_indices]
        
        logger.info(f"Semantic search found {len(results)} matches")
        
        return results
    
    def search(self, query, top_n=5, score_cutoff=50):
        """
        Search for services similar to the query using both fuzzy and semantic search.
        
        Args:
            query (str): The user's query
            top_n (int): Number of results to return
            score_cutoff (int): Minimum similarity score (0-100)
            
        Returns:
            list: List of tuples (service_name, similarity_score)
        """
        # Check if service names exist
        if not self.service_names:
            logger.error("Error: No service names available to search")
            return []
            
        # Correct spelling in the query
        corrected_query = self._correct_spelling(query)
        
        # Detect gender preference
        gender_preference = self._detect_gender_preference(corrected_query)
        
        logger.info(f"Searching for: '{corrected_query}'")
        
        all_matches = []
        
        # Try semantic search first if enabled
        if self.semantic_search_enabled:
            semantic_results = self._semantic_search(corrected_query, top_n=top_n*2)
            for service, score in semantic_results:
                if score >= score_cutoff:
                    all_matches.append((service, score, "semantic"))
            logger.info(f"Semantic search found {len(semantic_results)} matches above threshold")
        
        # Try different fuzzy matching strategies
        strategies = [
            ("token_set_ratio", fuzz.token_set_ratio),
            ("partial_ratio", fuzz.partial_ratio),
            ("token_sort_ratio", fuzz.token_sort_ratio),
            ("ratio", fuzz.ratio)
        ]
        
        # Try each strategy
        for strategy_name, scorer in strategies:
            # Perform fuzzy search
            matches = process.extract(
                corrected_query,
                self.service_names,
                scorer=scorer,
                score_cutoff=score_cutoff,
                limit=top_n * 3
            )
            
            logger.info(f"Strategy '{strategy_name}' found {len(matches)} matches")
            
            # Add to all matches
            for service, score, _ in matches:
                # Check if this service is already in all_matches
                existing = next((item for item in all_matches if item[0] == service), None)
                if existing:
                    # Update score if higher
                    if score > existing[1]:
                        all_matches.remove(existing)
                        all_matches.append((service, score, strategy_name))
                else:
                    all_matches.append((service, score, strategy_name))
        
        # Sort by score (descending)
        all_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Filter results based on gender preference if applicable
        if gender_preference and all_matches:
            filtered_matches = []
            for service, score, strategy in all_matches:
                service_gender = self.service_gender[service]
                
                # Include the service if it matches the gender preference or is neutral
                if service_gender == gender_preference or service_gender == 'neutral':
                    filtered_matches.append((service, score, strategy))
            
            # If we have enough matches after filtering, use them
            if filtered_matches:
                logger.info(f"After gender filtering: {len(filtered_matches)} matches")
                all_matches = filtered_matches
        
        # Return the top N results
        final_results = [(service, score) for service, score, _ in all_matches[:top_n]]
        
        # If no results, try backup search on individual words
        if not final_results and len(corrected_query.split()) > 1:
            logger.info("No direct matches found. Trying individual word matching...")
            words = corrected_query.split()
            
            for word in words:
                if len(word) > 3:  # Only try with meaningful words
                    word_matches = self.search(word, top_n=2, score_cutoff=60)
                    for service, score in word_matches:
                        if (service, score) not in final_results:
                            final_results.append((service, score))
            
            # Sort and limit again
            final_results.sort(key=lambda x: x[1], reverse=True)
            final_results = final_results[:top_n]
        
        return final_results
    
    def get_service_details(self, service_name):
        """
        Get the details of a service by name.
        
        Args:
            service_name (str): The name of the service
            
        Returns:
            dict: Service details
        """
        # Find the service in the DataFrame
        service = self.df[self.df['Name'] == service_name]
        
        # If the service is found, return its details
        if not service.empty:
            return service.iloc[0].to_dict()
        
        return None

# Initialize the retriever when the application starts
retriever = None

@app.before_request
def initialize_retriever():
    global retriever
    if 'retriever' not in globals():
        try:
            retriever = ServiceRetriever()
            logger.info("ServiceRetriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ServiceRetriever: {e}")

@app.route('/api/search', methods=['GET'])
def search_services():
    """API endpoint for searching services"""
    try:
        # Get query parameters
        query = request.args.get('q', '')
        limit = request.args.get('limit', 5)
        search_type = request.args.get('type', 'hybrid')  # 'fuzzy', 'semantic', or 'hybrid'
        
        # Convert limit to integer
        try:
            limit = int(limit)
        except ValueError:
            limit = 5
        
        # Check if query is provided
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query parameter "q" is required',
                'results': []
            }), 400
        
        # Check if retriever is initialized
        global retriever
        if retriever is None:
            try:
                retriever = ServiceRetriever()
            except Exception as e:
                logger.error(f"Failed to initialize ServiceRetriever: {e}")
                return jsonify({
                    'status': 'error',
                    'message': 'Service retriever is not available',
                    'results': []
                }), 500
        
        # Search for services
        results = retriever.search(query, top_n=limit)
        
        # Format results
        formatted_results = []
        for service, score in results:
            service_details = retriever.get_service_details(service)
            result = {
                'name': service,
                'score': round(score, 1)
            }
            
            # Add service details if available
            if service_details:
                for key, value in service_details.items():
                    if key != 'Name':  # Skip duplicate name field
                        result[key.lower()] = value
            
            formatted_results.append(result)
        
        # Return results
        return jsonify({
            'status': 'success',
            'query': query,
            'search_type': search_type,
            'results_count': len(formatted_results),
            'results': formatted_results
        })
        
    except Exception as e:
        logger.error(f"Error processing search request: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'results': []
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global retriever
    health_status = {
        'status': 'ok' if retriever is not None else 'error',
        'service': 'Service Retrieval API',
        'data_loaded': retriever is not None,
    }
    
    if retriever is not None:
        health_status['services_count'] = len(retriever.service_names)
        health_status['semantic_search_enabled'] = retriever.semantic_search_enabled
    
    return jsonify(health_status)

if __name__ == '__main__':
    # Create example data if newproduct.csv doesn't exist
    if not os.path.exists('newproduct.csv'):
        logger.warning("newproduct.csv not found. Creating example dataset.")
        example_services = [
            "Custom Tattoo", "Traditional Tattoo", "3D Tattoo", "Black & Grey Tattoo",
            "Women's Haircut", "Men's Haircut", "Hair Coloring", "Hair Styling",
            "Manicure", "Pedicure", "Nail Art", "Gel Nails",
            "Men's Spa Treatment", "Women's Spa Day", "Facial Treatment", "Massage Therapy",
            "Mehendi Design", "Bridal Mehendi", "Henna Tattoo", "Body Art",
            "Hair Salon", "Beauty Salon", "Barbershop", "Spa Salon"
        ]
        
        categories = ["Tattoo", "Hair", "Nails", "Spa", "Art", "Salon"]
        
        example_df = pd.DataFrame({
            'Name': example_services,
            'Category': [categories[i % len(categories)] for i in range(len(example_services))]
        })
        
        example_df.to_csv('newproduct.csv', index=False)
        logger.info(f"Created example dataset with {len(example_services)} services at newproduct.csv")
    
    # Initialize retriever here to make it available immediately
    try:
        retriever = ServiceRetriever()
        logger.info("ServiceRetriever initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ServiceRetriever: {e}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
