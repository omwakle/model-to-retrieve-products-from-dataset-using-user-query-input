# Enhanced Spa Service Search System

## Overview

This system is designed to help users find spa services based on natural language queries. It uses advanced text processing and machine learning techniques to match user queries with available spa services, even when the query doesn't exactly match service names or categories.

Key features:
- Multi-service query handling (e.g., "massage and facial")
- Fuzzy matching for partial or misspelled queries
- Gender-specific service filtering
- AI-powered query enhancement (optional)
- High-performance search with optimized algorithms

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/spa-service-search.git
   cd spa-service-search
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download NLTK data (required for sentence tokenization):
   ```python
   import nltk
   nltk.download('punkt')
   ```

5. Set up your Google Generative AI API key (if using AI-powered query enhancement):
   - Create a `.env` file with your API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```
   - Or set the environment variable:
     ```
     export GOOGLE_API_KEY=your_api_key_here  # On Windows: set GOOGLE_API_KEY=your_api_key_here
     ```

6. Prepare your spa service data:
   - Ensure your data is in a CSV file named `newproduct.csv`
   - Required columns: `Name`, `Category`
   - Optional columns: `Description`

## Usage

### Running the application

Execute the main script to start the interactive search console:

```
python spa_search.py
```

### Query examples

- Single service query: `"hot stone massage"`
- Multiple service query: `"facial and pedicure"`
- Gender-specific query: `"men's haircut"`
- Category-based query: `"all massage services"`
- Specific service with gender: `"women's body massage"`

### Commands

- `exit`: Quit the application
- `toggle`: Switch AI query enhancement on/off

## How It Works

1. **Query Processing:**
   - Segments multi-service queries into individual requests
   - Detects gender preferences
   - Identifies query intent (specific service, category, or general)
   - Expands queries with synonyms

2. **Search Algorithm:**
   - Uses TF-IDF vectorization to convert text to numerical form
   - Calculates cosine similarity between query and services
   - Applies fuzzy matching as a fallback for low-confidence matches
   - Filters results based on gender if specified

3. **Result Presentation:**
   - Groups results by query segment for multi-service queries
   - Sorts by relevance score
   - Displays name, category, gender, and relevance

## Customization

### Modifying the stopwords list

Edit the `spa_stopwords` set in the `load_and_process_data()` function to add or remove words that should be ignored during processing.

### Adjusting similarity thresholds

Modify the threshold values in the `get_recommendations()` function to make the matching more or less strict:
- Higher values (e.g., 0.2) require closer matches
- Lower values (e.g., 0.01) allow more distantly related results

### Adding new synonyms

Extend the synonym lists in the `optimize_query()` function to improve matching for specific service types.

## Performance Optimization

The system uses several techniques to maintain high performance:
- Function caching with `@functools.lru_cache`
- Vectorized operations with pandas
- Concurrent processing for data loading
- Partial sorting for large result sets
- Pre-computation of TF-IDF matrices

## Troubleshooting

### Common issues

1. **No results found:**
   - Try more general terms
   - Check for typos in your query
   - Try using synonyms or alternative terms

2. **API key errors:**
   - Verify your Google API key is correct
   - Use the `toggle` command to disable AI if you don't have a key

3. **Performance issues:**
   - For large datasets, increase the system's memory allocation
   - Consider reducing the dataset size or filtering irrelevant entries

### Logs

The system prints error messages to the console. For more detailed logging, you can add:

```python
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- FuzzyWuzzy library for fuzzy string matching
- Google Generative AI for query enhancement
- NLTK for natural language processing
