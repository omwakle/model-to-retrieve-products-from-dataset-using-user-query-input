# Service Retriever API

A Flask-based API for intelligent service search using fuzzy matching, semantic search, and natural language processing. This application helps users find services based on text queries, handling misspellings, related terms, and gender preferences.

## Features

- **Fuzzy Search**: Find services that approximately match user queries
- **Semantic Search**: Match services based on meaning, not just keywords
- **Spell Correction**: Automatically correct spelling errors in search queries
- **Gender Preference Detection**: Filter results based on detected gender preferences
- **Multiple Matching Strategies**: Uses various scoring algorithms for optimal results
- **RESTful API**: Simple HTTP endpoints for integration with any frontend

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/service-retriever.git
   cd service-retriever
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Prepare your data:
   - Place your service data in a CSV file named `newproduct.csv`
   - The CSV must contain at least a `Name` column
   - For example:
     ```
     Name,Category,Price
     Women's Haircut,Hair,50
     Men's Haircut,Hair,35
     Manicure,Nails,25
     ```
   - If no file is provided, an example dataset will be created automatically

## Usage

### Starting the Server

Run the application:
```
python app.py
```

The server will start on http://localhost:5000 by default.

### API Endpoints

#### 1. Search Services
```
GET /api/search?q={query}&limit={limit}
```

Parameters:
- `q` (required): Search query
- `limit` (optional): Maximum number of results to return (default: 5)
- `type` (optional): Search type - 'fuzzy', 'semantic', or 'hybrid' (default: 'hybrid')

Example Request:
```
GET /api/search?q=women%20haircut&limit=3
```

Example Response:
```json
{
  "status": "success",
  "query": "women haircut",
  "search_type": "hybrid",
  "results_count": 2,
  "results": [
    {
      "name": "Women's Haircut",
      "score": 95.3,
      "category": "Hair",
      "price": 50
    },
    {
      "name": "Hair Styling",
      "score": 62.7,
      "category": "Hair",
      "price": 40
    }
  ]
}
```

#### 2. Health Check
```
GET /api/health
```

Example Response:
```json
{
  "status": "ok",
  "service": "Service Retrieval API",
  "data_loaded": true,
  "services_count": 24,
  "semantic_search_enabled": true
}
```

## How It Works

1. **Initialization**: The application loads service data from the CSV file and prepares various search mechanisms.

2. **Query Processing**:
   - Spelling correction using PySpellChecker
   - Gender preference detection using keyword analysis
   
3. **Search Process**:
   - Semantic search using sentence embeddings (if enabled)
   - Multiple fuzzy matching strategies (token set ratio, partial ratio, etc.)
   - Results aggregation and scoring
   - Optional gender-based filtering
   - Fallback to word-by-word matching for difficult queries

4. **Result Formatting**:
   - Service details are retrieved from the original dataset
   - Results are ranked by similarity score

## Deployment

For production deployment:

1. Set `debug=False` in the `app.run()` call
2. Use Gunicorn or a similar WSGI server:
   ```
   gunicorn -w 4 app:app
   ```
3. Consider using environment variables for configuration

## Customization

- Modify the `ServiceRetriever` class parameters to tune search behavior
- Adjust the `score_cutoff` parameter to control match precision
- Change the semantic model by modifying the `model_name` parameter
- Add additional fields to your CSV for more detailed service information

## Limitations

- Semantic search requires significant memory for the embedding model
- Performance may degrade with very large service catalogs
- The default spell checker works best with English language queries

## License

[MIT License](LICENSE)
