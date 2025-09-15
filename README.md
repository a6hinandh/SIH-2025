# Groundwater Data Vector Search System

This project fetches groundwater data from the Indian Institute of Technology Hyderabad (IITH) API, processes it into vector embeddings using Hugging Face's `all-mpnet-base-v2` model, and stores them in Pinecone for efficient semantic search. The system includes an intelligent query interface powered by Google's Gemini AI for natural language responses about groundwater data across Indian states, districts, and blocks.

## Features

- **Data Fetching**: Retrieves groundwater data from IITH API for states, districts, and blocks
- **Vector Embeddings**: Converts geographical and hydrological data into searchable vector embeddings  
- **Pinecone Integration**: Stores and queries embeddings in Pinecone vector database
- **AI-Powered Responses**: Uses Google Gemini AI for intelligent, context-aware answers
- **Multi-level Geography**: Supports country, state, district, and block level data
- **Data Validation**: Includes debugging and validation tools for data integrity

## Project Structure

```
├── fetch_states.py         # Fetches data from IITH API (states, districts, blocks)
├── embeddings.py          # Creates vector embeddings from JSON data
├── pinecone_setup.py      # Initializes Pinecone index
├── insert_data.py         # Inserts embeddings into Pinecone
├── query_index.py         # Basic query interface for searching data
├── generate_response.py           # AI-powered query interface with Gemini integration
├── check_uuids.py         # Validates UUID integrity in data
├── debug_data.py          # Debugging tool for data structure analysis
├── delete_index.py        # Utility to delete Pinecone indexes
├── output/                # Directory for country-level data
│   ├── india.json        # Raw JSON data for all Indian states
│   └── india.csv         # Processed CSV data for states
├── states/               # Directory for state-level district data
└── KERALA/              # Directory for Kerala district block data
```

## Data Schema

The system processes groundwater data with the following key metrics:
- **Area**: Total geographical area
- **Rainfall**: Annual rainfall data
- **Recharge**: Groundwater recharge rates
- **Draft**: Groundwater extraction/draft data
- **Stage of Extraction**: Current extraction levels
- **Category**: Groundwater availability category

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `pinecone-client`
- `sentence-transformers`
- `google-generativeai`
- Additional dependencies as listed in `requirements.txt`

### 2. Configure API Keys

#### Pinecone API Key
Update your Pinecone API key in the relevant files:
- `insert_data.py`
- `delete_index.py` 
- `query_index.py`
- `generate_response.py`

Replace `"YOUR_PINECONE_API_KEY"` with your actual API key.

#### Google Gemini API Key
Update your Google Gemini API key in `generate_response.py`:
```python
genai.configure(api_key="YOUR_GOOGLE_GEMINI_API_KEY")
```

### 3. Fetch Data (Optional)
If you want to fetch fresh data from the API:
```bash
python fetch_states.py
```
This will create:
- `output/india.json` and `output/india.csv` - State-level data
- `states/[STATE_NAME].json` and `states/[STATE_NAME].csv` - District data for each state
- `KERALA/[DISTRICT_NAME].json` - Block data for Kerala districts

## Usage

### 1. Debug and Validate Data
```bash
# Check data structure and content
python debug_data.py

# Validate UUIDs integrity
python check_uuids.py
```

### 2. Create and Insert Embeddings
```bash
# Create embeddings and insert into Pinecone
python insert_data.py
```

### 3. Query the System

#### Basic Vector Search
```bash
# Basic query interface (returns raw vector search results)
python query_index.py
```

#### AI-Powered Query Interface
```bash
# Intelligent query interface with natural language responses
python generate_response.py
```

The AI-powered interface provides:
- **Natural Language Processing**: Ask questions in plain English
- **Context-Aware Responses**: Answers based on retrieved groundwater data
- **Professional Analysis**: Technical explanations with exact figures and units
- **Structured Output**: Well-organized, readable responses

Example queries for the AI interface:
- "Which states in India have the highest groundwater recharge?"
- "What is the groundwater situation in Kerala?"
- "Compare rainfall and recharge data for northern states"
- "Which areas are facing critical groundwater extraction?"
- "Show me districts with good groundwater availability"

## Data Processing Pipeline

1. **Data Fetching** (`fetch_states.py`): Retrieves data from IITH API
2. **Data Validation** (`debug_data.py`, `check_uuids.py`): Validates data integrity
3. **Embedding Creation** (`embeddings.py`): Converts data to vector embeddings
4. **Database Setup** (`pinecone_setup.py`): Initializes Pinecone index
5. **Data Insertion** (`insert_data.py`): Stores embeddings with metadata
6. **Querying**: 
   - **Basic Search** (`query_index.py`): Raw vector similarity search
   - **AI-Enhanced Search** (`generate_response.py`): Intelligent responses with Gemini

## Technical Details

- **Embedding Model**: `all-mpnet-base-v2` (768 dimensions)
- **Vector Database**: Pinecone (cosine similarity)
- **AI Model**: Google Gemini 1.5 Flash for response generation
- **RAG Architecture**: Retrieval-Augmented Generation for accurate, context-based responses
- **Batch Processing**: 100 vectors per batch for efficient insertion
- **Data Format**: JSON with nested structure for geographical hierarchy

## AI Response System

The RAG query system (`generate_response.py`) implements:
- **Vector Retrieval**: Finds most relevant groundwater data based on semantic similarity
- **Context Injection**: Provides retrieved data as context to the AI model
- **Prompt Engineering**: Specialized prompts for groundwater expertise
- **Response Generation**: Natural language answers with technical accuracy

## Troubleshooting

### Common Issues

1. **Missing UUIDs**: Run `check_uuids.py` to identify entries with missing location UUIDs
2. **API Rate Limits**: `fetch_states.py` includes 1.5-second delays between requests
3. **Index Conflicts**: Use `delete_index.py` to clean up existing indexes
4. **Data Structure Issues**: Use `debug_data.py` to analyze data format problems
5. **API Key Errors**: Ensure both Pinecone and Google Gemini API keys are valid and properly configured
6. **Model Loading Issues**: Ensure `sentence-transformers` is properly installed and has internet access

### Index Management
```bash
# Delete existing index if needed
python delete_index.py
```

## API Data Source

Data is fetched from: `https://ingres.iith.ac.in/api/gec/getBusinessDataForUserOpen`

The API provides groundwater data with the following parameters:
- **Component**: recharge, draft, rainfall data
- **Period**: annual data
- **Year**: 2024-2025
- **Geography**: Country → State → District → Block hierarchy

## Security Notes

- Keep your API keys secure and never commit them to version control
- Consider using environment variables for API key management
- The current implementation includes API keys directly in code for demonstration - update for production use

## Contributing

When adding new features:
1. Follow the existing code structure
2. Add appropriate error handling for API calls
3. Update validation scripts for new data fields
4. Test with small datasets before full processing
5. Ensure new AI prompts maintain technical accuracy

## License

This project processes public groundwater data from IITH for research and analysis purposes.