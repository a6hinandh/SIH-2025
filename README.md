# Groundwater Data Vector Search System

This project fetches groundwater data from the Indian Institute of Technology Hyderabad (IITH) API, processes it into vector embeddings using Hugging Face's `all-mpnet-base-v2` model, and stores them in Pinecone for efficient semantic search. The system covers groundwater recharge, draft, rainfall, and extraction data across Indian states, districts, and blocks.

## Features

- **Data Fetching**: Retrieves groundwater data from IITH API for states, districts, and blocks
- **Vector Embeddings**: Converts geographical and hydrological data into searchable vector embeddings  
- **Pinecone Integration**: Stores and queries embeddings in Pinecone vector database
- **Multi-level Geography**: Supports country, state, district, and block level data
- **Data Validation**: Includes debugging and validation tools for data integrity

## Project Structure

```
├── fetch_states.py         # Fetches data from IITH API (states, districts, blocks)
├── embeddings.py          # Creates vector embeddings from JSON data
├── pinecone_setup.py      # Initializes Pinecone index
├── insert_data.py         # Inserts embeddings into Pinecone
├── query_index.py         # Query interface for searching data
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

### 2. Configure Pinecone API Key
Update your Pinecone API key in the relevant files:
- `insert_data.py`
- `delete_index.py` 
- `query_index.py`

Replace `"YOUR_PINECONE_API_KEY"` with your actual API key.

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
```bash
# Interactive query interface
python query_index.py
```

Example queries:
- "States with high rainfall"
- "Areas with critical groundwater extraction"
- "Kerala groundwater data"
- "States with good groundwater recharge"

## Data Processing Pipeline

1. **Data Fetching** (`fetch_states.py`): Retrieves data from IITH API
2. **Data Validation** (`debug_data.py`, `check_uuids.py`): Validates data integrity
3. **Embedding Creation** (`embeddings.py`): Converts data to vector embeddings
4. **Database Setup** (`pinecone_setup.py`): Initializes Pinecone index
5. **Data Insertion** (`insert_data.py`): Stores embeddings with metadata
6. **Querying** (`query_index.py`): Semantic search interface

## Technical Details

- **Embedding Model**: `all-mpnet-base-v2` (768 dimensions)
- **Vector Database**: Pinecone (cosine similarity)
- **Batch Processing**: 100 vectors per batch for efficient insertion
- **Data Format**: JSON with nested structure for geographical hierarchy

## Troubleshooting

### Common Issues

1. **Missing UUIDs**: Run `check_uuids.py` to identify entries with missing location UUIDs
2. **API Rate Limits**: `fetch_states.py` includes 1.5-second delays between requests
3. **Index Conflicts**: Use `delete_index.py` to clean up existing indexes
4. **Data Structure Issues**: Use `debug_data.py` to analyze data format problems

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

## Contributing

When adding new features:
1. Follow the existing code structure
2. Add appropriate error handling
3. Update validation scripts for new data fields
4. Test with small datasets before full processing

## License

This project processes public groundwater data from IITH for research and analysis purposes.