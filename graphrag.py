"""
graphrag.py -- GraphRAG example: Pinecone (semantic) + Neo4j (graph) + Gemini (LLM)
Updated with consistent fetching and coding methods similar to generate_graph_response.py
"""

import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import traceback

load_dotenv()

# ---------- CONFIG ----------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "gw-index")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# Basic checks
if not GENAI_API_KEY:
    raise SystemExit("GENAI_API_KEY is required in .env")
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASS:
    raise SystemExit("Neo4j credentials required in .env")
if not PINECONE_API_KEY:
    raise SystemExit("Pinecone credentials required in .env")

# ---------- Initialize clients ----------
# 1) Neo4j - Using same pattern as generate_graph_response.py
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# 2) Pinecone v4
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    raise SystemExit(f"Pinecone index '{PINECONE_INDEX}' not found. Create it or update PINECONE_INDEX in .env.")
pine_index = pc.Index(PINECONE_INDEX)

# 3) Embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# 4) Configure Gemini (same as generate_graph_response.py)
genai.configure(api_key=GENAI_API_KEY)

# ---------- Knowledge Graph Schema (Enhanced from generate_graph_response.py) ----------
SCHEMA = """
We have a Neo4j knowledge graph with these entities:

Nodes:
(:Availability - command, non_command, poor_quality, total)
(:GroundWaterAvailability - command, non_command, poor_quality, total)
(:Aquifer - dynamic_gw, in_storage_gw, total, type)
(:Loss - command, non_command, poor_quality, total, et, evaporation, transpiration)
(:Rainfall - command, non_command, poor_quality, total)
(:AdditionalRecharge - floodProneArea, shallowArea, springDischarge, total)
(:Recharge - agriculture, artificial_structure, canal, gw_irrigation, pipeline, rainfall, sewage, streamRecharge, surface_irrigation, total, water_body)
(:BlockSummary - Hilly Area, critical, over_exploited, safe, semi_critical, salinity)
(:Area - type(non_recharge_worthy, recharge_worthy, total), commandArea, forestArea, hillyArea, nonCommandArea, pavedArea, poorQualityArea, totalArea, unpavedArea, uuid)
(:Draft - agriculture, domestic, industry, total)
(:Allocation - domestic, industry, total)
(:State - name, uuid)
(:StageOfExtraction - command, non_command, poor_quality, total)
(:FutureUse - command, non_command, poor_quality, total)
(:District - name, uuid)
(:Category - name) -- categories: Safe, Semi-Critical, Critical, Over-Exploited

Relationships:
(State)-[:HAS_RAINFALL]->(Rainfall)
(State)-[:HAS_RECHARGE]->(Recharge)
(State)-[:HAS_DRAFT]->(Draft)
(State)-[:HAS_ALLOCATION]->(Allocation)
(State)-[:HAS_AVAILABILITY]->(Availability)
(State)-[:HAS_STAGE]->(StageOfExtraction)
(State)-[:HAS_GROUND_WATER]->(GroundWaterAvailability)
(State)-[:HAS_FUTURE_USE]->(FutureUse)
(State)-[:HAS_ADDITIONAL_RECHARGE]->(AdditionalRecharge)
(State)-[:HAS_AQUIFER]->(Aquifer)
(State)-[:HAS_District]->(District)
(District)-[:HAS_RAINFALL]->(Rainfall)
(District)-[:HAS_RECHARGE]->(Recharge)
(District)-[:HAS_DRAFT]->(Draft)
(District)-[:HAS_ALLOCATION]->(Allocation)
(District)-[:HAS_AVAILABILITY]->(Availability)
(District)-[:HAS_STAGE]->(StageOfExtraction)
(District)-[:HAS_GROUND_WATER]->(GroundWaterAvailability)
(District)-[:HAS_FUTURE_USE]->(FutureUse)
(District)-[:HAS_ADDITIONAL_RECHARGE]->(AdditionalRecharge)
(District)-[:HAS_AQUIFER]->(Aquifer)
(District)-[:HAS_CATEGORY {year:int}]->(Category)

Notes:
- "India" is the only Country.
- States like Kerala, Tamil Nadu, Gujarat are (:State).
- Places like Ernakulam, Kottayam, Thrissur are (:District) of Kerala.
- Convert states and districts to CAPITAL LETTERS when querying.
"""

# ---------- Step 1: Convert natural language → Cypher (Same method as generate_graph_response.py) ----------
def query_to_cypher(user_query):
    """
    Convert natural language to Cypher query using the same method as generate_graph_response.py
    """
    prompt = f"""
    You are an assistant that converts natural language into Cypher queries.
    Schema:
    {SCHEMA}

    Convert this question into a Cypher query:
    "{user_query}"

    Only return the Cypher query (no explanation).
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ---------- Step 2: Run Cypher on Neo4j (Same method as generate_graph_response.py) ----------
def run_cypher(cypher_query):
    """
    Execute Cypher query against Neo4j - same method as generate_graph_response.py
    """
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

# ---------- Step 3: Semantic Retrieval from Pinecone ----------
def query_pinecone_index(query_text, top_k=5):
    """
    Query Pinecone for semantic similarity - following generate_response.py pattern
    """
    query_vector = embed_model.encode([query_text])[0].tolist()
    result = pine_index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    # Format results similar to generate_response.py
    formatted_results = []
    for match in result.matches:
        formatted_results.append({
            "id": match.id,
            "score": match.score,
            "metadata": dict(match.metadata) if match.metadata else {},
        })
    return formatted_results

# ---------- Step 4: Generate Response (Enhanced from generate_graph_response.py pattern) ----------
def generate_graphrag_response(semantic_results, graph_results, query):
    """
    Generate final response using both semantic and graph results
    Similar to generate_response pattern but enhanced for GraphRAG
    """
    prompt = f"""
    You are an expert in groundwater assessment and management. Your role is to provide accurate, data-driven responses based on both semantic search results and knowledge graph data.

    Rules:
    - Provide answers that are clear, concise, and factual.
    - Always include units, figures, and values exactly as they appear in the context.
    - Combine insights from both semantic search and graph traversal results.
    - If graph results contain errors, focus on semantic results but mention the limitation.
    - Structure responses logically for readability.

    Style:
    - Use a professional and technical tone.
    - Avoid speculation or assumptions.
    - Prioritize data-driven insights.

    Semantic Search Results:
    {json.dumps(semantic_results, indent=2)}

    Knowledge Graph Results:
    {json.dumps(graph_results, indent=2)}

    Query: {query}
    Answer:
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ---------- Main GraphRAG Function (Following generate_graph_response.py chatbot pattern) ----------
def graphrag_chatbot(user_query):
    """
    Main GraphRAG function that combines semantic and graph retrieval
    Following the chatbot pattern from generate_graph_response.py
    """
    # Step 1: Handle direct Cypher input for debugging (same as generate_graph_response.py)
    if user_query.lower().startswith("cypher:"):
        cypher = user_query[len("cypher:"):].strip()
        semantic_results = []
    else:
        # Step 2: Get semantic results from Pinecone
        semantic_results = query_pinecone_index(user_query, top_k=5)
        
        # Step 3: Convert to Cypher for graph traversal
        cypher = query_to_cypher(user_query)

    # Step 4: Execute graph query
    try:
        graph_results = run_cypher(cypher) if cypher else []
        error_info = None
    except Exception as e:
        graph_results = []
        error_info = f"⚠️ Error running graph query:\nCypher: {cypher}\nError: {e}"

    # Step 5: Generate combined response
    if error_info:
        # If graph query failed, use semantic results only
        if semantic_results:
            final_response = generate_graphrag_response(semantic_results, [], user_query)
            final_response += f"\n\nNote: Graph query encountered an issue: {str(error_info).split('Error: ')[-1]}"
        else:
            final_response = error_info
    elif not semantic_results and not graph_results:
        final_response = "No results found from either semantic search or knowledge graph."
    else:
        final_response = generate_graphrag_response(semantic_results, graph_results, user_query)

    return {
        "query": user_query,
        "cypher_used": cypher,
        "semantic_results": semantic_results,
        "graph_results": graph_results,
        "final_answer": final_response,
        "error": error_info
    }

# ---------- CLI Interface (Same pattern as generate_graph_response.py) ----------
if __name__ == "__main__":
    print("🚀 GraphRAG Chatbot started (Pinecone + Neo4j + Gemini)")
    print("Type natural language queries OR `cypher: <query>` for direct queries")
    print("Type 'exit' to quit\n")

    try:
        while True:
            q = input("Ask: ")
            if q.lower() in ["exit", "quit"]:
                break

            result = graphrag_chatbot(q)
            
            print(f"🔍 Cypher used: {result['cypher_used']}")
            print(f"📊 Semantic hits: {len(result['semantic_results'])}")
            print(f"🔗 Graph results: {len(result['graph_results'])}")
            
            if result['error']:
                print(f"⚠️ {result['error']}")
            
            print("💡 Answer:", result['final_answer'])
            print("-" * 80)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        try:
            driver.close()
        except:
            pass