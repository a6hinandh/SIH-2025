# """
# graphrag.py -- GraphRAG example: Pinecone (semantic) + Neo4j (graph) + Gemini (LLM)
# Updated with consistent fetching and coding methods similar to generate_graph_response.py
# """

# import os
# import json
# from dotenv import load_dotenv
# from neo4j import GraphDatabase
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# import traceback

# load_dotenv()

# # ---------- CONFIG ----------
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USER = os.getenv("NEO4J_USER")
# NEO4J_PASS = os.getenv("NEO4J_PASS")

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX", "gw-index")

# EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
# GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# # Basic checks
# if not GENAI_API_KEY:
#     raise SystemExit("GENAI_API_KEY is required in .env")
# if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASS:
#     raise SystemExit("Neo4j credentials required in .env")
# if not PINECONE_API_KEY:
#     raise SystemExit("Pinecone credentials required in .env")

# # ---------- Initialize clients ----------
# # 1) Neo4j - Using same pattern as generate_graph_response.py
# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# # 2) Pinecone v4
# pc = Pinecone(api_key=PINECONE_API_KEY)
# if PINECONE_INDEX not in pc.list_indexes().names():
#     raise SystemExit(f"Pinecone index '{PINECONE_INDEX}' not found. Create it or update PINECONE_INDEX in .env.")
# pine_index = pc.Index(PINECONE_INDEX)

# # 3) Embedding model
# embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# # 4) Configure Gemini (same as generate_graph_response.py)
# genai.configure(api_key=GENAI_API_KEY)

# # ---------- Knowledge Graph Schema (Enhanced from generate_graph_response.py) ----------
# SCHEMA = """
# We have a Neo4j knowledge graph with these entities:

# Nodes:
# (:Availability - command, non_command, poor_quality, total)
# (:GroundWaterAvailability - command, non_command, poor_quality, total)
# (:Aquifer - dynamic_gw, in_storage_gw, total, type)
# (:Loss - command, non_command, poor_quality, total, et, evaporation, transpiration)
# (:Rainfall - command, non_command, poor_quality, total)
# (:AdditionalRecharge - floodProneArea, shallowArea, springDischarge, total)
# (:Recharge - agriculture, artificial_structure, canal, gw_irrigation, pipeline, rainfall, sewage, streamRecharge, surface_irrigation, total, water_body)
# (:BlockSummary - Hilly Area, critical, over_exploited, safe, semi_critical, salinity)
# (:Area - type(non_recharge_worthy, recharge_worthy, total), commandArea, forestArea, hillyArea, nonCommandArea, pavedArea, poorQualityArea, totalArea, unpavedArea, uuid)
# (:Draft - agriculture, domestic, industry, total)
# (:Allocation - domestic, industry, total)
# (:State - name, uuid)
# (:StageOfExtraction - command, non_command, poor_quality, total)
# (:FutureUse - command, non_command, poor_quality, total)
# (:District - name, uuid)
# (:Category - name) -- categories: Safe, Semi-Critical, Critical, Over-Exploited

# Relationships:
# (State)-[:HAS_RAINFALL]->(Rainfall)
# (State)-[:HAS_RECHARGE]->(Recharge)
# (State)-[:HAS_DRAFT]->(Draft)
# (State)-[:HAS_ALLOCATION]->(Allocation)
# (State)-[:HAS_AVAILABILITY]->(Availability)
# (State)-[:HAS_STAGE]->(StageOfExtraction)
# (State)-[:HAS_GROUND_WATER]->(GroundWaterAvailability)
# (State)-[:HAS_FUTURE_USE]->(FutureUse)
# (State)-[:HAS_ADDITIONAL_RECHARGE]->(AdditionalRecharge)
# (State)-[:HAS_AQUIFER]->(Aquifer)
# (State)-[:HAS_District]->(District)
# (District)-[:HAS_RAINFALL]->(Rainfall)
# (District)-[:HAS_RECHARGE]->(Recharge)
# (District)-[:HAS_DRAFT]->(Draft)
# (District)-[:HAS_ALLOCATION]->(Allocation)
# (District)-[:HAS_AVAILABILITY]->(Availability)
# (District)-[:HAS_STAGE]->(StageOfExtraction)
# (District)-[:HAS_GROUND_WATER]->(GroundWaterAvailability)
# (District)-[:HAS_FUTURE_USE]->(FutureUse)
# (District)-[:HAS_ADDITIONAL_RECHARGE]->(AdditionalRecharge)
# (District)-[:HAS_AQUIFER]->(Aquifer)
# (District)-[:HAS_CATEGORY {year:int}]->(Category)

# Notes:
# - "India" is the only Country.
# - States like Kerala, Tamil Nadu, Gujarat are (:State).
# - Places like Ernakulam, Kottayam, Thrissur are (:District) of Kerala.
# - Convert states and districts to CAPITAL LETTERS when querying.
# """

# # ---------- Step 1: Convert natural language → Cypher (Same method as generate_graph_response.py) ----------
# def query_to_cypher(user_query):
#     """
#     Convert natural language to Cypher query using the same method as generate_graph_response.py
#     """
#     prompt = f"""
#     You are an assistant that converts natural language into Cypher queries.
#     Schema:
#     {SCHEMA}

#     Convert this question into a Cypher query:
#     "{user_query}"

#     Only return the Cypher query (no explanation).
#     Remove ```cypher from beginnng and ``` from end
#     """
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     response = model.generate_content(prompt)
#     return response.text.strip()

# # ---------- Step 2: Run Cypher on Neo4j (Same method as generate_graph_response.py) ----------
# def run_cypher(cypher_query):
#     """
#     Execute Cypher query against Neo4j - same method as generate_graph_response.py
#     """
#     with driver.session() as session:
#         result = session.run(cypher_query)
#         return [record.data() for record in result]

# # ---------- Step 3: Semantic Retrieval from Pinecone ----------
# def query_pinecone_index(query_text, top_k=5):
#     """
#     Query Pinecone for semantic similarity - following generate_response.py pattern
#     """
#     query_vector = embed_model.encode([query_text])[0].tolist()
#     result = pine_index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
#     # Format results similar to generate_response.py
#     formatted_results = []
#     for match in result.matches:
#         formatted_results.append({
#             "id": match.id,
#             "score": match.score,
#             "metadata": dict(match.metadata) if match.metadata else {},
#         })
#     return formatted_results

# # ---------- Step 4: Generate Response (Enhanced from generate_graph_response.py pattern) ----------
# def generate_graphrag_response(semantic_results, graph_results, query):
#     """
#     Generate final response using both semantic and graph results
#     Similar to generate_response pattern but enhanced for GraphRAG
#     """
#     prompt = f"""
#     You are an expert in groundwater assessment and management. Your role is to provide accurate, data-driven responses based on both semantic search results and knowledge graph data.

#     Rules:
#     - Provide answers that are clear, concise, and factual.
#     - Always include units, figures, and values exactly as they appear in the context.
#     - Combine insights from both semantic search and graph traversal results.
#     - If graph results contain errors, focus on semantic results but mention the limitation.
#     - Structure responses logically for readability.

#     Style:
#     - Use a professional and technical tone.
#     - Avoid speculation or assumptions.
#     - Prioritize data-driven insights.

#     Semantic Search Results:
#     {json.dumps(semantic_results, indent=2)}

#     Knowledge Graph Results:
#     {json.dumps(graph_results, indent=2)}

#     Query: {query}
#     Answer:
#     """
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     response = model.generate_content(prompt)
#     return response.text

# # ---------- Main GraphRAG Function (Following generate_graph_response.py chatbot pattern) ----------
# def graphrag_chatbot(user_query):
#     """
#     Main GraphRAG function that combines semantic and graph retrieval
#     Following the chatbot pattern from generate_graph_response.py
#     """
#     # Step 1: Handle direct Cypher input for debugging (same as generate_graph_response.py)
#     if user_query.lower().startswith("cypher:"):
#         cypher = user_query[len("cypher:"):].strip()
#         semantic_results = []
#     else:
#         # Step 2: Get semantic results from Pinecone
#         semantic_results = query_pinecone_index(user_query, top_k=5)
        
#         # Step 3: Convert to Cypher for graph traversal
#         cypher = query_to_cypher(user_query)

#     # Step 4: Execute graph query
#     try:
#         graph_results = run_cypher(cypher) if cypher else []
#         error_info = None
#     except Exception as e:
#         graph_results = []
#         error_info = f"⚠️ Error running graph query:\nCypher: {cypher}\nError: {e}"

#     # Step 5: Generate combined response
#     if error_info:
#         # If graph query failed, use semantic results only
#         if semantic_results:
#             final_response = generate_graphrag_response(semantic_results, [], user_query)
#             final_response += f"\n\nNote: Graph query encountered an issue: {str(error_info).split('Error: ')[-1]}"
#         else:
#             final_response = error_info
#     elif not semantic_results and not graph_results:
#         final_response = "No results found from either semantic search or knowledge graph."
#     else:
#         final_response = generate_graphrag_response(semantic_results, graph_results, user_query)

#     return {
#         "query": user_query,
#         "cypher_used": cypher,
#         "semantic_results": semantic_results,
#         "graph_results": graph_results,
#         "final_answer": final_response,
#         "error": error_info
#     }

# # ---------- CLI Interface (Same pattern as generate_graph_response.py) ----------
# if __name__ == "__main__":
#     print("🚀 GraphRAG Chatbot started (Pinecone + Neo4j + Gemini)")
#     print("Type natural language queries OR `cypher: <query>` for direct queries")
#     print("Type 'exit' to quit\n")

#     try:
#         while True:
#             q = input("Ask: ")
#             if q.lower() in ["exit", "quit"]:
#                 break

#             result = graphrag_chatbot(q)
            
#             print(f"🔍 Cypher used: {result['cypher_used']}")
#             print(f"📊 Semantic hits: {len(result['semantic_results'])}")
#             print(f"🔗 Graph results: {len(result['graph_results'])}")
            
#             if result['error']:
#                 print(f"⚠️ {result['error']}")
            
#             print("💡 Answer:", result['final_answer'])
#             print("-" * 80)

#     except KeyboardInterrupt:
#         print("\nExiting...")
#     finally:
#         try:
#             driver.close()
#         except:
#             pass



# """
# graphrag.py -- Enhanced GraphRAG: Pinecone (semantic) + Neo4j (graph) + Gemini (LLM)
# Updated with improvements from SIH 2025 document:
# - Robust Cypher handling with sanitization
# - Better error handling and caching
# - Improved prompts with few-shot examples
# - Performance optimizations
# - Provenance and citations
# """

# import os
# import json
# import re
# import time
# from typing import Dict, List, Any, Optional, Tuple
# from functools import lru_cache
# from dotenv import load_dotenv
# from neo4j import GraphDatabase
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# import traceback

# load_dotenv()

# # ---------- CONFIG ----------
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USER = os.getenv("NEO4J_USER")
# NEO4J_PASS = os.getenv("NEO4J_PASS")

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX", "gw-index")

# EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
# GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# # Performance settings
# CACHE_SIZE = 100
# QUERY_TIMEOUT = 30
# MAX_RETRIES = 3

# # Basic checks
# if not GENAI_API_KEY:
#     raise SystemExit("GENAI_API_KEY is required in .env")
# if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASS:
#     raise SystemExit("Neo4j credentials required in .env")
# if not PINECONE_API_KEY:
#     raise SystemExit("Pinecone credentials required in .env")

# # ---------- Initialize clients ----------
# # 1) Neo4j with timeout and retry settings
# driver = GraphDatabase.driver(
#     NEO4J_URI, 
#     auth=(NEO4J_USER, NEO4J_PASS),
#     connection_timeout=QUERY_TIMEOUT,
#     max_connection_lifetime=300
# )

# # 2) Pinecone v4
# pc = Pinecone(api_key=PINECONE_API_KEY)
# if PINECONE_INDEX not in pc.list_indexes().names():
#     raise SystemExit(f"Pinecone index '{PINECONE_INDEX}' not found. Create it or update PINECONE_INDEX in .env.")
# pine_index = pc.Index(PINECONE_INDEX)

# # 3) Embedding model
# embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# # 4) Configure Gemini
# genai.configure(api_key=GENAI_API_KEY)

# # 5) Warm-up call to reduce first-request lag
# def warmup_pinecone():
#     """Warm-up call to reduce first-request lag"""
#     try:
#         dummy_vector = embed_model.encode(["warmup"])[0].tolist()
#         pine_index.query(vector=dummy_vector, top_k=1)
#         print("🔥 Pinecone warmed up successfully")
#     except Exception as e:
#         print(f"⚠️ Warmup failed: {e}")

# warmup_pinecone()

# # ---------- Knowledge Graph Schema with Enhanced Documentation ----------
# SCHEMA = """
# We have a Neo4j knowledge graph with these entities:

# Nodes:
# Nodes and their properties:
# (:Availability - command, non_command, poor_quality, total)
# (:GroundWaterAvailability - command, non_command, poor_quality, total)
# (:Aquifer - dynamic_gw, in_storage_gw, total, type)
# (:Loss - command, non_command, poor_quality, total, et, evaporation, transpiration)
# (:Rainfall - command, non_command, poor_quality, total)
# (:AdditionalRecharge - floodProneArea, shallowArea, springDischarge, total)
# (:Recharge - agriculture, artificial_structure, canal, gw_irrigation, pipeline, rainfall,sewage, streamRecharge, surface_irrigation, total, water_body)
# (:BlockSummary - Hilly Area, critical, over_exploited, safe, semi_critical, salinity)
# (:Area - type(non_recharge_worthy, recharge_worthy, total), commandArea, forestArea, hillyArea, nonCommandArea, pavedArea, poorQualityArea, totalArea, unpavedArea, uuid)
# (:Draft - agriculture, domestic, industry, total)
# (:Allocation - domestic, industry, total)
# (:State - name, uuid)
# (:StageOfExtraction - command, non_command, poor_quality, total)
# (:FutureUse - command, non_command, poor_quality, total)
# (:District - name, uuid, status(safe,critical,exploited,etc), category(safe,critical,exploited,etc))
# (:Year - year, uuid)
# (:Country - name, uuid)

# Relationships:
# "(Country)-[:HAS_YEAR]->(Year)"
# "(Country)-[:HAS_STATE]->(State)"
# "(State)-[:HAS_YEAR]->(Year)"
# "(State)-[:HAS_DISTRICT]->(District)"
# "(District)-[:HAS_YEAR]->(Year)"
# "(Year)-[:HAS_AREA]->(Area)"
# "(Year)-[:HAS_LOSS]->(Loss)"
# "(Year)-[:HAS_BLOCK_SUMMARY]->(BlockSummary)"
# "(Year)-[:HAS_RECHARGE]->(Recharge)"
# "(Year)-[:HAS_DRAFT]->(Draft)"
# "(Year)-[:HAS_ALLOCATION]->(Allocation)"
# "(Year)-[:HAS_AVAILABILITY]->(Availability)"
# "(Year)-[:HAS_STAGE]->(StageOfExtraction)"
# "(Year)-[:HAS_RAINFALL]->(Rainfall)"
# "(Year)-[:HAS_GROUND_WATER]->(GroundWaterAvailability)"
# "(Year)-[:HAS_FUTURE_USE]->(FutureUse)"
# "(Year)-[:HAS_ADDITIONAL_RECHARGE]->(AdditionalRecharge)"
# "(Year)-[:HAS_AQUIFER]->(Aquifer)"

# IMPORTANT RULES:
# 1. NEVER use exists() function - use "property IS NOT NULL" instead
# 2. Always convert state and district names to UPPERCASE
# 3. Use proper Neo4j 5+ syntax
# 4. Return only valid Cypher without code fences or explanations
# 5. If asked for recharge_worthy or non_recharge_worthy_area, mention it in the type property of (:Area)
# 6. If asked for total area, remember the type is "total"
# 7. If have to use BlockSummary node, simply return the value of what is asked
# 8. If no year is mentioned, use ONLY 2024 as the year value on the Year node
# 9. The unit of rainfall is "mm", unit of area is "ha" and units for other ground water data is "ham" 
# """

# # ---------- Few-shot Examples for Better Cypher Generation ----------
# FEW_SHOT_EXAMPLES = """
# Example 1:
# Question: "What is the rainfall in Kerala?"
# Cypher: MATCH (c:Country {name:"India"})-[:HAS_STATE]->(s:State {name:"KERALA"})-[:HAS_YEAR]->(y:Year {year:2024})-[:HAS_RAINFALL]->(r:Rainfall)
# RETURN r.total AS rainfall


# Example 2:
# Question: "Show districts in Kerala with critical groundwater status"
# Cypher: MATCH (s:State {name: "KERALA"})-[:HAS_District]->(d:District)-[:HAS_CATEGORY]->(c:Category {name: "Critical"}) RETURN d.name, c.name

# Example 3:
# Question: "Rainfall data for Kottayam district in 2023"
# Cypher: MATCH (c:Country {name:"India"})-[:HAS_STATE]->(:State {name:"KERALA"})-[:HAS_DISTRICT]->(d:District {name:"KOTTAYAM"})-[:HAS_YEAR]->(y:Year {year:2023})-[:HAS_RAINFALL]->(r:Rainfall)
# RETURN d.name AS District, y.year AS Year, r.total AS Rainfall

# Example 4:
# Question: "Compare groundwater draft between Kerala and Tamil Nadu"
# Cypher: MATCH (c:Country {name:"India"})-[:HAS_STATE]->(s:State)-[:HAS_YEAR]->(y:Year)-[:HAS_DRAFT]->(d:Draft) WHERE s.name IN ["KERALA", "TAMIL NADU"]
# RETURN s.name AS State, y.year AS Year, d.total AS Draft
# ORDER BY y.year, State

# """

# # ---------- Cypher Sanitization Functions ----------
# def sanitize_cypher(cypher_query: str) -> str:
#     """
#     Sanitize Cypher query to fix common issues:
#     1. Replace deprecated exists() with IS NOT NULL
#     2. Uppercase state/district names
#     3. Strip code fences and extra text
#     4. Basic validation
#     """
#     # Remove code fences
#     cypher_query = re.sub(r'```cypher\s*', '', cypher_query)
#     cypher_query = re.sub(r'```\s*$', '', cypher_query)
    
#     # Replace deprecated exists() syntax
#     cypher_query = re.sub(r'exists\(([^)]+)\)', r'\1 IS NOT NULL', cypher_query)
    
#     # Uppercase state and district names in quotes
#     def uppercase_names(match):
#         return match.group(0).upper()
    
#     # Pattern to match quoted state/district names
#     cypher_query = re.sub(r'"[^"]*"', uppercase_names, cypher_query)
    
#     # Remove multiple whitespaces and newlines
#     cypher_query = re.sub(r'\s+', ' ', cypher_query).strip()
    
#     return cypher_query

# def validate_cypher(cypher_query: str) -> Tuple[bool, str]:
#     """
#     Validate Cypher query for basic safety and correctness
#     """
#     cypher_upper = cypher_query.upper()
    
#     # Check for RETURN statement
#     if 'RETURN' not in cypher_upper:
#         return False, "Query must contain a RETURN statement"
    
#     # Block destructive operations
#     destructive_ops = ['DELETE', 'CREATE', 'DROP', 'MERGE', 'SET']
#     for op in destructive_ops:
#         if op in cypher_upper:
#             return False, f"Destructive operation '{op}' not allowed"
    
#     # Basic syntax check
#     if cypher_query.count('(') != cypher_query.count(')'):
#         return False, "Unmatched parentheses in query"
    
#     return True, "Valid"

# # ---------- Enhanced Cypher Generation with Few-shot Examples ----------
# def query_to_cypher(user_query: str) -> str:
#     """
#     Convert natural language to Cypher query with improved prompts and few-shot examples
#     """
#     prompt = f"""
#     You are a Cypher query expert. Convert natural language questions into valid Neo4j Cypher queries.

#     Schema:
#     {SCHEMA}

#     Few-shot Examples:
#     {FEW_SHOT_EXAMPLES}

#     STRICT RULES:
#     1. NEVER use exists() - use "property IS NOT NULL" instead
#     2. ALWAYS convert state/district names to UPPERCASE in quotes
#     3. Return ONLY the Cypher query, no explanations or code fences
#     4. Use Neo4j 5+ compatible syntax only
#     5. Always include a RETURN statement
#     6. If asked for recharge_worthy or non_recharge_worthy_area, mention it in the type property of (:Area)
#     7. If asked for total area, remember the type is "total"
#     8. If have to use BlockSummary node, simply return the value of what is asked
#     9. If no year is mentioned, use ONLY 2024 as the year value on the Year node
#     10. The unit of rainfall is "mm", unit of area is "ha" and units for other ground water data is "ham" 

#     Question: "{user_query}"

#     Cypher:
#     """
    
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     try:
#         response = model.generate_content(prompt)
#         raw_cypher = response.text.strip()
        
#         # Sanitize the generated Cypher
#         sanitized_cypher = sanitize_cypher(raw_cypher)
        
#         # Validate the query
#         is_valid, error_msg = validate_cypher(sanitized_cypher)
#         if not is_valid:
#             print(f"⚠️ Cypher validation failed: {error_msg}")
#             print(f"Original: {raw_cypher}")
#             print(f"Sanitized: {sanitized_cypher}")
#             return None
        
#         return sanitized_cypher
#     except Exception as e:
#         print(f"❌ Error generating Cypher: {e}")
#         return None

# # ---------- Robust Neo4j Query Execution with Retries ----------
# def run_cypher(cypher_query: str) -> List[Dict[str, Any]]:
#     """
#     Execute Cypher query with retry logic and timeout handling
#     """
#     if not cypher_query:
#         return []
    
#     for attempt in range(MAX_RETRIES):
#         try:
#             with driver.session() as session:
#                 result = session.run(cypher_query, timeout=QUERY_TIMEOUT)
#                 return [record.data() for record in result]
#         except Exception as e:
#             if attempt == MAX_RETRIES - 1:
#                 raise e
#             print(f"⚠️ Attempt {attempt + 1} failed, retrying: {e}")
#             time.sleep(1)  # Wait before retry
    
#     return []

# # ---------- Enhanced Semantic Retrieval with Caching ----------
# @lru_cache(maxsize=CACHE_SIZE)
# def query_pinecone_index(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
#     """
#     Query Pinecone for semantic similarity with caching
#     """
#     try:
#         query_vector = embed_model.encode([query_text])[0].tolist()
#         result = pine_index.query(
#             vector=query_vector, 
#             top_k=top_k, 
#             include_metadata=True
#         )
        
#         formatted_results = []
#         for match in result.matches:
#             formatted_results.append({
#                 "id": match.id,
#                 "score": round(match.score, 4),
#                 "metadata": dict(match.metadata) if match.metadata else {},
#             })
#         return formatted_results
#     except Exception as e:
#         print(f"❌ Pinecone query failed: {e}")
#         return []

# # ---------- Enhanced Response Generation with Provenance ----------
# def generate_graphrag_response(semantic_results: List[Dict], graph_results: List[Dict], 
#                              query: str, cypher_used: Optional[str] = None, 
#                              debug_mode: bool = False) -> str:
#     """
#     Generate final response with provenance and citations
#     """
#     # Prepare provenance information
#     graph_provenance = []
#     if graph_results and cypher_used:
#         graph_provenance.append(f"Graph Query: {cypher_used}")
#         graph_provenance.append(f"Graph Results Count: {len(graph_results)}")
    
#     semantic_provenance = []
#     if semantic_results:
#         semantic_provenance.append(f"Semantic Results Count: {len(semantic_results)}")
#         for i, result in enumerate(semantic_results[:3]):  # Top 3 sources
#             metadata = result.get('metadata', {})
#             source = metadata.get('source', f"Document_{result['id']}")
#             score = result['score']
#             semantic_provenance.append(f"Source {i+1}: {source} (similarity: {score})")

#     prompt = f"""
#     You are an expert in groundwater assessment and management. Provide accurate, data-driven responses based on both semantic search results and knowledge graph data.

#     RESPONSE GUIDELINES:
#     - Be clear, concise, and factual
#     - Include specific numbers, units, and values from the data
#     - Combine insights from both semantic and graph results and give only the most appropriate result
#     - Dont include detailed explanation. ONLY answer the query of the user
#     - Use professional, technical tone
#     - If the results are negative, respond non-affirmatively in a polite way

#     USER QUERY: {query}

#     KNOWLEDGE GRAPH DATA:
#     {json.dumps(graph_results, indent=2) if graph_results else "No graph results available"}

#     SEMANTIC SEARCH DATA:
#     {json.dumps(semantic_results, indent=2) if semantic_results else "No semantic results available"}

#     PROVENANCE INFORMATION:
#     Graph Sources: {'; '.join(graph_provenance) if graph_provenance else 'None'}
#     Semantic Sources: {'; '.join(semantic_provenance) if semantic_provenance else 'None'}

#     Generate a comprehensive response that:
#     1. Directly answers the user's question
#     2. Dont include detailed explanation. ONLY answer the query of the user
#     3. Includes relevant numerical data with units
#     4. Answer in complete sentences using these units - The unit of rainfall is "mm", unit of area is "ha" and units for other ground water data is "ham" 
#     5. Dont mention the source from which information is taken
#     6. If the results are negative, respond non-affirmatively in a polite way
#     Response:
#     """
    
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     try:
#         response = model.generate_content(prompt)
#         answer = response.text
        
#         # Add debug information if requested
#         if debug_mode and (graph_provenance or semantic_provenance):
#             debug_info = "\n\n--- DEBUG INFORMATION ---"
#             if graph_provenance:
#                 debug_info += f"\n🔗 Graph Sources: {'; '.join(graph_provenance)}"
#             if semantic_provenance:
#                 debug_info += f"\n📚 Semantic Sources: {'; '.join(semantic_provenance[:3])}"
#             answer += debug_info
        
#         return answer
#     except Exception as e:
#         return f"❌ Error generating response: {e}"

# # ---------- Role-based Response Formatting ----------
# def format_response_for_role(response: str, role: str = "general") -> str:
#     """
#     Format response based on user role (farmer, policymaker, researcher)
#     """
#     if role.lower() == "farmer":
#         # Add practical recommendations
#         if "rainfall" in response.lower() or "water" in response.lower():
#             response += "\n\n💡 Recommendation: Consider water-conserving irrigation methods during low rainfall periods."
#     elif role.lower() == "policymaker":
#         # Add policy implications
#         if "critical" in response.lower() or "over-exploited" in response.lower():
#             response += "\n\n📋 Policy Note: Areas showing critical status may require immediate intervention and sustainable groundwater management policies."
#     elif role.lower() == "researcher":
#         # Add data export suggestion
#         response += "\n\n📊 Data Export: For detailed analysis, specific data can be exported in CSV format upon request."
    
#     return response

# # ---------- Main Enhanced GraphRAG Function ----------
# def graphrag_chatbot(user_query: str, role: str = "general", debug_mode: bool = False) -> Dict[str, Any]:
#     """
#     Enhanced GraphRAG function with robust error handling and caching
#     """
#     start_time = time.time()
    
#     # Handle direct Cypher input for debugging
#     if user_query.lower().startswith("cypher:"):
#         cypher = user_query[len("cypher:"):].strip()
#         semantic_results = []
#     else:
#         # Get semantic results from Pinecone
#         try:
#             semantic_results = query_pinecone_index(user_query, top_k=5)
#         except Exception as e:
#             semantic_results = []
#             print(f"⚠️ Semantic search failed: {e}")
        
#         # Convert to Cypher for graph traversal
#         cypher = query_to_cypher(user_query)

#     # Execute graph query with robust error handling
#     graph_results = []
#     error_info = None
    
#     if cypher:
#         try:
#             graph_results = run_cypher(cypher)
#         except Exception as e:
#             error_info = f"Graph query failed: {str(e)}"
#             print(f"❌ {error_info}")
#     else:
#         error_info = "Could not generate valid Cypher query"

#     # Generate enhanced response
#     if not semantic_results and not graph_results:
#         if error_info:
#             final_response = f"No results found. {error_info}"
#         else:
#             final_response = "No results found from either semantic search or knowledge graph."
#     else:
#         final_response = generate_graphrag_response(
#             semantic_results, graph_results, user_query, cypher, debug_mode
#         )
        
#         # Add error note if graph query failed but semantic results exist
#         # if error_info and semantic_results:
#         #     final_response += f"\n\n⚠️ Note: Graph query unavailable - {error_info}"
    
#     # Format response based on role
#     final_response = format_response_for_role(final_response, role)
    
#     processing_time = round(time.time() - start_time, 2)
    
#     return {
#         "query": user_query,
#         "cypher_used": cypher,
#         "semantic_results": semantic_results,
#         "graph_results": graph_results,
#         "final_answer": final_response,
#         "error": error_info,
#         "processing_time": processing_time,
#         "role": role,
#         "debug_mode": debug_mode
#     }

# # ---------- Enhanced CLI Interface with Role Selection ----------
# def get_user_role():
#     """Get user role for personalized responses"""
#     print("\nSelect your role:")
#     print("1. Farmer (practical recommendations)")
#     print("2. Policymaker (policy insights)")
#     print("3. Researcher (detailed data)")
#     print("4. General user")
    
#     while True:
#         choice = input("Enter choice (1-4): ").strip()
#         role_map = {"1": "farmer", "2": "policymaker", "3": "researcher", "4": "general"}
#         if choice in role_map:
#             return role_map[choice]
#         print("Invalid choice. Please enter 1-4.")

# if __name__ == "__main__":
#     print("🚀 Enhanced GraphRAG Chatbot (Pinecone + Neo4j + Gemini)")
#     print("✨ Features: Robust Cypher handling, caching, provenance, role-based responses")
    
#     # Get user role
#     user_role = get_user_role()
#     print(f"\n👤 Role selected: {user_role.capitalize()}")
    
#     print("\nCommands:")
#     print("- Natural language queries")
#     print("- 'cypher: <query>' for direct Cypher")
#     print("- 'debug on/off' to toggle debug mode")
#     print("- 'role <farmer/policymaker/researcher/general>' to change role")
#     print("- 'exit' to quit\n")
    
#     debug_mode = False
    
#     try:
#         while True:
#             q = input("Ask: ").strip()
            
#             if q.lower() in ["exit", "quit"]:
#                 break
#             elif q.lower().startswith("debug "):
#                 debug_mode = "on" in q.lower()
#                 print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
#                 continue
#             elif q.lower().startswith("role "):
#                 new_role = q[5:].strip().lower()
#                 if new_role in ["farmer", "policymaker", "researcher", "general"]:
#                     user_role = new_role
#                     print(f"👤 Role changed to: {user_role.capitalize()}")
#                 else:
#                     print("Invalid role. Use: farmer, policymaker, researcher, or general")
#                 continue
#             elif not q:
#                 continue

#             # Process query
#             result = graphrag_chatbot(q, role=user_role, debug_mode=debug_mode)
            
#             # Display results
#             print(f"\n🔍 Query processed in {result['processing_time']}s")
#             print(f"🔗 Cypher used: {result['cypher_used'] or 'None'}")
#             print(f"📊 Semantic hits: {len(result['semantic_results'])}")
#             print(f"📈 Graph results: {len(result['graph_results'])}")
            
#             if result['error'] and debug_mode:
#                 print(f"⚠️ Error: {result['error']}")
            
#             print(f"\n💡 Answer ({user_role}):")
#             print(result['final_answer'])
#             print("=" * 80)

#     except KeyboardInterrupt:
#         print("\n👋 Exiting...")
#     finally:
#         try:
#             driver.close()
#             print("🔌 Neo4j connection closed")
#         except:
#             pass



"""
graphrag.py -- Enhanced GraphRAG: Pinecone (semantic) + Neo4j (graph) + Gemini (LLM)
Updated with role-aware insights and interpretive responses from SIH 2025 improvements:
- Role-based prompt injection for tailored responses
- Interpretive insights (high/low/normal context)
- Enhanced thresholds and context rules
- Improved response generation with role guidelines
"""

import os
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from langdetect import detect
from googletrans import Translator
import traceback

load_dotenv()

translator = Translator()

# ---------- CONFIG ----------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "gw-index")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# Performance settings
CACHE_SIZE = 100
QUERY_TIMEOUT = 30
MAX_RETRIES = 3

# Basic checks
if not GENAI_API_KEY:
    raise SystemExit("GENAI_API_KEY is required in .env")
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASS:
    raise SystemExit("Neo4j credentials required in .env")
if not PINECONE_API_KEY:
    raise SystemExit("Pinecone credentials required in .env")

# ---------- Initialize clients ----------
# 1) Neo4j with timeout and retry settings
driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=(NEO4J_USER, NEO4J_PASS),
    connection_timeout=QUERY_TIMEOUT,
    max_connection_lifetime=300
)

# 2) Pinecone v4
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    raise SystemExit(f"Pinecone index '{PINECONE_INDEX}' not found. Create it or update PINECONE_INDEX in .env.")
pine_index = pc.Index(PINECONE_INDEX)

# 3) Embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# 4) Configure Gemini
genai.configure(api_key=GENAI_API_KEY)

# 5) Warm-up call to reduce first-request lag
def warmup_pinecone():
    """Warm-up call to reduce first-request lag"""
    try:
        dummy_vector = embed_model.encode(["warmup"])[0].tolist()
        pine_index.query(vector=dummy_vector, top_k=1)
        print("🔥 Pinecone warmed up successfully")
    except Exception as e:
        print(f"⚠️ Warmup failed: {e}")

warmup_pinecone()

# ---------- Knowledge Graph Schema with Enhanced Documentation ----------
SCHEMA = """
We have a Neo4j knowledge graph with these entities:

Nodes:
Nodes and their properties:
(:Availability - command, non_command, poor_quality, total)
(:GroundWaterAvailability - command, non_command, poor_quality, total)
(:Aquifer - dynamic_gw, in_storage_gw, total, type)
(:Loss - command, non_command, poor_quality, total, et, evaporation, transpiration)
(:Rainfall - command, non_command, poor_quality, total)
(:AdditionalRecharge - floodProneArea, shallowArea, springDischarge, total)
(:Recharge - agriculture, artificial_structure, canal, gw_irrigation, pipeline, rainfall,sewage, streamRecharge, surface_irrigation, total, water_body)
(:BlockSummary - Hilly Area, critical, over_exploited, safe, semi_critical, salinity)
(:Area - type(non_recharge_worthy, recharge_worthy, total), commandArea, forestArea, hillyArea, nonCommandArea, pavedArea, poorQualityArea, totalArea, unpavedArea, uuid)
(:Draft - agriculture, domestic, industry, total)
(:Allocation - domestic, industry, total)
(:State - name, uuid)
(:StageOfExtraction - command, non_command, poor_quality, total)
(:FutureUse - command, non_command, poor_quality, total)
(:District - name, uuid, status(safe,critical,exploited,etc), category(safe,critical,exploited,etc))
(:Year - year, uuid)
(:Country - name, uuid)

Relationships:
"(Country)-[:HAS_YEAR]->(Year)"
"(Country)-[:HAS_STATE]->(State)"
"(State)-[:HAS_YEAR]->(Year)"
"(State)-[:HAS_DISTRICT]->(District)"
"(District)-[:HAS_YEAR]->(Year)"
"(Year)-[:HAS_AREA]->(Area)"
"(Year)-[:HAS_LOSS]->(Loss)"
"(Year)-[:HAS_BLOCK_SUMMARY]->(BlockSummary)"
"(Year)-[:HAS_RECHARGE]->(Recharge)"
"(Year)-[:HAS_DRAFT]->(Draft)"
"(Year)-[:HAS_ALLOCATION]->(Allocation)"
"(Year)-[:HAS_AVAILABILITY]->(Availability)"
"(Year)-[:HAS_STAGE]->(StageOfExtraction)"
"(Year)-[:HAS_RAINFALL]->(Rainfall)"
"(Year)-[:HAS_GROUND_WATER]->(GroundWaterAvailability)"
"(Year)-[:HAS_FUTURE_USE]->(FutureUse)"
"(Year)-[:HAS_ADDITIONAL_RECHARGE]->(AdditionalRecharge)"
"(Year)-[:HAS_AQUIFER]->(Aquifer)"

IMPORTANT RULES:
1. NEVER use exists() function - use "property IS NOT NULL" instead
2. Always convert state and district names to UPPERCASE
3. Use proper Neo4j 5+ syntax
4. Return only valid Cypher without code fences or explanations
5. If asked for recharge_worthy or non_recharge_worthy_area, mention it in the type property of (:Area)
6. If asked for total area, remember the type is "total"
7. If have to use BlockSummary node, simply return the value of what is asked
8. If no year is mentioned, use ONLY 2024 as the year value on the Year node
9. The unit of rainfall is "mm", unit of area is "ha" and units for other ground water data is "ham" 
"""

# ---------- Context Thresholds for Interpretive Insights ----------
CONTEXT_THRESHOLDS = {
    "rainfall": {
        "low": 500,      # mm - Below 500mm is considered low
        "normal_low": 1000,  # mm - 500-1000mm is below normal
        "normal": 1500,   # mm - 1000-1500mm is normal
        "normal_high": 2500, # mm - 1500-2500mm is above normal
        "high": 3000,     # mm - Above 3000mm is very high
    },
    "groundwater_draft": {
        "low": 10,        # ham - Below 10 ham is low usage
        "normal": 50,     # ham - 10-50 ham is normal
        "high": 100,      # ham - 50-100 ham is high
        "critical": 150   # ham - Above 150 ham is critical
    },
    "recharge": {
        "low": 20,        # ham - Below 20 ham is low
        "normal": 80,     # ham - 20-80 ham is normal
        "good": 150,      # ham - 80-150 ham is good
        "excellent": 200  # ham - Above 200 ham is excellent
    },
    "stage_of_extraction": {
        "safe": 70,       # % - Below 70% is safe
        "semi_critical": 90, # % - 70-90% is semi-critical
        "critical": 100,  # % - 90-100% is critical
        "over_exploited": 100 # % - Above 100% is over-exploited
    }
}

def interpret_value(value: float, metric_type: str) -> str:
    """
    Interpret numerical values based on context thresholds
    """
    if metric_type not in CONTEXT_THRESHOLDS:
        return "normal"
    
    thresholds = CONTEXT_THRESHOLDS[metric_type]
    
    if metric_type == "rainfall":
        if value < thresholds["low"]:
            return "very low"
        elif value < thresholds["normal_low"]:
            return "below normal"
        elif value < thresholds["normal"]:
            return "normal"
        elif value < thresholds["normal_high"]:
            return "above normal"
        elif value < thresholds["high"]:
            return "high"
        else:
            return "very high"
    
    elif metric_type == "groundwater_draft":
        if value < thresholds["low"]:
            return "low"
        elif value < thresholds["normal"]:
            return "normal"
        elif value < thresholds["high"]:
            return "high"
        elif value < thresholds["critical"]:
            return "concerning"
        else:
            return "critical"
    
    elif metric_type == "recharge":
        if value < thresholds["low"]:
            return "poor"
        elif value < thresholds["normal"]:
            return "normal"
        elif value < thresholds["good"]:
            return "good"
        else:
            return "excellent"
    
    elif metric_type == "stage_of_extraction":
        if value < thresholds["safe"]:
            return "safe"
        elif value < thresholds["semi_critical"]:
            return "semi-critical"
        elif value < thresholds["critical"]:
            return "critical"
        else:
            return "over-exploited"
    
    return "normal"

# ---------- Few-shot Examples for Better Cypher Generation ----------
FEW_SHOT_EXAMPLES = """
Example 1:
Question: "What is the rainfall in Kerala?"
Cypher: MATCH (c:Country {name:"INDIA"})-[:HAS_STATE]->(s:State {name:"KERALA"})-[:HAS_YEAR]->(y:Year {year:2024})-[:HAS_RAINFALL]->(r:Rainfall)
RETURN r.total AS rainfall

Example 2:
Question: "Show districts in Kerala with critical groundwater status"
Cypher: MATCH (s:State {name: "KERALA"})-[:HAS_District]->(d:District)-[:HAS_CATEGORY]->(c:Category {name: "Critical"}) RETURN d.name, c.name

Example 3:
Question: "Rainfall data for Kottayam district in 2023"
Cypher: MATCH (c:Country {name:"INDIA"})-[:HAS_STATE]->(:State {name:"KERALA"})-[:HAS_DISTRICT]->(d:District {name:"KOTTAYAM"})-[:HAS_YEAR]->(y:Year {year:2023})-[:HAS_RAINFALL]->(r:Rainfall)
RETURN d.name AS District, y.year AS Year, r.total AS Rainfall

Example 4:
Question: "Compare groundwater draft between Kerala and Tamil Nadu"
Cypher: MATCH (c:Country {name:"INDIA"})-[:HAS_STATE]->(s:State)-[:HAS_YEAR]->(y:Year)-[:HAS_DRAFT]->(d:Draft) WHERE s.name IN ["KERALA", "TAMIL NADU"]
RETURN s.name AS State, y.year AS Year, d.total AS Draft
ORDER BY y.year, State
"""

# ---------- Cypher Sanitization Functions ----------
def sanitize_cypher(cypher_query: str) -> str:
    """
    Sanitize Cypher query to fix common issues:
    1. Replace deprecated exists() with IS NOT NULL
    2. Uppercase state/district names
    3. Strip code fences and extra text
    4. Basic validation
    """
    # Remove code fences
    cypher_query = re.sub(r'```cypher\s*', '', cypher_query)
    cypher_query = re.sub(r'```\s*$', '', cypher_query)
    
    # Replace deprecated exists() syntax
    cypher_query = re.sub(r'exists\(([^)]+)\)', r'\1 IS NOT NULL', cypher_query)
    
    # Uppercase state and district names in quotes
    def uppercase_names(match):
        return match.group(0).upper()
    
    # Pattern to match quoted state/district names
    cypher_query = re.sub(r'"[^"]*"', uppercase_names, cypher_query)
    
    # Remove multiple whitespaces and newlines
    cypher_query = re.sub(r'\s+', ' ', cypher_query).strip()
    
    return cypher_query

def validate_cypher(cypher_query: str) -> Tuple[bool, str]:
    """
    Validate Cypher query for basic safety and correctness
    """
    cypher_upper = cypher_query.upper()
    
    # Check for RETURN statement
    if 'RETURN' not in cypher_upper:
        return False, "Query must contain a RETURN statement"
    
    # Block destructive operations
    destructive_ops = ['DELETE', 'CREATE', 'DROP', 'MERGE', 'SET']
    for op in destructive_ops:
        if op in cypher_upper:
            return False, f"Destructive operation '{op}' not allowed"
    
    # Basic syntax check
    if cypher_query.count('(') != cypher_query.count(')'):
        return False, "Unmatched parentheses in query"
    
    return True, "Valid"

# ---------- Enhanced Cypher Generation with Few-shot Examples ----------
def query_to_cypher(user_query: str) -> str:
    """
    Convert natural language to Cypher query with improved prompts and few-shot examples
    """
    prompt = f"""
    You are a Cypher query expert. Convert natural language questions into valid Neo4j Cypher queries.

    Schema:
    {SCHEMA}

    Few-shot Examples:
    {FEW_SHOT_EXAMPLES}

    STRICT RULES:
    1. NEVER use exists() - use "property IS NOT NULL" instead
    2. ALWAYS convert country/state/district names to UPPERCASE in quotes
    3. Return ONLY the Cypher query, no explanations or code fences
    4. Use Neo4j 5+ compatible syntax only
    5. Always include a RETURN statement
    6. If asked for recharge_worthy or non_recharge_worthy_area, mention it in the type property of (:Area)
    7. If asked for total area, remember the type is "total"
    8. If have to use BlockSummary node, simply return the value of what is asked
    9. If no year is mentioned, use ONLY 2024 as the year value on the Year node
    10. The unit of rainfall is "mm", unit of area is "ha" and units for other ground water data is "ham" 

    Question: "{user_query}"

    Cypher:
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        raw_cypher = response.text.strip()
        
        # Sanitize the generated Cypher
        sanitized_cypher = sanitize_cypher(raw_cypher)
        
        # Validate the query
        is_valid, error_msg = validate_cypher(sanitized_cypher)
        if not is_valid:
            print(f"⚠️ Cypher validation failed: {error_msg}")
            print(f"Original: {raw_cypher}")
            print(f"Sanitized: {sanitized_cypher}")
            return None
        
        return sanitized_cypher
    except Exception as e:
        print(f"❌ Error generating Cypher: {e}")
        return None

# ---------- Robust Neo4j Query Execution with Retries ----------
def run_cypher(cypher_query: str) -> List[Dict[str, Any]]:
    """
    Execute Cypher query with retry logic and timeout handling
    """
    if not cypher_query:
        return []
    
    for attempt in range(MAX_RETRIES):
        try:
            with driver.session() as session:
                result = session.run(cypher_query, timeout=QUERY_TIMEOUT)
                return [record.data() for record in result]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise e
            print(f"⚠️ Attempt {attempt + 1} failed, retrying: {e}")
            time.sleep(1)  # Wait before retry
    
    return []

# ---------- Enhanced Semantic Retrieval with Caching ----------
@lru_cache(maxsize=CACHE_SIZE)
def query_pinecone_index(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query Pinecone for semantic similarity with caching
    """
    try:
        query_vector = embed_model.encode([query_text])[0].tolist()
        result = pine_index.query(
            vector=query_vector, 
            top_k=top_k, 
            include_metadata=True
        )
        
        formatted_results = []
        for match in result.matches:
            formatted_results.append({
                "id": match.id,
                "score": round(match.score, 4),
                "metadata": dict(match.metadata) if match.metadata else {},
            })
        return formatted_results
    except Exception as e:
        print(f"❌ Pinecone query failed: {e}")
        return []

# ---------- Enhanced Role-Aware Response Generation ----------
def generate_graphrag_response(semantic_results: List[Dict], graph_results: List[Dict], 
                             query: str, cypher_used: Optional[str] = None, 
                             role: str = "general", debug_mode: bool = False) -> str:
    """
    Generate final response with role-aware insights and interpretive context
    """
    # Role-specific guidelines
    role_guidelines = {
        "farmer": (
            "- Translate technical data into simple, actionable insights\n"
            "- Suggest practical irrigation, crop, and water management strategies\n"
            "- Focus on what farmers can do with this information\n"
            "- Use everyday language and avoid technical jargon"
        ),
        "policymaker": (
            "- Assess sustainability risks and governance implications\n"
            "- Mention safe/critical/over-exploited categories and their policy significance\n"
            "- Highlight areas needing intervention or regulatory attention\n"
            "- Focus on broader regional and administrative implications"
        ),
        "researcher": (
            "- Highlight anomalies, trends, and areas needing further research\n"
            "- Provide precise technical details and statistical context\n"
            "- Mention data gaps, uncertainties, or research opportunities\n"
            "- Focus on analytical insights and scientific interpretation"
        ),
        "general": (
            "- Provide clear, everyday explanations using terms like high, low, or normal\n"
            "- Make technical information accessible to non-experts\n"
            "- Explain what the numbers mean in practical terms\n"
            "- Use simple comparisons and analogies when helpful"
        )
    }
    
    # Extract numerical values for interpretation
    interpretation_context = ""
    if graph_results:
        for result in graph_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and value > 0:
                    # Determine metric type based on key name
                    metric_type = None
                    if "rainfall" in key.lower():
                        metric_type = "rainfall"
                    elif "draft" in key.lower():
                        metric_type = "groundwater_draft"
                    elif "recharge" in key.lower():
                        metric_type = "recharge"
                    elif "stage" in key.lower() or "extraction" in key.lower():
                        metric_type = "stage_of_extraction"
                    
                    if metric_type:
                        interpretation = interpret_value(value, metric_type)
                        interpretation_context += f"{key}: {value} ({interpretation}) | "
    
    prompt = f"""
    You are an expert in groundwater assessment and management. Provide both factual data and interpretive insights based on the user's role.

    USER QUERY: {query}
    ROLE: {role.upper()}

    ROLE GUIDELINES:
    {role_guidelines.get(role.lower(), role_guidelines["general"])}

    CONTEXT INTERPRETATIONS:
    {interpretation_context if interpretation_context else "No numerical context available"}

    KNOWLEDGE GRAPH DATA:
    {graph_results if graph_results else "No graph results available"}

    SEMANTIC SEARCH DATA:
    {semantic_results if semantic_results else "No semantic results available"}

    REQUIREMENTS:
    - Answer directly and include retrieved values with correct units (mm for rainfall, ha for area, ham for groundwater data)
    - First refer the KONOWLEDGE GRAPH DATA, and generate the response. If no graph data is available, refer to SEMANTIC SEARCH DATA
    - Add interpretation: Is the value high, low, normal, critical, etc.?
    - Tailor explanation specifically to the {role.upper()} role guidelines
    - Keep response concise but informative (2-4 sentences)
    - Do not mention data sources or technical backend details
    - If results show concerning trends, mention them appropriately for the role
    - For multiple data points, prioritize the most relevant to the query

    RESPONSE STYLE BY ROLE:
    - FARMER: Simple language, practical advice, focus on actions
    - POLICYMAKER: Policy implications, governance perspective, regional impact
    - RESEARCHER: Technical accuracy, analytical insights, research gaps
    - GENERAL: Plain explanations, accessible language, clear context

    Response:
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        # Add debug information if requested
        if debug_mode:
            debug_info = f"\n\n--- DEBUG INFO ---"
            if graph_results:
                debug_info += f"\n🔗 Graph Results: {len(graph_results)} items"
            if semantic_results:
                debug_info += f"\n📚 Semantic Results: {len(semantic_results)} items"
            if cypher_used:
                debug_info += f"\n💾 Cypher: {cypher_used}"
            if interpretation_context:
                debug_info += f"\n📊 Interpretations: {interpretation_context}"
            answer += debug_info
        
        return answer
    except Exception as e:
        return f"❌ Error generating response: {e}"

# ---------- Simplified Role-based Formatting (UI Only) ----------
def format_response_for_role(response: str, role: str = "general") -> str:
    """
    Simple formatting adjustments for UI presentation only
    (Main role logic is now handled in generate_graphrag_response)
    """
    role_emojis = {
        "farmer": "🌾",
        "policymaker": "🏛️",
        "researcher": "🔬",
        "general": "💡"
    }
    
    emoji = role_emojis.get(role.lower(), "💡")
    return f"{emoji} {response}"

# ---------- Main Enhanced GraphRAG Function ----------
def graphrag_chatbot(user_query: str, role: str = "general", debug_mode: bool = False) -> Dict[str, Any]:
    """
    Enhanced GraphRAG function with role-aware insights and interpretive responses
    """
    start_time = time.time()

    lang_code = detect(user_query)
    if lang_code!="en":
        result = translator.translate(user_query, src=lang_code, dest="en")
        user_query = result.text
    
    # Handle direct Cypher input for debugging
    if user_query.lower().startswith("cypher:"):
        cypher = user_query[len("cypher:"):].strip()
        semantic_results = []
    else:
        # Get semantic results from Pinecone
        try:
            semantic_results = query_pinecone_index(user_query, top_k=5)
        except Exception as e:
            semantic_results = []
            print(f"⚠️ Semantic search failed: {e}")
        
        # Convert to Cypher for graph traversal
        cypher = query_to_cypher(user_query)

    # Execute graph query with robust error handling
    graph_results = []
    error_info = None
    
    if cypher:
        try:
            graph_results = run_cypher(cypher)
        except Exception as e:
            error_info = f"Graph query failed: {str(e)}"
            print(f"❌ {error_info}")
    else:
        error_info = "Could not generate valid Cypher query"

    # Generate role-aware response with interpretive insights
    if not semantic_results and not graph_results:
        if error_info:
            final_response = f"I couldn't find specific data for your query. This might be because the requested information isn't available in our current dataset or the query needs to be more specific."
        else:
            final_response = "No results found from either semantic search or knowledge graph."
    else:
        final_response = generate_graphrag_response(
            semantic_results, graph_results, user_query, cypher, role, debug_mode
        )
    
    # Apply simple UI formatting
    final_response = format_response_for_role(final_response, role)

    if lang_code!="en":
        final_result = translator.translate(final_response, src="en", dest=lang_code)
    
    processing_time = round(time.time() - start_time, 2)
    
    return {
        "query": user_query,
        "cypher_used": cypher,
        "semantic_results": semantic_results,
        "graph_results": graph_results,
        "final_answer": final_result.text if lang_code!="en" else final_response,
        "error": error_info,
        "processing_time": processing_time,
        "role": role,
        "debug_mode": debug_mode,
        "interpretation_applied": len(graph_results) > 0  # Flag indicating if interpretation was applied
    }

# ---------- Enhanced CLI Interface with Role Selection ----------
def get_user_role():
    """Get user role for personalized responses"""
    print("\nSelect your role for personalized insights:")
    print("1. 🌾 Farmer (practical recommendations and irrigation advice)")
    print("2. 🏛️ Policymaker (governance insights and sustainability assessment)")
    print("3. 🔬 Researcher (detailed analysis and research perspectives)")
    print("4. 💡 General user (clear explanations in everyday language)")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        role_map = {"1": "farmer", "2": "policymaker", "3": "researcher", "4": "general"}
        if choice in role_map:
            return role_map[choice]
        print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    print("🚀 Enhanced GraphRAG Chatbot with Role-Aware Insights")
    print("✨ Features: Role-based interpretations, contextual insights, robust Cypher handling")
    
    # Get user role
    user_role = get_user_role()
    role_descriptions = {
        "farmer": "practical farming insights",
        "policymaker": "governance and policy perspectives", 
        "researcher": "technical analysis and research context",
        "general": "clear everyday explanations"
    }
    print(f"\n👤 Role selected: {user_role.capitalize()} ({role_descriptions[user_role]})")
    
    print("\nCommands:")
    print("- Ask natural language questions about groundwater and rainfall")
    print("- 'cypher: <query>' for direct Cypher execution")
    print("- 'debug on/off' to toggle debug information")
    print("- 'role <farmer/policymaker/researcher/general>' to change role")
    print("- 'help' to see example queries")
    print("- 'exit' to quit\n")
    
    debug_mode = False
    
    try:
        while True:
            q = input(f"({user_role}) Ask: ").strip()
            
            if q.lower() in ["exit", "quit"]:
                break
            elif q.lower() == "help":
                print("\n📝 Example queries:")
                print("• What is the rainfall in Kerala?")
                print("• Show groundwater draft for Kottayam district")
                print("• Which districts in Karnataka have critical groundwater status?")
                print("• Compare recharge rates between Kerala and Tamil Nadu")
                print("• What's the stage of groundwater extraction in Punjab?\n")
                continue
            elif q.lower().startswith("debug "):
                debug_mode = "on" in q.lower()
                print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue
            elif q.lower().startswith("role "):
                new_role = q[5:].strip().lower()
                if new_role in ["farmer", "policymaker", "researcher", "general"]:
                    user_role = new_role
                    print(f"👤 Role changed to: {user_role.capitalize()} ({role_descriptions[user_role]})")
                else:
                    print("Invalid role. Use: farmer, policymaker, researcher, or general")
                continue
            elif not q:
                continue

            # Process query with role-aware insights
            result = graphrag_chatbot(q, role=user_role, debug_mode=debug_mode)
            
            # Display results
            print(f"\n🔍 Processed in {result['processing_time']}s | Role: {user_role.capitalize()}")
            if debug_mode:
                print(f"🔗 Cypher: {result['cypher_used'] or 'None'}")
                print(f"📊 Results: {len(result['semantic_results'])} semantic, {len(result['graph_results'])} graph")
                if result['interpretation_applied']:
                    print("🎯 Contextual interpretation applied")
            
            print(f"\n{result['final_answer']}")
            print("=" * 80)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        try:
            driver.close()
            print("🔌 Neo4j connection closed")
        except:
            pass