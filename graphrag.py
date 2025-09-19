"""
graphrag.py -- GraphRAG example: Pinecone (semantic) + Neo4j (graph) + Gemini (LLM)
Updated for Pinecone v4.x API.
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
# 1) Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# 2) Pinecone v4
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in pc.list_indexes().names():
    raise SystemExit(f"Pinecone index '{PINECONE_INDEX}' not found. Create it or update PINECONE_INDEX in .env.")
pine_index = pc.Index(PINECONE_INDEX)

# 3) Embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# 4) Gemini (google.generativeai)
genai.configure(api_key=GENAI_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-flash")

# ---------- Helpers ----------
def embed_text(texts):
    """Return list of vectors corresponding to texts"""
    vecs = embed_model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vecs]

def semantic_retrieve(query_text, top_k=5):
    """Query Pinecone, return list of dicts {id, score, metadata}"""
    qvec = embed_text([query_text])[0]
    resp = pine_index.query(vector=qvec, top_k=top_k, include_metadata=True)
    results = []
    for m in resp.matches:   # v4: matches is an attribute, not dict
        results.append({
            "id": m.id,
            "score": m.score,
            "metadata": dict(m.metadata) if m.metadata else {},
        })
    return results

def query_to_cypher(user_query, schema_text):
    """
    Ask Gemini to produce a Cypher query when the question is structural/time-series.
    Returns Cypher string or None.
    """
    prompt = f"""
You are an assistant that converts groundwater-related natural language questions into Cypher queries.
Schema:
{schema_text}

Instructions:
- If the user question requires structured traversal or exact comparisons (e.g. "Which districts moved from Safe to Semi-Critical in the last 5 years?"),
  produce a single Cypher query only (no explanation).
- If the question is purely open-ended or requires long-form narrative (e.g. "explain groundwater trends"), respond with the single token: NO_CYPHER.
- Use the node/relationship naming style from the schema. Include YEAR filters when relevant.
Return ONLY the Cypher query or the token NO_CYPHER.
User question: \"{user_query}\"
"""
    try:
        res = llm.generate_content(prompt)
        text = res.text.strip()
        if text.upper().startswith("NO_CYPHER"):
            return None
        if any(k in text.upper() for k in ["MATCH ", "RETURN ", "WHERE ", "CREATE ", "WITH "]):
            text = text.replace("```cypher", "").replace("```", "").strip()
            return text
        return None
    except Exception as e:
        print("Error calling LLM for cypher conversion:", e)
        return None

def run_cypher(cypher_query):
    """Run query against Neo4j and return list of dicts"""
    if not cypher_query:
        return []
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            rows = [record.data() for record in result]
            return rows
    except Exception as e:
        return {"__error__": str(e), "__trace__": traceback.format_exc()}

def llm_fuse_and_answer(query, semantic_hits, graph_hits):
    """
    Merge semantic retrieval hits and graph results into a final answer.
    """
    prompt = f"""
You are an expert groundwater assistant that must merge semantic retrieval results and knowledge-graph query outputs into a clear, factual answer.

User question:
{query}

Semantic retrieval (top hits): {json.dumps(semantic_hits, indent=2)}

Graph query results: {json.dumps(graph_hits, indent=2)}

Rules:
- Use only the information provided above. Do not hallucinate.
- If graph results contain an "__error__" field, explain that the graph query failed and return the semantic-only summary.
- When possible, show the graph path or district names returned.
- Keep answer concise (max ~6 sentences) and technical.
- If no results were found, say "No results found" politely.

Provide the answer now:
"""
    r = llm.generate_content(prompt)
    return r.text

# ---------- Knowledge Graph Schema summary ----------
SCHEMA_SUMMARY = """
Nodes:
(:State {name, uuid})
(:District {name, uuid})
(:Block {name, uuid})
(:Category {name}) -- categories: Safe, Semi-Critical, Critical, Over-Exploited
Relationships with properties:
- (d:District)-[:HAS_CATEGORY {year:int}]->(c:Category)
- (d:District)-[:HAS_RAINFALL {year:int, value:float}]->(:Rainfall)
- (State)-[:HAS_DISTRICT]->(District)
- (District)-[:HAS_BLOCK]->(Block)
"""

# ---------- Main GraphRAG pipeline ----------
def graphrag_query(user_query):
    semantic_hits = semantic_retrieve(user_query, top_k=5)
    cypher = query_to_cypher(user_query, SCHEMA_SUMMARY)
    if cypher:
        graph_results = run_cypher(cypher)
    else:
        graph_results = []
    final_answer = llm_fuse_and_answer(user_query, semantic_hits, graph_results)
    return {
        "query": user_query,
        "cypher_attempted": bool(cypher),
        "cypher": cypher,
        "semantic_hits": semantic_hits,
        "graph_results": graph_results,
        "final_answer": final_answer
    }

# ---------- CLI ----------
if __name__ == "__main__":
    print("GraphRAG CLI. Type a question and press enter. Type exit to quit.")
    try:
        while True:
            q = input("Ask: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            out = graphrag_query(q)
            print("\n=== FINAL ANSWER ===\n")
            print(out["final_answer"])
            print("\n=== DEBUG (cypher & hits) ===")
            print("Cypher attempted:", out["cypher_attempted"])
            if out["cypher"]:
                print("Cypher:", out["cypher"])
            print("Semantic hits:", json.dumps(out["semantic_hits"], indent=2))
            print("Graph results:", json.dumps(out["graph_results"], indent=2))
            print("\n")
    finally:
        try:
            driver.close()
        except:
            pass
