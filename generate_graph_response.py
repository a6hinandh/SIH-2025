from neo4j import GraphDatabase
import google.generativeai as genai
from dotenv import load_dotenv
import os

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# Configure Gemini
genai.configure(api_key=GENAI_API_KEY)

# Connect Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# -----------------------------
# Knowledge Graph Schema (for Gemini prompt)
# -----------------------------
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

Notes:
- "India" is the only Country.
- States like Kerala, Tamil Nadu, Gujarat are (:State).
- Places like Ernakulam, Kottayam, Thrissur are (:District) of Kerala.
- Convert states and districts to CAPITAL LETTERS when querying.
"""

# -----------------------------
# Step 1: Convert natural language → Cypher
# -----------------------------
def query_to_cypher(user_query):
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

# -----------------------------
# Step 2: Run Cypher on Neo4j
# -----------------------------
def run_cypher(cypher_query):
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

# -----------------------------
# Step 3: Summarize results
# -----------------------------
def generate_response(answer, query):
    prompt = f"""
    You are given results of a Cypher query.
    Convert them into a meaningful human-readable answer.

    Context:
    {answer}

    Query: {query}
    Answer:
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# -----------------------------
# Chatbot Interface
# -----------------------------
def chatbot(user_query):
    # Allow direct Cypher input for debugging
    if user_query.lower().startswith("cypher:"):
        cypher = user_query[len("cypher:"):].strip()
    else:
        cypher = query_to_cypher(user_query)

    try:
        results = run_cypher(cypher)
        return results if results else "No results found.", cypher
    except Exception as e:
        return f"⚠️ Error running query:\nCypher: {cypher}\nError: {e}", cypher

# -----------------------------
# Main Loop
# -----------------------------
if __name__ == "__main__":
    print("🚀 Neo4j + Gemini Chatbot started")
    print("Type natural language queries OR `cypher: <query>` for direct queries")
    print("Type 'exit' to quit\n")

    while True:
        q = input("Ask: ")
        if q.lower() in ["exit", "quit"]:
            break

        answer, cypher = chatbot(q)
        print(f"📝 Cypher used: {cypher}")

        if isinstance(answer, str) and answer.startswith("⚠️"):
            print(answer)
        else:
            response = generate_response(answer, q)
            print("💡 Answer:", response)

