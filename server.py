# # server.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from graphrag import graphrag_chatbot

# # Define input format
# class ChatRequest(BaseModel):
#     query: str
#     role: str = "general"
#     debug: bool = False

# # Initialize app
# app = FastAPI()

# # Allow frontend (React) to connect
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # ⚠️ for dev only, restrict in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     result = graphrag_chatbot(request.query, role=request.role, debug_mode=request.debug)
#     return result

# @app.get("/")
# async def root():
#     return {"message": "INGRES AI ChatBot API is running", "status": "healthy"}

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "service": "INGRES AI ChatBot"}

# # Add this at the end to run with python server.py
# if __name__ == "__main__":
#     import uvicorn
#     print("🚀 Starting INGRES AI ChatBot API...")
#     print("📡 Server will be available at: http://localhost:8000")
#     print("📚 API docs at: http://localhost:8000/docs")
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from graphrag import graphrag_chatbot
from typing import Optional
import time

# Define enhanced input format with role support
class ChatRequest(BaseModel):
    query: str = Field(..., description="User's natural language query")
    role: str = Field(default="general", description="User role: farmer, policymaker, researcher, or general")
    debug: bool = Field(default=False, description="Enable debug mode for detailed response info")

class ChatResponse(BaseModel):
    query: str
    role: str
    final_answer: str
    processing_time: float
    cypher_used: Optional[str]
    semantic_results_count: int
    graph_results_count: int
    interpretation_applied: bool
    error: Optional[str]
    debug_info: Optional[dict]

# Initialize app
app = FastAPI(
    title="INGRES AI ChatBot API",
    description="Enhanced GraphRAG chatbot with role-aware insights for groundwater data analysis",
    version="2.0.0"
)

# Allow frontend (React) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ for dev only, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Role validation
VALID_ROLES = ["farmer", "policymaker", "researcher", "general"]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Enhanced chat endpoint with role-aware responses
    """
    # Validate role
    if request.role.lower() not in VALID_ROLES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"
        )
    
    try:
        # Process query with enhanced GraphRAG
        result = graphrag_chatbot(
            request.query, 
            role=request.role.lower(), 
            debug_mode=request.debug
        )
        
        # Prepare debug info if requested
        debug_info = None
        if request.debug:
            debug_info = {
                "semantic_results": result.get("semantic_results", []),
                "graph_results": result.get("graph_results", []),
                "cypher_query": result.get("cypher_used"),
                "processing_details": {
                    "role_applied": result.get("role"),
                    "interpretation_applied": result.get("interpretation_applied", False)
                }
            }
        
        # Return structured response
        return ChatResponse(
            query=result["query"],
            role=result["role"],
            final_answer=result["final_answer"],
            processing_time=result["processing_time"],
            cypher_used=result.get("cypher_used"),
            semantic_results_count=len(result.get("semantic_results", [])),
            graph_results_count=len(result.get("graph_results", [])),
            interpretation_applied=result.get("interpretation_applied", False),
            error=result.get("error"),
            debug_info=debug_info
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "INGRES AI ChatBot API v2.0 is running", 
        "status": "healthy",
        "features": [
            "Role-aware responses (farmer, policymaker, researcher, general)",
            "Interpretive insights (high/low/normal context)",
            "Enhanced GraphRAG with Pinecone + Neo4j + Gemini",
            "Robust Cypher handling and validation"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "INGRES AI ChatBot v2.0",
        "timestamp": time.time()
    }

@app.get("/roles")
async def get_roles():
    """
    Get available user roles and their descriptions
    """
    return {
        "roles": {
            "farmer": {
                "description": "Practical recommendations and irrigation advice",
                "focus": "Actionable insights for agricultural decision-making",
                "example_response": "Simple language with farming strategies"
            },
            "policymaker": {
                "description": "Governance insights and sustainability assessment", 
                "focus": "Policy implications and regulatory perspectives",
                "example_response": "Administrative impact and intervention needs"
            },
            "researcher": {
                "description": "Detailed analysis and research perspectives",
                "focus": "Technical accuracy and analytical insights",
                "example_response": "Precise data with research opportunities"
            },
            "general": {
                "description": "Clear explanations in everyday language",
                "focus": "Accessible information for non-experts",
                "example_response": "Plain explanations using high/low/normal context"
            }
        }
    }

@app.get("/metrics")
async def get_metrics_info():
    """
    Get information about supported metrics and their interpretation thresholds
    """
    return {
        "supported_metrics": {
            "rainfall": {
                "unit": "mm",
                "thresholds": {
                    "very_low": "< 500mm",
                    "below_normal": "500-1000mm", 
                    "normal": "1000-1500mm",
                    "above_normal": "1500-2500mm",
                    "high": "2500-3000mm",
                    "very_high": "> 3000mm"
                }
            },
            "groundwater_draft": {
                "unit": "ham", 
                "thresholds": {
                    "low": "< 10 ham",
                    "normal": "10-50 ham",
                    "high": "50-100 ham", 
                    "concerning": "100-150 ham",
                    "critical": "> 150 ham"
                }
            },
            "recharge": {
                "unit": "ham",
                "thresholds": {
                    "poor": "< 20 ham",
                    "normal": "20-80 ham",
                    "good": "80-150 ham",
                    "excellent": "> 150 ham"
                }
            },
            "stage_of_extraction": {
                "unit": "%",
                "thresholds": {
                    "safe": "< 70%",
                    "semi_critical": "70-90%", 
                    "critical": "90-100%",
                    "over_exploited": "> 100%"
                }
            }
        }
    }

@app.post("/validate-query")
async def validate_query(request: dict):
    """
    Validate a query without processing it - useful for frontend validation
    """
    query = request.get("query", "").strip()
    role = request.get("role", "general").lower()
    
    if not query:
        return {"valid": False, "message": "Query cannot be empty"}
    
    if role not in VALID_ROLES:
        return {"valid": False, "message": f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"}
    
    if len(query) > 500:
        return {"valid": False, "message": "Query too long. Maximum 500 characters."}
    
    return {"valid": True, "message": "Query is valid"}

# Add this at the end to run with python server.py
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced INGRES AI ChatBot API v2.0...")
    print("🔗 Server will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("🎭 Role-aware responses: farmer, policymaker, researcher, general")
    print("🎯 Interpretive insights: high/low/normal context for all metrics")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)