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



# # server.py
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from fastapi.middleware.cors import CORSMiddleware
# from graphrag import graphrag_chatbot
# from typing import Optional
# import time

# # Define enhanced input format with role support
# class ChatRequest(BaseModel):
#     query: str = Field(..., description="User's natural language query")
#     role: str = Field(default="general", description="User role: farmer, policymaker, researcher, or general")
#     debug: bool = Field(default=False, description="Enable debug mode for detailed response info")

# class ChatResponse(BaseModel):
#     query: str
#     role: str
#     final_answer: str
#     processing_time: float
#     cypher_used: Optional[str]
#     semantic_results_count: int
#     graph_results_count: int
#     interpretation_applied: bool
#     error: Optional[str]
#     debug_info: Optional[dict]

# # Initialize app
# app = FastAPI(
#     title="INGRES AI ChatBot API",
#     description="Enhanced GraphRAG chatbot with role-aware insights for groundwater data analysis",
#     version="2.0.0"
# )

# # Allow frontend (React) to connect
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # ⚠️ for dev only, restrict in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Role validation
# VALID_ROLES = ["farmer", "policymaker", "researcher", "general"]

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     """
#     Enhanced chat endpoint with role-aware responses
#     """
#     # Validate role
#     if request.role.lower() not in VALID_ROLES:
#         raise HTTPException(
#             status_code=400, 
#             detail=f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"
#         )
    
#     try:
#         # Process query with enhanced GraphRAG
#         result = graphrag_chatbot(
#             request.query, 
#             role=request.role.lower(), 
#             debug_mode=request.debug
#         )
        
#         # Prepare debug info if requested
#         debug_info = None
#         if request.debug:
#             debug_info = {
#                 "semantic_results": result.get("semantic_results", []),
#                 "graph_results": result.get("graph_results", []),
#                 "cypher_query": result.get("cypher_used"),
#                 "processing_details": {
#                     "role_applied": result.get("role"),
#                     "interpretation_applied": result.get("interpretation_applied", False)
#                 }
#             }
        
#         # Return structured response
#         return ChatResponse(
#             query=result["query"],
#             role=result["role"],
#             final_answer=result["final_answer"],
#             processing_time=result["processing_time"],
#             cypher_used=result.get("cypher_used"),
#             semantic_results_count=len(result.get("semantic_results", [])),
#             graph_results_count=len(result.get("graph_results", [])),
#             interpretation_applied=result.get("interpretation_applied", False),
#             error=result.get("error"),
#             debug_info=debug_info
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/")
# async def root():
#     return {
#         "message": "INGRES AI ChatBot API v2.0 is running", 
#         "status": "healthy",
#         "features": [
#             "Role-aware responses (farmer, policymaker, researcher, general)",
#             "Interpretive insights (high/low/normal context)",
#             "Enhanced GraphRAG with Pinecone + Neo4j + Gemini",
#             "Robust Cypher handling and validation"
#         ]
#     }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy", 
#         "service": "INGRES AI ChatBot v2.0",
#         "timestamp": time.time()
#     }

# @app.get("/roles")
# async def get_roles():
#     """
#     Get available user roles and their descriptions
#     """
#     return {
#         "roles": {
#             "farmer": {
#                 "description": "Practical recommendations and irrigation advice",
#                 "focus": "Actionable insights for agricultural decision-making",
#                 "example_response": "Simple language with farming strategies"
#             },
#             "policymaker": {
#                 "description": "Governance insights and sustainability assessment", 
#                 "focus": "Policy implications and regulatory perspectives",
#                 "example_response": "Administrative impact and intervention needs"
#             },
#             "researcher": {
#                 "description": "Detailed analysis and research perspectives",
#                 "focus": "Technical accuracy and analytical insights",
#                 "example_response": "Precise data with research opportunities"
#             },
#             "general": {
#                 "description": "Clear explanations in everyday language",
#                 "focus": "Accessible information for non-experts",
#                 "example_response": "Plain explanations using high/low/normal context"
#             }
#         }
#     }

# @app.get("/metrics")
# async def get_metrics_info():
#     """
#     Get information about supported metrics and their interpretation thresholds
#     """
#     return {
#         "supported_metrics": {
#             "rainfall": {
#                 "unit": "mm",
#                 "thresholds": {
#                     "very_low": "< 500mm",
#                     "below_normal": "500-1000mm", 
#                     "normal": "1000-1500mm",
#                     "above_normal": "1500-2500mm",
#                     "high": "2500-3000mm",
#                     "very_high": "> 3000mm"
#                 }
#             },
#             "groundwater_draft": {
#                 "unit": "ham", 
#                 "thresholds": {
#                     "low": "< 10 ham",
#                     "normal": "10-50 ham",
#                     "high": "50-100 ham", 
#                     "concerning": "100-150 ham",
#                     "critical": "> 150 ham"
#                 }
#             },
#             "recharge": {
#                 "unit": "ham",
#                 "thresholds": {
#                     "poor": "< 20 ham",
#                     "normal": "20-80 ham",
#                     "good": "80-150 ham",
#                     "excellent": "> 150 ham"
#                 }
#             },
#             "stage_of_extraction": {
#                 "unit": "%",
#                 "thresholds": {
#                     "safe": "< 70%",
#                     "semi_critical": "70-90%", 
#                     "critical": "90-100%",
#                     "over_exploited": "> 100%"
#                 }
#             }
#         }
#     }

# @app.post("/validate-query")
# async def validate_query(request: dict):
#     """
#     Validate a query without processing it - useful for frontend validation
#     """
#     query = request.get("query", "").strip()
#     role = request.get("role", "general").lower()
    
#     if not query:
#         return {"valid": False, "message": "Query cannot be empty"}
    
#     if role not in VALID_ROLES:
#         return {"valid": False, "message": f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"}
    
#     if len(query) > 500:
#         return {"valid": False, "message": "Query too long. Maximum 500 characters."}
    
#     return {"valid": True, "message": "Query is valid"}

# # Add this at the end to run with python server.py
# if __name__ == "__main__":
#     import uvicorn
#     print("🚀 Starting Enhanced INGRES AI ChatBot API v2.0...")
#     print("🔗 Server will be available at: http://localhost:8000")
#     print("📚 API docs at: http://localhost:8000/docs")
#     print("🎭 Role-aware responses: farmer, policymaker, researcher, general")
#     print("🎯 Interpretive insights: high/low/normal context for all metrics")
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
























































# # server.py - Enhanced with Data Visualization Endpoints
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from fastapi.middleware.cors import CORSMiddleware
# from graphrag import graphrag_chatbot, run_cypher
# from typing import Optional, List, Dict, Any
# import time
# import json

# # Define enhanced input formats
# class ChatRequest(BaseModel):
#     query: str = Field(..., description="User's natural language query")
#     role: str = Field(default="general", description="User role: farmer, policymaker, researcher, or general")
#     debug: bool = Field(default=False, description="Enable debug mode for detailed response info")

# class DataVisualizationRequest(BaseModel):
#     chart_type: str = Field(..., description="Type of chart: bar, line, pie, doughnut, radar, scatter")
#     comparison_type: str = Field(..., description="Type of comparison: state, district, yearly, metric")
#     states: Optional[List[str]] = Field(default=None, description="List of states to compare")
#     districts: Optional[List[str]] = Field(default=None, description="List of districts to compare")
#     years: Optional[List[int]] = Field(default=None, description="List of years to compare")
#     metrics: Optional[List[str]] = Field(default=None, description="List of metrics to visualize")
#     filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")

# class ChatResponse(BaseModel):
#     query: str
#     role: str
#     final_answer: str
#     processing_time: float
#     cypher_used: Optional[str]
#     semantic_results_count: int
#     graph_results_count: int
#     interpretation_applied: bool
#     error: Optional[str]
#     debug_info: Optional[dict]

# class DataVisualizationResponse(BaseModel):
#     chart_type: str
#     data: Dict[str, Any]
#     metadata: Dict[str, Any]
#     processing_time: float
#     error: Optional[str]

# # Initialize app
# app = FastAPI(
#     title="JALMITRA AI ChatBot API",
#     description="Enhanced GraphRAG chatbot with data visualization for groundwater analysis",
#     version="3.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Constants
# VALID_ROLES = ["farmer", "policymaker", "researcher", "general"]
# VALID_CHART_TYPES = ["bar", "line", "pie", "doughnut", "radar", "scatter", "area"]
# VALID_METRICS = ["rainfall", "recharge", "draft", "availability", "stage_extraction", "groundwater", "area"]

# # Data visualization query generators
# def generate_state_comparison_query(states: List[str], metric: str, year: int = 2024):
#     """Generate Cypher query for state-wise comparison"""
#     states_str = '", "'.join([s.upper() for s in states])
    
#     metric_mapping = {
#         "rainfall": ("Rainfall", "n.total"),
#         "recharge": ("Recharge", "n.total"),
#         "draft": ("Draft", "n.total"),
#         "availability": ("Availability", "n.total"),
#         "groundwater": ("GroundWaterAvailability", "n.total")
#     }

#     node_type, property_path = metric_mapping.get(metric, ("Rainfall", "n.total"))

#     return f'''
#     MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State)-[:HAS_YEAR]->(y:Year {{year:{year}}})-[:HAS_{metric.upper()}]->(n:{node_type})
#     WHERE s.name IN ["{states_str}"]
#     RETURN s.name AS state, {property_path} AS value
#     ORDER BY s.name
#     '''

# def generate_district_comparison_query(state: str, districts: List[str], metric: str, year: int = 2024):
#     """Generate Cypher query for district-wise comparison"""
#     districts_str = '", "'.join([d.upper() for d in districts])
    
#     metric_mapping = {
#         "rainfall": ("Rainfall", "n.total"),
#         "recharge": ("Recharge", "n.total"),
#         "draft": ("Draft", "n.total"),
#         "availability": ("Availability", "n.total"),
#         "groundwater": ("GroundWaterAvailability", "n.total")
#     }

#     node_type, property_path = metric_mapping.get(metric, ("Rainfall", "n.total"))

#     return f'''
#     MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})-[:HAS_DISTRICT]->(d:District)-[:HAS_YEAR]->(y:Year {{year:{year}}})-[:HAS_{metric.upper()}]->(n:{node_type})
#     WHERE d.name IN ["{districts_str}"]
#     RETURN d.name AS district, {property_path} AS value
#     ORDER BY d.name
#     '''

# def generate_yearly_trend_query(entity: str, entity_type: str, metric: str, years: List[int]):
#     """Generate Cypher query for yearly trend analysis"""
#     years_str = ', '.join(map(str, years))
    
#     metric_mapping = {
#         "rainfall": ("Rainfall", "n.total"),
#         "recharge": ("Recharge", "n.total"),
#         "draft": ("Draft", "n.total"),
#         "availability": ("Availability", "n.total"),
#         "groundwater": ("GroundWaterAvailability", "n.total")
#     }

#     node_type, property_path = metric_mapping.get(metric, ("Rainfall", "n.total"))

#     if entity_type == "state":
#         return f'''
#         MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"{entity.upper()}"}})-[:HAS_YEAR]->(y:Year)-[:HAS_{metric.upper()}]->(n:{node_type})
#         WHERE y.year IN [{years_str}]
#         RETURN y.year AS year, {property_path} AS value
#         ORDER BY y.year
#         '''
#     else:  # district
#         state_name = entity.split(',')[1].strip() if ',' in entity else "KERALA"  # Default fallback
#         district_name = entity.split(',')[0].strip()
#         return f'''
#         MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"{state_name.upper()}"}})-[:HAS_DISTRICT]->(d:District {{name:"{district_name.upper()}"}})-[:HAS_YEAR]->(y:Year)-[:HAS_{metric.upper()}]->(n:{node_type})
#         WHERE y.year IN [{years_str}]
#         RETURN y.year AS year, {property_path} AS value
#         ORDER BY y.year
#         '''

# def generate_multi_metric_query(entity: str, entity_type: str, metrics: List[str], year: int = 2024):
#     """Generate Cypher query for multi-metric comparison"""
#     if entity_type == "state":
#         base_match = f'MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"{entity.upper()}"}})-[:HAS_YEAR]->(y:Year {{year:{year}}})'
#     else:
#         state_name = entity.split(',')[1].strip() if ',' in entity else "KERALA"
#         district_name = entity.split(',')[0].strip()
#         base_match = f'MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"{state_name.upper()}"}})-[:HAS_DISTRICT]->(d:District {{name:"{district_name.upper()}"}})-[:HAS_YEAR]->(y:Year {{year:{year}}})'
    
#     metric_clauses = []
#     return_clauses = []
    
#     for metric in metrics:
#         metric_mapping = {
#             "rainfall": ("Rainfall", "r", "rainfall"),
#             "recharge": ("Recharge", "rec", "recharge"),
#             "draft": ("Draft", "dr", "draft"),
#             "availability": ("Availability", "av", "availability"),
#             "groundwater": ("GroundWaterAvailability", "gw", "groundwater")
#         }
        
#         if metric in metric_mapping:
#             node_type, alias, return_name = metric_mapping[metric]
#             metric_clauses.append(f'OPTIONAL MATCH (y)-[:HAS_{metric.upper()}]->({alias}:{node_type})')
#             return_clauses.append(f'{alias}.total AS {return_name}')
    
#     query = f'''
#     {base_match}
#     {' '.join(metric_clauses)}
#     RETURN {', '.join(return_clauses)}
#     '''
    
#     return query

# # Chart.js data formatters
# def format_bar_chart_data(data: List[Dict], x_field: str, y_field: str, title: str):
#     """Format data for bar chart"""
#     return {
#         "type": "bar",
#         "data": {
#             "labels": [item[x_field] for item in data],
#             "datasets": [{
#                 "label": title,
#                 "data": [item[y_field] if item[y_field] is not None else 0 for item in data],
#                 "backgroundColor": [
#                     "rgba(59, 130, 246, 0.8)",
#                     "rgba(16, 185, 129, 0.8)",
#                     "rgba(245, 158, 11, 0.8)",
#                     "rgba(239, 68, 68, 0.8)",
#                     "rgba(139, 92, 246, 0.8)",
#                     "rgba(236, 72, 153, 0.8)"
#                 ],
#                 "borderColor": [
#                     "rgba(59, 130, 246, 1)",
#                     "rgba(16, 185, 129, 1)",
#                     "rgba(245, 158, 11, 1)",
#                     "rgba(239, 68, 68, 1)",
#                     "rgba(139, 92, 246, 1)",
#                     "rgba(236, 72, 153, 1)"
#                 ],
#                 "borderWidth": 2,
#                 "borderRadius": 8,
#                 "borderSkipped": False
#             }]
#         },
#         "options": {
#             "responsive": True,
#             "plugins": {
#                 "title": {
#                     "display": True,
#                     "text": title,
#                     "font": {"size": 18, "weight": "bold"}
#                 },
#                 "legend": {
#                     "display": False
#                 }
#             },
#             "scales": {
#                 "y": {
#                     "beginAtZero": True,
#                     "grid": {"color": "rgba(0, 0, 0, 0.1)"},
#                     "ticks": {"font": {"size": 12}}
#                 },
#                 "x": {
#                     "grid": {"display": False},
#                     "ticks": {"font": {"size": 12}}
#                 }
#             }
#         }
#     }

# def format_line_chart_data(data: List[Dict], x_field: str, y_field: str, title: str):
#     """Format data for line chart"""
#     return {
#         "type": "line",
#         "data": {
#             "labels": [str(item[x_field]) for item in data],
#             "datasets": [{
#                 "label": title,
#                 "data": [item[y_field] if item[y_field] is not None else 0 for item in data],
#                 "borderColor": "rgba(59, 130, 246, 1)",
#                 "backgroundColor": "rgba(59, 130, 246, 0.1)",
#                 "borderWidth": 3,
#                 "fill": True,
#                 "tension": 0.4,
#                 "pointBackgroundColor": "rgba(59, 130, 246, 1)",
#                 "pointBorderColor": "#ffffff",
#                 "pointBorderWidth": 2,
#                 "pointRadius": 6,
#                 "pointHoverRadius": 8
#             }]
#         },
#         "options": {
#             "responsive": True,
#             "plugins": {
#                 "title": {
#                     "display": True,
#                     "text": title,
#                     "font": {"size": 18, "weight": "bold"}
#                 }
#             },
#             "scales": {
#                 "y": {
#                     "beginAtZero": True,
#                     "grid": {"color": "rgba(0, 0, 0, 0.1)"}
#                 },
#                 "x": {
#                     "grid": {"color": "rgba(0, 0, 0, 0.1)"}
#                 }
#             }
#         }
#     }

# def format_pie_chart_data(data: List[Dict], label_field: str, value_field: str, title: str):
#     """Format data for pie chart"""
#     return {
#         "type": "pie",
#         "data": {
#             "labels": [item[label_field] for item in data],
#             "datasets": [{
#                 "data": [item[value_field] if item[value_field] is not None else 0 for item in data],
#                 "backgroundColor": [
#                     "rgba(59, 130, 246, 0.8)",
#                     "rgba(16, 185, 129, 0.8)",
#                     "rgba(245, 158, 11, 0.8)",
#                     "rgba(239, 68, 68, 0.8)",
#                     "rgba(139, 92, 246, 0.8)",
#                     "rgba(236, 72, 153, 0.8)"
#                 ],
#                 "borderColor": "#ffffff",
#                 "borderWidth": 3,
#                 "hoverBorderWidth": 4
#             }]
#         },
#         "options": {
#             "responsive": True,
#             "plugins": {
#                 "title": {
#                     "display": True,
#                     "text": title,
#                     "font": {"size": 18, "weight": "bold"}
#                 },
#                 "legend": {
#                     "position": "right",
#                     "labels": {"font": {"size": 12}}
#                 }
#             }
#         }
#     }

# def format_multi_metric_data(data: List[Dict], metrics: List[str], title: str):
#     """Format data for multi-metric radar chart"""
#     if not data or not data[0]:
#         return {"error": "No data available"}
    
#     record = data[0]
    
#     return {
#         "type": "radar",
#         "data": {
#             "labels": [metric.title() for metric in metrics],
#             "datasets": [{
#                 "label": title,
#                 "data": [record.get(metric, 0) or 0 for metric in metrics],
#                 "backgroundColor": "rgba(59, 130, 246, 0.2)",
#                 "borderColor": "rgba(59, 130, 246, 1)",
#                 "borderWidth": 3,
#                 "pointBackgroundColor": "rgba(59, 130, 246, 1)",
#                 "pointBorderColor": "#ffffff",
#                 "pointBorderWidth": 2,
#                 "pointRadius": 6
#             }]
#         },
#         "options": {
#             "responsive": True,
#             "plugins": {
#                 "title": {
#                     "display": True,
#                     "text": title,
#                     "font": {"size": 18, "weight": "bold"}
#                 }
#             },
#             "scales": {
#                 "r": {
#                     "beginAtZero": True,
#                     "grid": {"color": "rgba(0, 0, 0, 0.1)"},
#                     "pointLabels": {"font": {"size": 12}}
#                 }
#             }
#         }
#     }

# # API Endpoints
# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     """Enhanced chat endpoint with role-aware responses"""
#     if request.role.lower() not in VALID_ROLES:
#         raise HTTPException(
#             status_code=400, 
#             detail=f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"
#         )
    
#     try:
#         result = graphrag_chatbot(
#             request.query, 
#             role=request.role.lower(), 
#             debug_mode=request.debug
#         )
        
#         debug_info = None
#         if request.debug:
#             debug_info = {
#                 "semantic_results": result.get("semantic_results", []),
#                 "graph_results": result.get("graph_results", []),
#                 "cypher_query": result.get("cypher_used"),
#                 "processing_details": {
#                     "role_applied": result.get("role"),
#                     "interpretation_applied": result.get("interpretation_applied", False)
#                 }
#             }
        
#         return ChatResponse(
#             query=result["query"],
#             role=result["role"],
#             final_answer=result["final_answer"],
#             processing_time=result["processing_time"],
#             cypher_used=result.get("cypher_used"),
#             semantic_results_count=len(result.get("semantic_results", [])),
#             graph_results_count=len(result.get("graph_results", [])),
#             interpretation_applied=result.get("interpretation_applied", False),
#             error=result.get("error"),
#             debug_info=debug_info
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/visualize", response_model=DataVisualizationResponse)
# async def data_visualization_endpoint(request: DataVisualizationRequest):
#     """Data visualization endpoint"""
#     start_time = time.time()
    
#     try:
#         # Validate request
#         if request.chart_type not in VALID_CHART_TYPES:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid chart type. Must be one of: {', '.join(VALID_CHART_TYPES)}"
#             )
        
#         # Generate appropriate query based on comparison type
#         cypher_query = None
#         chart_data = None
        
#         if request.comparison_type == "state" and request.states and request.metrics:
#             metric = request.metrics[0]
#             year = request.filters.get("year", 2024) if request.filters else 2024
#             cypher_query = generate_state_comparison_query(request.states, metric, year)
            
#         elif request.comparison_type == "district" and request.districts and request.metrics:
#             state = request.filters.get("state", "Kerala") if request.filters else "Kerala"
#             metric = request.metrics[0]
#             year = request.filters.get("year", 2024) if request.filters else 2024
#             cypher_query = generate_district_comparison_query(state, request.districts, metric, year)
            
#         elif request.comparison_type == "yearly" and request.years and request.metrics:
#             entity = request.filters.get("entity", "Kerala") if request.filters else "Kerala"
#             entity_type = request.filters.get("entity_type", "state") if request.filters else "state"
#             metric = request.metrics[0]
#             cypher_query = generate_yearly_trend_query(entity, entity_type, metric, request.years)
            
#         elif request.comparison_type == "metric" and request.metrics:
#             entity = request.filters.get("entity", "Kerala") if request.filters else "Kerala"
#             entity_type = request.filters.get("entity_type", "state") if request.filters else "state"
#             year = request.filters.get("year", 2024) if request.filters else 2024
#             cypher_query = generate_multi_metric_query(entity, entity_type, request.metrics, year)
        
#         if not cypher_query:
#             raise HTTPException(status_code=400, detail="Could not generate appropriate query")
        
#         # Execute query
#         raw_data = run_cypher(cypher_query)
        
#         if not raw_data:
#             raise HTTPException(status_code=404, detail="No data found for the specified parameters")
        
#         # Format data based on chart type
#         title = f"{request.metrics[0].title()} Comparison" if request.metrics else "Data Visualization"
        
#         if request.chart_type in ["bar", "column"]:
#             if request.comparison_type == "state":
#                 chart_data = format_bar_chart_data(raw_data, "state", "value", f"State-wise {request.metrics[0].title()}")
#             elif request.comparison_type == "district":
#                 chart_data = format_bar_chart_data(raw_data, "district", "value", f"District-wise {request.metrics[0].title()}")
#             elif request.comparison_type == "yearly":
#                 chart_data = format_bar_chart_data(raw_data, "year", "value", f"Yearly {request.metrics[0].title()} Trend")
                
#         elif request.chart_type == "line":
#             if request.comparison_type == "yearly":
#                 chart_data = format_line_chart_data(raw_data, "year", "value", f"Yearly {request.metrics[0].title()} Trend")
#             else:
#                 # Convert to line chart format
#                 x_field = "state" if request.comparison_type == "state" else "district"
#                 chart_data = format_line_chart_data(raw_data, x_field, "value", title)
                
#         elif request.chart_type in ["pie", "doughnut"]:
#             x_field = "state" if request.comparison_type == "state" else "district" if request.comparison_type == "district" else "year"
#             chart_data = format_pie_chart_data(raw_data, x_field, "value", title)
#             if request.chart_type == "doughnut":
#                 chart_data["type"] = "doughnut"
                
#         elif request.chart_type == "radar" and request.comparison_type == "metric":
#             chart_data = format_multi_metric_data(raw_data, request.metrics, f"Multi-metric Analysis")
        
#         processing_time = round(time.time() - start_time, 2)
        
#         return DataVisualizationResponse(
#             chart_type=request.chart_type,
#             data=chart_data,
#             metadata={
#                 "query_used": cypher_query,
#                 "data_points": len(raw_data),
#                 "comparison_type": request.comparison_type,
#                 "parameters": {
#                     "states": request.states,
#                     "districts": request.districts,
#                     "years": request.years,
#                     "metrics": request.metrics,
#                     "filters": request.filters
#                 }
#             },
#             processing_time=processing_time,
#             error=None
#         )
        
#     except Exception as e:
#         processing_time = round(time.time() - start_time, 2)
#         return DataVisualizationResponse(
#             chart_type=request.chart_type,
#             data={},
#             metadata={},
#             processing_time=processing_time,
#             error=str(e)
#         )

# @app.get("/visualization/options")
# async def get_visualization_options():
#     """Get available options for data visualization"""
#     return {
#         "chart_types": VALID_CHART_TYPES,
#         "comparison_types": ["state", "district", "yearly", "metric"],
#         "metrics": VALID_METRICS,
#         "sample_states": ["Kerala", "Karnataka", "Tamil Nadu", "Andhra Pradesh", "Maharashtra", "Gujarat"],
#         "sample_districts": {
#             "Kerala": ["Kottayam", "Ernakulam", "Thrissur", "Palakkad", "Kozhikode"],
#             "Karnataka": ["Bangalore", "Mysore", "Hubli", "Belgaum", "Mangalore"],
#             "Tamil Nadu": ["Chennai", "Madurai", "Coimbatore", "Trichy", "Salem"]
#         },
#         "available_years": [2020, 2021, 2022, 2023, 2024]
#     }

# @app.get("/")
# async def root():
#     return {
#         "message": "JALMITRA AI ChatBot API v3.0 with Data Visualization is running", 
#         "status": "healthy",
#         "features": [
#             "Role-aware responses (farmer, policymaker, researcher, general)",
#             "Interactive data visualization with Chart.js",
#             "State, district, yearly, and multi-metric comparisons",
#             "Enhanced GraphRAG with Pinecone + Neo4j + Gemini",
#             "Robust Cypher handling and validation"
#         ]
#     }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy", 
#         "service": "JALMITRA AI ChatBot v3.0",
#         "timestamp": time.time()
#     }

# if __name__ == "__main__":
#     import uvicorn
#     print("🚀 Starting Enhanced JALMITRA AI ChatBot API v3.0 with Data Visualization...")
#     print("🔗 Server will be available at: http://localhost:8000")
#     print("📚 API docs at: http://localhost:8000/docs")
#     print("📊 Data visualization endpoint: /visualize")
#     print("🎯 Visualization options: /visualization/options")
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
























# server.py - Fixed Data Visualization with Enhanced Neo4j Queries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from graphrag import graphrag_chatbot, run_cypher
from typing import Optional, List, Dict, Any
import time
import json

# Define enhanced input formats
class ChatRequest(BaseModel):
    query: str = Field(..., description="User's natural language query")
    role: str = Field(default="general", description="User role: farmer, policymaker, researcher, or general")
    debug: bool = Field(default=False, description="Enable debug mode for detailed response info")

class DataVisualizationRequest(BaseModel):
    chart_type: str = Field(..., description="Type of chart: bar, line, pie, doughnut, radar")
    comparison_type: str = Field(..., description="Type of comparison: state, district, yearly, metric")
    states: Optional[List[str]] = Field(default=None, description="List of states to compare")
    districts: Optional[List[str]] = Field(default=None, description="List of districts to compare")
    years: Optional[List[int]] = Field(default=None, description="List of years to compare")
    metrics: Optional[List[str]] = Field(default=None, description="List of metrics to visualize")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")

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

class DataVisualizationResponse(BaseModel):
    chart_type: str
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
    processing_time: float
    error: Optional[str]

# Initialize app
app = FastAPI(
    title="JALMITRA AI ChatBot API",
    description="Enhanced GraphRAG chatbot with data visualization for groundwater analysis",
    version="3.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
VALID_ROLES = ["farmer", "policymaker", "researcher", "general"]
VALID_METRICS = ["rainfall", "recharge", "draft", "availability", "groundwater"]

# Enhanced chart type validation with restricted combinations
CHART_TYPE_RESTRICTIONS = {
    "radar": ["metric"],  # Radar charts only work with multi-metric comparisons
    "pie": ["state", "district"],  # Pie charts work best with categorical comparisons
    "doughnut": ["state", "district"],  # Same as pie charts
}

def validate_chart_comparison_combo(chart_type: str, comparison_type: str) -> bool:
    """Validate if chart type is compatible with comparison type"""
    if chart_type in CHART_TYPE_RESTRICTIONS:
        return comparison_type in CHART_TYPE_RESTRICTIONS[chart_type]
    return True

# Enhanced Neo4j query generators with better error handling
def generate_state_comparison_query(states: List[str], metric: str, year: int = 2024):
    """Generate Cypher query for state-wise comparison with better node matching"""
    states_str = '", "'.join([s.upper() for s in states])
    
    metric_mapping = {
        "rainfall": ("RAINFALL", "Rainfall", "total"),
        "recharge": ("RECHARGE", "Recharge", "total"), 
        "draft": ("DRAFT", "Draft", "total"),
        "availability": ("AVAILABILITY", "Availability", "total"),
        "groundwater": ("GROUND_WATER", "GroundWaterAvailability", "total")
    }

    if metric not in metric_mapping:
        return None
        
    rel_type, node_type, property_name = metric_mapping[metric]

    return f'''
    MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State)-[:HAS_YEAR]->(y:Year {{year:{year}}})-[:HAS_{rel_type}]->(n:{node_type})
    WHERE s.name IN ["{states_str}"] AND n.{property_name} IS NOT NULL
    RETURN s.name AS entity, n.{property_name} AS value
    ORDER BY s.name
    '''

def generate_district_comparison_query(state: str, districts: List[str], metric: str, year: int = 2024):
    """Generate Cypher query for district-wise comparison"""
    districts_str = '", "'.join([d.upper() for d in districts])
    
    metric_mapping = {
        "rainfall": ("RAINFALL", "Rainfall", "total"),
        "recharge": ("RECHARGE", "Recharge", "total"),
        "draft": ("DRAFT", "Draft", "total"),
        "availability": ("AVAILABILITY", "Availability", "total"),
        "groundwater": ("GROUND_WATER", "GroundWaterAvailability", "total")
    }

    if metric not in metric_mapping:
        return None
        
    rel_type, node_type, property_name = metric_mapping[metric]

    return f'''
    MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"{state.upper()}"}})-[:HAS_DISTRICT]->(d:District)-[:HAS_YEAR]->(y:Year {{year:{year}}})-[:HAS_{rel_type}]->(n:{node_type})
    WHERE d.name IN ["{districts_str}"] AND n.{property_name} IS NOT NULL
    RETURN d.name AS entity, n.{property_name} AS value
    ORDER BY d.name
    '''

def generate_yearly_trend_query(entity: str, entity_type: str, metric: str, years: List[int]):
    """Generate Cypher query for yearly trend analysis"""
    years_str = ', '.join(map(str, years))
    
    metric_mapping = {
        "rainfall": ("RAINFALL", "Rainfall", "total"),
        "recharge": ("RECHARGE", "Recharge", "total"),
        "draft": ("DRAFT", "Draft", "total"),
        "availability": ("AVAILABILITY", "Availability", "total"),
        "groundwater": ("GROUND_WATER", "GroundWaterAvailability", "total")
    }

    if metric not in metric_mapping:
        return None
        
    rel_type, node_type, property_name = metric_mapping[metric]

    if entity_type == "state":
        return f'''
        MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"{entity.upper()}"}})-[:HAS_YEAR]->(y:Year)-[:HAS_{rel_type}]->(n:{node_type})
        WHERE y.year IN [{years_str}] AND n.{property_name} IS NOT NULL
        RETURN y.year AS entity, n.{property_name} AS value
        ORDER BY y.year
        '''
    else:  # district
        # For district, assume Kerala as default state since only Kerala districts are loaded
        return f'''
        MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"KERALA"}})-[:HAS_DISTRICT]->(d:District {{name:"{entity.upper()}"}})-[:HAS_YEAR]->(y:Year)-[:HAS_{rel_type}]->(n:{node_type})
        WHERE y.year IN [{years_str}] AND n.{property_name} IS NOT NULL
        RETURN y.year AS entity, n.{property_name} AS value
        ORDER BY y.year
        '''

def generate_multi_metric_query(entity: str, entity_type: str, metrics: List[str], year: int = 2024):
    """Generate Cypher query for multi-metric comparison"""
    
    if entity_type == "state":
        base_match = f'MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"{entity.upper()}"}})-[:HAS_YEAR]->(y:Year {{year:{year}}})'
    else:
        # For districts, use Kerala as the state
        base_match = f'MATCH (c:Country {{name:"INDIA"}})-[:HAS_STATE]->(s:State {{name:"KERALA"}})-[:HAS_DISTRICT]->(d:District {{name:"{entity.upper()}"}})-[:HAS_YEAR]->(y:Year {{year:{year}}})'
    
    metric_mapping = {
        "rainfall": ("RAINFALL", "Rainfall", "r", "rainfall"),
        "recharge": ("RECHARGE", "Recharge", "rec", "recharge"),
        "draft": ("DRAFT", "Draft", "dr", "draft"),
        "availability": ("AVAILABILITY", "Availability", "av", "availability"),
        "groundwater": ("GROUND_WATER", "GroundWaterAvailability", "gw", "groundwater")
    }
    
    metric_clauses = []
    return_clauses = []
    
    for metric in metrics:
        if metric in metric_mapping:
            rel_type, node_type, alias, return_name = metric_mapping[metric]
            metric_clauses.append(f'OPTIONAL MATCH (y)-[:HAS_{rel_type}]->({alias}:{node_type})')
            return_clauses.append(f'{alias}.total AS {return_name}')
    
    if not return_clauses:
        return None
    
    query = f'''
    {base_match}
    {' '.join(metric_clauses)}
    RETURN {', '.join(return_clauses)}
    '''
    
    return query

# Enhanced Chart.js data formatters
def format_bar_chart_data(data: List[Dict], title: str):
    """Format data for bar chart"""
    if not data:
        return None
        
    return {
        "type": "bar",
        "data": {
            "labels": [str(item.get('entity', 'Unknown')) for item in data],
            "datasets": [{
                "label": title,
                "data": [float(item.get('value', 0)) if item.get('value') is not None else 0 for item in data],
                "backgroundColor": [
                    "rgba(59, 130, 246, 0.8)",
                    "rgba(16, 185, 129, 0.8)",
                    "rgba(245, 158, 11, 0.8)",
                    "rgba(239, 68, 68, 0.8)",
                    "rgba(139, 92, 246, 0.8)",
                    "rgba(236, 72, 153, 0.8)",
                    "rgba(6, 182, 212, 0.8)",
                    "rgba(251, 113, 133, 0.8)"
                ],
                "borderColor": [
                    "rgba(59, 130, 246, 1)",
                    "rgba(16, 185, 129, 1)",
                    "rgba(245, 158, 11, 1)",
                    "rgba(239, 68, 68, 1)",
                    "rgba(139, 92, 246, 1)",
                    "rgba(236, 72, 153, 1)",
                    "rgba(6, 182, 212, 1)",
                    "rgba(251, 113, 133, 1)"
                ],
                "borderWidth": 2,
                "borderRadius": 8,
                "borderSkipped": False
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "title": {
                    "display": True,
                    "text": title,
                    "font": {"size": 18, "weight": "bold"}
                },
                "legend": {
                    "display": False
                }
            },
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "grid": {"color": "rgba(0, 0, 0, 0.1)"},
                    "ticks": {"font": {"size": 12}}
                },
                "x": {
                    "grid": {"display": False},
                    "ticks": {"font": {"size": 12}}
                }
            }
        }
    }

def format_line_chart_data(data: List[Dict], title: str):
    """Format data for line chart"""
    if not data:
        return None
        
    return {
        "type": "line",
        "data": {
            "labels": [str(item.get('entity', 'Unknown')) for item in data],
            "datasets": [{
                "label": title,
                "data": [float(item.get('value', 0)) if item.get('value') is not None else 0 for item in data],
                "borderColor": "rgba(59, 130, 246, 1)",
                "backgroundColor": "rgba(59, 130, 246, 0.1)",
                "borderWidth": 3,
                "fill": True,
                "tension": 0.4,
                "pointBackgroundColor": "rgba(59, 130, 246, 1)",
                "pointBorderColor": "#ffffff",
                "pointBorderWidth": 2,
                "pointRadius": 6,
                "pointHoverRadius": 8
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "title": {
                    "display": True,
                    "text": title,
                    "font": {"size": 18, "weight": "bold"}
                }
            },
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "grid": {"color": "rgba(0, 0, 0, 0.1)"}
                },
                "x": {
                    "grid": {"color": "rgba(0, 0, 0, 0.1)"}
                }
            }
        }
    }

def format_pie_chart_data(data: List[Dict], title: str):
    """Format data for pie chart"""
    if not data:
        return None
        
    return {
        "type": "pie",
        "data": {
            "labels": [str(item.get('entity', 'Unknown')) for item in data],
            "datasets": [{
                "data": [float(item.get('value', 0)) if item.get('value') is not None else 0 for item in data],
                "backgroundColor": [
                    "rgba(59, 130, 246, 0.8)",
                    "rgba(16, 185, 129, 0.8)",
                    "rgba(245, 158, 11, 0.8)",
                    "rgba(239, 68, 68, 0.8)",
                    "rgba(139, 92, 246, 0.8)",
                    "rgba(236, 72, 153, 0.8)",
                    "rgba(6, 182, 212, 0.8)",
                    "rgba(251, 113, 133, 0.8)"
                ],
                "borderColor": "#ffffff",
                "borderWidth": 3,
                "hoverBorderWidth": 4
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "title": {
                    "display": True,
                    "text": title,
                    "font": {"size": 18, "weight": "bold"}
                },
                "legend": {
                    "position": "right",
                    "labels": {"font": {"size": 12}}
                }
            }
        }
    }

def format_multi_metric_data(data: List[Dict], metrics: List[str], title: str):
    """Format data for multi-metric radar chart"""
    if not data or not data[0]:
        return None
    
    record = data[0]
    metric_values = []
    valid_metrics = []
    
    for metric in metrics:
        value = record.get(metric)
        if value is not None:
            metric_values.append(float(value))
            valid_metrics.append(metric.title())
        else:
            metric_values.append(0)
            valid_metrics.append(metric.title())
    
    return {
        "type": "radar",
        "data": {
            "labels": valid_metrics,
            "datasets": [{
                "label": title,
                "data": metric_values,
                "backgroundColor": "rgba(59, 130, 246, 0.2)",
                "borderColor": "rgba(59, 130, 246, 1)",
                "borderWidth": 3,
                "pointBackgroundColor": "rgba(59, 130, 246, 1)",
                "pointBorderColor": "#ffffff",
                "pointBorderWidth": 2,
                "pointRadius": 6
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "title": {
                    "display": True,
                    "text": title,
                    "font": {"size": 18, "weight": "bold"}
                }
            },
            "scales": {
                "r": {
                    "beginAtZero": True,
                    "grid": {"color": "rgba(0, 0, 0, 0.1)"},
                    "pointLabels": {"font": {"size": 12}}
                }
            }
        }
    }

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Enhanced chat endpoint with role-aware responses"""
    if request.role.lower() not in VALID_ROLES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"
        )
    
    try:
        result = graphrag_chatbot(
            request.query, 
            role=request.role.lower(), 
            debug_mode=request.debug
        )
        
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

@app.post("/visualize", response_model=DataVisualizationResponse)
async def data_visualization_endpoint(request: DataVisualizationRequest):
    """Enhanced data visualization endpoint with proper error handling"""
    start_time = time.time()
    
    try:
        # Validate chart type and comparison type combination
        if not validate_chart_comparison_combo(request.chart_type, request.comparison_type):
            valid_types = CHART_TYPE_RESTRICTIONS.get(request.chart_type, [])
            raise HTTPException(
                status_code=400,
                detail=f"Chart type '{request.chart_type}' is only compatible with: {', '.join(valid_types)}"
            )
        
        # Validate required parameters
        if not request.metrics or len(request.metrics) == 0:
            raise HTTPException(status_code=400, detail="At least one metric must be selected")
        
        # Generate appropriate query based on comparison type
        cypher_query = None
        
        if request.comparison_type == "state" and request.states:
            metric = request.metrics[0]
            year = request.filters.get("year", 2024) if request.filters else 2024
            # Ensure only years 2023-2024 are used
            if year not in [2023, 2024]:
                year = 2024
            cypher_query = generate_state_comparison_query(request.states, metric, year)
            
        elif request.comparison_type == "district" and request.districts:
            metric = request.metrics[0]
            year = request.filters.get("year", 2024) if request.filters else 2024
            # Ensure only years 2023-2024 are used
            if year not in [2023, 2024]:
                year = 2024
            cypher_query = generate_district_comparison_query("Kerala", request.districts, metric, year)
            
        elif request.comparison_type == "yearly" and request.years:
            entity = request.filters.get("entity", "Kerala") if request.filters else "Kerala"
            entity_type = request.filters.get("entity_type", "state") if request.filters else "state"
            metric = request.metrics[0]
            # Filter years to only 2023-2024
            valid_years = [y for y in request.years if y in [2023, 2024]]
            if not valid_years:
                valid_years = [2023, 2024]
            cypher_query = generate_yearly_trend_query(entity, entity_type, metric, valid_years)
            
        elif request.comparison_type == "metric" and request.metrics:
            entity = request.filters.get("entity", "Kerala") if request.filters else "Kerala"
            entity_type = request.filters.get("entity_type", "state") if request.filters else "state"
            year = request.filters.get("year", 2024) if request.filters else 2024
            # Ensure only years 2023-2024 are used
            if year not in [2023, 2024]:
                year = 2024
            cypher_query = generate_multi_metric_query(entity, entity_type, request.metrics, year)
        
        if not cypher_query:
            raise HTTPException(status_code=400, detail="Could not generate appropriate query for the given parameters")
        
        # Execute query
        raw_data = run_cypher(cypher_query)
        
        if not raw_data:
            processing_time = round(time.time() - start_time, 2)
            return DataVisualizationResponse(
                chart_type=request.chart_type,
                data=None,
                metadata={
                    "query_used": cypher_query,
                    "data_points": 0,
                    "comparison_type": request.comparison_type,
                    "parameters": {
                        "states": request.states,
                        "districts": request.districts, 
                        "years": request.years,
                        "metrics": request.metrics,
                        "filters": request.filters
                    }
                },
                processing_time=processing_time,
                error="No data found for the specified parameters. Please check if data exists for the selected entities and time period."
            )
        
        # Format data based on chart type
        chart_data = None
        title = f"{request.metrics[0].title()} Analysis"
        
        if request.comparison_type == "state":
            title = f"State-wise {request.metrics[0].title()}"
        elif request.comparison_type == "district":
            title = f"District-wise {request.metrics[0].title()}"
        elif request.comparison_type == "yearly":
            title = f"Yearly {request.metrics[0].title()} Trend"
        elif request.comparison_type == "metric":
            title = f"Multi-metric Analysis"
        
        if request.chart_type == "bar":
            chart_data = format_bar_chart_data(raw_data, title)
        elif request.chart_type == "line":
            chart_data = format_line_chart_data(raw_data, title)
        elif request.chart_type in ["pie", "doughnut"]:
            chart_data = format_pie_chart_data(raw_data, title)
            if request.chart_type == "doughnut" and chart_data:
                chart_data["type"] = "doughnut"
        elif request.chart_type == "radar" and request.comparison_type == "metric":
            chart_data = format_multi_metric_data(raw_data, request.metrics, title)
        
        processing_time = round(time.time() - start_time, 2)
        
        return DataVisualizationResponse(
            chart_type=request.chart_type,
            data=chart_data,
            metadata={
                "query_used": cypher_query,
                "data_points": len(raw_data),
                "comparison_type": request.comparison_type,
                "parameters": {
                    "states": request.states,
                    "districts": request.districts,
                    "years": request.years,
                    "metrics": request.metrics,
                    "filters": request.filters
                }
            },
            processing_time=processing_time,
            error=None
        )
        
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        return DataVisualizationResponse(
            chart_type=request.chart_type,
            data=None,
            metadata={},
            processing_time=processing_time,
            error=f"Error generating visualization: {str(e)}"
        )

@app.get("/visualization/options")
async def get_visualization_options():
    """Get available options for data visualization"""
    
    # Get available states from Neo4j
    states_query = """
    MATCH (c:Country {name:"INDIA"})-[:HAS_STATE]->(s:State)
    RETURN DISTINCT s.name AS state_name
    ORDER BY s.name
    """
    
    # Get available districts (Kerala only)
    districts_query = """
    MATCH (s:State {name:"KERALA"})-[:HAS_DISTRICT]->(d:District)
    RETURN DISTINCT d.name AS district_name
    ORDER BY d.name
    """
    
    try:
        states_data = run_cypher(states_query)
        districts_data = run_cypher(districts_query)
        
        available_states = [state['state_name'] for state in states_data] if states_data else []
        available_districts = [district['district_name'] for district in districts_data] if districts_data else []
        
    except Exception as e:
        # Fallback to sample data if query fails
        available_states = ["KERALA", "KARNATAKA", "TAMIL NADU", "ANDHRA PRADESH", "MAHARASHTRA", "GUJARAT"]
        available_districts = ["KOTTAYAM", "ERNAKULAM", "THRISSUR", "PALAKKAD", "KOZHIKODE"]
    
    return {
        "chart_types": [
            {"id": "bar", "label": "Bar Chart", "compatible_comparisons": ["state", "district", "yearly"]},
            {"id": "line", "label": "Line Chart", "compatible_comparisons": ["state", "district", "yearly"]},
            {"id": "pie", "label": "Pie Chart", "compatible_comparisons": ["state", "district"]},
            {"id": "doughnut", "label": "Doughnut Chart", "compatible_comparisons": ["state", "district"]},
            {"id": "radar", "label": "Radar Chart", "compatible_comparisons": ["metric"]}
        ],
        "comparison_types": ["state", "district", "yearly", "metric"],
        "metrics": VALID_METRICS,
        "sample_states": available_states[:10],  # Limit to first 10 for UI
        "sample_districts": {
            "Kerala": available_districts
        },
        "available_years": [2023, 2024],
        "restrictions": {
            "radar_chart_only_for_multi_metric": True,
            "pie_charts_for_categorical_only": True,
            "data_available_for": "2023-2024 only",
            "districts_available_for": "Kerala only"
        }
    }

@app.get("/")
async def root():
    return {
        "message": "JALMITRA AI ChatBot API v3.1 with Enhanced Data Visualization", 
        "status": "healthy",
        "features": [
            "Role-aware responses (farmer, policymaker, researcher, general)",
            "Interactive data visualization with Chart.js",
            "State, district, yearly, and multi-metric comparisons",
            "Enhanced Neo4j queries with proper error handling",
            "Chart type restrictions for better UX",
            "Data validation for 2023-2024 only"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "JALMITRA AI ChatBot v3.1",
        "timestamp": time.time(),
        "data_availability": {
            "years": [2023, 2024],
            "states": "All Indian states and UTs",
            "districts": "Kerala districts only"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced JALMITRA AI ChatBot API v3.1...")
    print("📊 Enhanced Data Visualization with Neo4j integration")
    print("🔗 Server will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("📊 Data visualization endpoint: /visualize") 
    print("🎯 Visualization options: /visualization/options")
    print("⚠️  Data available for: 2023-2024 years only")
    print("🗺️  Districts available for: Kerala only")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)