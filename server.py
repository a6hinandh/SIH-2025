# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from graphrag import graphrag_chatbot

# Define input format
class ChatRequest(BaseModel):
    query: str
    role: str = "general"
    debug: bool = False

# Initialize app
app = FastAPI()

# Allow frontend (React) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ for dev only, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    result = graphrag_chatbot(request.query, role=request.role, debug_mode=request.debug)
    return result

@app.get("/")
async def root():
    return {"message": "INGRES AI ChatBot API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "INGRES AI ChatBot"}

# Add this at the end to run with python server.py
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting INGRES AI ChatBot API...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)