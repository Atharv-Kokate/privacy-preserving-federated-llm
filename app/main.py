import uvicorn
from fastapi import FastAPI
from app.api import router as inference_router
from app.training_api import router as training_router
from utils.config import API_HOST, API_PORT
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(
    title="Privacy-Preserving Federated LLM API",
    description="Healthcare Text Analytics Inference Server",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_router)
app.include_router(training_router)

# Ensure frontend directory exists
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Federated LLM API is running."}

if __name__ == "__main__":
    print(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=False)
