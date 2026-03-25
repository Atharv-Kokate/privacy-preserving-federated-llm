import uvicorn
from fastapi import FastAPI
from app.api import router
from utils.config import API_HOST, API_PORT

app = FastAPI(
    title="Privacy-Preserving Federated LLM API",
    description="Healthcare Text Analytics Inference Server",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Federated LLM API is running."}

if __name__ == "__main__":
    print(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=False)
