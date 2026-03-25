from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from app.inference import InferenceHandler

# Global instance for inference
inference_handler = InferenceHandler()

# Model Definitions
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str
    risk_level: Optional[str] = "Unknown"
    tests: List[str] = []

router = APIRouter()

@router.on_event("startup")
async def startup_event():
    # Load model when API starts
    inference_handler.load_model()

@router.post("/medical_query", response_model=QueryResponse)
async def process_query(req: QueryRequest):
    """Answers a medical query using the federated trained LLM."""
    
    # Generate full raw response
    raw_response = inference_handler.generate_response(req.question)
    
    # Very basic parsing based on the desired format
    # In a full app, this would use regex or strict Pydantic parsing internally from the LLM
    response_text = raw_response
    risk_level = "Unknown"
    tests = []

    if "Risk Level:" in raw_response:
        try:
            risk_part = raw_response.split("Risk Level:")[1].split("\n")[0].strip()
            risk_level = risk_part
        except IndexError:
            pass
            
    if "Possible Tests:" in raw_response:
        try:
            tests_part = raw_response.split("Possible Tests:")[1].split("\n")[0].strip()
            tests = [t.strip() for t in tests_part.split(";")]
        except IndexError:
            pass

    return QueryResponse(
        response=response_text,
        risk_level=risk_level,
        tests=tests
    )
