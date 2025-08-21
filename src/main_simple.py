import os
import uuid
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Optional
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API setup
app = FastAPI(title="Bikeshare Analytics Assistant", version="1.0.0")

# Initialize grader mode at startup
GRADER_MODE = os.getenv('GRADER_MODE', '0') == '1'
if GRADER_MODE:
    logger.info("Grader mode: ON")
else:
    logger.info("Grader mode: OFF")

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracing"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    sql: str
    result: Any
    error: Optional[str]

@app.get("/ping")
def ping():
    #Health check
    return {"status": "ok"}

@app.get("/ready")
def ready():
    #Readiness check - simple version
    return {"status": "ready", "message": "API is ready"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    #Simple test endpoint
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.info(f"Processing query request {request_id}: {req.question}")
    
    return QueryResponse(
        sql="SELECT 1 as test",
        result=[["test"]],
        error=None
    )
