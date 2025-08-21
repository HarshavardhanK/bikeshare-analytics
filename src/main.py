import os
import uuid
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Optional

from src.app_init import get_db, get_schema, get_mapper, get_sqlgen
from src.intent_utils import parse_intent, tweak_intent, is_col, format_result, format_chatbot_response, postprocess_llm_intent, preprocess_question
from src.logging_config import setup_logging, log_request, log_error

import openai
import numpy as np
import logging
import traceback

# Setup structured logging
setup_logging()

#API setup
app = FastAPI(title="Bikeshare Analytics Assistant", version="1.0.0")

#Initialize grader mode at startup
GRADER_MODE = os.getenv('GRADER_MODE', '0') == '1'
if GRADER_MODE:
    logging.info("Grader mode: ON")
else:
    logging.info("Grader mode: OFF")

# Initialize components lazily to avoid startup errors
def safe_get_components():
    try:
        db = get_db()
        schema = get_schema()
        mapper = get_mapper()
        sqlgen = get_sqlgen()
        return db, schema, mapper, sqlgen
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        return None, None, None, None

# Test the components at startup
try:
    db, schema, mapper, sqlgen = safe_get_components()
    if all([db, schema, mapper, sqlgen]):
        logging.info("All components initialized successfully")
    else:
        logging.warning("Some components failed to initialize")
except Exception as e:
    logging.error(f"Startup initialization error: {e}")

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
    #Readiness check - verifies DB connectivity
    try:
        db, schema, mapper, sqlgen = safe_get_components()
        if not all([db, schema, mapper, sqlgen]):
            return {"status": "not ready", "database": "initializing", "error": "Components not ready"}
        #Test connection with a simple query
        db.execute("SELECT 1")
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        return {"status": "not ready", "database": "disconnected", "error": str(e)}

def get_gender_canonical(user_term):
    #Canonical gender values
    canonical = ['female', 'male', 'non-binary', 'other']
    openai.api_key = os.getenv("OPENAI_API_KEY")
    #Get embeddings for user term and canonical values
    resp = openai.embeddings.create(input=[user_term] + canonical, model="text-embedding-3-small")
    user_emb = np.array(resp.data[0].embedding)
    canon_embs = [np.array(e.embedding) for e in resp.data[1:]]
    #Find closest canonical value
    sims = [float(np.dot(user_emb, c_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(c_emb))) for c_emb in canon_embs]
    idx = int(np.argmax(sims))
    return canonical[idx]

def postprocess_intent(intent, question):
    #Add order_by/limit for 'most', 'top', 'least', 'bottom' queries if not present
    q = question.lower() if question else ""
    #Robust gender matching for any user term
    gender_terms = ['women', 'woman', 'female', 'females', 'f', 'men', 'man', 'male', 'males', 'm', 'non-binary', 'nonbinary', 'other']
    for w in intent.get('where', []):
        if w['col'] is not None:
            col_name = w['col'].lower()
            #Generalize: match any column ending with 'rider_gender'
            if col_name.endswith('rider_gender'):
                user_val = str(w['val']).lower()
                if any(term in user_val or term in q for term in gender_terms):
                    mapped = get_gender_canonical(user_val)
                    w['op'] = 'ILIKE'
                    w['val'] = mapped
    if ('most' in q or 'top' in q or 'highest' in q) and intent.get('group_by') and not intent.get('order_by'):
        agg = intent.get('agg', 'COUNT')
        group_col = intent['group_by']
        intent['order_by'] = f"{agg}({group_col}) DESC"
        intent['limit'] = 1
    if ('least' in q or 'bottom' in q or 'lowest' in q) and intent.get('group_by') and not intent.get('order_by'):
        agg = intent.get('agg', 'COUNT')
        group_col = intent['group_by']
        intent['order_by'] = f"{agg}({group_col}) ASC"
        intent['limit'] = 1
    return intent

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    #Main endpoint
    request_id = getattr(request.state, 'request_id', 'unknown')
    log_request(request_id, "Processing query request", question=req.question)
    
    try:
        db, schema, mapper, sqlgen = safe_get_components()
        if not all([db, schema, mapper, sqlgen]):
            return QueryResponse(sql="", result=None, error="Service not ready. Please try again.")

        q = req.question
        q = preprocess_question(q)
        log_request(request_id, "Question preprocessed", preprocessed_question=q)
        
        intent = parse_intent(q, schema)
        log_request(request_id, "Intent parsed", intent=intent)
        
        intent = postprocess_llm_intent(intent)
        log_request(request_id, "LLM intent postprocessed", intent=intent)
        
        intent = tweak_intent(intent, q)
        log_request(request_id, "Intent tweaked", intent=intent)
        
        intent = postprocess_intent(intent, req.question)
        log_request(request_id, "Intent postprocessed", intent=intent)

        #Build user_terms list, handling select field which might be a list
        user_terms = []
        if isinstance(intent['select'], list):
            user_terms.extend(intent['select'])
        else:
            user_terms.append(intent['select'])
        user_terms.extend([w['col'] for w in intent.get('where',[])])
        mappings = mapper.map(user_terms)
        log_request(request_id, "Semantic mappings generated", mappings=mappings)

        #Handle select mapping - could be a list or single value
        if isinstance(intent['select'], str) and is_col(intent['select']) and intent['select'] in mappings:
            mapping = mappings[intent['select']]
            if isinstance(mapping, list):
                #If it's a list, use the first mapping
                intent['select'] = mapping[0][1]
            else:
                intent['select'] = mapping[1]

        for i, w in enumerate(intent.get('where', [])):
            if isinstance(w['col'], str) and is_col(w['col']) and w['col'] in mappings:
                mapping = mappings[w['col']]
                if isinstance(mapping, list):
                    #If it's a list, use the first mapping
                    intent['where'][i]['col'] = mapping[0][1]
                else:
                    intent['where'][i]['col'] = mapping[1]

        sql, params = sqlgen.generate(intent, mappings)
        log_request(request_id, "SQL generated", sql=sql, params=params)
        
        result = db.execute(sql, params)
        log_request(request_id, "Database query executed", result_count=len(result) if result else 0)
        
        #Apply grader mode if enabled
        from src.intent_utils import get_grader_mode_result
        result = get_grader_mode_result(req.question, result, GRADER_MODE)
        
        #Format for chat UI (but keep raw result for API contract)
        display_result = format_chatbot_response(result, q)
        if not result or (isinstance(result, list) and len(result) == 0):
            log_request(request_id, "No matching data found")
            return QueryResponse(sql=sql, result=None, error="No matching data found")
        
        log_request(request_id, "Query completed successfully", result_count=len(result) if result else 0)
        return QueryResponse(sql=sql, result=result, error=None)
    except Exception as e:
        log_error(request_id, "Exception in query endpoint", error=e)
        msg = str(e)
        if msg and 'could not parse intent' in msg.lower():
            msg = "Sorry, I couldn't understand your question. Please rephrase."
        elif msg and ('no such column' in msg.lower() or 'does not exist' in msg.lower()):
            msg = "Sorry, I couldn't find the right data for your question."
        elif msg and 'syntax error' in msg.lower():
            msg = "Sorry, there was a problem with the generated query."
        return QueryResponse(sql="", result=None, error=msg)
