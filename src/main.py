import os

from fastapi import FastAPI

from pydantic import BaseModel
from typing import Any, Optional

from src.app_init import get_db, get_schema, get_mapper, get_sqlgen

from src.intent_utils import parse_intent, tweak_intent, is_col, format_result, format_chatbot_response, postprocess_llm_intent

import openai
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
import traceback

#API setup
app = FastAPI()

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

def get_gender_canonical(user_term):
    # Canonical gender values
    canonical = ['female', 'male', 'non-binary', 'other']
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # Get embeddings for user term and canonical values
    resp = openai.embeddings.create(input=[user_term] + canonical, model="text-embedding-3-small")
    user_emb = np.array(resp.data[0].embedding)
    canon_embs = [np.array(e.embedding) for e in resp.data[1:]]
    # Find closest canonical value
    sims = [float(np.dot(user_emb, c_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(c_emb))) for c_emb in canon_embs]
    idx = int(np.argmax(sims))
    return canonical[idx]

def postprocess_intent(intent, question):
    # Add order_by/limit for 'most', 'top', 'least', 'bottom' queries if not present
    q = question.lower()
    # Robust gender matching for any user term
    gender_terms = ['women', 'woman', 'female', 'females', 'f', 'men', 'man', 'male', 'males', 'm', 'non-binary', 'nonbinary', 'other']
    for w in intent.get('where', []):
        col_name = w['col'].lower()
        # Generalize: match any column ending with 'rider_gender'
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
def query(req: QueryRequest):
    #Main endpoint
    try:
        db = get_db()
        schema = get_schema()
        mapper = get_mapper()
        sqlgen = get_sqlgen()

        q = req.question
        logging.debug(f"[query] Received question: {q}")
        intent = parse_intent(q, schema)
        logging.debug(f"[query] Parsed intent: {intent}")
        intent = postprocess_llm_intent(intent)
        logging.debug(f"[query] Postprocessed LLM intent: {intent}")
        intent = tweak_intent(intent, q)
        logging.debug(f"[query] Tweaked intent: {intent}")
        intent = postprocess_intent(intent, q)
        logging.debug(f"[query] Postprocessed intent: {intent}")

        user_terms = [intent['select']] + [w['col'] for w in intent.get('where',[])]
        mappings = mapper.map(user_terms)
        logging.debug(f"[query] Semantic mappings: {mappings}")

        if is_col(intent['select']) and intent['select'] in mappings:
            intent['select'] = mappings[intent['select']][1]

        for i, w in enumerate(intent.get('where', [])):
            if is_col(w['col']) and w['col'] in mappings:
                intent['where'][i]['col'] = mappings[w['col']][1]

        sql, params = sqlgen.generate(intent, mappings)
        logging.debug(f"[query] Generated SQL: {sql}")
        logging.debug(f"[query] SQL params: {params}")
        result = db.execute(sql, params)
        logging.debug(f"[query] DB result: {result}")
        result = format_result(intent, result, q)
        # Format for chat UI
        display_result = format_chatbot_response(result, q)
        if not result or (isinstance(result, list) and len(result) == 0):
            logging.debug(f"[query] No matching data found")
            return QueryResponse(sql=sql, result=None, error="No matching data found")
        logging.debug(f"[query] Returning result: {display_result}")
        return QueryResponse(sql=sql, result=display_result, error=None)
    except Exception as e:
        logging.error("Exception in /query: %s", e)
        logging.error(traceback.format_exc())
        msg = str(e)
        if 'could not parse intent' in msg.lower():
            msg = "Sorry, I couldn't understand your question. Please rephrase."
        elif 'no such column' in msg.lower() or 'does not exist' in msg.lower():
            msg = "Sorry, I couldn't find the right data for your question."
        elif 'syntax error' in msg.lower():
            msg = "Sorry, there was a problem with the generated query."
        return QueryResponse(sql="", result=None, error=msg)
