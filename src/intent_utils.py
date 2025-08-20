import os
import re
import json

import openai

from decimal import Decimal

#Intent parsing

def fix_json(txt):

    s = txt
    s = re.sub(r'"(\w+)"\s*,\s*"([<>=!]+)"', r'"\1":"\2"', s)
    s = re.sub(r'"(\w+)"\s*,\s*"([\w\-:\.]+)"', r'"\1":"\2"', s)
    s = re.sub(r',\s*([}\]])', r'\1', s)
    s = re.sub(r'([,{]\s*)(\w+)(\s*:)', r'\1"\2"\3', s)
    s = re.sub(r',\s*,', ',', s)
    s = re.sub(r'[\x00-\x1F]+', '', s)
    
    return s

def parse_intent(q, schema):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    schema_str = "\n".join([f"{t}: {', '.join(cols)}" for t, cols in schema.items()])
    # Add explicit superlative example
    prompt = f"""
Given the following database schema:
{schema_str}

Parse the following question into a JSON object with keys: select, from, where (list of dicts with col, op, val), group_by, join (list), agg, order_by, limit. Use only columns and tables from the schema. If aggregation is needed, set agg to SUM, AVG, etc. If a filter is needed, add to where. If the question asks for the youngest/oldest/highest/lowest, use ORDER BY and LIMIT 1, and select all requested columns. Example outputs:
{{"select": "trip_distance_km", "from": "trips", "where": [{{"col": "rider_gender", "op": "=", "val": "F"}}], "agg": "SUM", "group_by": null, "join": []}}
{{"select": ["trip_id", "rider_birth_year"], "from": "trips", "where": [{{"col": "started_at", "op": ">=", "val": "2025-06-01"}}, {{"col": "started_at", "op": "<", "val": "2025-07-01"}}], "order_by": "rider_birth_year DESC", "limit": 1, "group_by": null, "join": []}}

Question: {q}
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024,
            timeout=30,
            prompt_cache_key="intent-parsing-v1"
        )
        text = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
    except Exception as e:
        raise ValueError(f"OpenAI API error: {e}")
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        # Fallback: retry with a more explicit prompt for superlative queries
        if any(word in q.lower() for word in ["youngest", "oldest", "highest", "lowest", "top", "most", "least"]):
            prompt2 = prompt + "\nIf the question asks for the youngest/oldest, always use ORDER BY and LIMIT 1, and select all requested columns."
            resp2 = openai.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt2}],
                max_completion_tokens=1024,
                timeout=30,
                prompt_cache_key="intent-parsing-v1"
            )
            text2 = resp2.choices[0].message.content if resp2.choices and resp2.choices[0].message else ""
            match2 = re.search(r'\{[\s\S]*\}', text2)
            if not match2:
                raise ValueError(f"Could not parse intent from LLM response. Raw response: {text2}")
            raw_json = match2.group(0)
        else:
            raise ValueError(f"Could not parse intent from LLM response. Raw response: {text}")
    else:
        raw_json = match.group(0)
    try:
        intent = json.loads(raw_json)
        if not isinstance(intent, dict):
            raise ValueError(f"Intent is not a dict: {intent}")
    except Exception:
        repaired = fix_json(raw_json)
        intent = json.loads(repaired)
        if not isinstance(intent, dict):
            raise ValueError(f"Intent is not a dict after repair: {intent}")
    return intent

def postprocess_llm_intent(intent, question=None):
    # Fix select: if string with comma, split into list
    if isinstance(intent.get('select'), str) and ',' in intent['select']:
        intent['select'] = [s.strip() for s in intent['select'].split(',')]
    # Fix group_by: if string with comma, split into list
    if isinstance(intent.get('group_by'), str) and ',' in intent.get('group_by', ''):
        intent['group_by'] = [s.strip() for s in intent['group_by'].split(',')]
    # If select is a list and agg is present, remove agg (use ORDER BY/LIMIT 1 instead)
    if isinstance(intent.get('select'), list) and intent.get('agg'):
        intent.pop('agg', None)
    # For superlative queries (youngest/oldest/highest/lowest/top/most/least), add order_by/limit if not present
    if question:
        q = question.lower()
        superlative = any(word in q for word in ["youngest", "oldest", "highest", "lowest", "top", "most", "least"])
        if superlative and isinstance(intent.get('select'), list):
            # Guess the right column for order_by (e.g., birth_year for youngest/oldest)
            col = None
            if any('birth_year' in s for s in intent['select']):
                col = [s for s in intent['select'] if 'birth_year' in s][0]
                if 'youngest' in q or 'highest' in q or 'top' in q or 'most' in q:
                    intent['order_by'] = f"{col} DESC"
                else:
                    intent['order_by'] = f"{col} ASC"
            elif len(intent['select']) > 0:
                col = intent['select'][0]
                intent['order_by'] = f"{col} DESC"
            intent['limit'] = 1
        # Always remove agg for multi-column select
        if isinstance(intent.get('select'), list) and intent.get('agg'):
            intent.pop('agg', None)
    # Fix malformed join: if join is a string with '=', convert to dict
    join = intent.get('join')
    if join:
        if isinstance(join, list):
            new_join = []
            for j in join:
                if isinstance(j, str) and '=' in j:
                    if ' ON ' in j:
                        table, on = j.split(' ON ', 1)
                        new_join.append({'table': table.strip(), 'on': on.strip()})
                    else:
                        new_join.append({'table': intent.get('from'), 'on': j.strip()})
                else:
                    new_join.append(j)
            intent['join'] = new_join
        elif isinstance(join, str) and '=' in join:
            if ' ON ' in join:
                table, on = join.split(' ON ', 1)
                intent['join'] = [{'table': table.strip(), 'on': on.strip()}]
            else:
                intent['join'] = [{'table': intent.get('from'), 'on': join.strip()}]
    # Remove invalid IS NOT NULL/IS NOT patterns in where
    if 'where' in intent:
        new_where = []
        for w in intent['where']:
            if w.get('op', '').upper() in ('IS NOT NULL', 'IS NOT'):
                w.pop('val', None)
                new_where.append(w)
            elif w.get('op', '').lower() == 'between' and isinstance(w.get('val'), (list, tuple)) and len(w['val']) == 2:
                new_where.append(w)
            elif w.get('val') is not None:
                new_where.append(w)
        intent['where'] = new_where
    # For group_by/order_by/count: if group_by is a list, use only the first col in order_by/count
    if isinstance(intent.get('group_by'), list):
        group_col = intent['group_by'][0]
        if 'order_by' in intent and 'COUNT' in str(intent['order_by']):
            # Fix COUNT([list]) to COUNT(first_col)
            m = re.match(r"COUNT\(\[([^\]]+)\]\)", str(intent['order_by']))
            if m:
                first_col = m.group(1).split(',')[0].replace("'", "").strip()
                intent['order_by'] = f"COUNT({first_col}) DESC"
            else:
                intent['order_by'] = re.sub(r'COUNT\([^)]+\)', f'COUNT({group_col})', str(intent['order_by']))
    # For order_by: if order_by is a count on multiple columns, use only the first
    if 'order_by' in intent and 'COUNT' in str(intent['order_by']):
        m = re.match(r'COUNT\(([^)]+)\)', str(intent['order_by']))
        if m and ',' in m.group(1):
            first_col = m.group(1).split(',')[0].strip()
            intent['order_by'] = re.sub(r'COUNT\([^)]+\)', f'COUNT({first_col})', str(intent['order_by']))
    return intent

def tweak_intent(intent, q):
    #Add order/limit for 'most', 'top', etc
    
    s = q.lower()
    
    if ("most" in s or "top" in s or "highest" in s) and intent.get('group_by') and not intent.get('order_by'):
        agg = intent.get('agg', 'COUNT')
        group_col = intent['group_by']
        intent['order_by'] = f"{agg}({group_col}) DESC"
        intent['limit'] = 1

    if ("least" in s or "bottom" in s or "lowest" in s) and intent.get('group_by') and not intent.get('order_by'):
        agg = intent.get('agg', 'COUNT')
        group_col = intent['group_by']
        intent['order_by'] = f"{agg}({group_col}) ASC"
        intent['limit'] = 1
        
    return intent

def is_col(term):
    #Single word/col only
    return re.match(r'^[a-zA-Z0-9_.]+$', term) is not None

def format_result(intent, result, question):
    #Format output, add units
    
    def fmt_val(val):
        if isinstance(val, (float, Decimal)):
            return round(float(val), 2)
        
        return val

    def add_units(val, intent, q):
        
        s = q.lower()
        
        if 'kilometre' in s or 'km' in s:
            return f"{val} km"
        if 'minute' in s or 'ride time' in s:
            return f"{val} min"
        
        return val

    def get_col(col):
        
        if '.' in col:
            return col.split('.')[-1]
        return col

    if intent.get('group_by') and isinstance(result, list) and all(isinstance(row, (list, tuple)) and len(row) == 2 for row in result):
        group_key = get_col(intent['select'])
        agg_key = intent.get('agg', 'value').lower()
        
        formatted = [
            {str(group_key): fmt_val(row[0]), str(agg_key): fmt_val(row[1])}
            for row in result
        ]
        result = formatted
        
    else:
        if isinstance(result, list):
            formatted = [tuple(fmt_val(v) for v in row) if isinstance(row, (list, tuple)) else fmt_val(row) for row in result]
            if len(formatted) == 1 and isinstance(formatted[0], tuple) and len(formatted[0]) == 1:
                formatted = formatted[0][0]
            result = formatted

    if isinstance(result, (float, int, Decimal, str)):
        result = add_units(result, intent, question)
        
    return result

def format_chatbot_response(result, question=None):
    #Format the result as a human-readable string for the chatbot UI
    import pandas as pd
    if result is None:
        return "No data found."
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], dict):
            row = result[0]
            # Try to make a natural sentence for two columns
            if len(row) == 2:
                keys = list(row.keys())
                # Try to use the question for context if available
                if question:
                    return f"{row[keys[0]]} had {row[keys[1]]} {keys[1].replace('_', ' ')}."
                return f"{row[keys[0]]} ({row[keys[1]]})"
            return ", ".join(f"{k}: {v}" for k, v in row.items())
        elif all(isinstance(r, dict) for r in result):
            # Multi-row table
            df = pd.DataFrame(result)
            return df.to_markdown(index=False)
        else:
            return ", ".join(str(x) for x in result)
    elif isinstance(result, dict):
        return ", ".join(f"{k}: {v}" for k, v in result.items())
    else:
        return str(result)
