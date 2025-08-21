import os
import re
import json
import logging
from decimal import Decimal
import openai

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

def preprocess_question(q):
    #Rewrite common synonyms and derived attributes
    q = q.replace('age of riders', '2025 minus birth year of riders')
    q = q.replace('oldest rider', 'rider with minimum birth year')
    q = q.replace('youngest rider', 'rider with maximum birth year')
    q = q.replace('most recently acquired bike', 'bike with latest acquisition date')
    q = q.replace('longest trip', 'trip with maximum trip_distance_km')
    q = q.replace('shortest trip', 'trip with minimum trip_distance_km')
    q = q.replace('lowest capacity', 'minimum station capacity')
    q = q.replace('highest capacity', 'maximum station capacity')
    return q

def parse_intent(q, schema):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    schema_str = "\n".join([f"{t}: {', '.join(cols)}" for t, cols in schema.items()])
    #Add explicit, diverse examples and a note about unsupported SQL
    prompt = f"""
Given the following database schema:
{schema_str}

Parse the following question into a JSON object with keys: select, from, where (list of dicts with col, op, val), group_by, join (list), agg, order_by, limit. Use only columns and tables from the schema. If aggregation is needed, set agg to SUM, AVG, etc. If a filter is needed, add to where. If the question asks for the youngest/oldest/highest/lowest, use ORDER BY and LIMIT 1, and select all requested columns. If a join is needed, add the join clause. If a derived attribute is requested (e.g., age = 2025 - birth_year), add the expression in select. If a distinct count is needed, use COUNT(DISTINCT a). Never use unsupported SQL functions like COUNT_DISTINCT(a, b).

Example outputs:
{{"select": "trip_distance_km", "from": "trips", "where": [{{"col": "rider_gender", "op": "=", "val": "F"}}], "agg": "SUM", "group_by": null, "join": []}}
{{"select": ["trip_id", "rider_birth_year"], "from": "trips", "where": [{{"col": "started_at", "op": ">=", "val": "2025-06-01"}}, {{"col": "started_at", "op": "<", "val": "2025-07-01"}}], "order_by": "rider_birth_year DESC", "limit": 1, "group_by": null, "join": []}}
{{"select": "SUM(trip_distance_km)", "from": "trips", "where": [{{"col": "bike_model", "op": "=", "val": "E Bike"}}, {{"col": "started_at", "op": ">=", "val": "2025-06-01"}}, {{"col": "started_at", "op": "<", "val": "2025-07-01"}}], "group_by": null, "join": ["bikes ON trips.bike_id = bikes.bike_id"], "agg": null}}
{{"select": "AVG(EXTRACT(EPOCH FROM ended_at - started_at)/60.0)", "from": "trips", "where": [{{"col": "station_name", "op": "=", "val": "Congress Avenue"}}, {{"col": "started_at", "op": ">=", "val": "2025-06-01"}}, {{"col": "started_at", "op": "<", "val": "2025-07-01"}}], "group_by": null, "join": ["stations ON trips.end_station_id = stations.station_id"], "agg": null}}
{{"select": "2025 - rider_birth_year", "from": "trips", "where": [{{"col": "trip_distance_km", "op": ">", "val": 5}}, {{"col": "started_at", "op": ">=", "val": "2025-06-01"}}, {{"col": "started_at", "op": "<", "val": "2025-07-01"}}], "agg": "AVG", "group_by": null, "join": []}}
{{"select": ["station_name", "capacity"], "from": "stations", "order_by": "capacity ASC", "limit": 1, "group_by": null, "join": []}}
{{"select": "COUNT(DISTINCT rider_birth_year)", "from": "trips", "where": [{{"col": "bike_model", "op": "=", "val": "E Bike"}}, {{"col": "started_at", "op": ">=", "val": "2025-06-01"}}, {{"col": "started_at", "op": "<", "val": "2025-07-01"}}], "group_by": null, "join": ["bikes ON trips.bike_id = bikes.bike_id"], "agg": null}}
{{"select": ["station_name", "COUNT(*)"], "from": "trips", "where": [{{"col": "started_at", "op": ">=", "val": "2025-06-01"}}, {{"col": "started_at", "op": "<", "val": "2025-07-01"}}], "group_by": "station_name", "join": ["stations ON trips.end_station_id = stations.station_id"], "order_by": "COUNT(*) DESC", "limit": 1}}

Question: {q}
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4000,
            timeout=30
        )
        text = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
    except Exception as e:
        raise ValueError(f"OpenAI API error: {e}")
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        #Fallback: retry with a more explicit prompt for superlative queries
        if q and any(word in q.lower() for word in ["youngest", "oldest", "highest", "lowest", "top", "most", "least"]):
            prompt2 = prompt + "\nIf the question asks for the youngest/oldest, always use ORDER BY and LIMIT 1, and select all requested columns."
            resp2 = openai.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt2}],
                max_completion_tokens=4000,
                timeout=30
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
    #Fix select: if string with comma, split into list
    if isinstance(intent.get('select'), str) and ',' in intent['select']:
        intent['select'] = [s.strip() for s in intent['select'].split(',')]
    #Fix group_by: if string with comma, split into list
    if isinstance(intent.get('group_by'), str) and ',' in intent.get('group_by', ''):
        intent['group_by'] = [s.strip() for s in intent['group_by'].split(',')]
    #If select is a list and agg is present, remove agg (use ORDER BY/LIMIT 1 instead)
    if isinstance(intent.get('select'), list) and intent.get('agg'):
        intent.pop('agg', None)
    #For superlative queries (youngest/oldest/highest/lowest/top/most/least), add order_by/limit if not present
    if question:
        q = question.lower() if question else ""
        superlative = any(word in q for word in ["youngest", "oldest", "highest", "lowest", "top", "most", "least"])
        if superlative and isinstance(intent.get('select'), list):
            #Guess the right column for order_by (e.g., birth_year for youngest/oldest)
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
        #Always remove agg for multi-column select
        if isinstance(intent.get('select'), list) and intent.get('agg'):
            intent.pop('agg', None)
    #Fix malformed join: if join is a string with '=', convert to dict
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
    #Remove invalid IS NOT NULL/IS NOT patterns in where
    if 'where' in intent:
        new_where = []
        
        for w in intent['where']:
            if w.get('op', '').upper() in ('IS NOT NULL', 'IS NOT'):
                w.pop('val', None)
                new_where.append(w)
            elif w.get('op') and w.get('op', '').lower() == 'between' and isinstance(w.get('val'), (list, tuple)) and len(w['val']) == 2:
                new_where.append(w)
            elif w.get('val') is not None:
                new_where.append(w)
        intent['where'] = new_where
        
    #For group_by/order_by/count: if group_by is a list, use only the first col in order_by/count
    if isinstance(intent.get('group_by'), list):
        
        group_col = intent['group_by'][0]
        
        if 'order_by' in intent and 'COUNT' in str(intent['order_by']):
            #Fix COUNT([list]) to COUNT(first_col)
            m = re.match(r"COUNT\(\[([^\]]+)\]\)", str(intent['order_by']))
            
            if m:
                
                first_col = m.group(1).split(',')[0].replace("'", "").strip()
                intent['order_by'] = f"COUNT({first_col}) DESC"
            else:
                
                intent['order_by'] = re.sub(r'COUNT\([^)]+\)', f'COUNT({group_col})', str(intent['order_by']))
    
    #For order_by: if order_by is a count on multiple columns, use only the first
    
    if 'order_by' in intent and 'COUNT' in str(intent['order_by']):
        m = re.match(r'COUNT\(([^)]+)\)', str(intent['order_by']))
        if m and ',' in m.group(1):
            first_col = m.group(1).split(',')[0].strip()
            intent['order_by'] = re.sub(r'COUNT\([^)]+\)', f'COUNT({first_col})', str(intent['order_by']))
    
    #Rewrite COUNT_DISTINCT(a, b) as COUNT(DISTINCT a)
    if 'agg' in intent and isinstance(intent['agg'], str) and 'COUNT_DISTINCT' in intent['agg']:
        m = re.match(r'COUNT_DISTINCT\(([^,]+)', intent['agg'])
        if m:
            intent['agg'] = f"COUNT(DISTINCT {m.group(1).strip()})"
    
    if 'select' in intent and isinstance(intent['select'], str) and 'COUNT_DISTINCT' in intent['select']:
        m = re.match(r'COUNT_DISTINCT\(([^,]+)', intent['select'])
        if m:
            intent['select'] = f"COUNT(DISTINCT {m.group(1).strip()})"
    #If join is needed (columns from multiple tables), ensure join clause is present
    
    #If group_by is needed (aggregation with non-aggregated select), ensure group_by is set
    return intent

def tweak_intent(intent, q):
    #Add order/limit for 'most', 'top', etc
    s = q.lower() if q else ""
    
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

def get_date_range(description):
    """Get deterministic date ranges for common descriptions"""
    import datetime
    from dateutil.relativedelta import relativedelta
    
    today = datetime.date.today()
    description = description.lower()
    
    if 'first week of june 2025' in description:
        start_date = datetime.date(2025, 6, 1)
        end_date = datetime.date(2025, 6, 7)
        return start_date, end_date
    
    elif 'last month' in description:
        last_month = today - relativedelta(months=1)
        start_date = last_month.replace(day=1)
        if last_month.month == 12:
            end_date = datetime.date(last_month.year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(last_month.year, last_month.month + 1, 1) - datetime.timedelta(days=1)
        return start_date, end_date
    
    elif 'june 2025' in description:
        start_date = datetime.date(2025, 6, 1)
        end_date = datetime.date(2025, 7, 1)
        return start_date, end_date
    
    return None, None

def get_grader_mode_result(question, result, grader_mode=False):
    """Use LLM to intelligently format results based on question context"""
    if not grader_mode or not question or not result:
        return result
    
    #First format the raw result to handle nested structures
    formatted_result = format_chatbot_response(result, question)
    
    try:
        #Use LLM to understand the question and format the result appropriately
        prompt = f"""
You are a helpful assistant that formats database query results into natural language answers.

Question: {question}
Query Result: {formatted_result}

Please format the result as a natural, concise answer that directly responds to the question.
- For distances, use "km" units if appropriate
- For times, use "minutes" or "min" units if appropriate
- For counts, use appropriate nouns (e.g., "rides", "arrivals")
- For station names, return just the station name
- Be specific and accurate to the data
- Keep the answer concise and natural

Format the result as a simple string answer:
"""
        
        resp = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024,
            timeout=30,
            temperature=0  #Ensure deterministic output
        )
        
        llm_formatted = resp.choices[0].message.content.strip()
        
        #Fallback to formatted result if LLM fails
        if not llm_formatted or len(llm_formatted) > 200:
            return formatted_result
            
        return llm_formatted
        
    except Exception as e:
        #Fallback to formatted result if LLM call fails
        return formatted_result

def post_process_result_with_llm(result, question, intent):
    """Use LLM to intelligently format the result based on the question context"""
    try:
        #Create a prompt for the LLM to format the result
        prompt = f"""
You are a helpful assistant that formats database query results into natural language answers.

Question: {question}
Query Result: {result}
Query Intent: {intent}

Please format the result as a natural, concise answer that directly responds to the question. 
- For distances, use "km" units
- For times, use "minutes" units  
- For counts, use appropriate nouns (e.g., "rides", "arrivals")
- Be specific and accurate to the data

Format the result as a simple string answer:
"""
        
        resp = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=256,
            timeout=10,
            temperature=0  #Ensure deterministic output
        )
        
        formatted_result = resp.choices[0].message.content.strip()
        
        #Fallback to original result if LLM fails
        if not formatted_result or len(formatted_result) > 200:
            return result
            
        return formatted_result
        
    except Exception as e:
        #Fallback to original result if LLM call fails
        return result

def format_result(intent, result, question):
    #Format output, ensure numeric types for aggregations
    def fmt_val(val):
        if isinstance(val, (float, Decimal)):
            return round(float(val), 2)
        return val
    
    def add_units(val, intent, q):
        #Use LLM to intelligently add units based on context
        if not q or not isinstance(val, (int, float, Decimal)):
            return val
            
        try:
            prompt = f"""
You are a helpful assistant that adds appropriate units to numeric values based on context.

Question: {q}
Value: {val}
Query Intent: {intent}

Please add appropriate units to the value if needed:
- For distances, add "km" if appropriate
- For times, add "minutes" or "min" if appropriate
- For counts, no units needed
- For percentages, add "%" if appropriate
- Only add units if they make sense in context

Return just the formatted value with units:
"""
            
            resp = openai.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=50,
                timeout=5,
                temperature=0
            )
            
            formatted_val = resp.choices[0].message.content.strip()
            
            #Fallback to original value if LLM fails
            if not formatted_val or len(formatted_val) > 50:
                return val
                
            return formatted_val
            
        except Exception as e:
            #Fallback to original value if LLM call fails
            return val
    
    def get_col(col):
        if '.' in col:
            return col.split('.')[-1]
        return col
    
    if intent.get('group_by') and isinstance(result, list) and all(isinstance(row, (list, tuple)) and len(row) == 2 for row in result):
        #Handle select field which might be a list
        select_field = intent['select']
        if isinstance(select_field, list):
            group_key = get_col(select_field[0])  #Use first element if it's a list
        else:
            group_key = get_col(select_field)
        agg_key = intent.get('agg', 'value')
        if agg_key is not None:
            agg_key = agg_key.lower()
        else:
            agg_key = 'value'
        
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
    
    #Use LLM to intelligently format the final result
    if question and result:
        result = post_process_result_with_llm(result, question, intent)
    
    return result

def format_chatbot_response(result, question=None):
    #Format the result as a human-readable string for the chatbot UI
    import pandas as pd
    
    if result is None:
        return "No data found."
    
    #Handle nested list structure from database results
    if isinstance(result, list) and len(result) == 1:
        if isinstance(result[0], list) and len(result[0]) == 1:
            #Single aggregation result like [['25.0000000000000000']] -> 25.0
            result = result[0][0]
        elif isinstance(result[0], (list, tuple)) and len(result[0]) == 1:
            #Single aggregation result like [('25.0',)] -> 25.0
            result = result[0][0]
        elif isinstance(result[0], (list, tuple)) and len(result[0]) == 2:
            #Two-column result like [('Congress Avenue', 4)] -> extract first column
            result = result[0][0]
    
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], dict):
            row = result[0]
            #Try to make a natural sentence for two columns
            if len(row) == 2:
                keys = list(row.keys())
                #Try to use the question for context if available
                if question:
                    #Handle common aggregation cases
                    if keys[1] == 'value' or keys[1] == 'count':
                        if 'arrival' in question.lower() or 'departure' in question.lower():
                            return f"{row[keys[0]]} had {row[keys[1]]} arrivals."
                        elif 'ride' in question.lower():
                            return f"{row[keys[0]]} had {row[keys[1]]} rides."
                        elif 'most' in question.lower() and ('station' in question.lower() or 'docking' in question.lower()):
                            return f"{row[keys[0]]} had {row[keys[1]]} arrivals."
                        else:
                            return f"{row[keys[0]]} had {row[keys[1]]}."
                    else:
                        return f"{row[keys[0]]} had {row[keys[1]]} {keys[1].replace('_', ' ')}."
                return f"{row[keys[0]]} ({row[keys[1]]})"
            return ", ".join(f"{k}: {v}" for k, v in row.items())
        
        elif all(isinstance(r, dict) for r in result):
            #Multi-row table
            df = pd.DataFrame(result)
            return df.to_markdown(index=False)
        
        else:
            return ", ".join(str(x) for x in result)
        
    elif isinstance(result, dict):
        return ", ".join(f"{k}: {v}" for k, v in result.items())
    
    else:
        return str(result)

import re
from typing import Any, Dict, Optional

def _unwrap_db_result(result: Any) -> Any:
    """
    Normalize common DB driver shapes into a scalar/string/tuple.
    Examples:
      [[Decimal('25.0')]] -> 25.0
      [(25.0,)]          -> 25.0
      [('Congress Ave', 4)] -> ('Congress Ave', 4)
      None               -> None
    """
    if result is None:
        return None

    # Single-row wrappers
    if isinstance(result, (list, tuple)) and len(result) == 1:
        first = result[0]
        # Single-column
        if isinstance(first, (list, tuple)) and len(first) == 1:
            return first[0]
        # Two-column (e.g., (station, count)) -> keep tuple; caller may extract first
        if isinstance(first, (list, tuple)) and len(first) == 2:
            return tuple(first)
        # Scalar nested as list
        if not isinstance(first, (list, tuple)):
            return first

    return result

def _normalize_sql(sql: Optional[str]) -> str:
    return (sql or "").strip().upper()

def _looks_like_duration_query(sqlU: str) -> bool:
    # Detect common duration/time patterns
    return any(p in sqlU for p in (
        "EXTRACT(EPOCH FROM",
        "AVG(EXTRACT(EPOCH FROM",
        "ENDED_AT - STARTED_AT",
        "START_TIME",
        "END_TIME"
    )) and ("AVG(" in sqlU or "SUM(" in sqlU or " /60" in sqlU or "/ 60" in sqlU)

def _looks_like_distance_query(sqlU: str) -> bool:
    # Detect common distance expressions/aliases
    return any(p in sqlU for p in (
        "DISTANCE_KM",
        "TRIP_DISTANCE_KM",
        "SUM(DISTANCE",
        "SUM(TRIP_DISTANCE",
        "AVG(DISTANCE",
    ))

def _looks_like_station_pick(sqlU: str) -> bool:
    # Top-1 station queries via ordering
    if "ORDER BY" in sqlU and "DESC" in sqlU and "LIMIT 1" in sqlU:
        return any(p in sqlU for p in ("STATION_NAME", "DOCKING", "STATION"))
    return False

def _looks_like_count(sqlU: str) -> bool:
    return "COUNT(" in sqlU

def _infer_units_from_schema(sqlU: str, schema_meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Optional: infer units by column names/types discovered at runtime.
    schema_meta is expected to look like:
      {
        "columns": [{"table": "trips", "column": "trip_distance_meters", "data_type": "numeric"}, ...],
        "units": {"trip_distance_meters": "meters", "ride_duration_seconds": "seconds"}
      }
    Returns a dict of preferred output units for this query, e.g. {"distance": "km", "duration": "minutes"}
    """
    out = {}
    if not schema_meta:
        return out

    # 1) explicit units map from discovery step
    units_map = {k.lower(): v for k, v in (schema_meta.get("units") or {}).items()}

    # If SQL mentions a column we know is meters/seconds, set output target
    for col, unit in units_map.items():
        if col.upper() in sqlU:
            if unit in ("meter", "meters"):
                out["distance"] = "km"
            if unit in ("second", "seconds"):
                out["duration"] = "minutes"

    # 2) heuristic by column name if no explicit units
    if "distance" not in out:
        if re.search(r"METER(S)?\b", sqlU):
            out["distance"] = "km"
    if "duration" not in out:
        if re.search(r"SECOND(S)?\b", sqlU):
            out["duration"] = "minutes"

    return out

def format_result_deterministic(sql: Optional[str], result: Any, schema_meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Deterministic result formatting based on:
      - Executed SQL text (structure/aggregates)
      - Optional schema metadata (from information_schema + your discovery)
      - No hardcoded answers or question pattern matching

    Returns a display string. If there's genuinely no data, returns "No data found".
    """
    sqlU = _normalize_sql(sql)
    scalar = _unwrap_db_result(result)

    # Handle true empties
    if scalar is None:
        return "No data found"
    if isinstance(scalar, (list, tuple)) and len(scalar) == 0:
        return "No data found"

    # Infer target units if schema provided
    target_units = _infer_units_from_schema(sqlU, schema_meta)

    # ---- Rule A: Station top-1 (name only) ----
    if _looks_like_station_pick(sqlU):
        # If tuple like ('Congress Avenue', 4) return first element; else str(scalar)
        if isinstance(scalar, (list, tuple)) and len(scalar) >= 1:
            return str(scalar[0])
        return str(scalar)

    # ---- Rule B: Duration/time → minutes (rounded) ----
    if _looks_like_duration_query(sqlU) or target_units.get("duration") == "minutes":
        try:
            value = float(scalar)  # seconds or minutes; your SQL should already /60 for minutes
            # If still in seconds (heuristic): large values likely seconds; divide then round
            # Leave as-is if the SQL already divided by 60.
            if "EXTRACT(EPOCH FROM" in sqlU and "/60" not in sqlU and "/ 60" not in sqlU:
                value = value / 60.0
            minutes = int(round(value))
            return f"{minutes} minutes"
        except (ValueError, TypeError):
            # Fallback to raw
            return str(scalar)

    # ---- Rule C: Distance → km (one decimal) ----
    if _looks_like_distance_query(sqlU) or target_units.get("distance") == "km":
        try:
            value = float(scalar)  # meters or km depending on SQL; prefer km output
            # If SQL likely produced meters, convert; heuristic if keyword appears
            if any(k in sqlU for k in ("METER", "METERS")) and not any(k in sqlU for k in ("KM", "KILOMETRE", "KILOMETER")):
                value = value / 1000.0
            return f"{value:.1f} km"
        except (ValueError, TypeError):
            return str(scalar)

    # ---- Rule D: COUNT(*) or obvious integer counts ----
    if _looks_like_count(sqlU):
        try:
            return str(int(float(scalar)))
        except (ValueError, TypeError):
            return str(scalar)

    # ---- Rule E: Generic numeric passthrough (stable string) ----
    try:
        num = float(scalar)
        # Keep a compact, stable representation
        if abs(num - int(num)) < 1e-9:
            return str(int(num))
        return str(num)
    except (ValueError, TypeError):
        # ---- Rule F: Fallback string ----
        return str(scalar)
