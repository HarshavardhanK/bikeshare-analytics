import os
import math
from typing import Iterable, Optional, Dict, Set

import openai
import numpy as np

def _norm(s: str) -> str:
    """Normalize string for comparison"""
    return "".join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace())

def _jaro_winkler(a: str, b: str) -> float:
    """Simple deterministic similarity scorer"""
    a, b = _norm(a), _norm(b)
    if a == b: 
        return 1.0
    if a and b and (a.startswith(b) or b.startswith(a)): 
        return 0.92
    # Basic token overlap
    aset, bset = set(a.split()), set(b.split())
    if aset and bset:
        return len(aset & bset) / len(aset | bset)
    return 0.0

def resolve_value_to_domain(
    column_fqn: str,  # e.g., "trips.rider_gender"
    value_text: str,
    domain_values: Iterable[str],
    embeddings=None  # optional local/LLM embeddings; must be deterministic fallback
) -> Optional[str]:
    """
    Return the best-matching *domain value* for a column, or None.
    Deterministic ranking: exact (case-insensitive) > prefix > fuzzy > (optional) embeddings tie-break.
    """
    vt = _norm(value_text)
    candidates = list({_norm(v) for v in domain_values if v is not None})

    # 1) exact ILIKE match
    for v_raw in candidates:
        if vt == v_raw:
            return v_raw

    # 2) common expansions (data-driven, not hardcoded synonyms): 
    # Try single-letter abbreviations if there's a unique domain starting with it.
    if len(vt) == 1:
        starts = [v for v in candidates if v.startswith(vt)]
        if len(starts) == 1:
            return starts[0]  # 'f' -> 'female' if unique

    # 3) prefix match
    pref = [v for v in candidates if v.startswith(vt) or vt.startswith(v)]
    if len(pref) == 1:
        return pref[0]

    # 4) fuzzy
    scored = sorted(candidates, key=lambda v: _jaro_winkler(vt, v), reverse=True)
    if scored and _jaro_winkler(vt, scored[0]) >= 0.8:
        return scored[0]

    return None

#Maps user terms to schema columns using embedding similarity
class SemanticMapper:
    def __init__(self, schema, db_connection=None):
        self.schema = schema
        self.db_connection = db_connection
        self.columns = self._get_columns()
        self.column_embeddings = self._embed_columns()
        self.domain_cache = {}  # Cache for categorical column domains
        self._populate_domain_cache()

    def _populate_domain_cache(self):
        """Populate domain cache for low-cardinality categorical columns"""
        if not self.db_connection:
            return
            
        try:
            # Find candidate categorical columns
            categorical_columns = []
            for table, columns in self.schema.items():
                for col in columns:
                    # Simple heuristic: text columns are likely categorical
                    if any(keyword in col.lower() for keyword in ['gender', 'status', 'type', 'category', 'condition']):
                        categorical_columns.append(f"{table}.{col}")
            
            # Fetch domains for categorical columns
            for column_fqn in categorical_columns:
                table, column = column_fqn.split('.')
                try:
                    # Get distinct values for this column
                    query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 100"
                    result = self.db_connection.execute(query)
                    if result:
                        domain_values = set()
                        for row in result:
                            if isinstance(row, (list, tuple)) and len(row) > 0:
                                domain_values.add(str(row[0]))
                            else:
                                domain_values.add(str(row))
                        self.domain_cache[column_fqn] = domain_values
                except Exception as e:
                    # Skip columns that can't be queried
                    continue
                    
        except Exception as e:
            # If domain caching fails, continue without it
            pass

    def _get_columns(self):
        cols = []
        for table, columns in self.schema.items():
            for col in columns:
                cols.append((table, col))
        return cols

    def _embed_columns(self):
        names = [col[1] for col in self.columns]
        return self._get_embeddings(names)

    def _get_embeddings(self, texts):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        resp = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        return [np.array(e.embedding) for e in resp.data]

    def map(self, user_terms):
        #Flatten user_terms if they contain lists
        flat_terms = []
        term_mapping = {}  #Map flat index back to original term
        
        for i, term in enumerate(user_terms):
            if isinstance(term, list):
                for j, subterm in enumerate(term):
                    flat_terms.append(str(subterm))
                    term_mapping[len(flat_terms) - 1] = (i, j)
            else:
                flat_terms.append(str(term))
                term_mapping[len(flat_terms) - 1] = (i, None)
        
        #Ensure all terms are strings and not empty
        flat_terms = [term for term in flat_terms if term and term.strip()]
        
        if not flat_terms:
            return {}
        
        #Use LLM to enhance term understanding for better mapping
        enhanced_terms = self._enhance_terms_with_llm(flat_terms)
        
        user_embeds = self._get_embeddings(enhanced_terms)
        mapping = {}
        
        for i, u_emb in enumerate(user_embeds):
            sims = [self._cosine(u_emb, c_emb) for c_emb in self.column_embeddings]
            max_sim = max(sims)
            idx = int(np.argmax(sims))
            
            #Only map if similarity is above threshold
            if max_sim >= 0.3:
                orig_idx, sub_idx = term_mapping[i]
                if sub_idx is not None:
                    #This was part of a list, map back to the list element
                    if user_terms[orig_idx] not in mapping:
                        mapping[user_terms[orig_idx]] = []
                    mapping[user_terms[orig_idx]].append(self.columns[idx])
                else:
                    #This was a single term
                    mapping[user_terms[orig_idx]] = self.columns[idx]
        
        return mapping
    
    def resolve_value(self, column_fqn: str, value_text: str) -> Optional[str]:
        """
        Resolve a user-provided value to an actual domain value for a column.
        Returns the resolved domain value or None if no match found.
        """
        if column_fqn not in self.domain_cache:
            return None
            
        domain_values = self.domain_cache[column_fqn]
        return resolve_value_to_domain(column_fqn, value_text, domain_values)
    
    def _enhance_terms_with_llm(self, terms):
        """Use LLM to enhance term understanding for better semantic mapping"""
        try:
            #Create context about the database schema
            schema_context = "Available database columns: " + ", ".join([col[1] for col in self.columns])
            
            prompt = f"""
You are helping to map user terms to database columns. Given these user terms and available columns, suggest enhanced versions that would better match the database schema.

User Terms: {terms}
{schema_context}

For each term, suggest a more specific or related term that would better match database columns. Consider:
- Synonyms and related concepts
- More specific terminology
- Common database naming conventions

Return a list of enhanced terms in the same order, separated by commas:
"""
            
            resp = openai.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                timeout=10,
                temperature=0
            )
            
            enhanced = resp.choices[0].message.content.strip()
            if enhanced and "," in enhanced:
                return [term.strip() for term in enhanced.split(",")]
            else:
                return terms
                
        except Exception as e:
            #Fallback to original terms if LLM fails
            return terms

    def _cosine(self, a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
