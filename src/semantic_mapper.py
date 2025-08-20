import os

import openai
import numpy as np

#Maps user terms to schema columns using embedding similarity
class SemanticMapper:
    def __init__(self, schema):
        self.schema = schema
        self.columns = self._get_columns()
        self.column_embeddings = self._embed_columns()

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
        user_embeds = self._get_embeddings(user_terms)
        mapping = {}
        for i, u_emb in enumerate(user_embeds):
            sims = [self._cosine(u_emb, c_emb) for c_emb in self.column_embeddings]
            idx = int(np.argmax(sims))
            mapping[user_terms[i]] = self.columns[idx]
        return mapping

    def _cosine(self, a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
