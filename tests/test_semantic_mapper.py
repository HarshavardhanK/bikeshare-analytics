import os
from dotenv import load_dotenv
load_dotenv()

#Tests for semantic mapping logic.
import pytest
import numpy as np
from semantic_mapper import SemanticMapper

class DummySemanticMapper(SemanticMapper):
    def _get_embeddings(self, texts):
        #Returns embeddings so that user_terms[0] matches columns[0], user_terms[1] matches columns[1], etc.
        # Pad with zeros if needed
        emb_dim = 5
        embs = []
        for i, _ in enumerate(texts):
            arr = np.zeros(emb_dim)
            arr[i % emb_dim] = 1.0
            embs.append(arr)
        return embs

def test_semantic_mapping():
    schema = {
        'trips': ['trip_distance_km', 'rider_gender'],
        'stations': ['station_name']
    }
    mapper = DummySemanticMapper(schema)
    user_terms = ['kilometres', 'gender', 'station']
    mapping = mapper.map(user_terms)
    assert mapping['kilometres'][1] == 'trip_distance_km'
    assert mapping['gender'][1] == 'rider_gender'
    assert mapping['station'][1] == 'station_name'

def test_synonym_mapping():
    schema = {
        'trips': ['trip_distance_km', 'rider_gender'],
    }
    # Use the real SemanticMapper with OpenAI embeddings
    mapper = SemanticMapper(schema)
    user_terms = ['females']
    mapping = mapper.map(user_terms)
    # Should map to rider_gender
    assert mapping['females'][1] == 'rider_gender'

def test_derived_attribute_mapping():
    schema = {
        'trips': ['trip_distance_km', 'rider_gender'],
    }
    mapper = DummySemanticMapper(schema)
    user_terms = ['meters']
    mapping = mapper.map(user_terms)
    # Should map to trip_distance_km (conversion logic is in main pipeline)
    assert mapping['meters'][1] == 'trip_distance_km'
