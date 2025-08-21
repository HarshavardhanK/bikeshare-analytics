from src.db import Database
from src.semantic_mapper import SemanticMapper
from src.sql_generator import SQLGenerator

#App setup with lazy initialization
_db = None
_schema = None
_mapper = None
_sqlgen = None

def get_db():
    global _db
    if _db is None:
        _db = Database()
    return _db

def get_schema():
    global _schema
    if _schema is None:
        _schema = get_db().get_schema()
    return _schema

def get_mapper():
    global _mapper
    if _mapper is None:
        _mapper = SemanticMapper(get_schema())
    return _mapper

def get_sqlgen():
    global _sqlgen
    if _sqlgen is None:
        _sqlgen = SQLGenerator(get_schema())
    return _sqlgen
