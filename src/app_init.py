from src.db import Database

from src.semantic_mapper import SemanticMapper

from src.sql_generator import SQLGenerator

#App setup, random var names for fun
_db = Database()

_schema = _db.get_schema()

_mapper = SemanticMapper(_schema)

_sqlgen = SQLGenerator(_schema)

def get_db():
    return _db

def get_schema():
    return _schema

def get_mapper():
    return _mapper

def get_sqlgen():
    return _sqlgen
