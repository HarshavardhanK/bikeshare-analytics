#Handles DB connection and schema stuff

import os
import logging
from dotenv import load_dotenv
load_dotenv()
import psycopg2
from psycopg2.pool import SimpleConnectionPool

class Database:
    def __init__(self):
        dbname = os.getenv("POSTGRES_DB")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT", 5432)
        sslmode = os.getenv("PGSSLMODE", "prefer")
        
        logging.info(f"Connecting to DB: dbname={dbname}, user={user}, host={host}, port={port}, sslmode={sslmode}")
        
        #Connection parameters with SSL support
        conn_params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port,
            'sslmode': sslmode
        }
        
        #Create connection pool for better performance
        self.pool = SimpleConnectionPool(1, 10, **conn_params)
        self.conn = self.pool.getconn()

    def get_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public';
        """)
        rows = cur.fetchall()
        schema = {}
        for table, col in rows:
            schema.setdefault(table, []).append(col)
        cur.close()
        return schema

    def execute(self, sql, params=None):
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(sql, params or [])
            try:
                result = cur.fetchall()
            except Exception:
                result = None
            cur.close()
            self.pool.putconn(conn)
            return result
        except Exception as e:
            logging.error(f"DB error: {e}")
            conn.rollback()
            cur.close()
            self.pool.putconn(conn)
            raise
