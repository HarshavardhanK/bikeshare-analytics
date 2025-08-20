#Handles DB connection and schema stuff

import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2

class Database:
    def __init__(self):
        dbname = os.getenv("POSTGRES_DB")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT", 5432)
        print(f"Connecting to DB: dbname={dbname}, user={user}, host={host}, port={port}")
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )

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
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params or [])
            try:
                result = cur.fetchall()
            except Exception:
                result = None
            cur.close()
            return result
        except Exception as e:
            print(f"[ERROR] DB error: {e}")
            self.conn.rollback()
            cur.close()
            raise
