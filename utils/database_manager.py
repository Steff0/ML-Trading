#!/usr/bin/env python3
"""
Gestione database SQLite per il sistema di predizione
"""

import sqlite3
import pandas as pd

class DatabaseManager:
    """Gestore semplificato per operazioni sul database"""

    def __init__(self, db_path):
        self.db_path = db_path

    def execute(self, query, params=None):
        """Esegue una query INSERT/UPDATE"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or [])
            conn.commit()

    def fetch_dataframe(self, query, params=None):
        """Esegue una SELECT e restituisce un DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params or [])
