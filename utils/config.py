#!/usr/bin/env python3
"""
Configurazioni globali per il sistema di predizione azionaria
"""

import os

class Config:
    """Parametri e percorsi di configurazione"""
    
    DATA_DIR = "data"
    DB_PATH = os.path.join(DATA_DIR, "predictions.db")
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    SAMPLE_FILE = os.path.join(DATA_DIR, "sample_data.csv")

    FOREIGN_MARKETS = {
        'SP500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DOW': '^DJI',
        'FTSE': '^FTSE',
        'DAX': '^GDAXI',
        'NIKKEI': '^N225',
        'HANG_SENG': '^HSI',
        'CAC40': '^FCHI',
        'EUR_USD': 'EURUSD=X',
        'GBP_USD': 'GBPUSD=X',
        'USD_JPY': 'USDJPY=X',
        'VIX': '^VIX',
        'GOLD': 'GC=F',
        'OIL': 'CL=F'
    }
