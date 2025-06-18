#!/usr/bin/env python3
"""
Funzioni di utilità per analisi e trasformazione dei dati
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_returns(series: pd.Series, periods: int = 1):
    """Calcola i rendimenti su n periodi"""
    return series.pct_change(periods=periods)

def normalize_data(df: pd.DataFrame):
    """Normalizza i dati usando Z-score"""
    return (df - df.mean()) / df.std()

def validate_symbol(symbol: str) -> bool:
    """Valida un simbolo azionario"""
    return symbol.isalnum() and 1 <= len(symbol) <= 10

def format_currency(value: float) -> str:
    """Formatta il valore come valuta"""
    return "${:,.2f}".format(value)

def get_trading_days(start_date: str, end_date: str):
    """Restituisce i giorni di trading tra due date"""
    return pd.bdate_range(start=start_date, end=end_date).to_pydatetime().tolist()
