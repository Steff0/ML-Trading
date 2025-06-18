
#!/usr/bin/env python3
"""
Modulo utils per il sistema di predizione azionaria
Contiene funzioni di utilità, helper e configurazioni
"""

from .data_validator import DataValidator
from .database_manager import DatabaseManager
from .logging_config import setup_logging
from .config import Config
from .helpers import (
    calculate_returns,
    normalize_data,
    validate_symbol,
    format_currency,
    get_trading_days
)

__version__ = "1.0.0"
__author__ = "Stock Prediction System"

# Esporta le classi e funzioni principali
__all__ = [
    'DataValidator',
    'DatabaseManager',
    'setup_logging',
    'Config',
    'calculate_returns',
    'normalize_data',
    'validate_symbol',
    'format_currency',
    'get_trading_days'
]
