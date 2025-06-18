#!/usr/bin/env python3
"""
Modulo models per il sistema di predizione azionaria
Contiene le classi e funzioni per i modelli di machine learning
"""

from .ensemble_predictor import EnsemblePredictor
from .technical_indicators import TechnicalIndicators
from .market_analyzer import MarketAnalyzer

__version__ = "1.0.0"
__author__ = "Stock Prediction System"

# Esporta le classi principali
__all__ = [
    'EnsemblePredictor',
    'TechnicalIndicators', 
    'MarketAnalyzer'
]
