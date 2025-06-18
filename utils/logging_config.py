#!/usr/bin/env python3
"""
Configurazione centralizzata del logging
"""

import logging

def setup_logging(level=logging.INFO):
    """Configura il logging di sistema"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("StockPredictionSystem")
