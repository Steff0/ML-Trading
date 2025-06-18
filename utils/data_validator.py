#!/usr/bin/env python3
"""
Modulo per la validazione dei dati nel sistema di predizione azionaria
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List, Tuple, Any

class DataValidator:
    """
    Classe per validare dati finanziari e parametri del sistema di predizione
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Simboli validi per mercati esteri
        self.valid_foreign_symbols = {
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
    
    def validate_stock_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Valida un DataFrame contenente dati azionari
        
        Args:
            df: DataFrame con dati azionari
            
        Returns:
            Tuple (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Verifica che sia un DataFrame
            if not isinstance(df, pd.DataFrame):
                errors.append("I dati devono essere un DataFrame pandas")
                return False, errors
            
            # Verifica che non sia vuoto
            if df.empty:
                errors.append("DataFrame vuoto")
                return False, errors
            
            # Verifica colonne richieste
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                errors.append(f"Colonne mancanti: {', '.join(missing_columns)}")
            
            # Verifica tipi di dati numerici
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(f"Colonna '{col}' deve essere numerica")
            
            # Verifica valori logici dei prezzi
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                # High >= max(Open, Close) and Low <= min(Open, Close)
                invalid_high = df['High'] < df[['Open', 'Close']].max(axis=1)
                invalid_low = df['Low'] > df[['Open', 'Close']].min(axis=1)
                
                if invalid_high.any():
                    errors.append(f"Trovati {invalid_high.sum()} record con High < max(Open, Close)")
                
                if invalid_low.any():
                    errors.append(f"Trovati {invalid_low.sum()} record con Low > min(Open, Close)")
            
            # Verifica valori negativi nei prezzi
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in df.columns:
                    negative_values = df[col] <= 0
                    if negative_values.any():
                        errors.append(f"Trovati {negative_values.sum()} valori non positivi in '{col}'")
            
            # Verifica volume negativo
            if 'Volume' in df.columns:
                negative_volume = df['Volume'] < 0
                if negative_volume.any():
                    errors.append(f"Trovati {negative_volume.sum()} valori negativi nel Volume")
            
            # Verifica date se presente indice temporale
            if isinstance(df.index, pd.DatetimeIndex):
                if not df.index.is_monotonic_increasing:
                    errors.append("Le date non sono in ordine crescente")
                
                # Verifica duplicati nelle date
                if df.index.duplicated().any():
                    errors.append("Trovate date duplicate nell'indice")
            
            # Verifica valori NaN eccessivi
            for col in df.columns:
                nan_percentage = df[col].isna().sum() / len(df) * 100
                if nan_percentage > 50:
                    errors.append(f"Colonna '{col}' ha {nan_percentage:.1f}% di valori NaN")
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            self.logger.error(f"Errore durante validazione dati: {e}")
            errors.append(f"Errore interno di validazione: {str(e)}")
            return False, errors
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        Valida un simbolo azionario
        
        Args:
            symbol: Simbolo da validare
            
        Returns:
            Tuple (is_valid, message)
        """
        if not isinstance(symbol, str):
            return False, "Il simbolo deve essere una stringa"
        
        symbol = symbol.strip().upper()
        
        if not symbol:
            return False, "Simbolo vuoto"
        
        if len(symbol) < 1 or len(symbol) > 10:
            return False, "Lunghezza simbolo non valida (1-10 caratteri)"
        
        # Verifica caratteri validi (lettere, numeri, alcuni simboli speciali)
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=')
        if not set(symbol).issubset(valid_chars):
            return False, "Simbolo contiene caratteri non validi"
        
        return True, "Simbolo valido"
    
    def validate_date_range(self, start_date: str, end_date: str) -> Tuple[bool, str]:
        """
        Valida un range di date
        
        Args:
            start_date: Data di inizio (YYYY-MM-DD)
            end_date: Data di fine (YYYY-MM-DD)
            
        Returns:
            Tuple (is_valid, message)
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            if start >= end:
                return False, "Data di inizio deve essere precedente alla data di fine"
            
            # Verifica che le date non siano troppo nel futuro
            today = datetime.now()
            if end > today:
                return False, "Data di fine non può essere nel futuro"
            
            # Verifica che il range non sia troppo lungo (max 10 anni)
            if (end - start).days > 3650:
                return False, "Range di date troppo ampio (massimo 10 anni)"
            
            # Verifica che il range non sia troppo corto
            if (end - start).days < 1:
                return False, "Range di date troppo breve (minimo 1 giorno)"
            
            return True, "Range di date valido"
            
        except Exception as e:
            return False, f"Errore parsing date: {str(e)}"
    
    def validate_model_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valida parametri per l'addestramento del modello
        
        Args:
            params: Dizionario con parametri del modello
            
        Returns:
            Tuple (is_valid, error_messages)
        """
        errors = []
        
        # Parametri richiesti con i loro tipi e range validi
        required_params = {
            'test_size': (float, 0.1, 0.5),
            'random_state': (int, 0, 2147483647),
            'n_estimators': (int, 10, 1000),
            'min_samples_split': (int, 2, 100),
            'min_samples_leaf': (int, 1, 50)
        }
        
        # Parametri opzionali
        optional_params = {
            'max_depth': (int, 1, 50),
            'learning_rate': (float, 0.01, 1.0),
            'subsample': (float, 0.1, 1.0),
            'max_features': (str, ['auto', 'sqrt', 'log2'])
        }
        
        # Verifica parametri richiesti
        for param_name, (param_type, min_val, max_val) in required_params.items():
            if param_name not in params:
                errors.append(f"Parametro richiesto mancante: {param_name}")
                continue
            
            value = params[param_name]
            
            if not isinstance(value, param_type):
                errors.append(f"Parametro '{param_name}' deve essere di tipo {param_type.__name__}")
                continue
            
            if isinstance(max_val, (int, float)) and not (min_val <= value <= max_val):
                errors.append(f"Parametro '{param_name}' fuori range [{min_val}, {max_val}]")
        
        # Verifica parametri opzionali
        for param_name, validation in optional_params.items():
            if param_name not in params:
                continue
            
            value = params[param_name]
            
            if len(validation) == 3:  # Tipo e range numerico
                param_type, min_val, max_val = validation
                if not isinstance(value, param_type):
                    errors.append(f"Parametro '{param_name}' deve essere di tipo {param_type.__name__}")
                elif isinstance(max_val, (int, float)) and not (min_val <= value <= max_val):
                    errors.append(f"Parametro '{param_name}' fuori range [{min_val}, {max_val}]")
            
            elif len(validation) == 2:  # Tipo e valori validi
                param_type, valid_values = validation
                if not isinstance(value, param_type):
                    errors.append(f"Parametro '{param_name}' deve essere di tipo {param_type.__name__}")
                elif value not in valid_values:
                    errors.append(f"Parametro '{param_name}' deve essere uno di: {valid_values}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_prediction_confidence(self, confidence: float) -> Tuple[bool, str]:
        """
        Valida il confidence score di una predizione
        
        Args:
            confidence: Score di confidenza
            
        Returns:
            Tuple (is_valid, message)
        """
        if not isinstance(confidence, (int, float)):
            return False, "Confidence deve essere numerico"
        
        if not (0 <= confidence <= 1):
            return False, "Confidence deve essere tra 0 e 1"
        
        if confidence < 0.1:
            return False, "Confidence troppo basso (< 0.1)"
        
        return True, "Confidence valido"
    
    def validate_csv_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Valida un file CSV prima dell'importazione
        
        Args:
            file_path: Percorso del file CSV
            
        Returns:
            Tuple (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Verifica esistenza file
            import os
            if not os.path.exists(file_path):
                errors.append(f"File non trovato: {file_path}")
                return False, errors
            
            # Verifica estensione
            if not file_path.lower().endswith('.csv'):
                errors.append("Il file deve avere estensione .csv")
            
            # Prova a leggere il file
            try:
                df = pd.read_csv(file_path, nrows=5)  # Leggi solo prime 5 righe per test
            except Exception as e:
                errors.append(f"Impossibile leggere il CSV: {str(e)}")
                return False, errors
            
            # Valida il contenuto
            is_valid, validation_errors = self.validate_stock_data(df)
            errors.extend(validation_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Errore validazione CSV: {e}")
            errors.append(f"Errore interno: {str(e)}")
            return False, errors
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizza la qualità dei dati
        
        Args:
            df: DataFrame da analizzare
            
        Returns:
            Dizionario con metriche di qualità
        """
        quality_report = {
            'total_rows': len(df),
            'date_range': None,
            'missing_data': {},
            'data_consistency': {},
            'outliers': {},
            'overall_score': 0
        }
        
        try:
            # Range di date
            if isinstance(df.index, pd.DatetimeIndex):
                quality_report['date_range'] = {
                    'start': df.index.min().strftime('%Y-%m-%d'),
                    'end': df.index.max().strftime('%Y-%m-%d'),
                    'days': (df.index.max() - df.index.min()).days
                }
            
            # Dati mancanti
            for col in df.columns:
                missing_pct = df[col].isna().sum() / len(df) * 100
                quality_report['missing_data'][col] = round(missing_pct, 2)
            
            # Consistenza dati (prezzi)
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                # Verifica relazioni logiche
                high_low_ok = (df['High'] >= df['Low']).mean() * 100
                high_prices_ok = (df['High'] >= df[['Open', 'Close']].max(axis=1)).mean() * 100
                low_prices_ok = (df['Low'] <= df[['Open', 'Close']].min(axis=1)).mean() * 100
                
                quality_report['data_consistency'] = {
                    'high_low_relation': round(high_low_ok, 2),
                    'high_prices_logic': round(high_prices_ok, 2),
                    'low_prices_logic': round(low_prices_ok, 2)
                }
            
            # Outliers (usando IQR)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | 
                           (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_pct = outliers / len(df) * 100
                quality_report['outliers'][col] = round(outlier_pct, 2)
            
            # Score complessivo (0-100)
            scores = []
            
            # Penalizza dati mancanti
            avg_missing = np.mean(list(quality_report['missing_data'].values()))
            missing_score = max(0, 100 - avg_missing * 2)
            scores.append(missing_score)
            
            # Premia consistenza
            if quality_report['data_consistency']:
                consistency_score = np.mean(list(quality_report['data_consistency'].values()))
                scores.append(consistency_score)
            
            # Penalizza outliers eccessivi
            avg_outliers = np.mean(list(quality_report['outliers'].values()))
            outlier_score = max(0, 100 - avg_outliers)
            scores.append(outlier_score)
            
            quality_report['overall_score'] = round(np.mean(scores), 1)
            
        except Exception as e:
            self.logger.error(f"Errore analisi qualità dati: {e}")
            quality_report['error'] = str(e)
        
        return quality_report