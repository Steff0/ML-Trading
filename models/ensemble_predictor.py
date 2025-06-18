#!/usr/bin/env python3
"""
Ensemble predictor per combinare più modelli di machine learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import logging

class EnsemblePredictor:
    """
    Classe per gestire ensemble di modelli di predizione
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def initialize_models(self):
        """Inizializza i modelli base dell'ensemble"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
    def train_ensemble(self, X, y, validation_split=0.2):
        """
        Addestra l'ensemble di modelli
        
        Args:
            X: Features di training
            y: Target di training
            validation_split: Percentuale per validazione
        """
        try:
            self.logger.info("Inizio addestramento ensemble...")
            
            if not self.models:
                self.initialize_models()
            
            # Normalizza i dati
            X_scaled = self.scaler.fit_transform(X)
            
            # Split per validazione
            split_idx = int(len(X_scaled) * (1 - validation_split))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            trained_models = {}
            model_scores = {}
            
            # Addestra ogni modello
            for name, model in self.models.items():
                self.logger.info(f"Addestramento {name}...")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                          scoring='neg_mean_squared_error')
                
                # Addestramento finale
                model.fit(X_train, y_train)
                
                # Validazione
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                trained_models[name] = model
                model_scores[name] = {
                    'cv_mse': -cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'val_mse': mse,
                    'val_r2': r2
                }
                
                self.logger.info(f"{name} - R²: {r2:.4f}, MSE: {mse:.4f}")
            
            # Calcola pesi basati su performance
            self._calculate_weights(model_scores)
            
            self.models = trained_models
            self.is_trained = True
            
            self.logger.info("✅ Ensemble addestrato con successo")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Errore addestramento ensemble: {e}")
            return False
    
    def _calculate_weights(self, scores):
        """Calcola pesi per l'ensemble basati su performance"""
        # Usa R² come metrica principale
        r2_scores = {name: info['val_r2'] for name, info in scores.items()}
        
        # Normalizza pesi (R² più alto = peso maggiore)
        min_r2 = min(r2_scores.values())
        adjusted_scores = {name: max(0.1, score - min_r2 + 0.1) 
                          for name, score in r2_scores.items()}
        
        total = sum(adjusted_scores.values())
        self.weights = {name: score/total for name, score in adjusted_scores.items()}
        
        self.logger.info(f"Pesi calcolati: {self.weights}")
    
    def predict(self, X):
        """
        Effettua predizioni con l'ensemble
        
        Args:
            X: Features per predizione
            
        Returns:
            Predizione pesata dell'ensemble
        """
        if not self.is_trained:
            raise ValueError("Ensemble non ancora addestrato")
            
        try:
            # Normalizza input
            X_scaled = self.scaler.transform(X)
            
            # Predizioni da ogni modello
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X_scaled)
            
            # Media pesata
            final_prediction = np.zeros(len(X_scaled))
            for name, pred in predictions.items():
                final_prediction += pred * self.weights[name]
            
            return final_prediction
            
        except Exception as e:
            self.logger.error(f"Errore predizione ensemble: {e}")
            return None
    
    def predict_single(self, X):
        """Predizione per un singolo campione"""
        pred = self.predict(X.reshape(1, -1))
        return pred[0] if pred is not None else None
    
    def get_model_importance(self):
        """Restituisce l'importanza delle features per Random Forest"""
        if 'random_forest' in self.models:
            return self.models['random_forest'].feature_importances_
        return None
    
    def get_ensemble_info(self):
        """Restituisce informazioni sull'ensemble"""
        return {
            'models': list(self.models.keys()),
            'weights': self.weights,
            'is_trained': self.is_trained,
            'scaler_fitted': hasattr(self.scaler, 'scale_')
        }