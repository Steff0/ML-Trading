#!/usr/bin/env python3
"""
Analizzatore di mercato per correlazioni e sentiment
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats

class MarketAnalyzer:
    """
    Classe per analizzare correlazioni e trend di mercato
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.correlation_cache = {}
        
    def calculate_correlation_matrix(self, data_dict, window=252):
        """
        Calcola matrice di correlazione tra diversi asset
        
        Args:
            data_dict: Dizionario {nome_asset: serie_prezzi}
            window: Finestra temporale per correlazione
        """
        try:
            # Crea DataFrame combinato
            combined_df = pd.DataFrame(data_dict)
            
            # Calcola rendimenti
            returns_df = combined_df.pct_change().dropna()
            
            # Correlazione rolling
            correlations = {}
            for asset in combined_df.columns:
                correlations[asset] = {}
                for other_asset in combined_df.columns:
                    if asset != other_asset:
                        rolling_corr = returns_df[asset].rolling(window).corr(
                            returns_df[other_asset]
                        )
                        correlations[asset][other_asset] = rolling_corr.iloc[-1]
            
            self.logger.info("✅ Matrice correlazioni calcolata")
            return correlations
            
        except Exception as e:
            self.logger.error(f"❌ Errore calcolo correlazioni: {e}")
            return {}
    
    def detect_market_regime(self, price_data, volatility_threshold=0.02):
        """
        Identifica il regime di mercato (bull/bear/sideways)
        
        Args:
            price_data: Serie storica dei prezzi
            volatility_threshold: Soglia volatilità
        """
        try:
            # Calcola rendimenti e volatilità
            returns = price_data.pct_change().dropna()
            
            # Trend (media mobile 50 vs 200)
            ma_short = price_data.rolling(50).mean()
            ma_long = price_data.rolling(200).mean()
            
            # Volatilità rolling
            volatility = returns.rolling(20).std()
            
            # Determinazione regime
            latest_return = returns.rolling(20).mean().iloc[-1]
            latest_volatility = volatility.iloc[-1]
            trend_signal = 1 if ma_short.iloc[-1] > ma_long.iloc[-1] else -1
            
            if latest_volatility > volatility_threshold:
                regime = "high_volatility"
            elif latest_return > 0.001 and trend_signal > 0:
                regime = "bull_market"
            elif latest_return < -0.001 and trend_signal < 0:
                regime = "bear_market"
            else:
                regime = "sideways"
            
            regime_info = {
                'regime': regime,
                'volatility': latest_volatility,
                'trend_strength': abs(latest_return),
                'trend_direction': trend_signal
            }
            
            self.logger.info(f"Regime identificato: {regime}")
            return regime_info
            
        except Exception as e:
            self.logger.error(f"❌ Errore identificazione regime: {e}")
            return {'regime': 'unknown'}
    
    def calculate_beta(self, asset_returns, market_returns, window=252):
        """
        Calcola il beta dell'asset rispetto al mercato
        
        Args:
            asset_returns: Rendimenti dell'asset
            market_returns: Rendimenti del mercato
            window: Finestra temporale
        """
        try:
            # Allinea le serie
            aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
            
            if len(aligned_data) < window:
                window = len(aligned_data)
            
            # Calcola beta rolling
            recent_data = aligned_data.tail(window)
            covariance = recent_data.cov().iloc[0, 1]
            market_variance = recent_data.iloc[:, 1].var()
            
            beta = covariance / market_variance if market_variance != 0 else 1.0
            
            # Calcola anche alpha
            asset_mean = recent_data.iloc[:, 0].mean()
            market_mean = recent_data.iloc[:, 1].mean()
            alpha = asset_mean - (beta * market_mean)
            
            return {
                'beta': beta,
                'alpha': alpha,
                'correlation': recent_data.corr().iloc[0, 1]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Errore calcolo beta: {e}")
            return {'beta': 1.0, 'alpha': 0.0, 'correlation': 0.0}
    
    def analyze_support_resistance(self, price_data, window=20, min_touches=2):
        """
        Identifica livelli di supporto e resistenza
        
        Args:
            price_data: Serie storica dei prezzi
            window: Finestra per identificare massimi/minimi locali
            min_touches: Minimo numero di "tocchi" per validare il livello
        """
        try:
            # Trova massimi e minimi locali
            highs = price_data.rolling(window).max()
            lows = price_data.rolling(window).min()
            
            # Identifica punti di svolta
            resistance_levels = []
            support_levels = []
            
            for i in range(window, len(price_data) - window):
                current_price = price_data.iloc[i]
                
                # Controlla se è un massimo locale
                if current_price == highs.iloc[i]:
                    # Conta quante volte il prezzo ha "toccato" questo livello
                    touches = 0
                    tolerance = current_price * 0.02  # 2% di tolleranza
                    
                    for j in range(max(0, i-100), min(len(price_data), i+100)):
                        if abs(price_data.iloc[j] - current_price) <= tolerance:
                            touches += 1
                    
                    if touches >= min_touches:
                        resistance_levels.append({
                            'level': current_price,
                            'date': price_data.index[i],
                            'touches': touches,
                            'strength': min(touches / 5, 1.0)  # Normalizza la forza
                        })
                
                # Controlla se è un minimo locale
                if current_price == lows.iloc[i]:
                    touches = 0
                    tolerance = current_price * 0.02
                    
                    for j in range(max(0, i-100), min(len(price_data), i+100)):
                        if abs(price_data.iloc[j] - current_price) <= tolerance:
                            touches += 1
                    
                    if touches >= min_touches:
                        support_levels.append({
                            'level': current_price,
                            'date': price_data.index[i],
                            'touches': touches,
                            'strength': min(touches / 5, 1.0)
                        })
            
            # Ordina per forza e prendi i più significativi
            resistance_levels = sorted(resistance_levels, 
                                     key=lambda x: x['strength'], reverse=True)[:5]
            support_levels = sorted(support_levels, 
                                  key=lambda x: x['strength'], reverse=True)[:5]
            
            self.logger.info(f"Identificati {len(resistance_levels)} livelli di resistenza "
                           f"e {len(support_levels)} livelli di supporto")
            
            return {
                'resistance': resistance_levels,
                'support': support_levels
            }
            
        except Exception as e:
            self.logger.error(f"❌ Errore analisi supporto/resistenza: {e}")
            return {'resistance': [], 'support': []}
    
    def calculate_market_sentiment(self, foreign_markets_data):
        """
        Calcola sentiment di mercato basato su indici globali
        
        Args:
            foreign_markets_data: Dizionario con dati mercati esteri
        """
        try:
            sentiment_scores = {}
            
            for market, data in foreign_markets_data.items():
                if len(data) < 20:
                    continue
                
                # Calcola rendimenti
                returns = data.pct_change().dropna()
                
                # Score basato su diversi fattori
                recent_return = returns.tail(5).mean()  # Rendimento recente
                volatility = returns.tail(20).std()     # Volatilità
                trend = (data.iloc[-1] / data.iloc[-20] - 1)  # Trend 20 giorni
                
                # Combina in un sentiment score
                sentiment = (
                    recent_return * 0.4 +  # 40% peso al rendimento recente
                    trend * 0.4 +          # 40% peso al trend
                    (-volatility * 0.2)    # 20% peso alla stabilità (bassa volatilità)
                )
                
                sentiment_scores[market] = {
                    'score': sentiment,
                    'recent_return': recent_return,
                    'volatility': volatility,
                    'trend': trend
                }
            
            # Calcola sentiment globale
            if sentiment_scores:
                global_sentiment = np.mean([s['score'] for s in sentiment_scores.values()])
                
                # Classifica sentiment
                if global_sentiment > 0.01:
                    sentiment_label = "Molto Positivo"
                elif global_sentiment > 0.005:
                    sentiment_label = "Positivo"
                elif global_sentiment > -0.005:
                    sentiment_label = "Neutrale"
                elif global_sentiment > -0.01:
                    sentiment_label = "Negativo"
                else:
                    sentiment_label = "Molto Negativo"
                
                result = {
                    'global_sentiment': global_sentiment,
                    'sentiment_label': sentiment_label,
                    'individual_markets': sentiment_scores
                }
                
                self.logger.info(f"Sentiment globale: {sentiment_label} ({global_sentiment:.4f})")
                return result
            
            return {'global_sentiment': 0, 'sentiment_label': 'Neutrale'}
            
        except Exception as e:
            self.logger.error(f"❌ Errore calcolo sentiment: {e}")
            return {'global_sentiment': 0, 'sentiment_label': 'Sconosciuto'}