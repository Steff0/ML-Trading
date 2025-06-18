#!/usr/bin/env python3
"""
Modulo per il calcolo di indicatori tecnici avanzati
"""

import pandas as pd
import numpy as np
import logging

class TechnicalIndicators:
    """
    Classe per calcolare indicatori tecnici completi
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, window=20, std_dev=2):
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    @staticmethod
    def stochastic(high, low, close, k_window=14, d_window=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high, low, close, window=14):
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def atr(high, low, close, window=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def cci(high, low, close, window=20):
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def obv(close, volume):
        """On Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def adx(high, low, close, window=14):
        """Average Directional Index"""
        atr_val = TechnicalIndicators.atr(high, low, close, window)
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr_val)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr_val)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    def calculate_all_indicators(self, df):
        """
        Calcola tutti gli indicatori tecnici per un DataFrame
        
        Args:
            df: DataFrame con colonne OHLCV
            
        Returns:
            DataFrame con tutti gli indicatori
        """
        try:
            result_df = df.copy()
            
            high = df['High']
            low = df['Low'] 
            close = df['Close']
            volume = df['Volume']
            
            # Moving Averages
            for window in [5, 10, 20, 50, 200]:
                result_df[f'SMA_{window}'] = self.sma(close, window)
                result_df[f'EMA_{window}'] = self.ema(close, window)
            
            # RSI
            result_df['RSI'] = self.rsi(close)
            
            # MACD
            macd, signal, hist = self.macd(close)
            result_df['MACD'] = macd
            result_df['MACD_Signal'] = signal
            result_df['MACD_Hist'] = hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.bollinger_bands(close)
            result_df['BB_Upper'] = bb_upper
            result_df['BB_Middle'] = bb_middle
            result_df['BB_Lower'] = bb_lower
            result_df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            result_df['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic
            stoch_k, stoch_d = self.stochastic(high, low, close)
            result_df['Stoch_K'] = stoch_k
            result_df['Stoch_D'] = stoch_d
            
            # Williams %R
            result_df['Williams_R'] = self.williams_r(high, low, close)
            
            # ATR
            result_df['ATR'] = self.atr(high, low, close)
            
            # CCI
            result_df['CCI'] = self.cci(high, low, close)
            
            # OBV
            result_df['OBV'] = self.obv(close, volume)
            
            # ADX
            result_df['ADX'] = self.adx(high, low, close)
            
            # Price-based indicators
            result_df['Price_Change'] = close.pct_change()
            result_df['Price_Range'] = (high - low) / close
            result_df['Body_Size'] = abs(close - df['Open']) / close
            
            # Volume indicators
            result_df['Volume_SMA'] = self.sma(volume, 20)
            result_df['Volume_Ratio'] = volume / result_df['Volume_SMA']
            
            # Volatility
            result_df['Volatility'] = close.pct_change().rolling(20).std()
            
            # Momentum indicators
            for period in [5, 10, 20]:
                result_df[f'Momentum_{period}'] = close / close.shift(period) - 1
            
            self.logger.info("✅ Tutti gli indicatori tecnici calcolati")
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Errore calcolo indicatori: {e}")
            return df