#!/usr/bin/env python3
"""
Sistema AI di Predizione Azionaria Multi-Mercato con Focus su Milano
- Confronto simultaneo stesso titolo su mercati globali
- Integrazione Borsa Italiana (Milano)
- Analisi geopolitica specifica per titoli italiani/europei
- Auto-affinamento predittivo
- Generazione automatica CSV/Tabelle
- Predizioni breve e lungo termine
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
import pickle
import warnings
import requests
import json
import time
from textblob import TextBlob
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import feedparser
import re

warnings.filterwarnings("ignore")


class MilanoStockPredictionAI:
    """
    Sistema AI avanzato per predizioni azionarie con focus su Milano
    - Multi-mercato simultaneo
    - Auto-affinamento
    - Analisi geopolitica
    - Predizioni breve/lungo termine
    """

    def __init__(self, data_dir="milan_data"):
        """Inizializza il sistema AI"""
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "milan_predictions.db")
        self.models_dir = os.path.join(data_dir, "ai_models")
        self.csv_dir = os.path.join(data_dir, "csv_exports")
        self.reports_dir = os.path.join(data_dir, "reports")

        # Azioni Milano/Europa con codici internazionali
        self.milan_stocks = {
            # Azioni italiane principali
            "ENI.MI": {
                "name": "Eni S.p.A.",
                "sector": "Energy",
                "global_codes": ["E", "ENI.PA", "ENI.L"],
            },
            "TIT.MI": {
                "name": "Telecom Italia",
                "sector": "Telecoms",
                "global_codes": ["TIAMF", "TIM.MI"],
            },
            "UCG.MI": {
                "name": "UniCredit",
                "sector": "Banking",
                "global_codes": ["UNCFF", "UCG.F"],
            },
            "ISP.MI": {
                "name": "Intesa Sanpaolo",
                "sector": "Banking",
                "global_codes": ["IITSF", "ISP.F"],
            },
            "ENEL.MI": {
                "name": "Enel",
                "sector": "Utilities",
                "global_codes": ["ENLAY", "ENEL.PA"],
            },
            "FCA.MI": {
                "name": "Stellantis (ex-FCA)",
                "sector": "Automotive",
                "global_codes": ["STLA", "STLA.PA"],
            },
            "G.MI": {
                "name": "Generali",
                "sector": "Insurance",
                "global_codes": ["ARZGY", "G.PA"],
            },
            "RACE.MI": {
                "name": "Ferrari",
                "sector": "Automotive",
                "global_codes": ["RACE", "RACE.F"],
            },
            "TERNA.MI": {
                "name": "Terna",
                "sector": "Utilities",
                "global_codes": ["TERNY"],
            },
            "LUX.MI": {
                "name": "Luxottica (EssilorLuxottica)",
                "sector": "Consumer",
                "global_codes": ["EL", "EL.PA"],
            },
            # ETF e indici Milano
            "FTSEMIB.MI": {
                "name": "FTSE MIB Index",
                "sector": "Index",
                "global_codes": ["^FTSEMIB"],
            },
            "ITB.MI": {
                "name": "iShares FTSE MIB ETF",
                "sector": "ETF",
                "global_codes": [],
            },
        }

        # Mercati globali estesi per confronto
        self.global_markets = {
            # Indici principali
            "SP500": "^GSPC",
            "NASDAQ": "^IXIC",
            "DOW": "^DJI",
            "FTSE_MIB": "^FTSEMIB",  # Milano
            "FTSE100": "^FTSE",  # Londra
            "DAX": "^GDAXI",  # Germania
            "CAC40": "^FCHI",  # Francia
            "IBEX35": "^IBEX",  # Spagna
            "AEX": "^AEX",  # Olanda
            "SMI": "^SSMI",  # Svizzera
            "NIKKEI": "^N225",  # Giappone
            "HANG_SENG": "^HSI",  # Hong Kong
            "SHANGHAI": "000001.SS",  # Cina
            "BSE_SENSEX": "^BSESN",  # India
            "BOVESPA": "^BVSP",  # Brasile
            "MOEX": "IMOEX.ME",  # Russia
            # Commodities
            "GOLD": "GC=F",
            "SILVER": "SI=F",
            "OIL_BRENT": "BZ=F",  # Importante per ENI
            "OIL_WTI": "CL=F",
            "NATURAL_GAS": "NG=F",  # Importante per utilities italiane
            "COPPER": "HG=F",
            # Valute (importante per export italiano)
            "EUR_USD": "EURUSD=X",
            "EUR_GBP": "EURGBP=X",
            "EUR_JPY": "EURJPY=X",
            "EUR_CHF": "EURCHF=X",
            "USD_CNY": "USDCNY=X",
            # Crypto (settore emergente)
            "BITCOIN": "BTC-USD",
            "ETHEREUM": "ETH-USD",
            # Bond (importanti per banche italiane)
            "BTP_10Y": "ITGB10YR=X",  # BTP italiano
            "BUND_10Y": "TNX",  # Bund tedesco
            "US_10Y": "^TNX",  # Treasury US
        }

        # Keywords geopolitici specifici per Italia/Europa
        self.geopolitical_keywords = {
            "italy_specific": [
                "italia",
                "italy",
                "governo italiano",
                "italian government",
                "draghi",
                "meloni",
                "mattarella",
                "parlamento italiano",
                "debito italia",
                "italian debt",
                "spread btp",
                "rating italia",
                "pnrr",
                "recovery fund",
                "next generation eu",
            ],
            "europe_wide": [
                "europa",
                "europe",
                "european union",
                "ue",
                "eu",
                "bce",
                "ecb",
                "lagarde",
                "euro",
                "eurozona",
                "eurozone",
                "brexit",
                "recovery fund",
                "green deal",
                "fit for 55",
            ],
            "energy_geopolitics": [
                "russia",
                "ucraina",
                "ukraine",
                "gas russo",
                "russian gas",
                "nord stream",
                "sanzioni russia",
                "russia sanctions",
                "gazprom",
                "eni",
                "energia",
                "energy crisis",
            ],
            "banking_regulation": [
                "vigilanza bancaria",
                "banking supervision",
                "stress test",
                "npl",
                "crediti deteriorati",
                "bad loans",
                "basilea",
                "basel",
                "capital ratio",
                "tier 1",
            ],
            "automotive": [
                "automotive",
                "auto",
                "electric vehicle",
                "ev",
                "stellantis",
                "ferrari",
                "co2 emission",
                "green transition",
            ],
            "general_risk": [
                "guerra",
                "war",
                "conflict",
                "crisi",
                "crisis",
                "inflazione",
                "inflation",
                "recessione",
                "recession",
                "covid",
                "pandemic",
                "lockdown",
                "supply chain",
            ],
        }

        # RSS feeds per news italiane/europee
        self.news_sources = {
            "italian_financial": [
                "https://www.milanofinanza.it/rss",
                "https://www.soldionline.it/rss",
                "https://www.borsaitaliana.it/rss/notizie.rss",
            ],
            "european_financial": [
                "https://feeds.financial-times.com/feeds/rss/european-business.xml",
                "https://www.reuters.com/cmbs/world/europe/feed",
            ],
            "global_financial": [
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://www.investing.com/rss/news.rss",
            ],
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(data_dir, "milan_ai.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Crea directories
        for directory in [data_dir, self.models_dir, self.csv_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)

        # Setup database
        self._setup_enhanced_database()

        # Cache per modelli AI
        self.ai_models = {}
        self.scalers = {}
        self.performance_history = {}

        # Auto-aggiornamento all'avvio
        self.startup_update()

    def _setup_enhanced_database(self):
        """Setup database esteso per Milano AI"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Tabella azioni multi-mercato
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS multi_market_stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    primary_symbol TEXT NOT NULL,
                    market_symbol TEXT NOT NULL,
                    market_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    adj_close REAL,
                    currency TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(primary_symbol, market_symbol, date)
                )
            """
            )

            # Tabella predizioni breve/lungo termine
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,  -- 'short_term', 'long_term'
                    prediction_date TIMESTAMP NOT NULL,
                    target_date DATE NOT NULL,
                    predicted_price REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    actual_price REAL,
                    accuracy_score REAL,
                    model_used TEXT,
                    features_count INTEGER,
                    sentiment_score REAL,
                    geopolitical_score REAL,
                    multi_market_correlation REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Tabella performance modelli AI
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    training_date TIMESTAMP NOT NULL,
                    r2_score REAL,
                    mse_score REAL,
                    mae_score REAL,
                    validation_accuracy REAL,
                    feature_importance TEXT,  -- JSON
                    hyperparameters TEXT,     -- JSON
                    data_points_used INTEGER,
                    prediction_horizon INTEGER,  -- giorni
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Tabella correlazioni multi-mercato
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS market_correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    primary_symbol TEXT NOT NULL,
                    correlated_market TEXT NOT NULL,
                    correlation_coefficient REAL,
                    correlation_type TEXT,  -- 'pearson', 'spearman'
                    calculation_date DATE NOT NULL,
                    data_period INTEGER,  -- giorni usati per calcolo
                    significance_level REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(primary_symbol, correlated_market, calculation_date)
                )
            """
            )

            # Tabella news e sentiment geopolitico
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS geopolitical_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    news_source TEXT,
                    headline TEXT,
                    content_summary TEXT,
                    overall_sentiment REAL,
                    italy_relevance REAL,
                    europe_relevance REAL,
                    energy_impact REAL,
                    banking_impact REAL,
                    automotive_impact REAL,
                    keywords_matched TEXT,  -- JSON
                    affected_symbols TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Tabella auto-affinamento
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_refinement (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    refinement_date TIMESTAMP NOT NULL,
                    old_accuracy REAL,
                    new_accuracy REAL,
                    improvement_pct REAL,
                    refinement_type TEXT,  -- 'feature_engineering', 'hyperparameter_tuning', 'model_selection'
                    changes_made TEXT,     -- JSON
                    validation_results TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            conn.close()
            self.logger.info("✅ Database avanzato configurato")

        except Exception as e:
            self.logger.error(f"❌ Errore setup database: {e}")

    def startup_update(self):
        """Aggiornamento automatico all'avvio"""
        self.logger.info("🚀 Avvio sistema AI Milano - Aggiornamento automatico...")

        try:
            # 1. Aggiorna mercati globali
            self.logger.info("🌍 Aggiornamento mercati globali...")
            self._update_global_markets()

            # 2. Aggiorna azioni Milano multi-mercato
            self.logger.info("🇮🇹 Aggiornamento azioni Milano multi-mercato...")
            self._update_milan_multi_market()

            # 3. Aggiorna sentiment geopolitico
            self.logger.info("📰 Aggiornamento sentiment geopolitico...")
            self._update_geopolitical_sentiment()

            # 4. Calcola correlazioni
            self.logger.info("📊 Calcolo correlazioni multi-mercato...")
            self._calculate_market_correlations()

            self.logger.info("✅ Aggiornamento automatico completato")

        except Exception as e:
            self.logger.error(f"❌ Errore aggiornamento automatico: {e}")

    def _update_global_markets(self):
        """Aggiorna tutti i mercati globali"""
        success_count = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._fetch_market_data, market, symbol, "3mo"): market
                for market, symbol in self.global_markets.items()
            }

            for future in as_completed(futures):
                market = futures[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        self._save_market_data(market, data)
                        success_count += 1
                        self.logger.info(f"✅ {market}: {len(data)} records")
                except Exception as e:
                    self.logger.warning(f"⚠️ {market}: {e}")

        self.logger.info(
            f"Mercati globali aggiornati: {success_count}/{len(self.global_markets)}"
        )

    def _update_milan_multi_market(self):
        """Aggiorna azioni Milano su mercati multipli"""
        for primary_symbol, stock_info in self.milan_stocks.items():
            try:
                # Simbolo principale (Milano)
                self._fetch_and_save_multi_market(
                    primary_symbol, primary_symbol, "Milano"
                )

                # Simboli su altri mercati
                for global_code in stock_info["global_codes"]:
                    if global_code:
                        market_name = self._detect_market_from_symbol(global_code)
                        self._fetch_and_save_multi_market(
                            primary_symbol, global_code, market_name
                        )
                        time.sleep(0.5)  # Rate limiting

            except Exception as e:
                self.logger.warning(f"Errore multi-mercato {primary_symbol}: {e}")

    def _fetch_and_save_multi_market(self, primary_symbol, market_symbol, market_name):
        """Scarica e salva dati per un simbolo su un mercato specifico"""
        try:
            data = self._fetch_market_data("temp", market_symbol, "6mo")
            if data is not None and not data.empty:
                conn = sqlite3.connect(self.db_path)

                for date, row in data.iterrows():
                    try:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO multi_market_stocks 
                            (primary_symbol, market_symbol, market_name, date, 
                             open_price, high_price, low_price, close_price, volume, adj_close, currency)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                primary_symbol,
                                market_symbol,
                                market_name,
                                date.strftime("%Y-%m-%d"),
                                (
                                    float(row["Open"])
                                    if not pd.isna(row["Open"])
                                    else None
                                ),
                                (
                                    float(row["High"])
                                    if not pd.isna(row["High"])
                                    else None
                                ),
                                float(row["Low"]) if not pd.isna(row["Low"]) else None,
                                (
                                    float(row["Close"])
                                    if not pd.isna(row["Close"])
                                    else None
                                ),
                                int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                                (
                                    float(row["Close"])
                                    if not pd.isna(row["Close"])
                                    else None
                                ),
                                self._detect_currency_from_market(market_name),
                            ),
                        )
                    except Exception:
                        continue

                conn.commit()
                conn.close()

                self.logger.info(
                    f"✅ {primary_symbol} su {market_name}: {len(data)} records"
                )
                return True

        except Exception as e:
            self.logger.warning(f"Errore fetch {market_symbol}: {e}")
            return False

    def _detect_market_from_symbol(self, symbol):
        """Rileva il mercato dal simbolo"""
        if ".MI" in symbol or ".F" in symbol:
            return "Europa"
        elif ".L" in symbol:
            return "Londra"
        elif ".PA" in symbol:
            return "Parigi"
        elif "NASDAQ" in symbol or not "." in symbol:
            return "USA"
        else:
            return "Altro"

    def _detect_currency_from_market(self, market_name):
        """Rileva la valuta dal mercato"""
        currency_map = {
            "Milano": "EUR",
            "Europa": "EUR",
            "Parigi": "EUR",
            "Londra": "GBP",
            "USA": "USD",
            "Altro": "USD",
        }
        return currency_map.get(market_name, "EUR")

    def _fetch_market_data(self, market_name, symbol, period):
        """Scarica dati di mercato generici"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data if not data.empty else None
        except Exception:
            return None

    def _save_market_data(self, market_name, data):
        """Salva dati di mercato nel database globale"""
        # Implementazione placeholder - da adattare alla struttura specifica
        pass

    def _update_geopolitical_sentiment(self):
        """Aggiorna sentiment geopolitico da news italiane/europee"""
        try:
            all_news = []

            # Scarica news da varie fonti
            for source_type, urls in self.news_sources.items():
                for url in urls:
                    try:
                        news_items = self._fetch_rss_news(url, source_type)
                        all_news.extend(news_items)
                        time.sleep(1)  # Rate limiting
                    except Exception as e:
                        self.logger.warning(f"Errore RSS {url}: {e}")

            # Analizza sentiment per ogni notizia
            if all_news:
                self._analyze_news_sentiment(all_news)
                self.logger.info(f"✅ Analizzate {len(all_news)} notizie")

        except Exception as e:
            self.logger.error(f"Errore sentiment geopolitico: {e}")

    def _fetch_rss_news(self, url, source_type):
        """Scarica news da RSS feed"""
        try:
            feed = feedparser.parse(url)
            news_items = []

            for entry in feed.entries[:10]:  # Limita a 10 news per fonte
                news_item = {
                    "source": source_type,
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                }
                news_items.append(news_item)

            return news_items

        except Exception:
            return []

    def _analyze_news_sentiment(self, news_items):
        """Analizza sentiment delle notizie"""
        conn = sqlite3.connect(self.db_path)
        today = datetime.now().date()

        for news in news_items:
            try:
                # Testo completo per analisi
                full_text = f"{news['title']} {news['summary']}"

                # Sentiment generale
                blob = TextBlob(full_text)
                overall_sentiment = blob.sentiment.polarity

                # Rilevanza per settori specifici
                italy_relevance = self._calculate_keyword_relevance(
                    full_text, self.geopolitical_keywords["italy_specific"]
                )
                europe_relevance = self._calculate_keyword_relevance(
                    full_text, self.geopolitical_keywords["europe_wide"]
                )
                energy_impact = self._calculate_keyword_relevance(
                    full_text, self.geopolitical_keywords["energy_geopolitics"]
                )
                banking_impact = self._calculate_keyword_relevance(
                    full_text, self.geopolitical_keywords["banking_regulation"]
                )
                automotive_impact = self._calculate_keyword_relevance(
                    full_text, self.geopolitical_keywords["automotive"]
                )

                # Keywords matchate
                matched_keywords = self._find_matched_keywords(full_text)
                affected_symbols = self._identify_affected_symbols(full_text)

                # Salva nel database
                conn.execute(
                    """
                    INSERT INTO geopolitical_sentiment 
                    (date, news_source, headline, content_summary, overall_sentiment,
                     italy_relevance, europe_relevance, energy_impact, banking_impact, 
                     automotive_impact, keywords_matched, affected_symbols)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        today,
                        news["source"],
                        news["title"],
                        news["summary"][:500],
                        overall_sentiment,
                        italy_relevance,
                        europe_relevance,
                        energy_impact,
                        banking_impact,
                        automotive_impact,
                        json.dumps(matched_keywords),
                        json.dumps(affected_symbols),
                    ),
                )

            except Exception as e:
                self.logger.warning(f"Errore analisi news: {e}")

        conn.commit()
        conn.close()

    def _calculate_keyword_relevance(self, text, keywords):
        """Calcola rilevanza basata su keywords"""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return min(matches / len(keywords), 1.0)  # Normalizza a [0,1]

    def _find_matched_keywords(self, text):
        """Trova keywords matchate"""
        matched = []
        text_lower = text.lower()

        for category, keywords in self.geopolitical_keywords.items():
            category_matches = [kw for kw in keywords if kw.lower() in text_lower]
            if category_matches:
                matched.extend(category_matches)

        return matched[:20]  # Limita per evitare JSON troppo grandi

    def _identify_affected_symbols(self, text):
        """Identifica simboli azionari potenzialmente affetti"""
        affected = []
        text_lower = text.lower()

        for symbol, info in self.milan_stocks.items():
            # Controlla nome azienda
            if info["name"].lower() in text_lower:
                affected.append(symbol)

            # Controlla settore
            sector_keywords = {
                "Energy": ["energia", "oil", "gas", "petrolio"],
                "Banking": ["banca", "bank", "credito", "lending"],
                "Automotive": ["auto", "automotive", "car"],
                "Telecoms": ["telecom", "telefonia", "mobile"],
                "Utilities": ["utilities", "elettrico", "electric"],
            }

            if info["sector"] in sector_keywords:
                for keyword in sector_keywords[info["sector"]]:
                    if keyword.lower() in text_lower and symbol not in affected:
                        affected.append(symbol)
                        break

        return affected

    def _calculate_market_correlations(self):
        """Calcola correlazioni tra mercati per ogni azione"""
        for primary_symbol in self.milan_stocks.keys():
            try:
                correlations = self._compute_symbol_correlations(primary_symbol)
                self._save_correlations(primary_symbol, correlations)
            except Exception as e:
                self.logger.warning(f"Errore correlazioni {primary_symbol}: {e}")

    def _compute_symbol_correlations(self, primary_symbol):
        """Computa correlazioni per un simbolo specifico"""
        try:
            # Recupera dati multi-mercato
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                """
                SELECT market_name, date, close_price
                FROM multi_market_stocks
                WHERE primary_symbol = ?
                ORDER BY date
            """,
                conn,
                params=(primary_symbol,),
            )
            conn.close()

            if df.empty:
                return {}

            # Pivot per avere mercati come colonne
            pivot_df = df.pivot(
                index="date", columns="market_name", values="close_price"
            )
            pivot_df = pivot_df.fillna(method="ffill").dropna()

            if pivot_df.shape[1] < 2:
                return {}

            # Calcola correlazioni
            correlations = {}

            if "Milano" in pivot_df.columns:
                milan_prices = pivot_df["Milano"]

                for market in pivot_df.columns:
                    if market != "Milano":
                        market_prices = pivot_df[market]

                        # Correlazione Pearson
                        pearson_corr, pearson_p = stats.pearsonr(
                            milan_prices, market_prices
                        )

                        # Correlazione Spearman (rank-based)
                        spearman_corr, spearman_p = stats.spearmanr(
                            milan_prices, market_prices
                        )

                        correlations[market] = {
                            "pearson": {"corr": pearson_corr, "p_value": pearson_p},
                            "spearman": {"corr": spearman_corr, "p_value": spearman_p},
                        }

            return correlations

        except Exception as e:
            self.logger.warning(f"Errore calcolo correlazioni: {e}")
            return {}

    def _save_correlations(self, primary_symbol, correlations):
        """Salva correlazioni nel database"""
        try:
            conn = sqlite3.connect(self.db_path)
            today = datetime.now().date()

            for market, corr_data in correlations.items():
                # Salva correlazione Pearson
                conn.execute(
                    """
                    INSERT OR REPLACE INTO market_correlations
                    (
                        primary_symbol,
                        correlated_market,
                        correlation_coefficient,
                        correlation_type,
                        calculation_date,
                        data_period,
                        significance_level
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        primary_symbol,
                        market,
                        corr_data["pearson"]["corr"],
                        "pearson",
                        today,
                        90,
                        corr_data["pearson"]["p_value"],
                    ),
                )

                # Salva correlazione Spearman
                conn.execute(
                    """
                    INSERT OR REPLACE INTO market_correlations
                    (primary_symbol, 
                        correlated_market, 
                        correlation_coefficient,
                        correlation_type,
                        calculation_date,
                        data_period,
                        significance_level
                    )
                    VALUES ( )
                    (primary_symbol, market, corr_data['spearman']['corr'],
                    'spearman', today, 90, corr_data['spearman']['p_value']
                """,
                    (),
                )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.warning(f"Errore salvataggio correlazioni: {e}")

    def generate_ai_predictions(
        self, symbol, prediction_horizon_days=30, long_term_days=180
    ):
        """Genera predizioni AI breve e lungo termine"""
        try:
            self.logger.info(f"🤖 Generazione predizioni AI per {symbol}")

            # Prepara dataset per training
            features_df = self._prepare_ai_features(symbol)
            if features_df.empty:
                self.logger.warning(f"Dati insufficienti per {symbol}")
                return None

            # Addestra modelli AI
            models_performance = self._train_ai_models(symbol, features_df)

            # Genera predizioni breve termine
            short_predictions = self._generate_short_term_predictions(
                symbol, features_df, models_performance, prediction_horizon_days
            )

            # Genera predizioni lungo termine
            long_predictions = self._generate_long_term_predictions(
                symbol, features_df, models_performance, long_term_days
            )

            # Salva predizioni
            self._save_predictions(symbol, short_predictions, long_predictions)

            # Auto-affinamento

            self._auto_refine_models(symbol, models_performance)

            # Esportazione automatica
            self.export_predictions_to_csv(symbol)

            return {
                "short_term": short_predictions,
                "long_term": long_predictions,
                "models_performance": models_performance,
            }

        except Exception as e:
            self.logger.error(f"Errore predizioni AI {symbol}: {e}")
            return None

    def _prepare_ai_features(self, symbol):
        """Prepara features per training AI"""
        try:
            # Recupera dati multi-mercato
            conn = sqlite3.connect(self.db_path)

            # Dati principali del simbolo
            main_df = pd.read_sql_query(
                """
                SELECT date, close_price, volume, open_price, high_price, low_price
                FROM multi_market_stocks
                WHERE primary_symbol = ? AND market_name = 'Milano'
                ORDER BY date
            """,
                conn,
                params=(symbol,),
            )

            if main_df.empty:
                return pd.DataFrame()

            main_df["date"] = pd.to_datetime(main_df["date"])
            main_df = main_df.set_index("date")

            # Features tecniche base
            main_df["returns"] = main_df["close_price"].pct_change()
            main_df["volatility"] = main_df["returns"].rolling(20).std()
            main_df["sma_20"] = main_df["close_price"].rolling(20).mean()
            main_df["sma_50"] = main_df["close_price"].rolling(50).mean()
            main_df["rsi"] = self._calculate_rsi(main_df["close_price"])
            main_df["bollinger_upper"], main_df["bollinger_lower"] = (
                self._calculate_bollinger_bands(main_df["close_price"])
            )
            main_df["volume_ma"] = main_df["volume"].rolling(20).mean()
            main_df["price_volume_trend"] = (
                (main_df["close_price"] - main_df["close_price"].shift(1))
                / main_df["close_price"].shift(1)
            ) * main_df["volume"]

            # Correlazioni multi-mercato
            correlations_df = pd.read_sql_query(
                """
                SELECT correlated_market, correlation_coefficient, correlation_type
                FROM market_correlations
                WHERE primary_symbol = ? AND calculation_date = (
                    SELECT MAX(calculation_date) FROM market_correlations WHERE primary_symbol = ?
                )
            """,
                conn,
                params=(symbol, symbol),
            )

            for _, row in correlations_df.iterrows():
                col_name = f"corr_{row['correlated_market']}_{row['correlation_type']}"
                main_df[col_name] = row["correlation_coefficient"]

            # Sentiment geopolitico
            sentiment_df = pd.read_sql_query(
                """
                SELECT date, 
                       AVG(overall_sentiment) as avg_sentiment,
                       AVG(italy_relevance) as italy_relevance,
                       AVG(europe_relevance) as europe_relevance,
                       AVG(energy_impact) as energy_impact,
                       AVG(banking_impact) as banking_impact,
                       AVG(automotive_impact) as automotive_impact
                FROM geopolitical_sentiment
                WHERE date >= ?
                GROUP BY date
                ORDER BY date
            """,
                conn,
                params=(main_df.index.min().date(),),
            )

            if not sentiment_df.empty:
                sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
                sentiment_df = sentiment_df.set_index("date")
                main_df = main_df.join(sentiment_df, how="left")
                main_df[sentiment_df.columns] = main_df[sentiment_df.columns].fillna(
                    method="ffill"
                )

            # Mercati globali correlati
            sector = self.milan_stocks[symbol]["sector"]
            global_markets_list = ["SP500", "FTSE_MIB", "EUR_USD"]

            if sector == "Energy":
                global_markets_list.extend(["OIL_BRENT", "NATURAL_GAS"])
            elif sector == "Banking":
                global_markets_list.extend(["BTP_10Y", "US_10Y"])
            elif sector == "Automotive":
                global_markets_list.extend(["COPPER"])

            # Aggiungi features macro-economiche simulate
            main_df["market_regime"] = np.where(
                main_df["volatility"] > main_df["volatility"].rolling(50).mean(), 1, 0
            )
            main_df["trend_strength"] = (
                main_df["sma_20"] - main_df["sma_50"]
            ) / main_df["sma_50"]

            conn.close()

            # Target variable (prezzo futuro)
            main_df["target_1d"] = main_df["close_price"].shift(-1)
            main_df["target_5d"] = main_df["close_price"].shift(-5)
            main_df["target_20d"] = main_df["close_price"].shift(-20)

            # Rimuovi NaN
            main_df = main_df.dropna()

            return main_df

        except Exception as e:
            self.logger.error(f"Errore preparazione features: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices, period=14):
        """Calcola RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcola Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _train_ai_models(self, symbol, features_df):
        """Addestra multipli modelli AI"""
        try:
            # Prepara dati per training
            feature_columns = [
                col for col in features_df.columns if not col.startswith("target_")
            ]
            X = features_df[feature_columns]
            y_1d = features_df["target_1d"]
            y_5d = features_df["target_5d"]
            y_20d = features_df["target_20d"]

            # Scaling
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[symbol] = scaler

            models_results = {}

            # Modelli da testare
            models = {
                "RandomForest": RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                ),
                "ExtraTrees": ExtraTreesRegressor(
                    n_estimators=100, random_state=42, n_jobs=-1
                ),
                "NeuralNetwork": MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
                ),
            }

            # Time series split per validazione
            tscv = TimeSeriesSplit(n_splits=5)

            for model_name, model in models.items():
                try:
                    # Training per diversi orizzonti temporali
                    horizons_performance = {}

                    for horizon, y_target in [
                        ("1d", y_1d),
                        ("5d", y_5d),
                        ("20d", y_20d),
                    ]:
                        # Rimuovi NaN per questo target
                        valid_idx = ~y_target.isna()
                        X_valid = X_scaled[valid_idx]
                        y_valid = y_target[valid_idx]

                        if len(y_valid) < 50:  # Dati insufficienti
                            continue

                        # Cross-validation temporale
                        cv_scores = []
                        feature_importances = []

                        for train_idx, val_idx in tscv.split(X_valid):
                            X_train, X_val = X_valid[train_idx], X_valid[val_idx]
                            y_train, y_val = (
                                y_valid.iloc[train_idx],
                                y_valid.iloc[val_idx],
                            )

                            # Training
                            model.fit(X_train, y_train)

                            # Predizione
                            y_pred = model.predict(X_val)

                            # Metriche
                            r2 = r2_score(y_val, y_pred)
                            mse = mean_squared_error(y_val, y_pred)
                            mae = mean_absolute_error(y_val, y_pred)

                            cv_scores.append({"r2": r2, "mse": mse, "mae": mae})

                            # Feature importance (se disponibile)
                            if hasattr(model, "feature_importances_"):
                                feature_importances.append(model.feature_importances_)

                        if cv_scores:
                            avg_scores = {
                                "r2": np.mean([s["r2"] for s in cv_scores]),
                                "mse": np.mean([s["mse"] for s in cv_scores]),
                                "mae": np.mean([s["mae"] for s in cv_scores]),
                            }

                            avg_feature_importance = (
                                np.mean(feature_importances, axis=0)
                                if feature_importances
                                else None
                            )

                            horizons_performance[horizon] = {
                                "scores": avg_scores,
                                "feature_importance": avg_feature_importance,
                                "model": model,
                            }

                    models_results[model_name] = horizons_performance

                except Exception as e:
                    self.logger.warning(f"Errore training {model_name}: {e}")

            # Salva performance nel database
            self._save_model_performance(symbol, models_results, X.columns.tolist())

            # Salva migliori modelli
            self.ai_models[symbol] = models_results

            return models_results

        except Exception as e:
            self.logger.error(f"Errore training modelli: {e}")
            return {}

    def _save_model_performance(self, symbol, models_results, feature_names):
        """Salva performance modelli nel database"""
        try:
            conn = sqlite3.connect(self.db_path)

            for model_name, horizons in models_results.items():
                for horizon, results in horizons.items():
                    scores = results["scores"]
                    feature_imp = results.get("feature_importance", [])

                    # Crea dizionario feature importance
                    if feature_imp is not None and len(feature_imp) == len(
                        feature_names
                    ):
                        feature_importance_dict = dict(
                            zip(feature_names, feature_imp.tolist())
                        )
                    else:
                        feature_importance_dict = {}

                    conn.execute(
                        """
                        INSERT INTO model_performance
                        (symbol, model_name, training_date, r2_score, mse_score, mae_score,
                         validation_accuracy, feature_importance, data_points_used, prediction_horizon)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            symbol,
                            f"{model_name}_{horizon}",
                            datetime.now(),
                            scores["r2"],
                            scores["mse"],
                            scores["mae"],
                            max(0, scores["r2"]),  # Accuracy come R2 normalizzato
                            json.dumps(feature_importance_dict),
                            len(feature_names),
                            int(horizon.replace("d", "")),
                        ),
                    )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.warning(f"Errore salvataggio performance: {e}")

    def _generate_short_term_predictions(
        self, symbol, features_df, models_performance, horizon_days
    ):
        """Genera predizioni breve termine"""
        try:
            predictions = []

            # Usa ultimi dati disponibili
            latest_features = features_df.iloc[-1:][
                self._get_feature_columns(features_df)
            ]
            latest_scaled = self.scalers[symbol].transform(latest_features)

            # Seleziona miglior modello per breve termine
            best_model_name, best_horizon = self._select_best_model(
                models_performance, "short"
            )

            if best_model_name and best_horizon in models_performance[best_model_name]:
                model = models_performance[best_model_name][best_horizon]["model"]

                # Genera predizioni per ogni giorno
                current_features = latest_scaled.copy()
                base_date = features_df.index[-1]

                for day in range(1, horizon_days + 1):
                    pred_price = model.predict(current_features)[0]
                    target_date = base_date + timedelta(days=day)

                    # Calcola confidence score basato su R2
                    confidence = models_performance[best_model_name][best_horizon][
                        "scores"
                    ]["r2"]
                    confidence = max(0, min(1, confidence))  # Normalizza [0,1]

                    predictions.append(
                        {
                            "target_date": target_date,
                            "predicted_price": pred_price,
                            "confidence": confidence,
                            "model_used": f"{best_model_name}_{best_horizon}",
                            "prediction_type": "short_term",
                        }
                    )

            return predictions

        except Exception as e:
            self.logger.error(f"Errore predizioni breve termine: {e}")
            return []

    def _generate_long_term_predictions(
        self, symbol, features_df, models_performance, horizon_days
    ):
        """Genera predizioni lungo termine"""
        try:
            predictions = []

            # Seleziona miglior modello per lungo termine
            best_model_name, best_horizon = self._select_best_model(
                models_performance, "long"
            )

            if best_model_name and best_horizon in models_performance[best_model_name]:
                model = models_performance[best_model_name][best_horizon]["model"]
                latest_features = features_df.iloc[-1:][
                    self._get_feature_columns(features_df)
                ]
                latest_scaled = self.scalers[symbol].transform(latest_features)

                # Predizioni mensili per lungo termine
                base_date = features_df.index[-1]
                monthly_steps = [30, 60, 90, 120, 150, 180]

                for days in monthly_steps:
                    if days <= horizon_days:
                        pred_price = model.predict(latest_scaled)[0]
                        target_date = base_date + timedelta(days=days)

                        # Confidence diminuisce con orizzonte temporale
                        base_confidence = models_performance[best_model_name][
                            best_horizon
                        ]["scores"]["r2"]
                        confidence = max(
                            0.1, base_confidence * (1 - days / 365)
                        )  # Decay temporale

                        predictions.append(
                            {
                                "target_date": target_date,
                                "predicted_price": pred_price,
                                "confidence": confidence,
                                "model_used": f"{best_model_name}_{best_horizon}",
                                "prediction_type": "long_term",
                            }
                        )

            return predictions

        except Exception as e:
            self.logger.error(f"Errore predizioni lungo termine: {e}")
            return []

    def _get_feature_columns(self, features_df):
        """Ottieni colonne delle features (esclusi target)"""
        return [col for col in features_df.columns if not col.startswith("target_")]

    def _select_best_model(self, models_performance, term_type):
        """Seleziona miglior modello basato su performance"""
        best_r2 = -np.inf
        best_model = None
        best_horizon = None

        horizon_preference = {"short": ["1d", "5d", "20d"], "long": ["20d", "5d", "1d"]}

        for model_name, horizons in models_performance.items():
            for horizon in horizon_preference[term_type]:
                if horizon in horizons:
                    r2_score = horizons[horizon]["scores"]["r2"]
                    if r2_score > best_r2:
                        best_r2 = r2_score
                        best_model = model_name
                        best_horizon = horizon
                        break

        return best_model, best_horizon

    def _save_predictions(self, symbol, short_predictions, long_predictions):
        """Salva predizioni nel database"""
        try:
            conn = sqlite3.connect(self.db_path)

            all_predictions = short_predictions + long_predictions

            for pred in all_predictions:
                # Calcola features aggiuntive
                sentiment_score = self._get_latest_sentiment_for_symbol(symbol)
                geopolitical_score = self._get_geopolitical_score_for_symbol(symbol)
                multi_market_corr = self._get_average_correlation(symbol)

                conn.execute(
                    """
                    INSERT INTO ai_predictions
                    (symbol, prediction_type, prediction_date, target_date, predicted_price,
                     confidence_score, model_used, sentiment_score, geopolitical_score,
                     multi_market_correlation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        pred["prediction_type"],
                        datetime.now(),
                        pred["target_date"],
                        pred["predicted_price"],
                        pred["confidence"],
                        pred["model_used"],
                        sentiment_score,
                        geopolitical_score,
                        multi_market_corr,
                    ),
                )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Errore salvataggio predizioni: {e}")

    def _get_latest_sentiment_for_symbol(self, symbol):
        """Ottieni ultimo sentiment per simbolo"""
        try:
            conn = sqlite3.connect(self.db_path)
            result = conn.execute(
                """
                SELECT AVG(overall_sentiment)
                FROM geopolitical_sentiment
                WHERE date >= ? AND affected_symbols LIKE ?
            """,
                ((datetime.now() - timedelta(days=7)).date(), f"%{symbol}%"),
            ).fetchone()
            conn.close()
            return result[0] if result and result[0] else 0.0
        except:
            return 0.0

    def _get_geopolitical_score_for_symbol(self, symbol):
        """Calcola score geopolitico per simbolo"""
        try:
            sector = self.milan_stocks[symbol]["sector"]
            conn = sqlite3.connect(self.db_path)

            # Score basato su settore
            sector_column_map = {
                "Energy": "energy_impact",
                "Banking": "banking_impact",
                "Automotive": "automotive_impact",
            }

            column = sector_column_map.get(sector, "europe_relevance")

            result = conn.execute(
                f"""
                SELECT AVG({column})
                FROM geopolitical_sentiment
                WHERE date >= ?
            """,
                ((datetime.now() - timedelta(days=7)).date(),),
            ).fetchone()

            conn.close()
            return result[0] if result and result[0] else 0.0
        except:
            return 0.0

    def _get_average_correlation(self, symbol):
        """Calcola correlazione media multi-mercato"""
        try:
            conn = sqlite3.connect(self.db_path)
            result = conn.execute(
                """
                SELECT AVG(ABS(correlation_coefficient))
                FROM market_correlations
                WHERE primary_symbol = ? AND correlation_type = 'pearson'
            """,
                (symbol,),
            ).fetchone()
            conn.close()
            return result[0] if result and result[0] else 0.0
        except:
            return 0.0

    def _auto_refine_models(self, symbol, models_performance):
        """Auto-affinamento modelli basato su performance"""
        try:
            # Verifica se ci sono predizioni passate da validare
            conn = sqlite3.connect(self.db_path)

            # Trova predizioni passate che possono essere validate
            past_predictions = pd.read_sql_query(
                """
                SELECT * FROM ai_predictions
                WHERE symbol = ? AND target_date <= ? AND actual_price IS NULL
                ORDER BY prediction_date DESC LIMIT 50
            """,
                conn,
                params=(symbol, datetime.now().date()),
            )

            if not past_predictions.empty:
                self._validate_past_predictions(symbol, past_predictions, conn)

            # Calcola accuracy trend
            accuracy_trend = self._calculate_accuracy_trend(symbol, conn)

            # Se accuracy è in diminuzione, prova refinement
            if accuracy_trend < -0.05:  # Soglia del 5% di peggioramento
                self._perform_model_refinement(symbol, models_performance, conn)

            conn.close()

        except Exception as e:
            self.logger.warning(f"Errore auto-affinamento {symbol}: {e}")

    def _validate_past_predictions(self, symbol, past_predictions, conn):
        """Valida predizioni passate con prezzi reali"""
        try:
            for _, pred in past_predictions.iterrows():
                target_date = pd.to_datetime(pred["target_date"]).date()

                # Cerca prezzo reale per quella data
                actual_price_result = conn.execute(
                    """
                    SELECT close_price FROM multi_market_stocks
                    WHERE primary_symbol = ? AND market_name = 'Milano' AND date = ?
                """,
                    (symbol, target_date),
                ).fetchone()

                if actual_price_result:
                    actual_price = actual_price_result[0]
                    predicted_price = pred["predicted_price"]

                    # Calcola accuracy
                    accuracy = 1 - abs(actual_price - predicted_price) / actual_price
                    accuracy = max(0, accuracy)  # Non negativa

                    # Aggiorna database
                    conn.execute(
                        """
                        UPDATE ai_predictions
                        SET actual_price = ?, accuracy_score = ?
                        WHERE id = ?
                    """,
                        (actual_price, accuracy, pred["id"]),
                    )

            conn.commit()

        except Exception as e:
            self.logger.warning(f"Errore validazione predizioni: {e}")

    def _calculate_accuracy_trend(self, symbol, conn):
        """Calcola trend di accuracy"""
        try:
            recent_accuracy = conn.execute(
                """
                SELECT AVG(accuracy_score) FROM ai_predictions
                WHERE symbol = ? AND accuracy_score IS NOT NULL
                AND prediction_date >= ?
            """,
                (symbol, (datetime.now() - timedelta(days=30)).date()),
            ).fetchone()

            older_accuracy = conn.execute(
                """
                SELECT AVG(accuracy_score) FROM ai_predictions
                WHERE symbol = ? AND accuracy_score IS NOT NULL
                AND prediction_date BETWEEN ? AND ?
            """,
                (
                    symbol,
                    (datetime.now() - timedelta(days=60)).date(),
                    (datetime.now() - timedelta(days=30)).date(),
                ),
            ).fetchone()

            if recent_accuracy[0] and older_accuracy[0]:
                return recent_accuracy[0] - older_accuracy[0]

            return 0.0

        except:
            return 0.0

    def _perform_model_refinement(self, symbol, models_performance, conn):
        """Esegue raffinamento del modello"""
        try:
            self.logger.info(f"🔧 Auto-raffinamento per {symbol}")

            # Registra refinement nel database
            old_accuracy = (
                conn.execute(
                    """
                SELECT AVG(accuracy_score) FROM ai_predictions
                WHERE symbol = ? AND accuracy_score IS NOT NULL
            """,
                    (symbol,),
                ).fetchone()[0]
                or 0.0
            )

            # Riaddestra con parametri ottimizzati
            features_df = self._prepare_ai_features(symbol)
            if not features_df.empty:
                new_models_performance = self._train_optimized_models(
                    symbol, features_df
                )

                # Calcola miglioramento
                best_new_r2 = max(
                    [
                        max([h["scores"]["r2"] for h in horizons.values()])
                        for horizons in new_models_performance.values()
                        if horizons
                    ],
                    default=0,
                )

                improvement = best_new_r2 - old_accuracy

                # Salva refinement
                conn.execute(
                    """
                    INSERT INTO model_refinement
                    (symbol, refinement_date, old_accuracy, new_accuracy, improvement_pct,
                     refinement_type, changes_made)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        datetime.now(),
                        old_accuracy,
                        best_new_r2,
                        improvement * 100,
                        "hyperparameter_tuning",
                        json.dumps(
                            {"method": "auto_refinement", "trigger": "accuracy_decline"}
                        ),
                    ),
                )

                # Aggiorna modelli in cache
                self.ai_models[symbol] = new_models_performance

                self.logger.info(
                    f"✅ Refinement completato: {improvement*100:.2f}% miglioramento"
                )

        except Exception as e:
            self.logger.error(f"Errore refinement: {e}")

    def _train_optimized_models(self, symbol, features_df):
        """Addestra modelli con iperparametri ottimizzati"""
        # Versione semplificata - in produzione usare GridSearch/RandomSearch
        try:
            feature_columns = self._get_feature_columns(features_df)
            X = features_df[feature_columns]
            y_1d = features_df["target_1d"]

            # Modelli ottimizzati
            optimized_models = {
                "RandomForest_Opt": RandomForestRegressor(
                    n_estimators=150,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                ),
                "GradientBoosting_Opt": GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42
                ),
            }

            # Riutilizza logica di training esistente
            return self._train_models_with_custom_set(symbol, X, y_1d, optimized_models)

        except Exception as e:
            self.logger.error(f"Errore training ottimizzato: {e}")
            return {}

    def _train_models_with_custom_set(self, symbol, X, y, models_dict):
        """Addestra set custom di modelli"""
        # Implementazione semplificata per brevità
        return self._train_ai_models(symbol, pd.concat([X, y], axis=1))

    def export_predictions_to_csv(self, symbol, days_ahead=30):
        """Esporta predizioni in CSV"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Query predizioni recenti
            df = pd.read_sql_query(
                """
                SELECT symbol, prediction_type, target_date, predicted_price,
                       confidence_score, model_used, sentiment_score,
                       geopolitical_score, multi_market_correlation,
                       actual_price, accuracy_score, created_at
                FROM ai_predictions
                WHERE symbol = ? AND target_date >= ?
                ORDER BY target_date, prediction_type
            """,
                conn,
                params=(symbol, datetime.now().date()),
            )

            if not df.empty:
                # Aggiungi informazioni aggiuntive
                df["stock_name"] = self.milan_stocks[symbol]["name"]
                df["sector"] = self.milan_stocks[symbol]["sector"]
                df["export_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Percorso file
                filename = f"predictions_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                filepath = os.path.join(self.csv_dir, filename)

                # Salva CSV
                df.to_csv(filepath, index=False, encoding="utf-8")

                self.logger.info(f"✅ CSV esportato: {filepath}")
                return filepath

            conn.close()

        except Exception as e:
            self.logger.error(f"Errore esportazione CSV: {e}")


import sqlite3
