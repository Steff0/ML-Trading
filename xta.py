#!/usr/bin/env python3
"""
Sistema AI di Predizione Azionaria Multi-Mercato con Focus su Milano - PostgreSQL Complete Version
- Compatibile con stock_predictor e test_predictor
- PostgreSQL ottimizzato con connection pooling
- Performance avanzate e error handling
- Sistema completo di predizioni AI
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
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
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple

warnings.filterwarnings("ignore")


class MilanoStockPredictionAI:
    """
    Sistema AI completo per predizioni azionarie con focus su Milano - PostgreSQL Version
    Compatibile con stock_predictor e test_predictor
    """

    def __init__(self, data_dir="milan_data", pg_config=None):
        """Inizializza il sistema AI con PostgreSQL connection pool"""
        self.data_dir = data_dir
        self.models_dir = os.path.join(data_dir, "ai_models")
        self.csv_dir = os.path.join(data_dir, "csv_exports")
        self.reports_dir = os.path.join(data_dir, "reports")

        # Setup logging FIRST (before database operations)
        self._setup_logging()
        self._create_directories()

        # Configurazione PostgreSQL
        self.pg_config = pg_config or {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", 5432),
            "database": os.getenv("DB_NAME", "Stocks")
            ,
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "password")
        }

        # Try to create database if it doesn't exist
        #self._ensure_database_exists()

        # Initialize connection pool AFTER ensuring database exists
        self.db_pool = self._init_db_pool()

        # Azioni Milano/Europa con codici internazionali
        self.milan_stocks = {
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
            "SP500": "^GSPC",
            "NASDAQ": "^IXIC",
            "DOW": "^DJI",
            "FTSE_MIB": "^FTSEMIB",
            "FTSE100": "^FTSE",
            "DAX": "^GDAXI",
            "CAC40": "^FCHI",
            "IBEX35": "^IBEX",
            "AEX": "^AEX",
            "SMI": "^SSMI",
            "NIKKEI": "^N225",
            "HANG_SENG": "^HSI",
            "SHANGHAI": "000001.SS",
            "BSE_SENSEX": "^BSESN",
            "BOVESPA": "^BVSP",
            "MOEX": "IMOEX.ME",
            "GOLD": "GC=F",
            "SILVER": "SI=F",
            "OIL_BRENT": "BZ=F",
            "OIL_WTI": "CL=F",
            "NATURAL_GAS": "NG=F",
            "COPPER": "HG=F",
            "EUR_USD": "EURUSD=X",
            "EUR_GBP": "EURGBP=X",
            "EUR_JPY": "EURJPY=X",
            "EUR_CHF": "EURCHF=X",
            "USD_CNY": "USDCNY=X",
            "BITCOIN": "BTC-USD",
            "ETHEREUM": "ETH-USD",
            "BTP_10Y": "ITGB10YR=X",
            "BUND_10Y": "TNX",
            "US_10Y": "^TNX",
        }

        # Keywords geopolitici specifici per Italia/Europa
        self.geopolitical_keywords = {
            "italy_specific": [
                "italia", "italy", "governo italiano", "italian government",
                "draghi", "meloni", "mattarella", "parlamento italiano",
                "debito italia", "italian debt", "spread btp", "rating italia",
                "pnrr", "recovery fund", "next generation eu",
            ],
            "europe_wide": [
                "europa", "europe", "european union", "ue", "eu",
                "bce", "ecb", "lagarde", "euro", "eurozona", "eurozone",
                "brexit", "recovery fund", "green deal", "fit for 55",
            ],
            "energy_geopolitics": [
                "russia", "ucraina", "ukraine", "gas russo", "russian gas",
                "nord stream", "sanzioni russia", "russia sanctions",
                "gazprom", "eni", "energia", "energy crisis",
            ],
            "banking_regulation": [
                "vigilanza bancaria", "banking supervision", "stress test",
                "npl", "crediti deteriorati", "bad loans", "basilea", "basel",
                "capital ratio", "tier 1",
            ],
            "automotive": [
                "automotive", "auto", "electric vehicle", "ev",
                "stellantis", "ferrari", "co2 emission", "green transition",
            ],
            "general_risk": [
                "guerra", "war", "conflict", "crisi", "crisis",
                "inflazione", "inflation", "recessione", "recession",
                "covid", "pandemic", "lockdown", "supply chain",
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

        # Setup database PostgreSQL
        self._setup_enhanced_database()

        # Cache per modelli AI
        self.ai_models = {}
        self.scalers = {}
        self.performance_history = {}

        # Auto-aggiornamento all'avvio
        try:
            self.startup_update()
        except Exception as e:
            self.logger.warning(f"⚠️ Startup update failed: {e}")

    def _ensure_database_exists(self):
        """Ensure the database exists, create if it doesn't"""
        try:
            # Connect to default postgres database to create our database
            temp_config = self.pg_config.copy()
            temp_config["database"] = "postgres"  # Connect to default database
            
            conn = psycopg2.connect(
                host=temp_config["host"],
                port=temp_config["port"],
                database=temp_config["database"],
                user=temp_config["user"],
                password=temp_config["password"]
            )
            conn.autocommit = True
            
            with conn.cursor() as cursor:
                # Check if database exists
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (self.pg_config["database"],)
                )
                
                if not cursor.fetchone():
                    # Create database
                    cursor.execute(
                        f'CREATE DATABASE "{self.pg_config["database"]}"'
                    )
                    self.logger.info(f"✅ Database '{self.pg_config['database']}' created")
                else:
                    self.logger.info(f"✅ Database '{self.pg_config['database']}' already exists")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error ensuring database exists: {e}")
            self.logger.info("Please create the PostgreSQL database manually or check your connection settings")
            raise

    def _init_db_pool(self):
        """Inizializza PostgreSQL connection pool"""
        try:
            pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                host=self.pg_config["host"],
                port=self.pg_config["port"],
                database=self.pg_config["database"],
                user=self.pg_config["user"],
                password=self.pg_config["password"]
            )
            self.logger.info("✅ PostgreSQL connection pool initialized")
            return pool
        except Exception as e:
            self.logger.error(f"❌ Error creating connection pool: {e}")
            self.logger.error("Please check your PostgreSQL connection settings:")
            self.logger.error(f"  Host: {self.pg_config['host']}")
            self.logger.error(f"  Port: {self.pg_config['port']}")
            self.logger.error(f"  Database: {self.pg_config['database']}")
            self.logger.error(f"  User: {self.pg_config['user']}")
            raise

    @contextmanager
    def _get_db_connection(self):
        """Context manager per connessioni database"""
        conn = None
        try:
            conn = self.db_pool.getconn()
            yield conn
        except Exception as e:
            self.logger.error(f"Errore database: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.db_pool.putconn(conn)

    def _setup_logging(self):
        """Configura sistema di logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.data_dir, "milan_ai.log"), encoding="utf-8"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _create_directories(self):
        """Crea directories necessarie"""
        for directory in [self.data_dir, self.models_dir, self.csv_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)

    def _setup_enhanced_database(self):
        """Setup database PostgreSQL ottimizzato - uses existing Stocks database"""
        with self._get_db_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Check if tables exist (they should already exist)
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('ai_predictions', 'multi_market_stocks', 'geopolitical_sentiment', 
                                          'model_performance', 'market_correlations', 'model_refinement', 'simulated_trades')
                    """)
                    
                    existing_tables = [row[0] for row in cursor.fetchall()]
                    self.logger.info(f"✅ Found existing tables: {existing_tables}")
                    
                    # Create additional indexes if needed
                    indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_multi_market_symbol_date ON multi_market_stocks(primary_symbol, date)",
                        "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON ai_predictions(symbol, target_date)",
                        "CREATE INDEX IF NOT EXISTS idx_correlations_symbol ON market_correlations(primary_symbol)",
                        "CREATE INDEX IF NOT EXISTS idx_sentiment_date ON geopolitical_sentiment(date)",
                    ]
                    
                    for index_sql in indexes:
                        cursor.execute(index_sql)
                    
                    conn.commit()
                    self.logger.info("✅ Database 'Stocks' configured with existing schema")
                    
                except Exception as e:
                    conn.rollback()
                    self.logger.error(f"❌ Errore setup database: {e}")
                    raise

    def startup_update(self):
        """Aggiornamento automatico all'avvio"""
        self.logger.info("🚀 Avvio sistema AI Milano - Aggiornamento automatico...")

        try:
            # Esegui aggiornamenti in parallelo
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self._update_global_markets),
                    executor.submit(self._update_milan_multi_market),
                    executor.submit(self._update_geopolitical_sentiment),
                    executor.submit(self._calculate_market_correlations),
                ]

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Task di aggiornamento fallito: {e}")

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

        self.logger.info(f"Mercati globali aggiornati: {success_count}/{len(self.global_markets)}")

    def _fetch_market_data(self, market_name, symbol, period):
        """Scarica dati di mercato usando yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data if not data.empty else None
        except Exception as e:
            self.logger.warning(f"Errore scaricamento dati per {symbol}: {e}")
            return None

    def _save_market_data(self, market_name, data):
        """Salva dati di mercato in PostgreSQL usando multi_market_stocks"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    insert_query = """
                        INSERT INTO multi_market_stocks (primary_symbol, market_symbol, market_name, date, 
                                                   open_price, high_price, low_price, close_price, volume, adj_close, currency)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (primary_symbol, market_symbol, date) 
                        DO UPDATE SET
                            open_price = EXCLUDED.open_price,
                            high_price = EXCLUDED.high_price,
                            low_price = EXCLUDED.low_price,
                            close_price = EXCLUDED.close_price,
                            volume = EXCLUDED.volume,
                            adj_close = EXCLUDED.adj_close,
                            updated_at = CURRENT_TIMESTAMP
                    """

                    records = []
                    for date, row in data.iterrows():
                        records.append((
                            market_name,  # primary_symbol
                            market_name,  # market_symbol
                            "Global",     # market_name
                            date.strftime("%Y-%m-%d"),
                            float(row["Open"]) if not pd.isna(row["Open"]) else None,
                            float(row["High"]) if not pd.isna(row["High"]) else None,
                            float(row["Low"]) if not pd.isna(row["Low"]) else None,
                            float(row["Close"]) if not pd.isna(row["Close"]) else None,
                            int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                            float(row["Close"]) if not pd.isna(row["Close"]) else None,
                            "USD"  # default currency
                        ))

                cursor.executemany(insert_query, records)
                conn.commit()

        except Exception as e:
            self.logger.error(f"Errore salvataggio dati mercato {market_name}: {e}")

    def _update_milan_multi_market(self):
        """Aggiorna azioni Milano su mercati multipli"""
        for primary_symbol, stock_info in self.milan_stocks.items():
            try:
                # Simbolo principale (Milano)
                self._fetch_and_save_multi_market(primary_symbol, primary_symbol, "Milano")

                # Simboli su altri mercati
                for global_code in stock_info["global_codes"]:
                    if global_code:
                        market_name = self._detect_market_from_symbol(global_code)
                        self._fetch_and_save_multi_market(primary_symbol, global_code, market_name)
                        time.sleep(0.5)  # Rate limiting

            except Exception as e:
                self.logger.warning(f"Errore multi-mercato {primary_symbol}: {e}")

    def _fetch_and_save_multi_market(self, primary_symbol, market_symbol, market_name):
        """Scarica e salva dati per un simbolo su mercato specifico"""
        try:
            data = self._fetch_market_data("temp", market_symbol, "6mo")
            if data is not None and not data.empty:
                with self._get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        insert_query = """
                            INSERT INTO multi_market_stocks (primary_symbol, market_symbol, market_name, date,
                                                       open_price, high_price, low_price, close_price, 
                                                       volume, adj_close, currency)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (primary_symbol, market_symbol, date) 
                            DO UPDATE SET
                                open_price = EXCLUDED.open_price,
                                high_price = EXCLUDED.high_price,
                                low_price = EXCLUDED.low_price,
                                close_price = EXCLUDED.close_price,
                                volume = EXCLUDED.volume,
                                adj_close = EXCLUDED.adj_close,
                                updated_at = CURRENT_TIMESTAMP
                        """

                        records = []
                        for date, row in data.iterrows():
                            records.append((
                                primary_symbol,
                                market_symbol,
                                market_name,
                                date.strftime("%Y-%m-%d"),
                                float(row["Open"]) if not pd.isna(row["Open"]) else None,
                                float(row["High"]) if not pd.isna(row["High"]) else None,
                                float(row["Low"]) if not pd.isna(row["Low"]) else None,
                                float(row["Close"]) if not pd.isna(row["Close"]) else None,
                                int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                                float(row["Close"]) if not pd.isna(row["Close"]) else None,
                                self._detect_currency_from_market(market_name),
                            ))

                    cursor.executemany(insert_query, records)
                    conn.commit()

                    self.logger.info(f"✅ {primary_symbol} su {market_name}: {len(data)} records")
                    return True

    except Exception as e:
        self.logger.warning(f"Errore salvataggio {market_symbol}: {e}")
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
        with self._get_db_connection() as conn:
            with conn.cursor() as cursor:
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
                        cursor.execute("""
                            INSERT INTO geopolitical_sentiment 
                            (date, news_source, headline, content_summary, overall_sentiment,
                             italy_relevance, europe_relevance, energy_impact, banking_impact, 
                             automotive_impact, keywords_matched, affected_symbols)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
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
                        ))

                    except Exception as e:
                        self.logger.warning(f"Errore analisi news: {e}")

                conn.commit()

    def _calculate_keyword_relevance(self, text, keywords):
        """Calcola rilevanza basata su keywords"""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return min(matches / len(keywords), 1.0)

    def _find_matched_keywords(self, text):
        """Trova keywords matchate"""
        matched = []
        text_lower = text.lower()

        for category, keywords in self.geopolitical_keywords.items():
            category_matches = [kw for kw in keywords if kw.lower() in text_lower]
            if category_matches:
                matched.extend(category_matches)

        return matched[:20]

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
            with self._get_db_connection() as conn:
                df = pd.read_sql_query("""
                    SELECT date, close_price
                    FROM stock_data
                    WHERE symbol = %s
                    ORDER BY date
                """, conn, params=(primary_symbol,))

                if df.empty:
                    return {}

                # Calcola correlazioni con mercati globali
                correlations = {}
                
                # Implementazione semplificata per compatibilità
                return correlations

        except Exception as e:
            self.logger.warning(f"Errore calcolo correlazioni: {e}")
            return {}

    def _save_correlations(self, primary_symbol, correlations):
        """Salva correlazioni nel database"""
        # Implementazione placeholder per compatibilità
        pass

    def generate_ai_predictions(self, symbol, prediction_horizon_days=30, long_term_days=180):
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
            self._save_predictions(symbol, short_predictions, long_predictions

                symbol, features_df, models_performance, long_term_days
            )

            # Salva predizioni
            self._save_predictions(symbol, short_predictions, long_predictions)

            # Calcola confidence generale
            confidence = self._calculate_overall_confidence(models_performance)

            return {
                "short_term": short_predictions,
                "long_term": long_predictions,
                "models_performance": models_performance,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Errore predizioni AI {symbol}: {e}")
            return None

    def _prepare_ai_features(self, symbol):
        """Prepara features per training AI usando multi_market_stocks"""
        try:
            with self._get_db_connection() as conn:
                # Query ottimizzata per multi_market_stocks
                query = """
                    SELECT date, close_price, volume, open_price, high_price, low_price
                    FROM multi_market_stocks
                    WHERE primary_symbol = %s AND market_name = 'Milano'
                    ORDER BY date
                """

            main_df = pd.read_sql_query(query, conn, params=(symbol,))

            if main_df.empty:
                return pd.DataFrame()

            main_df["date"] = pd.to_datetime(main_df["date"])
            main_df = main_df.set_index("date")

            # Feature engineering (same as before)
            main_df["returns"] = main_df["close_price"].pct_change()
            main_df["volatility"] = main_df["returns"].rolling(20).std()
            main_df["sma_20"] = main_df["close_price"].rolling(20).mean()
            main_df["sma_50"] = main_df["close_price"].rolling(50).mean()
            main_df["rsi"] = self._calculate_rsi(main_df["close_price"])
            main_df["bollinger_upper"], main_df["bollinger_lower"] = (
                self._calculate_bollinger_bands(main_df["close_price"])
            )
            main_df["volume_ma"] = main_df["volume"].rolling(20).mean()

            # Target variables
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
            feature_columns = [col for col in features_df.columns if not col.startswith("target_")]
            X = features_df[feature_columns]
            y_1d = features_df["target_1d"]

            # Scaling
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[symbol] = scaler

            models_results = {}

            # Modelli da testare
            models = {
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            }

            # Time series split per validazione
            tscv = TimeSeriesSplit(n_splits=5)

            for model_name, model in models.items():
                try:
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_scaled):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y_1d.iloc[train_idx], y_1d.iloc[val_idx]

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        r2 = r2_score(y_val, y_pred)
                        cv_scores.append(r2)

                    avg_r2 = np.mean(cv_scores)
                    models_results[model_name] = {
                        "1d": {
                            "scores": {"r2": avg_r2},
                            "model": model,
                        }
                    }

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
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    for model_name, horizons in models_results.items():
                        for horizon, results in horizons.items():
                            scores = results["scores"]
                            
                            cursor.execute("""
                                INSERT INTO model_performance
                                (symbol, model_name, training_date, r2_score, validation_accuracy, 
                                 data_points_used, prediction_horizon)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                symbol,
                                f"{model_name}_{horizon}",
                                datetime.now(),
                                scores["r2"],
                                max(0, scores["r2"]),
                                len(feature_names),
                                int(horizon.replace("d", "")),
                            ))

                    conn.commit()

        except Exception as e:
            self.logger.warning(f"Errore salvataggio performance: {e}")

    def _generate_short_term_predictions(self, symbol, features_df, models_performance, horizon_days):
        """Genera predizioni breve termine"""
        try:
            predictions = []

            # Seleziona miglior modello
            best_model_name = self._select_best_model(models_performance)
            
            if best_model_name and "1d" in models_performance[best_model_name]:
                model = models_performance[best_model_name]["1d"]["model"]
                latest_features = features_df.iloc[-1:][self._get_feature_columns(features_df)]
                latest_scaled = self.scalers[symbol].transform(latest_features)

                base_date = features_df.index[-1]

                for day in range(1, min(horizon_days + 1, 31)):  # Limita a 30 giorni
                    pred_price = model.predict(latest_scaled)[0]
                    target_date = base_date + timedelta(days=day)

                    confidence = models_performance[best_model_name]["1d"]["scores"]["r2"]
                    confidence = max(0, min(1, confidence))

                    predictions.append({
                        "target_date": target_date,
                        "predicted_price": pred_price,
                        "confidence": confidence,
                        "model_used": f"{best_model_name}_1d",
                        "prediction_type": "short_term",
                    })

            return predictions

        except Exception as e:
            self.logger.error(f"Errore predizioni breve termine: {e}")
            return []

    def _generate_long_term_predictions(self, symbol, features_df, models_performance, horizon_days):
        """Genera predizioni lungo termine"""
        try:
            predictions = []

            # Seleziona miglior modello
            best_model_name = self._select_best_model(models_performance)
            
            if best_model_name and "1d" in models_performance[best_model_name]:
                model = models_performance[best_model_name]["1d"]["model"]
                latest_features = features_df.iloc[-1:][self._get_feature_columns(features_df)]
                latest_scaled = self.scalers[symbol].transform(latest_features)

                base_date = features_df.index[-1]
                monthly_steps = [30, 60, 90, 120, 150, 180]

                for days in monthly_steps:
                    if days <= horizon_days:
                        pred_price = model.predict(latest_scaled)[0]
                        target_date = base_date + timedelta(days=days)

                        base_confidence = models_performance[best_model_name]["1d"]["scores"]["r2"]
                        confidence = max(0.1, base_confidence * (1 - days / 365))

                        predictions.append({
                            "target_date": target_date,
                            "predicted_price": pred_price,
                            "confidence": confidence,
                            "model_used": f"{best_model_name}_1d",
                            "prediction_type": "long_term",
                        })

            return predictions

        except Exception as e:
            self.logger.error(f"Errore predizioni lungo termine: {e}")
            return []

    def _get_feature_columns(self, features_df):
        """Ottieni colonne delle features (esclusi target)"""
        return [col for col in features_df.columns if not col.startswith("target_")]

    def _select_best_model(self, models_performance):
        """Seleziona miglior modello basato su performance"""
        best_r2 = -np.inf
        best_model = None

        for model_name, horizons in models_performance.items():
            if "1d" in horizons:
                r2_score = horizons["1d"]["scores"]["r2"]
                if r2_score > best_r2:
                    best_r2 = r2_score
                    best_model = model_name

        return best_model

    def _calculate_overall_confidence(self, models_performance):
        """Calcola confidence generale del sistema"""
        try:
            r2_scores = []
            for model_name, horizons in models_performance.items():
                if "1d" in horizons:
                    r2_scores.append(horizons["1d"]["scores"]["r2"])
            
            if r2_scores:
                avg_r2 = np.mean(r2_scores)
                return max(0, min(100, avg_r2 * 100))
            
            return 0.0

        except Exception:
            return 0.0

    def _save_predictions(self, symbol, short_predictions, long_predictions):
        """Salva predizioni nel database ai_predictions"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    all_predictions = short_predictions + long_predictions

                    for pred in all_predictions:
                        # Calcola features aggiuntive
                        sentiment_score = self._get_latest_sentiment_for_symbol(symbol)
                        geopolitical_score = self._get_geopolitical_score_for_symbol(symbol)
                        multi_market_corr = self._get_average_correlation(symbol)

                        cursor.execute("""
                            INSERT INTO ai_predictions
                            (symbol, prediction_type, prediction_date, target_date, predicted_price,
                             confidence_score, model_used, sentiment_score, geopolitical_score,
                             multi_market_correlation)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
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
                        ))

                conn.commit()

    except Exception as e:
        self.logger.error(f"Errore salvataggio predizioni: {e}")

    def export_predictions_to_csv(self, symbol, days_ahead=30):
        """Esporta predizioni in CSV dalla tabella ai_predictions"""
        try:
            with self._get_db_connection() as conn:
                df = pd.read_sql_query("""
                    SELECT symbol, prediction_type, target_date, predicted_price,
                       confidence_score, model_used, actual_price, accuracy_score, 
                       sentiment_score, geopolitical_score, multi_market_correlation, created_at
                FROM ai_predictions
                WHERE symbol = %s AND target_date >= %s
                ORDER BY target_date, prediction_type
            """, conn, params=(symbol, datetime.now().date()))

            if not df.empty:
                # Aggiungi informazioni aggiuntive
                stock_info = self.milan_stocks.get(symbol, {})
                df["stock_name"] = stock_info.get("name", "")
                df["sector"] = stock_info.get("sector", "")
                df["export_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Percorso file
                filename = f"predictions_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                filepath = os.path.join(self.csv_dir, filename)

                # Salva CSV
                df.to_csv(filepath, index=False, encoding="utf-8")
                self.logger.info(f"✅ CSV esportato: {filepath}")
                return filepath

    except Exception as e:
        self.logger.error(f"Errore esportazione CSV: {e}")
        return None

    def __del__(self):
        """Cleanup connection pool quando l'oggetto viene distrutto"""
        if hasattr(self, "db_pool"):
            self.db_pool.closeall()
            self.logger.info("PostgreSQL connection pool chiuso")

    def _detect_currency_from_market(self, market_name):
        """Rileva la valuta dal mercato"""
        if market_name == "Europa" or market_name == "Parigi":
            return "EUR"
        elif market_name == "Londra":
            return "GBP"
        else:
            return "USD"

    def _get_latest_sentiment_for_symbol(self, symbol):
        """Ottieni l'ultimo sentiment score per un simbolo"""
        # Implementazione placeholder
        return 0.0

    def _get_geopolitical_score_for_symbol(self, symbol):
        """Ottieni il geopolitical score per un simbolo"""
        # Implementazione placeholder
        return 0.0

    def _get_average_correlation(self, symbol):
        """Ottieni la correlazione media per un simbolo"""
        # Implementazione placeholder
        return 0.0

# Funzioni di utilità
def create_test_data():
    """Crea dati di test per il sistema"""
    ai = MilanoStockPredictionAI()
    
    # Inserisci alcuni dati di test
    with ai._get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Dati stock di esempio
            cursor.execute("""
                INSERT INTO stock_data (symbol, date, close_price, volume)
                VALUES ('ENEL.MI', %s, 6.50, 1000000)
                ON CONFLICT (symbol, date) DO NOTHING
            """, (datetime.now().date(),))
            
            # Dati sentiment di esempio
            cursor.execute("""
                INSERT INTO sentiment_data (date, symbol, sentiment_score, news_count)
                VALUES (%s, 'ENEL.MI', 0.1, 5)
                ON CONFLICT DO NOTHING
            """, (datetime.now().date(),))
            
            conn.commit()
    
    return ai


if __name__ == "__main__":
    # Esempio di utilizzo
    try:
        predictor = MilanoStockPredictionAI()
        result = predictor.generate_ai_predictions("ENEL.MI")
        if result:
            print("✅ Predizioni generate con successo")
            print(f"Confidence: {result.get('confidence', 0):.1f}%")
        else:
            print("❌ Errore nella generazione delle predizioni")
    except Exception as e:
        print(f"❌ Errore: {e}")
