#!/usr/bin/env python3
"""
Test completo per il sistema MilanoStockPredictionAI
"""

from xta import MilanoStockPredictionAI
from datetime import datetime
import os


def test_prediction(symbol="ENEL.MI"):
    print(f"\n📊 TEST: Predizione per {symbol}")
    ai = MilanoStockPredictionAI()
    result = ai.generate_ai_predictions(symbol)
    if result:
        print(f"✅ Predizioni generate per {symbol}")
        print(f" - Breve termine: {len(result['short_term'])} predizioni")
        print(f" - Lungo termine: {len(result['long_term'])} predizioni")
        export_path = ai.export_predictions_to_csv(symbol)
        print(f"📁 Esportato: {export_path}")
        return True
    else:
        print("❌ Nessuna predizione generata")
        return False


def test_sentiment_update():
    print("\n📰 TEST: Aggiornamento Sentiment Geopolitico")
    ai = MilanoStockPredictionAI()
    try:
        ai._update_geopolitical_sentiment()
        print("✅ Sentiment aggiornato correttamente")
        return True
    except Exception as e:
        print(f"❌ Errore aggiornamento sentiment: {e}")
        return False


def test_database_integrity():
    print("\n🗄️ TEST: Integrità Database")
    ai = MilanoStockPredictionAI()
    try:
        import sqlite3

        conn = sqlite3.connect(ai.db_path)
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        conn.close()
        print(f"📦 Tabelle trovate: {[t[0] for t in tables]}")
        return len(tables) >= 5
    except Exception as e:
        print(f"❌ Errore DB: {e}")
        return False


def run_all_tests():
    print("\n🚀 AVVIO TEST COMPLETO")
    print("=" * 60)
    print(f"🕒 Data/Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {
        "Predizione AI": test_prediction("ENEL.MI"),
        "Aggiorna Sentiment": test_sentiment_update(),
        "Database OK": test_database_integrity(),
    }

    print("\n📋 RISULTATO FINALE")
    for name, result in results.items():
        status = "✅ OK" if result else "❌ FAIL"
        print(f"{status} {name}")


if __name__ == "__main__":
    run_all_tests()
