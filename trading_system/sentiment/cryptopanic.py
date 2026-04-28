"""
Módulo de sentimiento de noticias.

Fuente: CryptoPanic API (gratis con registro)
  → https://cryptopanic.com/developers/api/

Fallback: FinBERT local o análisis de keywords simple.
"""
import re
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import CRYPTOPANIC_KEY, SENTIMENT


CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1/posts/"

# Keywords para análisis simple sin API
BULLISH_KEYWORDS = [
    "rally", "surge", "breakout", "all-time high", "ath", "bullish",
    "adoption", "etf", "institutional", "buy", "upgrade", "partnership",
    "launc", "record", "soar", "climb", "gain", "recovery",
]
BEARISH_KEYWORDS = [
    "crash", "dump", "hack", "ban", "regulation", "bearish", "fear",
    "sell", "drop", "decline", "loss", "lawsuit", "sec", "investigation",
    "scam", "fraud", "delist", "warning", "exploit", "breach",
]


def fetch_news(symbol: str = "BTC",
               hours: int = 24,
               limit: int = 50) -> list:
    """
    Descarga noticias recientes de CryptoPanic.

    Args:
        symbol: "BTC", "ETH", "XAU" (para Oro usa keywords)
        hours:  Ventana temporal hacia atrás
        limit:  Número máximo de noticias

    Returns:
        Lista de dicts con campos: title, published_at, votes, kind
    """
    if not CRYPTOPANIC_KEY:
        logger.debug("Sin API key de CryptoPanic. Usando fallback.")
        return []

    currencies_map = {"BTC": "BTC", "ETH": "ETH", "GOLD": "PAXG,XAUT"}
    currencies = currencies_map.get(symbol.upper(), "BTC")

    params = {
        "auth_token":  CRYPTOPANIC_KEY,
        "currencies":  currencies,
        "kind":        "news",
        "public":      "true",
        "limit":       limit,
    }
    try:
        resp = requests.get(CRYPTOPANIC_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        news = data.get("results", [])
        logger.debug(f"CryptoPanic: {len(news)} noticias para {symbol}")
        return news
    except Exception as e:
        logger.warning(f"CryptoPanic no disponible: {e}")
        return []


def compute_sentiment_score(
    news:   list,
    symbol: str = "BTC",
) -> float:
    """
    Calcula un score de sentimiento agregado de -1 a +1.

    Método:
      - Si hay votos de CryptoPanic: usa (bullish_votes - bearish_votes) / total
      - Si no hay votos: análisis de keywords en títulos
      - Pondera por tiempo (noticias recientes pesan más)

    Returns:
        float en [-1, +1]. 0 = neutral.
    """
    if not news:
        return 0.0

    now    = datetime.now(timezone.utc)
    scores = []
    weights= []

    for item in news:
        # Score por votos (si CryptoPanic los provee)
        votes = item.get("votes", {})
        if votes:
            bullish = votes.get("positive",  0) + votes.get("liked",   0)
            bearish = votes.get("negative",  0) + votes.get("disliked",0)
            total   = bullish + bearish
            if total > 0:
                score = (bullish - bearish) / total
            else:
                score = 0.0
        else:
            # Fallback: análisis de keywords en el título
            title = (item.get("title", "") + " " + item.get("slug", "")).lower()
            score = _keyword_score(title)

        # Peso temporal (noticias < 4h pesan 2x)
        pub_at_str = item.get("published_at", "")
        try:
            pub_at = datetime.fromisoformat(pub_at_str.replace("Z", "+00:00"))
            age_h  = (now - pub_at).total_seconds() / 3600
            weight = 2.0 if age_h < 4 else (1.0 if age_h < 12 else 0.5)
        except Exception:
            weight = 1.0

        scores.append(score)
        weights.append(weight)

    if not scores:
        return 0.0

    weighted_score = float(np.average(scores, weights=weights))
    # Normalizar a [-1, +1]
    return float(np.clip(weighted_score, -1.0, 1.0))


def _keyword_score(text: str) -> float:
    """Score de sentimiento basado en keywords. Rango [-1, +1]."""
    bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return (bull_count - bear_count) / total


def should_veto_signal(
    direction:       str,
    sentiment_score: float,
) -> tuple:
    """
    Decide si el sentimiento veta la señal.

    Returns:
        (vetoed: bool, reason: str)
    """
    veto = SENTIMENT["veto_threshold"]
    boost= SENTIMENT["boost_threshold"]

    if direction == "long" and sentiment_score < veto:
        return True, f"Sentimiento muy negativo ({sentiment_score:.2f} < {veto})"
    if direction == "short" and sentiment_score > -veto:
        return False, f"Sentimiento no confirma SHORT ({sentiment_score:.2f})"

    return False, "OK"


def get_sentiment_summary(symbol: str) -> dict:
    """
    Retorna un resumen del sentimiento para mostrar en el dashboard.
    """
    news  = fetch_news(symbol, hours=SENTIMENT["lookback_hours"])
    score = compute_sentiment_score(news, symbol)

    if score > SENTIMENT["boost_threshold"]:
        label, color = "ALCISTA", "green"
    elif score < SENTIMENT["veto_threshold"]:
        label, color = "BAJISTA", "red"
    else:
        label, color = "NEUTRAL", "gray"

    return {
        "score":      round(score, 3),
        "label":      label,
        "color":      color,
        "n_news":     len(news),
        "latest_headlines": [n.get("title", "")[:80] for n in news[:5]],
    }
