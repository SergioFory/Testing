"""
Módulo de sentimiento de mercado.

Fuente primaria:  Fear & Greed Index (Alternative.me) — gratuito, sin API key.
Fuente secundaria: análisis de keywords sobre noticias (fallback sin red).

El plan gratuito de CryptoPanic fue discontinuado el 1 de abril de 2026.

API pública mantenida:
  get_sentiment_summary(symbol)  →  dict con score, label, color, fg_value …
  should_veto_signal(direction, score) → (bool, reason)
"""
from loguru import logger
from config.settings import SENTIMENT
from sentiment.fear_greed import get_current_fng


def get_sentiment_summary(symbol: str) -> dict:
    """
    Retorna el resumen de sentimiento para un símbolo.

    Para BTC y ETH usa el Fear & Greed Index (contrarian).
    Para activos no-crypto (GOLD, SILVER, SP500) retorna neutral
    (el F&G Index es específico de crypto).

    Estructura del dict devuelto:
      score             float  [-1, +1] — contrarian score
      label             str    — "MIEDO EXTREMO" | "MIEDO" | "NEUTRAL" | "CODICIA" | "CODICIA EXTREMA"
      color             str    — "green" | "red" | "gray" | "orange"
      fg_value          int    — valor 0-100 del índice (0 si no aplica)
      fg_classification str    — clasificación en inglés de Alternative.me
      trend_7d          float  — cambio del índice vs hace 7 días
      n_news            int    — siempre 0 (no usamos feed de noticias)
      latest_headlines  list   — siempre []
    """
    sym_upper = symbol.upper().replace("USDT", "").replace("USD", "")

    non_crypto = ("GOLD", "XAU", "GC", "SILVER", "XAG", "SI",
                  "SP500", "SPX", "SPX500", "GSPC", "SPY")
    if sym_upper in non_crypto:
        return {
            "score": 0.0, "label": "NEUTRAL", "color": "gray",
            "fg_value": 0, "fg_classification": "N/A",
            "trend_7d": 0.0, "n_news": 0, "latest_headlines": [],
        }

    return get_current_fng()


def should_veto_signal(direction: str, sentiment_score: float) -> tuple:
    """
    Decide si el sentimiento veta la señal.

    Lógica contrarian:
      - LONG vetado cuando score < veto_threshold (mercado en euforia extrema)
      - SHORT vetado cuando score > -veto_threshold (mercado en pánico extremo)

    Returns:
        (vetoed: bool, reason: str)
    """
    veto = SENTIMENT["veto_threshold"]    # default -0.4

    if direction == "long" and sentiment_score < veto:
        return True, (
            f"Mercado en codicia extrema (score={sentiment_score:.2f}). "
            "Señal LONG vetada."
        )
    if direction == "short" and sentiment_score > abs(veto):
        return True, (
            f"Mercado en miedo extremo (score={sentiment_score:.2f}). "
            "Señal SHORT vetada."
        )

    return False, "OK"
