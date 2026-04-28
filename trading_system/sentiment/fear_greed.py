"""
Fear & Greed Index de Alternative.me — fuente gratuita, sin API key.

Endpoint: https://api.alternative.me/fng/?limit=7
Escala:   0 (Extreme Fear) → 100 (Extreme Greed)

Lógica contrarian para trading:
  - Extreme Fear  →  score positivo  (mercado en pánico = oportunidad de compra)
  - Extreme Greed →  score negativo  (mercado eufórico  = señal de precaución)
"""
import time
import requests
from loguru import logger

_FNG_URL   = "https://api.alternative.me/fng/"
_CACHE: dict = {}          # {ts_floor: (value, classification, score)}
_CACHE_TTL = 3600          # segundos — el índice se actualiza 1× al día


def fetch_fear_greed(days: int = 7) -> list[dict]:
    """
    Retorna los últimos `days` registros del Fear & Greed Index.

    Returns:
        Lista de dicts con:
          value            int  — 0–100
          classification   str  — "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"
          timestamp        int  — Unix timestamp UTC
    """
    cache_key = int(time.time()) // _CACHE_TTL
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    try:
        resp = requests.get(
            _FNG_URL,
            params={"limit": days, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        result = [
            {
                "value":          int(d["value"]),
                "classification": d["value_classification"],
                "timestamp":      int(d["timestamp"]),
            }
            for d in data
        ]
        _CACHE[cache_key] = result
        logger.debug(f"Fear & Greed: {result[0]['value']} ({result[0]['classification']})")
        return result

    except Exception as exc:
        logger.warning(f"Fear & Greed Index no disponible: {exc}")
        return []


def fng_to_sentiment_score(value: int) -> float:
    """
    Convierte el valor F&G (0–100) a score de sentimiento contrarian (-1 a +1).

    Contrarian porque:
      - Extreme Fear (compra cuando todos huyen)  →  score alto positivo
      - Extreme Greed (vende cuando todos eufóricos) →  score negativo
    """
    # Normalizar a [0, 1] y aplicar reflexión contrarian
    normalized = value / 100.0
    contrarian = 1.0 - normalized          # invierte la escala
    score = (contrarian - 0.5) * 2.0      # escalar a [-1, +1]
    return round(float(score), 3)


def get_current_fng() -> dict:
    """
    Retorna el estado actual del Fear & Greed Index listo para consumo.

    Returns:
        dict con:
          fg_value          int   — 0–100
          fg_classification str
          score             float — contrarian score [-1, +1]
          label             str   — "MIEDO EXTREMO" etc.
          color             str   — "green" | "red" | "gray"
          trend_7d          float — cambio del índice en los últimos 7 días
    """
    data = fetch_fear_greed(days=7)
    if not data:
        return _neutral()

    current = data[0]
    value   = current["value"]
    cls     = current["classification"]
    score   = fng_to_sentiment_score(value)

    # Tendencia 7 días (valor actual - valor hace 7 días)
    trend_7d = float(value - data[-1]["value"]) if len(data) >= 7 else 0.0

    label, color = _label_color(value)

    return {
        "fg_value":         value,
        "fg_classification":cls,
        "score":            score,
        "label":            label,
        "color":            color,
        "trend_7d":         trend_7d,
        "n_news":           0,
        "latest_headlines": [],
    }


def _label_color(value: int) -> tuple[str, str]:
    if value <= 25:
        return "MIEDO EXTREMO", "red"
    if value <= 46:
        return "MIEDO", "orange"
    if value <= 54:
        return "NEUTRAL", "gray"
    if value <= 75:
        return "CODICIA", "green"
    return "CODICIA EXTREMA", "red"


def _neutral() -> dict:
    return {
        "fg_value": 50, "fg_classification": "Neutral",
        "score": 0.0, "label": "NEUTRAL", "color": "gray",
        "trend_7d": 0.0, "n_news": 0, "latest_headlines": [],
    }
