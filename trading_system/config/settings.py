"""
Configuración central del sistema de trading.
Todos los parámetros operativos viven aquí.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# RUTAS
# =============================================================================
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data" / "cache"
DB_PATH    = BASE_DIR / "data" / "trading.db"
LOG_DIR    = BASE_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CREDENCIALES
# =============================================================================
def _clean_env(key: str) -> str:
    """Lee variable de entorno e ignora placeholders del .env.example."""
    val = os.getenv(key, "").strip()
    placeholders = {
        "tu_api_key_aqui", "tu_api_secret_aqui", "tu_key_aqui",
        "tu_bot_token", "tu_chat_id",
        "your_api_key_here", "your_secret_here", "changeme",
    }
    if val.lower() in placeholders:
        return ""
    return val


BINANCE_API_KEY    = _clean_env("BINANCE_API_KEY")
BINANCE_API_SECRET = _clean_env("BINANCE_API_SECRET")
ANTHROPIC_KEY      = _clean_env("ANTHROPIC_API_KEY")
TELEGRAM_TOKEN     = _clean_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _clean_env("TELEGRAM_CHAT_ID")

# CryptoPanic discontinuó su plan gratuito el 01-Apr-2026.
# Sentimiento ahora usa Fear & Greed Index (Alternative.me) — sin API key.
CRYPTOPANIC_KEY    = ""  # deprecated — mantenido para no romper imports antiguos

# =============================================================================
# ACTIVOS
# =============================================================================
ASSETS = {
    "BTCUSDT": {
        "label":        "BTC",
        "source":       "binance",
        "exchange":     "binanceusdm",
        "min_range_pct": 0.015,
        "pip_value":    1.0,
        "days_history": 1825,
        "forward_bars": 24,
        "ml_threshold": 0.55,   # era 0.40 — umbral más selectivo tras regularización
        "use_macro":    False,  # macro 30d añade ruido en 4H crypto: precisión bajó 0.193→0.128
        # R:R 1.5 (era 2.0): con 85% breakouts dominando el dataset, win rate al
        # 38% deja al ML aprendiendo sobre fakeouts. TP a 2.25×ATR sube el win
        # rate esperado a ~45-50%, mejorando el balance de clases del entrenamiento.
        # Break-even precision: 40% (era 33%, alcanzable con la mejora de win rate).
        "atr_target_mult": 2.25,
        "atr_stop_mult":   1.5,
        # Ventana acotada al régimen bull actual (Nov 2024 - presente).
        # Con 1095d el AUC era 0.474 (anti-predictivo): el modelo aprendía
        # patrones del bear 2022 que no transfieren. 548d ≈ 18 meses cubre
        # solo el bull post-Trump, régimen homogéneo. Esperamos ~110 muestras.
        "train_window_days": 548,
        # AUC 0.499 con 548d: esencialmente aleatorio. Sin ML, la tasa base
        # da EV = +0.15R (win rate 46% × R:R 1.5 - 54%). Se usa raw_score.
        "use_ml":          False,
    },
    "ETHUSDT": {
        "label":        "ETH",
        "source":       "binance",
        "exchange":     "binanceusdm",
        "min_range_pct": 0.020,
        "pip_value":    1.0,
        "days_history": 1825,
        "forward_bars": 24,
        "ml_threshold": 0.55,
        # ML precision 31% < break-even 33% para R:R 2.0 → modelo filtra al revés.
        # Sin ML: tasa base 39% × 2.0 - 61% = +0.17R (positivo).
        "use_ml":       False,
    },
    "XAUUSD": {
        "label":        "GOLD",
        "source":       "yfinance",
        "yf_ticker":    "GC=F",
        "yf_fallback":  ("GLD", 10.0),  # GLD cotiza ~1/10 del oro → x10
        "yf_min_price": 500,
        "min_range_pct": 0.008,
        "pip_value":    1.0,
        "days_history": 7300,           # ~20 años; GC=F tiene datos desde 2000
        "forward_bars": 20,
        "ml_threshold": 0.58,
        "train_window_days": 0,         # usar todo el histórico disponible
        "safe_haven":   True,           # correlación NEGATIVA con DXY → usa macro_dxy_alignment
        # TP a 2×ATR en lugar de 3×ATR: el oro se mueve en tendencias lentas.
        # Con 2×ATR el win rate histórico sube ~10pp, mejorando el dataset de entrenamiento.
        "atr_target_mult": 2.0,         # R:R resultante: 1.33
        "atr_stop_mult":   1.5,
    },
    "XAGUSD": {
        "label":        "SILVER",
        "source":       "yfinance",
        "yf_ticker":    "SI=F",
        "yf_fallback":  ("SLV", 1.0),
        "yf_min_price": 10,
        "min_range_pct": 0.012,
        "pip_value":    1.0,
        "days_history": 7300,
        "forward_bars": 20,
        "ml_threshold": 0.55,
        "train_window_days": 0,
        "safe_haven":   True,           # plata como metal refugio: mismo alineamiento que Gold
        # R:R 1.0: TP=1.5×ATR. Con R:R 1.5 (TP=2.25×ATR) el win rate cayó 47%→37%
        # (plata revierte antes de alcanzar targets amplios). Break-even a R:R 1.0 = 50%;
        # el raw_score selecciona el top 5% de setups que pueden superar ese umbral.
        "atr_target_mult": 1.5,         # R:R 1.0
        "atr_stop_mult":   1.5,
        # AUC 0.489 consistente (anti-predictivo): ML destruye EV positivo a tasa base.
        "use_ml":          False,
    },
    "SPX500": {
        "label":        "SP500",
        "source":       "yfinance",
        "yf_ticker":    "^GSPC",
        "yf_fallback":  ("SPY", 10.0),
        "yf_min_price": 1000,
        "min_range_pct": 0.005,
        "pip_value":    1.0,
        "days_history": 7300,
        "forward_bars": 20,
        "ml_threshold": 0.55,
        "train_window_days": 0,
        # SP500: índice con sesgo alcista histórico, tendencias más suaves.
        # TP a 2.5×ATR: más ambicioso que metales pero realista para el índice.
        "atr_target_mult": 2.5,         # R:R 1.67
        "atr_stop_mult":   1.5,
    },
    # -------------------------------------------------------------------------
    # FOREX (yfinance, datos diarios). Volumen no disponible en tickers "=X":
    # los detectores se ajustan en ASSET_PARAMS_OVERRIDE (vol_factor=0,
    # volume_confirm=False). R:R 1.67 (2.5/1.5) para superar el piso de 1.5
    # que exige build_trade_setup. dxy_corr define la correlación con el dólar:
    #   -1 → el par sube cuando el DXY cae (EUR/USD, GBP/USD)
    #   +1 → el par sube cuando el DXY sube (USD/JPY)
    # -------------------------------------------------------------------------
    "EURUSD": {
        "label":        "EURUSD",
        "source":       "twelvedata",   # 4H bars ≈ 3.2 años (5000 velas); yfinance solo daba diario
        "yf_ticker":    "EURUSD=X",     # fallback si Twelve Data falla
        "yf_fallback":  ("FXE", 0.01),
        "yf_min_price": 0.5,
        "min_range_pct": 0.004,
        "pip_value":    1.0,
        "days_history": 1825,           # ~5 años para macro; OHLCV lo define outputsize=5000
        "forward_bars": 120,            # 120 × 4H = 480h = 20 días (equiv. a 20 barras diarias)
        "ml_threshold": 0.55,
        "train_window_days": 0,
        "dxy_corr":     -1,             # EUR/USD ↑ cuando el dólar ↓
        "atr_target_mult": 2.5,         # R:R 1.67
        "atr_stop_mult":   1.5,
    },
    "GBPUSD": {
        "label":        "GBPUSD",
        "source":       "twelvedata",
        "yf_ticker":    "GBPUSD=X",
        "yf_fallback":  ("FXB", 0.01),
        "yf_min_price": 0.5,
        "min_range_pct": 0.004,
        "pip_value":    1.0,
        "days_history": 1825,
        "forward_bars": 120,
        "ml_threshold": 0.55,
        "train_window_days": 0,
        "dxy_corr":     -1,             # GBP/USD ↑ cuando el dólar ↓
        "atr_target_mult": 2.5,         # R:R 1.67
        "atr_stop_mult":   1.5,
    },
    "USDJPY": {
        "label":        "USDJPY",
        "source":       "twelvedata",
        "yf_ticker":    "USDJPY=X",
        "yf_fallback":  ("JPY=X", 1.0),
        "yf_min_price": 50,
        "min_range_pct": 0.004,
        "pip_value":    1.0,
        "days_history": 1825,
        "forward_bars": 120,
        "ml_threshold": 0.55,
        "train_window_days": 0,
        "dxy_corr":     1,              # USD/JPY ↑ cuando el dólar ↑
        "atr_target_mult": 2.5,         # R:R 1.67
        "atr_stop_mult":   1.5,
    },
}

# =============================================================================
# TIMEFRAMES
# =============================================================================
TIMEFRAME_PRIMARY = "4h"    # Timeframe para generar setups
TIMEFRAME_TREND   = "1d"    # Timeframe para filtro de tendencia
TIMEFRAME_ENTRY   = "1h"    # Timeframe para afinar entrada (futuro)
DAYS_HISTORY      = 730     # 2 años de historia para backtest

# =============================================================================
# FRESCURA DE LA SEÑAL
# -----------------------------------------------------------------------------
# El precio de entrada de un setup es el CIERRE de la barra donde se detectó.
# Si el precio actual ya se alejó más de MAX_ENTRY_DIST_ATR × ATR de ese nivel,
# la operación no es ejecutable (el precio no regresará al entry) y se descarta.
# Evita emitir señales obsoletas como "SHORT @ 7580" cuando el precio ya está
# en 7523 y cayendo. Configurable por activo con la clave "max_entry_dist_atr".
# =============================================================================
MAX_ENTRY_DIST_ATR = 0.5


# =============================================================================
# FORMATO DE PRECIOS
# -----------------------------------------------------------------------------
# Los decimales se ajustan a la magnitud del precio: forex necesita 4-5 decimales
# (EUR/USD 1.13869) mientras que BTC/Oro/SP500 con 2 basta. Evita mostrar
# "Entrada $1.14 | Stop $1.14" donde no se distingue la geometría de la operación.
# =============================================================================
def price_decimals(price: float) -> int:
    """Decimales apropiados según la magnitud del precio."""
    p = abs(float(price))
    if p >= 1000:  return 2    # BTC, Oro, SP500
    if p >= 100:   return 3    # USD/JPY (~161) → muestra pipettes
    if p >= 10:    return 2    # Plata (~59)
    if p >= 1:     return 5    # EUR/USD, GBP/USD (~1.14)
    return 6                   # activos sub-1


def fmt_price(price: float) -> str:
    """Precio con separador de miles y decimales según magnitud."""
    return f"{price:,.{price_decimals(price)}f}"

# =============================================================================
# SETUPS TÉCNICOS — parámetros base (calibrados para 4H crypto)
# =============================================================================
BREAKOUT_PARAMS = {
    "donchian_period":  20,     # Período del canal Donchian
    "vol_factor":       1.3,    # Volumen debe ser X veces el promedio
    "atr_min_move":     1.5,    # Movimiento mínimo en ATRs para confirmar
    "lookback_days":    5,
}

PULLBACK_PARAMS = {
    "ema_fast":         21,
    "ema_slow":         55,
    "rsi_period":       14,
    "rsi_oversold":     45,
    "rsi_overbought":   55,
    "min_trend_bars":   10,
    "max_pullback_pct": 0.05,
}

REVERSAL_PARAMS = {
    "rsi_period":       14,
    "rsi_extreme_low":  30,
    "rsi_extreme_high": 70,
    "bb_period":        20,
    "bb_std":           2.0,
    "volume_confirm":   True,
}

# =============================================================================
# OVERRIDES POR ACTIVO
# Permiten ajustar parámetros según el instrumento y timeframe.
# Gold usa datos diarios → filtros más relajados que crypto 4H.
# =============================================================================
ASSET_PARAMS_OVERRIDE = {
    # Crypto: solo breakouts en mercado en tendencia (ADX > 20 = trending)
    # Elimina los fakeouts que dominaban el dataset (249/282 breakouts en BTC)
    "BTCUSDT": {
        "BREAKOUT": {
            "adx_min_trend": 20,
        },
    },
    "ETHUSDT": {
        "BREAKOUT": {
            "adx_min_trend": 20,
        },
    },
    "XAUUSD": {
        "BREAKOUT": {
            # Diario: barras más pequeñas en ATR relativo; volumen de GC=F
            # no siempre tiene el spike típico de crypto → bajar filtros
            "vol_factor":   1.05,   # Casi cualquier volumen confirmado
            "atr_min_move": 0.6,    # Barra más pequeña aceptable en diario
        },
        "PULLBACK": {
            "min_trend_bars": 5,    # Menos barras para confirmar tendencia en diario
            "rsi_oversold":   50,   # RSI ≤ 50 en pullback alcista (zona media)
            "rsi_overbought": 50,   # RSI ≥ 50 en pullback bajista
        },
        "REVERSAL": {
            "rsi_extreme_low":  35, # RSI < 35 en diario (menos extremo que 30)
            "rsi_extreme_high": 65, # RSI > 65 en diario
            "volume_confirm":   False,  # No requerir spike de volumen en GC=F
        },
    },
    # Plata: comportamiento técnico similar al oro pero más volátil
    "XAGUSD": {
        "BREAKOUT": {
            "vol_factor":   1.10,
            "atr_min_move": 0.7,
        },
        "PULLBACK": {
            "min_trend_bars": 5,
            "rsi_oversold":   50,
            "rsi_overbought": 50,
        },
        "REVERSAL": {
            "rsi_extreme_low":  35,
            "rsi_extreme_high": 65,
            "volume_confirm":   False,
        },
    },
    # S&P 500: índice más estable, ATRs relativos pequeños
    "SPX500": {
        "BREAKOUT": {
            "vol_factor":   1.05,
            "atr_min_move": 0.5,
        },
        "PULLBACK": {
            "min_trend_bars": 5,
            "rsi_oversold":   50,
            "rsi_overbought": 50,
        },
        "REVERSAL": {
            "rsi_extreme_low":  35,
            "rsi_extreme_high": 65,
            "volume_confirm":   False,
        },
    },
    # Forex: yfinance no entrega volumen para tickers "=X" (llega 0).
    # vol_factor=0 desactiva la confirmación de volumen en breakout y
    # volume_confirm=False hace lo propio en reversal. Filtros de precio
    # (ATR, RSI, tendencia) se mantienen activos.
    "EURUSD": {
        "BREAKOUT": {
            "vol_factor":   0.0,
            "atr_min_move": 0.5,
        },
        "PULLBACK": {
            "min_trend_bars": 5,
            "rsi_oversold":   50,
            "rsi_overbought": 50,
        },
        "REVERSAL": {
            "rsi_extreme_low":  35,
            "rsi_extreme_high": 65,
            "volume_confirm":   False,
        },
    },
    "GBPUSD": {
        "BREAKOUT": {
            "vol_factor":   0.0,
            "atr_min_move": 0.5,
        },
        "PULLBACK": {
            "min_trend_bars": 5,
            "rsi_oversold":   50,
            "rsi_overbought": 50,
        },
        "REVERSAL": {
            "rsi_extreme_low":  35,
            "rsi_extreme_high": 65,
            "volume_confirm":   False,
        },
    },
    "USDJPY": {
        "BREAKOUT": {
            "vol_factor":   0.0,
            "atr_min_move": 0.5,
        },
        "PULLBACK": {
            "min_trend_bars": 5,
            "rsi_oversold":   50,
            "rsi_overbought": 50,
        },
        "REVERSAL": {
            "rsi_extreme_low":  35,
            "rsi_extreme_high": 65,
            "volume_confirm":   False,
        },
    },
}

# =============================================================================
# MODELO ML
# =============================================================================
ML_PARAMS = {
    # Modelo más simple y regularizado para datasets pequeños (< 500 muestras)
    "n_estimators":     150,    # era 400 — menos árboles, menos overfitting
    "learning_rate":    0.05,   # era 0.02 — más rápido con menos árboles
    "num_leaves":       15,     # era 31 — árbol más simple
    "max_depth":        4,      # era 6  — profundidad limitada
    "min_child_samples":30,     # era 20 — más muestras mínimas por hoja
    "subsample":        0.7,    # era 0.8
    "colsample_bytree": 0.6,    # era 0.8 — usar 60% de features por árbol
    "lambda_l1":        0.2,    # NUEVO: regularización L1 (LASSO)
    "lambda_l2":        0.2,    # NUEVO: regularización L2 (Ridge)
    "min_train_samples": 80,    # era 50 — requerir más datos mínimos
    "top_features":     20,     # Selección de top-N features por importancia
    "train_window_days": 730,   # Últimos 2 años; 1000 días incorpora 2023 que genera ruido en crypto
    "retrain_every":    7,      # Reentrenar cada N días
    "prob_threshold":   0.55,   # Umbral para métricas walk-forward
}

WALK_FORWARD = {
    "n_splits":         3,      # era 5 — con < 400 muestras, 5 folds dejan folds de 60 muestras (muy poco)
    "train_size":       240,
    "test_size":        60,
}

# =============================================================================
# GESTIÓN DE RIESGO
# =============================================================================
RISK = {
    "capital_inicial":  10_000,     # Capital inicial en USD (para backtest)
    "risk_per_trade":   0.01,       # 1% del capital por operación
    "atr_stop_mult":    1.5,        # Stop loss = 1.5 × ATR
    "atr_target_mult":  3.0,        # Take profit = 3.0 × ATR (R:R 2:1)
    "max_positions":    3,          # Máximo de posiciones abiertas simultáneas
    "max_daily_loss":   0.03,       # Si pierdas 3% en un día, parar
    "max_drawdown":     0.15,       # Si drawdown alcanza 15%, revisar sistema
    "atr_period":       14,
}

# =============================================================================
# SENTIMENT
# =============================================================================
SENTIMENT = {
    "min_news_score":   0.0,        # Score mínimo para no vetar una señal
    "veto_threshold":  -0.4,        # Si sentiment < -0.4 → no operar LONG
    "boost_threshold":  0.4,        # Si sentiment > 0.4 → boost de confianza
    "lookback_hours":   24,         # Horas de noticias a considerar
    "enabled":          True,
}

# =============================================================================
# BACKTEST
# =============================================================================
BACKTEST = {
    "start_date":       "2022-01-01",
    "end_date":         None,           # None = hasta hoy
    "commission":       0.0004,         # 0.04% (Binance Maker fee)
    "slippage":         0.0002,         # 0.02% slippage estimado
    "initial_capital":  10_000,
}
