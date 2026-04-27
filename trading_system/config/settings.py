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
CRYPTOPANIC_KEY    = _clean_env("CRYPTOPANIC_API_KEY")
ANTHROPIC_KEY      = _clean_env("ANTHROPIC_API_KEY")
TELEGRAM_TOKEN     = _clean_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _clean_env("TELEGRAM_CHAT_ID")

# =============================================================================
# ACTIVOS
# =============================================================================
ASSETS = {
    "BTCUSDT": {
        "label":        "BTC",
        "source":       "binance",
        "exchange":     "binanceusdm",
        "min_range_pct": 0.015,   # 1.5% rango mínimo para contar como "movimiento grande"
        "pip_value":    1.0,
    },
    "ETHUSDT": {
        "label":        "ETH",
        "source":       "binance",
        "exchange":     "binanceusdm",
        "min_range_pct": 0.020,
        "pip_value":    1.0,
    },
    "XAUUSD": {
        "label":        "GOLD",
        "source":       "yfinance",
        "yf_ticker":    "GC=F",
        "min_range_pct": 0.008,
        "pip_value":    1.0,
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
}

# =============================================================================
# MODELO ML
# =============================================================================
ML_PARAMS = {
    "n_estimators":     400,
    "learning_rate":    0.02,
    "num_leaves":       31,
    "max_depth":        6,
    "min_child_samples":20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_train_samples":100,    # Mínimo de setups históricos para entrenar
    "retrain_every":    7,      # Reentrenar cada N días
    "prob_threshold":   0.58,   # Umbral de confianza para emitir señal
}

WALK_FORWARD = {
    "n_splits":         5,
    "train_size":       180,    # Días de entrenamiento por fold
    "test_size":        30,     # Días de evaluación por fold
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
