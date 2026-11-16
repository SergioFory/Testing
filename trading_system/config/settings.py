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

# Latido opcional (Fase 1, punto 1.4): URL de un servicio externo gratuito tipo
# healthchecks.io. Al final de cada ciclo correcto el worker hace un GET (5 s,
# sin fallar si no responde); si los pings dejan de llegar, el servicio avisa.
# Vacía = desactivado. Ver monitoring/heartbeat.py.
HEARTBEAT_URL      = _clean_env("HEARTBEAT_URL")

# CryptoPanic discontinuó su plan gratuito el 01-Apr-2026.
# Sentimiento ahora usa Fear & Greed Index (Alternative.me) — sin API key.
CRYPTOPANIC_KEY    = ""  # deprecated — mantenido para no romper imports antiguos

# =============================================================================
# ACTIVOS
# =============================================================================
ASSETS = {
    # =========================================================================
    # ESPECIALIZACIÓN (validación de expectativa, 2026-07-08):
    #   Solo se operan los activos con E[R] > 0 tras costos. Los demás quedan
    #   apagados (enabled=False): sin señales, sin reentreno → menos RAM/OOM.
    #     ✓ BTC  E[R] +0.082 (Top25 +0.266)   ✓ ETH  E[R] +0.159 (Top25 +0.205)
    #     ~ Oro/Plata: solo BREAKOUT (su única parte con E[R] > 0)
    #     ✗ SP500, EUR/USD, GBP/USD, USD/JPY: E[R] negativo → apagados
    #   "enabled": False  → el activo se omite en señales y reentreno.
    #   "setups": [...]   → restringe los tipos de setup detectados (default: los 3).
    # =========================================================================
    "BTCUSDT": {
        "label":        "BTC",
        "enabled":      True,
        # Sin reversal: validación 2026-09 lo halló negativo de forma coincidente
        # en ambos activos cripto (BTC -0.095 n=36 | ETH -0.344 n=32). Mecanismo:
        # el reversal es contra-tendencia y en cripto la tendencia arrolla.
        # Sin pullback (Fase 1, config honesta): ningún pullback de BTC ni de ETH
        # llega al umbral 0.55 en la historia actual (máx. raw 0.513 en BTC y
        # 0.434 en ETH, snapshot 2026-11-15), así que nunca se emitía; quitarlo
        # no cambia ningún setup aprobado (check-live: BTC 64 / ETH 63).
        "setups":       ["breakout"],
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
        "enabled":      True,
        "setups":       ["breakout"],   # sin reversal (-0.344) ni pullback (ver BTC)
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
        "enabled":      True,
        # Solo breakout: la validación mostró breakout E[R] +0.100, pero pullback
        # (-0.023) y reversal (-0.150) hundían el conjunto a -0.030. Aislar el
        # breakout deja la única parte con expectativa positiva.
        "setups":       ["breakout"],
        # use_ml=False: con solo breakouts quedan ~58 muestras (< min 80), el ML
        # no puede entrenar; se usa raw_score directo (que fue lo validado).
        "use_ml":       False,
        "source":       "yfinance",
        "yf_ticker":    "GC=F",
        # Sin yf_fallback (Fase 0, 2026-11): el respaldo GLD×10 cotiza ~8.6 % por
        # debajo de GC=F y se activaba EN SILENCIO ante un 429 de Yahoo. Ahora, si
        # GC=F falla tras los reintentos, el ciclo se salta y se avisa por Telegram.
        "yf_min_price": 500,
        "min_range_pct": 0.008,
        "pip_value":    1.0,
        "days_history": 7300,           # ~20 años; GC=F tiene datos desde 2000
        "forward_bars": 20,
        # 0.20 (no 0.55): el raw_score del oro se distribuye bajo (mediana ~0.25) y
        # el edge validado (+0.100) está en TODOS los breakouts, no en los de mayor
        # score (Top25 fue peor: +0.022). 0.55 aprobaba 1/58 → nunca generaba señal.
        "ml_threshold": 0.20,           # umbral raw_score, calibrado a la distribución
        "train_window_days": 0,         # usar todo el histórico disponible
        "safe_haven":   True,           # correlación NEGATIVA con DXY → usa macro_dxy_alignment
        # TP a 2×ATR en lugar de 3×ATR: el oro se mueve en tendencias lentas.
        # Con 2×ATR el win rate histórico sube ~10pp, mejorando el dataset de entrenamiento.
        "atr_target_mult": 2.0,         # R:R resultante: 1.33
        "atr_stop_mult":   1.5,
    },
    "XAGUSD": {
        "label":        "SILVER",
        # APAGADA (2026-09-18). Aunque el conjunto de breakouts da E[R] +0.081,
        # su Top25 E[R] es NEGATIVO (-0.044): el raw_score selecciona justamente
        # los setups perdedores, y en vivo get_best_setup opera el de mayor score.
        # Es decir, lo que realmente se operaría tiene expectativa negativa.
        # Además aporta solo 48 setups en 20 años (~2.4/año): muestra insuficiente
        # para sostener un edge. Se reactiva si acumula evidencia favorable.
        "enabled":      False,
        "setups":       ["breakout"],
        "source":       "yfinance",
        "yf_ticker":    "SI=F",
        "yf_fallback":  ("SLV", 1.0),
        "yf_min_price": 10,
        "min_range_pct": 0.012,
        "pip_value":    1.0,
        "days_history": 7300,
        "forward_bars": 20,
        # 0.25 (no 0.55): mismo motivo que el oro — raw_score bajo (mediana ~0.31),
        # edge validado (+0.081) en todos los breakouts, Top25 peor (-0.044).
        "ml_threshold": 0.25,           # umbral raw_score, calibrado a la distribución
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
        # REACTIVADO con estrategia validada (variantes 2026-07-09):
        #   Baseline (todo, ambas direcciones): E[R] -0.089 ✗
        #   Solo SHORT: -0.205 ✗ (apostar contra el drift alcista pierde)
        #   Solo LONG: +0.082 (N=222) ✓
        #   LONG sin pullback (breakout+reversal): +0.246 (N=107) ✓✓ ← elegido
        #   (los pullback largos pierden -0.071; el filtro MA200 bajó el edge)
        "enabled":      True,
        # MODO SOMBRA (Fase 0, 2026-11): se detecta, se registra en la BD con
        # mode='shadow' y se resuelve, pero NO se notifica por Telegram. Motivo:
        # sin edge demostrable con costes de XM (swap) y walk-forward ≈ 0. Volver a
        # notificar solo si el E[R] neto con el swap real de XM llega a +0.10R en
        # al menos 2 de 3 épocas y pasa la puerta de validación.
        "notify":       False,
        "directions":   ["long"],              # solo compras: el índice sube estructuralmente
        "setups":       ["breakout", "reversal"],  # sin pullback (E[R] negativo)
        "use_ml":       False,                 # se usó raw_score en la validación; muestra pequeña
        "source":       "yfinance",
        "yf_ticker":    "^GSPC",
        # Sin yf_fallback (Fase 0): ver nota en XAUUSD.
        "yf_min_price": 1000,
        "min_range_pct": 0.005,
        "pip_value":    1.0,
        "days_history": 7300,
        "forward_bars": 20,
        # 0.12 (no 0.55): el raw_score del SP500 se distribuye muy bajo (mediana ~0.10).
        # 0.55 aprobaba 6/107 → casi nunca señal. 0.12 aprueba ~la mitad superior;
        # el raw_score SÍ ayuda aquí (Top25 +0.337 > +0.245), así que conserva algo de filtro.
        "ml_threshold": 0.12,                  # umbral raw_score, calibrado a la distribución
        "train_window_days": 0,
        # TP a 2.5×ATR (R:R 1.67): el R:R con el que se validó el +0.246.
        "atr_target_mult": 2.5,
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
        "enabled":      False,          # E[R] -0.450, PF 0.50 → pérdida grave, apagado
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
        "enabled":      False,          # E[R] -0.480, PF 0.47 → pérdida grave, apagado
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
        "enabled":      False,          # E[R] -0.396, PF 0.53 → pérdida grave, apagado
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


def notify_enabled(symbol: str) -> bool:
    """True si las señales del activo se notifican por Telegram (modo 'live').
    Con "notify": False el activo opera en MODO SOMBRA: se registra y se mide,
    pero no se envía."""
    return bool(ASSETS.get(symbol, {}).get("notify", True))


def active_assets() -> list:
    """Símbolos operativos (enabled != False). Los apagados se omiten en
    generación de señales y reentreno, reduciendo carga de RAM."""
    return [s for s, cfg in ASSETS.items() if cfg.get("enabled", True)]


def allowed_setups(symbol: str):
    """Tipos de setup permitidos para un activo, o None si todos.
    Permite restringir p.ej. Oro/Plata a solo 'breakout'."""
    return ASSETS.get(symbol, {}).get("setups")


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
#
# 1.0×ATR: la entrada debe estar dentro del rango de una barra promedio del
# precio actual — sigue siendo ejecutable (una vela normal puede alcanzarla)
# sin bloquear casi todos los setups como hacía 0.5×ATR (descartaba el 100% de
# los candidatos que no estuvieran en la barra más reciente).
# =============================================================================
MAX_ENTRY_DIST_ATR = 1.0


# =============================================================================
# VENTANA DE EMISIÓN (Fase 0, 2026-11)
# -----------------------------------------------------------------------------
# Solo se emiten setups de las últimas velas CERRADAS: la del setup debe ser la
# última vela cerrada o, como máximo, MAX_SETUP_AGE_BARS velas anteriores (para
# no perder un setup si un ciclo falla). Antes la ventana se contaba desde el
# último setup del histórico, no desde ahora, y un setup de hace semanas seguía
# "vigente" indefinidamente (re-emisiones). El validador opera cada setup UNA
# vez, en la vela siguiente: esto alinea el vivo con lo validado.
MAX_SETUP_AGE_BARS = 1

# Hora UTC a partir de la cual la vela DIARIA de yfinance (GC=F, ^GSPC) se da por
# cerrada. GC=F liquida a las 17:00 ET (21:00/22:00 UTC según horario de verano)
# y ^GSPC cierra a las 16:00 ET. 22:15 UTC cubre ambos con margen todo el año.
DAILY_BAR_CLOSE_UTC = (22, 15)

# Antigüedad máxima aceptable de la última vela cerrada antes de considerar que
# la fuente está congelada (se salta el ciclo y se avisa). Cripto: 3 velas 4H.
# Diarios: 5 días naturales (fines de semana y festivos largos).
MAX_DATA_STALENESS_HOURS = {"4h": 12, "1d_session": 24 * 5}

# =============================================================================
# HISTORIA DESCARGADA EN EL CICLO VIVO (Fase 1, 2026-11)
# -----------------------------------------------------------------------------
# main._load_data(history="live") pide a Gate.io solo estos días (cripto) en vez
# de la historia completa (~1.650 días de 4H = 10 páginas + 1.825 de diario por
# activo y hora). Los indicadores (ATR/ADX/RSI de Wilder, EMAs, ventanas móviles
# de hasta 100 velas) olvidan el pasado lejano: se comprobó con el arnés (snapshot
# 2026-11-15) que, para CADA uno de los 2.190 ciclos vivos de los últimos 12
# meses, los setups de las velas que mira el vivo (breakouts y pullbacks), su
# raw_score, su ATR y su aprobación son IDÉNTICOS (diferencia exactamente 0) con
# 300 días de 4H y 900 de diario que con la historia completa: BTC 66 setups (13
# aprobados), ETH 60 (8 aprobados), 0 discrepancias; tendencia diaria igual en
# 1.101 de 1.101 comparaciones por activo. Margen (BTC, muestreo): con 120 días
# de 4H ya es idéntico, con 60 hay diferencias de ~1e-12 y con 30 de ~1e-6; con
# 120 días de diario la tendencia cambia en 61 de 1.101 comparaciones y con 200
# en ninguna.
# validate, check-live, train y la limpieza siguen con la historia
# completa (history="full"). yfinance (oro, SPX) no cambia: es una sola petición.
LIVE_HISTORY_DAYS = {"4h": 300, "1d": 900}

# Señales live cuyo envío a Telegram falló (notified = 0): se reintentan en cada
# ciclo mientras tengan menos de estas horas; las más antiguas se quedan sin
# notificar y solo se registran en el log (una señal vieja ya no es ejecutable).
NOTIFY_RETRY_MAX_AGE_HOURS = 2.0
# Margen sobre ese límite (revisión de la Fase 1): `schedule` programa el ciclo
# siguiente 60 min después de TERMINAR el anterior, así que en el segundo
# reintento la señal ya tiene 2 h + lo que duraron los ciclos (siempre > 2 h) y
# en la práctica solo había UN reintento. Con 15 min de margen hay dos (a ~1 h y
# a ~2 h); una señal más vieja ya no se reenvía y se avisa de que no llegó.
NOTIFY_RETRY_GRACE_MINUTES = 15
# Durante cuántas horas se intenta avisar (una vez) de una señal que NUNCA llegó
# ("no te llegó la señal #N…"), por si Telegram vuelve tarde.
UNDELIVERED_NOTICE_MAX_AGE_HOURS = 48


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
    # atr_rank_min 0.50 (era 0.30): la validación de 2026-09 halló que la franja
    # ATR rank 0.30-0.50 destruía valor de forma coincidente en ambos activos
    # (entonces: BTC E[R] -0.234 n=78 | ETH -0.374 n=57). Esas cifras YA NO se
    # reproducen: Gate.io solo sirve las últimas ~10.000 velas (ventana móvil) y
    # la convención de salida cambió en la Fase 1. Medido de nuevo (snapshot
    # 2026-11-15, solo breakouts, convención Fase 1): franja 0.30-0.50
    # BTC -0.190 (n=61) | ETH -0.165 (n=46); con corte 2026-09-18 BTC -0.203
    # (n=59) | ETH -0.211 (n=45). El signo sigue siendo negativo en ambos, pero
    # la magnitud es menor que la citada. Valor 0.50 SIN CAMBIOS (config
    # congelada). NO se aprieta el ADX: su patrón resultó no-monotónico (ruido)
    # y combinarlo con volatilidad empeoraba BTC.
    "BTCUSDT": {
        "BREAKOUT": {
            "adx_min_trend": 20,
            "atr_rank_min":  0.50,
        },
    },
    "ETHUSDT": {
        "BREAKOUT": {
            "adx_min_trend": 20,
            "atr_rank_min":  0.50,
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
def _env_capital(key: str = "TRADING_CAPITAL", default: float = 10_000.0) -> float:
    """Capital de la cuenta en USD desde la variable de entorno (Fase 1, 1.1).
    Admite "25000", "25_000", "25,000" o "25000.50". Vacía o inválida → default."""
    raw = _clean_env(key).replace("_", "").replace(",", "").replace("$", "").strip()
    if not raw:
        return float(default)
    try:
        val = float(raw)
    except ValueError:
        val = 0.0
    if val <= 0:
        from loguru import logger
        logger.warning(f"{key}={raw!r} no es un capital válido; se usa {default:,.0f} USD.")
        return float(default)
    return val


# Capital REAL de la cuenta del usuario (USD). Con él se calcula el tamaño de la
# posición y el riesgo en % que muestra el mensaje de Telegram. Se configura en
# Render con la variable TRADING_CAPITAL; sin ella, 10.000 USD.
TRADING_CAPITAL = _env_capital()

# =============================================================================
# CONTRATOS DE XM (Fase 1, 1.1)
# -----------------------------------------------------------------------------
# Para mostrar el tamaño en LOTES de XM hacen falta, por activo:
#   contract_size: unidades del activo por 1 lote (p. ej. 1 BTC, 100 oz…)
#   min_lot:       lote mínimo que acepta XM
#   lot_step:      incremento de lote (p. ej. 0.01)
# PENDIENTE: el usuario aún no ha enviado la especificación de sus contratos, así
# que van en None y el mensaje dice "Lotes XM: pendiente de la especificación del
# contrato" y muestra el tamaño en unidades del activo (`unit`). Cuando lleguen
# los datos, basta con rellenarlos aquí (no hay que tocar nada más).
# =============================================================================
XM_CONTRACTS = {
    "BTCUSDT": {"unit": "BTC", "contract_size": None, "min_lot": None, "lot_step": None},
    "ETHUSDT": {"unit": "ETH", "contract_size": None, "min_lot": None, "lot_step": None},
    "XAUUSD":  {"unit": "oz",  "contract_size": None, "min_lot": None, "lot_step": None},
    "SPX500":  {"unit": "unidades del índice", "contract_size": None, "min_lot": None, "lot_step": None},
}

# Tope de riesgo de la regla 3 de REGLAS_DE_DECISION.md ("el riesgo no sube del
# 0,25 % hasta que el SPRT acepte el edge"). Desde la Fase 3 (3.2) el sistema lo
# cumple solo: RISK["risk_per_trade"] = este mismo valor. El mensaje solo avisa
# (línea "Para respetar el 0,25 %") si algún día el riesgo calculado lo superase.
RULES_MAX_RISK_PCT = 0.0025

RISK = {
    "capital_inicial":  10_000,     # Capital inicial en USD (para backtest)
    # Fase 3 (3.2, 16-nov-2026): 0,25 % explícito por operación (antes 1 % con un
    # tope del 10 % de nominal que casi siempre mandaba y dejaba el riesgo real
    # entre ~0,1 % y ~0,8 % según el stop). No cambia qué señales salen ni las
    # estadísticas en R; solo el tamaño. Debe ser ≤ RULES_MAX_RISK_PCT.
    "risk_per_trade":   0.0025,     # 0,25 % del capital por operación
    # Nominal máximo = capital × max_leverage (sustituye al tope del 10 % de nominal).
    # Con 0,25 % de riesgo solo actuaría con un stop a < 0,25 % del precio; si
    # actúa, el riesgo baja y el mensaje lo dice.
    "max_leverage":     1.0,
    "atr_stop_mult":    1.5,        # Stop loss = 1.5 × ATR
    "atr_target_mult":  3.0,        # Take profit = 3.0 × ATR (R:R 2:1)
    "max_positions":    3,          # Máximo de posiciones abiertas simultáneas
    "max_daily_loss":   0.03,       # Si pierdas 3% en un día, parar
    "max_drawdown":     0.15,       # Si drawdown alcanza 15%, revisar sistema
    "atr_period":       14,
}

# =============================================================================
# CONTROL DE CARTERA (Fase 3, punto 3.1; F11 y F19) — risk/portfolio.py
# -----------------------------------------------------------------------------
# Antes de cada señal se reconstruye desde la BD la cartera SEGÚN EL SISTEMA
# (señales live abiertas que el usuario no saltó con /skip) y se comprueban:
#   R1 máximo de posiciones abiertas;
#   R2 nunca LONG y SHORT a la vez en el mismo activo;
#   R3 grupo correlacionado BTC+ETH: riesgo abierto del grupo + la nueva ≤ 0,50 %
#      del capital (con el 0,25 % por operación: como mucho 2 cripto abiertas);
#   R4 si la pérdida realizada del día UTC llega al 3 % del capital, no abrir.
# PORTFOLIO_CONTROL_MODE:
#   "shadow"  (HOY): la señal SALE igual; se guarda qué regla la habría bloqueado
#             (columna portfolio_flag) y el mensaje lo cuenta ("en pruebas: no bloquea").
#   "enforce" (probado con tests, NO activado): la señal no se emite y el setup
#             queda en shadow_setups con motivo 'portfolio_R1'…'portfolio_R4'.
# Pasar a "enforce" es decisión del usuario, con la recomendación por regla de
# REGLAS_DE_DECISION.md ("Control de cartera"). (RISK["max_positions"] y
# RISK["max_daily_loss"] son del backtest antiguo, backtest/engine.py; el vivo usa esto.)
# =============================================================================
PORTFOLIO_CONTROL_MODE = "shadow"
PORTFOLIO_RULES = {
    "max_open_positions":          3,       # R1
    "no_opposite_same_symbol":     True,    # R2
    "correlated_groups": [                  # R3
        {"name": "BTC+ETH", "symbols": ["BTCUSDT", "ETHUSDT"], "max_risk_pct": 0.005},
    ],
    "max_daily_realized_loss_pct": 0.03,    # R4
}

# =============================================================================
# FILTRO DE EVENTOS ECONÓMICOS — QUITADO (Fase 3, punto 0; decisión del usuario
# del 16-nov-2026)
# -----------------------------------------------------------------------------
# Antes no se emitían señales de oro/SPX entre 2 h antes y 1 h después de un
# evento de alto impacto (EVENT_FILTER). La Fase 2 midió que NUNCA actúa: el vivo
# evalúa el oro y el SPX tras el cierre diario (22:15–23:00 o 04:00–07:00 UTC) y
# los FOMC/CPI/NFP son a las 12:30/13:30 y 18:00/19:00 UTC (0 de 64 señales de
# oro/SPX desde 2015 retrasadas o bloqueadas), y las operaciones con un evento
# cerca no son peores de forma concluyente. Por eso se quitó el bloqueo (y la
# configuración EVENT_FILTER). SE MANTIENEN: el calendario data/econ_events.csv,
# la línea de aviso de eventos en el mensaje y los eventos de 48 h del resumen
# nocturno. El motivo 'event_blackout' ya no se produce (las filas antiguas se
# conservan). Ver REGLAS_DE_DECISION.md, "Filtro de eventos macro".
# =============================================================================

# =============================================================================
# SENTIMENT
# -----------------------------------------------------------------------------
# APAGADO (Fase 1, 2026-11). El sentimiento NO filtra ni puntúa señales:
#   - Veto contrarian con Fear & Greed (should_veto_signal): DESCARTADO con datos.
#     En BTC las señales que vetaría rendían +0,751R frente a +0,332R de las que
#     conservaría (quitaría las mejores) y, fuera de muestra, las vetadas fueron
#     ganadoras (+1,4 a +1,9R). No se conecta en ningún sitio.
#   - Sentimiento de titulares: signo opuesto en BTC y ETH (229.000 titulares).
# Con enabled=False el worker no consulta Fear & Greed, no se guarda
# sentiment_score y el dashboard no muestra etiqueta "contrarian" ni veto.
# No reabrir sin evidencia nueva fuera de muestra y neta de costes.
# =============================================================================
SENTIMENT = {
    "min_news_score":   0.0,        # (sin uso) Score mínimo para no vetar una señal
    "veto_threshold":  -0.4,        # (sin uso) veto descartado, ver arriba
    "boost_threshold":  0.4,        # (sin uso)
    "lookback_hours":   24,         # (sin uso)
    "enabled":          False,
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

# =============================================================================
# COSTES POR BRÓKER Y PUERTA DE VALIDACIÓN (Fase 2, punto 2.1)
# -----------------------------------------------------------------------------
# Solo los usa el comando OFFLINE `python main.py validate --protocol`
# (backtest/costs.py y backtest/protocol.py). El ciclo vivo NO los lee: no
# cambian qué señales se emiten ni cuándo.
#
# Por activo, en la cuenta XM Standard del usuario:
#   spread:        en UNIDADES DE PRECIO (USD por BTC/ETH/oz, puntos del índice).
#                  Se cobra 1 spread por operación: medio a la entrada y medio a la
#                  salida, sobre los precios medios del histórico (no cambia qué
#                  vela toca el SL o el TP, igual que el slippage del validador).
#   commission:    fracción del precio por lado (0 en XM Standard).
#   swap_pct_year: swap anual sobre el nominal; None = DESCONOCIDO → el informe da
#                  siempre los escenarios de SWAP_SCENARIOS_PCT_YEAR. Siempre en
#                  contra (largos y cortos), por noche de rollover.
#   swap_nights:   "crypto"  → 7 noches por semana (todas las medianoches);
#                  "session" → 5 noches con triple el miércoles (oro, SPX).
#   origen:        de dónde sale cada cifra (los SUPUESTOS van marcados).
# =============================================================================
BROKER_COSTS = {
    "xm": {
        "BTCUSDT": {"spread": 5.0, "commission": 0.0, "swap_pct_year": None, "swap_nights": "crypto",
                    "origen": "spread que reporta el usuario (3–5 USD en el precio de entrada; cota alta)"},
        "ETHUSDT": {"spread": 5.0, "commission": 0.0, "swap_pct_year": None, "swap_nights": "crypto",
                    "origen": "spread que reporta el usuario (3–5 USD en el precio de entrada; cota alta)"},
        "XAUUSD":  {"spread": 0.45, "commission": 0.0, "swap_pct_year": None, "swap_nights": "session",
                    "origen": "SUPUESTO del informe final (0,45 USD/oz); no verificado con XM"},
        "SPX500":  {"spread": 1.0, "commission": 0.0, "swap_pct_year": None, "swap_nights": "session",
                    "origen": "SUPUESTO del informe final (1 punto); no verificado con XM"},
    },
}
# Swap desconocido: escenarios del 0 %, 5 % y 10 % anual sobre el nominal.
SWAP_SCENARIOS_PCT_YEAR = (0.0, 0.05, 0.10)
# Escenario con el que se decide (E[R] de decisión): XM con swap del 5 % anual.
COST_DECISION_SCENARIO = "xm_swap5"

# Reserva INTOCABLE (UTC, los 4 activos): ningún setup con fecha >= esta entra en
# la selección, la CPCV, el PBO ni el DSR. Solo `validate --protocol --final` la
# evalúa y cada acceso queda en backtest/trials_registry.jsonl.
PROTOCOL_RESERVE_START = "2025-11-15"

# =============================================================================
# MONITOR DE DATOS — DERIVA DEL DETECTOR (Fase 2, punto 2.3)
# =============================================================================
# Línea base de cuántos setups detecta y aprueba cada activo en 30 días, para
# comparar con lo que registra el vivo en shadow_setups (monitoring/data_monitor).
# Solo la lee el monitor (nota en el resumen / aviso): NO cambia ninguna señal.
#
# Cómo se calculó (2026-11-16, arnés local, snapshot congelado al 2026-11-15:
# Gate.io 4H 2022-05-10 → 2026-11-15; yfinance diario 2006-11-20 → 2026-11-13;
# script scratchpad/fase2/C-monitor/drift_baseline.py): detect_all_setups +
# score_setups con la configuración ACTUAL sobre todo el histórico; setups =
# claves únicas (vela, dirección) con entrada/SL/TP válidos (lo mismo que guarda
# shadow_setups); aprobados = raw_score >= umbral. Conteos en ventanas MÓVILES
# de 30 días (paso 1 día, tras 90 días de calentamiento: 1.531 ventanas en
# cripto, 7.179 en diarios); se guardan los percentiles p1, p5, p50, p95, p99.
# Fuera de [p5, p95] → nota en el resumen; fuera de [p1, p99] → aviso.
# OJO: en oro y SPX los setups son raros (mediana 0 por mes): la deriva solo
# puede avisar de "demasiados", nunca de "ninguno". Si cambia la estrategia o el
# umbral hay que recalcularla con el script (si no, avisará en falso).
DRIFT_BASELINE: dict = {
    "BTCUSDT": {"setups":   {"p1": 0.0, "p5": 1.0, "p50": 4.0, "p95": 10.0, "p99": 12.0},
                "approved": {"p1": 0.0, "p5": 0.0, "p50": 1.0, "p95": 4.0,  "p99": 6.0}},
    "ETHUSDT": {"setups":   {"p1": 0.0, "p5": 1.0, "p50": 4.0, "p95": 9.0,  "p99": 11.0},
                "approved": {"p1": 0.0, "p5": 0.0, "p50": 1.0, "p95": 3.0,  "p99": 3.0}},
    "XAUUSD":  {"setups":   {"p1": 0.0, "p5": 0.0, "p50": 0.0, "p95": 3.0,  "p99": 5.0},
                "approved": {"p1": 0.0, "p5": 0.0, "p50": 0.0, "p95": 1.0,  "p99": 2.0}},
    "SPX500":  {"setups":   {"p1": 0.0, "p5": 0.0, "p50": 0.0, "p95": 2.0,  "p99": 4.0},
                "approved": {"p1": 0.0, "p5": 0.0, "p50": 0.0, "p95": 1.0,  "p99": 2.0}},
}
