"""
Feature engineering para el modelo ML de priorización de setups.

Diferencia clave vs v4.0:
  - El modelo NO intenta predecir dirección diaria
  - El modelo CALIFICA si un setup técnico detectado tiene alta probabilidad
    de alcanzar el take profit antes del stop loss
  - Features se calculan sobre la barra del setup + contexto previo
"""
import pandas as pd
import numpy as np
import ta
from loguru import logger


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores técnicos sobre el OHLCV.
    Esta función se llama UNA VEZ sobre el DataFrame completo.

    Returns:
        df con decenas de columnas de features.
    """
    df = df.copy()

    # --- Retornos ---
    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_2"]  = df["close"].pct_change(2)
    df["ret_3"]  = df["close"].pct_change(3)
    df["ret_5"]  = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_20"] = df["close"].pct_change(20)

    # --- MAs y EMAs ---
    for n in [5, 10, 20, 50, 100, 200]:
        df[f"ma_{n}"]  = df["close"].rolling(n).mean()
    for n in [9, 21, 55]:
        df[f"ema_{n}"] = df["close"].ewm(span=n).mean()

    df["price_vs_ma20"]  = df["close"] / df["ma_20"]  - 1
    df["price_vs_ma50"]  = df["close"] / df["ma_50"]  - 1
    df["price_vs_ma200"] = df["close"] / df["ma_200"] - 1
    df["ma20_vs_ma50"]   = df["ma_20"] / df["ma_50"]  - 1
    df["ma50_vs_ma200"]  = df["ma_50"] / df["ma_200"] - 1
    df["ema21_vs_ema55"] = df["ema_21"] / df["ema_55"] - 1

    # --- ATR y volatilidad ---
    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()
    df["atr"]      = atr
    df["atr_pct"]  = atr / df["close"]
    df["atr_rank"] = df["atr_pct"].rolling(100).rank(pct=True)

    df["vol_5"]  = df["ret_1"].rolling(5).std()
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_ratio_5_20"] = df["vol_5"] / (df["vol_20"] + 1e-9)

    # --- Rango de la barra ---
    df["bar_range"]  = (df["high"] - df["low"]) / df["close"]
    df["bar_body"]   = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["bar_upper_wick"] = (df["high"] - df[["open","close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
    df["bar_lower_wick"] = (df[["open","close"]].min(axis=1) - df["low"])  / (df["high"] - df["low"] + 1e-9)
    df["is_bullish_bar"] = (df["close"] > df["open"]).astype(int)
    df["bar_range_atr"]  = (df["high"] - df["low"]) / (atr + 1e-9)  # Rango en ATRs

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_pband"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_above_upper"] = (df["close"] > bb.bollinger_hband()).astype(int)
    df["bb_below_lower"] = (df["close"] < bb.bollinger_lband()).astype(int)

    # --- RSI multi-período ---
    for w in [7, 14, 21]:
        df[f"rsi_{w}"] = ta.momentum.RSIIndicator(df["close"], window=w).rsi()
    df["rsi_slope"] = df["rsi_14"].diff(3)
    df["rsi_above_50"] = (df["rsi_14"] > 50).astype(int)

    # --- MACD ---
    macd = ta.trend.MACD(df["close"])
    df["macd"]       = macd.macd()
    df["macd_signal"]= macd.macd_signal()
    df["macd_hist"]  = macd.macd_diff()
    df["macd_cross"] = np.sign(df["macd_hist"]) - np.sign(df["macd_hist"].shift(1))

    # --- Stochastic ---
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # --- CCI ---
    df["cci_14"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=14).cci()
    df["cci_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()

    # --- ADX (fuerza de tendencia) ---
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"]       = adx.adx()
    df["adx_pos"]   = adx.adx_pos()
    df["adx_neg"]   = adx.adx_neg()
    df["adx_trend"] = (df["adx"] > 25).astype(int)

    # --- Donchian (posición relativa del precio) ---
    for n in [20, 50]:
        df[f"dc_upper_{n}"] = df["high"].rolling(n).max()
        df[f"dc_lower_{n}"] = df["low"].rolling(n).min()
        df[f"dc_pos_{n}"]   = (df["close"] - df[f"dc_lower_{n}"]) / (
            df[f"dc_upper_{n}"] - df[f"dc_lower_{n}"] + 1e-9
        )

    # --- Volumen ---
    df["vol_ma_20"]   = df["volume"].rolling(20).mean()
    df["vol_ratio"]   = df["volume"] / (df["vol_ma_20"] + 1e-9)
    df["vol_trend_5"] = df["volume"].pct_change(5)

    # --- Momentum / Aceleración ---
    df["mom_5"]  = df["close"] / (df["close"].shift(5)  + 1e-9) - 1
    df["mom_10"] = df["close"] / (df["close"].shift(10) + 1e-9) - 1
    df["mom_20"] = df["close"] / (df["close"].shift(20) + 1e-9) - 1
    df["accel"]  = df["ret_1"] - df["ret_1"].shift(3)

    # --- Calendario ---
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["is_monday"]  = (df.index.dayofweek == 0).astype(int)
    df["is_friday"]  = (df.index.dayofweek == 4).astype(int)

    # --- Skewness / Kurtosis ---
    df["ret_skew5"]  = df["ret_1"].rolling(5).skew()
    df["ret_kurt10"] = df["ret_1"].rolling(10).kurt()

    return df


def build_setup_feature_row(
    df_with_features: pd.DataFrame,
    setup_ts: pd.Timestamp,
    setup_type: str,
    direction: str,
    raw_score: float,
    macro_df: pd.DataFrame = None,
    funding_series: pd.Series = None,
    symbol: str = None,
) -> dict:
    """
    Construye el vector de features para UN setup específico.
    Usa la barra del setup + contexto de las últimas N barras.

    Args:
        df_with_features: DataFrame con compute_base_features() ya aplicado
        setup_ts:         Timestamp de la barra del setup
        setup_type:       "breakout" | "pullback" | "reversal"
        direction:        "long" | "short"
        raw_score:        Score técnico del detector (0-1)
        macro_df:         Retornos macro (opcional)

    Returns:
        dict con el vector de features para el modelo ML.
    """
    try:
        row = df_with_features.loc[setup_ts]
    except KeyError:
        # Si el timestamp exacto no está, buscar el más cercano
        idx = df_with_features.index.get_indexer([setup_ts], method="nearest")[0]
        if idx < 0:
            return {}
        row = df_with_features.iloc[idx]

    features = {}

    # Features de la barra actual
    for col in [
        "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
        "atr_pct", "atr_rank", "vol_ratio", "vol_ratio_5_20",
        "bar_range_atr", "bar_body", "bar_upper_wick", "bar_lower_wick",
        "is_bullish_bar", "bb_pband", "bb_width",
        "rsi_7", "rsi_14", "rsi_21", "rsi_slope", "rsi_above_50",
        "macd_hist", "macd_cross", "stoch_k", "stoch_d",
        "cci_14", "adx", "adx_trend",
        "dc_pos_20", "dc_pos_50",
        "price_vs_ma20", "price_vs_ma50", "price_vs_ma200",
        "ma20_vs_ma50", "ma50_vs_ma200", "ema21_vs_ema55",
        "mom_5", "mom_10", "mom_20", "accel",
        "vol_trend_5", "bb_above_upper", "bb_below_lower",
        "ret_skew5", "ret_kurt10",
        "hour", "dayofweek", "is_monday", "is_friday",
    ]:
        val = row.get(col, np.nan)
        features[col] = float(val) if pd.notna(val) else np.nan

    # Features del setup
    features["raw_score"]    = raw_score
    features["is_long"]      = 1 if direction == "long" else 0
    features["setup_break"]  = 1 if setup_type == "breakout" else 0
    features["setup_pull"]   = 1 if setup_type == "pullback" else 0
    features["setup_rev"]    = 1 if setup_type == "reversal" else 0

    # Features invertidas para SHORT (el modelo puede aprender de ambas direcciones)
    if direction == "short":
        for col in ["ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
                    "mom_5", "mom_10", "bb_pband", "dc_pos_20", "dc_pos_50",
                    "price_vs_ma20", "price_vs_ma50", "accel", "macd_hist"]:
            if col in features:
                features[col] = -features[col]

    # Funding rate (solo crypto): tasa actual, promedio 7d y señal direccional
    if funding_series is not None and not funding_series.empty:
        try:
            setup_date = setup_ts.date() if hasattr(setup_ts, "date") else pd.Timestamp(setup_ts).date()
            fr = funding_series.copy()
            fr.index = pd.to_datetime(fr.index).normalize()
            past = fr[fr.index.date <= setup_date].tail(7)
            if len(past) > 0:
                features["funding_current"] = float(past.iloc[-1])
                features["funding_7d_mean"] = float(past.mean())
                features["funding_7d_std"]  = float(past.std()) if len(past) > 1 else 0.0
                features["funding_trend"]   = float(past.iloc[-1] - past.iloc[0]) if len(past) > 1 else 0.0
                # Señal: funding positivo = mercado sobrecargado de longs (contrarian para LONG)
                features["funding_signal"] = -1.0 if (direction == "long" and past.iloc[-1] > 0.001) else \
                                              1.0 if (direction == "long" and past.iloc[-1] < -0.001) else 0.0
        except Exception:
            pass

    # Macro (alinear por fecha calendario)
    if macro_df is not None and not macro_df.empty:
        setup_date = setup_ts.date()
        macro_dates = [d.date() for d in macro_df.index]
        macro_map = dict(zip(macro_dates, range(len(macro_dates))))

        # Feature instántanea del día del setup
        if setup_date in macro_map:
            idx_m = macro_map[setup_date]
            for col in macro_df.columns:
                val = macro_df.iloc[idx_m][col]
                features[f"macro_{col}"] = float(val) if pd.notna(val) else np.nan

        # Features de TENDENCIA macro 30d: mucho más informativas que el valor diario.
        # Para Gold: dxy_trend negativo (DXY cayendo) → gold sube → señal LONG favorable.
        # Para SP500: sp500_trend positivo (mercado alcista) → pullbacks más fiables.
        past_mask = [d <= setup_date for d in macro_dates]
        past_macro = macro_df[past_mask].tail(30) if any(past_mask) else pd.DataFrame()
        if not past_macro.empty and len(past_macro) >= 10:
            for col in past_macro.columns:
                series = past_macro[col].dropna()
                if len(series) >= 5:
                    features[f"macro_{col}_trend30"] = float(series.mean())
                    # Momentum: compara última semana vs tendencia del mes
                    week_mean = float(series.tail(5).mean())
                    features[f"macro_{col}_accel"] = week_mean - features[f"macro_{col}_trend30"]

            # Alineamiento macro según tipo de activo.
            # Bug fix: la clave correcta es "macro_dxy_ret_trend30", no "dxy_ret_trend30".
            from config.settings import ASSETS as _assets
            is_safe_haven = _assets.get(symbol or "", {}).get("safe_haven", False)

            if is_safe_haven:
                # Gold, Silver: correlación NEGATIVA con DXY.
                # DXY cayendo + LONG = alineado → +1; DXY subiendo + LONG = contrario → -1
                if "macro_dxy_ret_trend30" in features:
                    dxy_dir = np.sign(features["macro_dxy_ret_trend30"])
                    features["macro_dxy_alignment"] = float(-dxy_dir if direction == "long" else dxy_dir)
                # Cross-asset: Gold ETF (GLD) como proxy de metales preciosos.
                # Para Silver: gold subiendo + LONG Silver = metales en tendencia → +1
                if "macro_gold_etf_ret_trend30" in features:
                    gold_dir = np.sign(features["macro_gold_etf_ret_trend30"])
                    features["macro_gold_alignment"] = float(gold_dir if direction == "long" else -gold_dir)
            else:
                # Crypto, SP500: correlación POSITIVA con SP500 (risk-on / risk-off).
                # SP500 subiendo + LONG crypto/equity = alineado → +1
                if "macro_sp500_ret_trend30" in features:
                    sp500_dir = np.sign(features["macro_sp500_ret_trend30"])
                    features["macro_sp500_alignment"] = float(sp500_dir if direction == "long" else -sp500_dir)

    return features


def build_training_dataset(
    df_setups: pd.DataFrame,
    df_ohlcv:  pd.DataFrame,
    forward_bars: int = 12,
    macro_df:  pd.DataFrame = None,
    funding_series: pd.Series = None,
    symbol: str = None,
) -> pd.DataFrame:
    """
    Construye el dataset completo de entrenamiento.
    Etiqueta cada setup: 1 si alcanzó take_profit, 0 si tocó stop_loss.

    Args:
        df_setups:    DataFrame de setups detectados (salida de detect_all_setups)
        df_ohlcv:     OHLCV completo del mismo timeframe
        forward_bars: Barras hacia adelante para evaluar el resultado
        macro_df:     Datos macro para features adicionales

    Returns:
        DataFrame con features + columna 'target' (1=win, 0=loss).
    """
    if df_setups.empty or df_ohlcv.empty:
        return pd.DataFrame()

    df_feat = compute_base_features(df_ohlcv)
    rows = []

    for _, setup in df_setups.iterrows():
        ts         = setup["ts"]
        direction  = setup["direction"]
        stop_loss  = setup["stop_loss"]
        take_profit= setup["take_profit"]
        entry      = setup["close"]

        # Buscar resultado en las siguientes forward_bars barras
        future = df_ohlcv[df_ohlcv.index > ts].head(forward_bars)
        if len(future) < 3:
            continue

        target = _label_outcome(future, direction, stop_loss, take_profit, entry)
        if target is None:
            continue  # No hubo resolución en el período

        feat_row = build_setup_feature_row(
            df_feat, ts,
            setup["setup_type"],
            direction,
            setup["raw_score"],
            macro_df,
            funding_series,
            symbol=symbol,
        )
        if not feat_row:
            continue

        feat_row["target"] = float(target)
        feat_row["ts"]     = ts
        rows.append(feat_row)

    if not rows:
        logger.warning("No hay muestras de entrenamiento suficientes")
        return pd.DataFrame()

    df_train = pd.DataFrame(rows)
    df_train = df_train.set_index("ts")

    n_win  = int(df_train["target"].sum())
    n_loss = len(df_train) - n_win
    logger.info(
        f"Dataset: {len(df_train)} setups | "
        f"WIN: {n_win} ({n_win/len(df_train)*100:.0f}%) | "
        f"LOSS: {n_loss} ({n_loss/len(df_train)*100:.0f}%)"
    )
    return df_train


def _label_outcome(
    future: pd.DataFrame,
    direction: str,
    stop_loss: float,
    take_profit: float,
    entry: float,
) -> int | None:
    """
    Etiqueta el outcome del setup mirando las barras futuras.
    Retorna 1 (win), 0 (loss), None (no resuelto).
    """
    for _, bar in future.iterrows():
        if direction == "long":
            if bar["low"]  <= stop_loss:   return 0
            if bar["high"] >= take_profit: return 1
        else:  # short
            if bar["high"] >= stop_loss:   return 0
            if bar["low"]  <= take_profit: return 1
    return None
