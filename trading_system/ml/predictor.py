"""
Inferencia: califica setups activos con el modelo entrenado.
"""
import pandas as pd
import numpy as np
from loguru import logger

from ml.trainer  import load_model
from ml.features import compute_base_features, build_setup_feature_row
from config.settings import ML_PARAMS


def score_setups(
    df_setups:      pd.DataFrame,
    df_ohlcv:       pd.DataFrame,
    symbol:         str,
    macro_df:       pd.DataFrame = None,
    funding_series: pd.Series = None,
    threshold:      float = None,
) -> pd.DataFrame:
    """
    Aplica el modelo ML a cada setup en df_setups y agrega la columna 'ml_score'.

    Args:
        df_setups: DataFrame de setups (salida de detect_all_setups)
        df_ohlcv:  OHLCV completo del mismo timeframe
        symbol:    Para cargar el modelo correcto
        macro_df:  Features macroeconómicas

    Returns:
        df_setups con columna 'ml_score' y 'ml_approved' añadidas.
    """
    if df_setups.empty:
        return df_setups

    from config.settings import ASSETS
    asset_cfg = ASSETS.get(symbol, {})

    # Bypass ML si el activo tiene use_ml=False (AUC demasiado bajo para añadir valor)
    if not asset_cfg.get("use_ml", True):
        effective_threshold = threshold if threshold is not None else \
            asset_cfg.get("ml_threshold", ML_PARAMS["prob_threshold"])
        logger.info(f"ML deshabilitado para {symbol}. Usando raw_score (umbral {effective_threshold:.2f})")
        df_setups = df_setups.copy()
        df_setups["ml_score"]    = df_setups["raw_score"]
        df_setups["ml_approved"] = df_setups["ml_score"] >= effective_threshold
        n_approved = df_setups["ml_approved"].sum()
        scores = df_setups["ml_score"]
        logger.info(
            f"raw_score {symbol}: {len(df_setups)} setups → "
            f"{n_approved} aprobados | "
            f"scores min={scores.min():.3f} med={scores.median():.3f} max={scores.max():.3f}"
        )
        return df_setups

    model, feature_cols, medians, adaptive_threshold = load_model(symbol)

    effective_threshold = threshold if threshold is not None else \
        adaptive_threshold if adaptive_threshold is not None else \
        asset_cfg.get("ml_threshold", ML_PARAMS["prob_threshold"])

    if model is None:
        logger.warning(f"No hay modelo entrenado para {symbol}. Usando raw_score como proxy.")
        df_setups = df_setups.copy()
        df_setups["ml_score"]    = df_setups["raw_score"]
        df_setups["ml_approved"] = df_setups["ml_score"] >= effective_threshold
        return df_setups

    df_feat = compute_base_features(df_ohlcv)
    scores  = []

    for _, setup in df_setups.iterrows():
        feat_row = build_setup_feature_row(
            df_feat,
            setup["ts"],
            setup["setup_type"],
            setup["direction"],
            setup["raw_score"],
            macro_df,
            funding_series,
            symbol=symbol,
        )
        if not feat_row:
            scores.append(0.5)
            continue

        # Alinear features con las que el modelo conoce
        feat_df = pd.DataFrame([feat_row])
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = medians.get(col, 0.0)
        feat_df = feat_df[feature_cols]

        # Imputar NaN con medianas del entrenamiento
        for col in feature_cols:
            if feat_df[col].isna().any():
                feat_df[col] = medians.get(col, 0.0)

        try:
            prob = float(model.predict_proba(feat_df)[0, 1])
        except Exception as e:
            logger.warning(f"Error en predicción: {e}")
            prob = 0.5
        scores.append(prob)

    df_setups = df_setups.copy()
    df_setups["ml_score"]    = scores
    df_setups["ml_approved"] = df_setups["ml_score"] >= effective_threshold

    n_approved = df_setups["ml_approved"].sum()
    scores = df_setups["ml_score"]
    logger.info(
        f"ML scoring {symbol}: {len(df_setups)} setups → "
        f"{n_approved} aprobados (umbral {effective_threshold:.2f}) | "
        f"scores min={scores.min():.3f} med={scores.median():.3f} max={scores.max():.3f}"
    )
    return df_setups


def get_best_setup(df_setups: pd.DataFrame,
                   n_last_bars: int = 3,
                   current_price: float = None,
                   max_dist_atr: float = None) -> pd.Series | None:
    """
    Retorna el mejor setup reciente aprobado por el ML.
    None si no hay setups válidos.

    Args:
        n_last_bars:   Ventana de barras recientes a considerar.
        current_price: Último precio conocido. Si se provee, se aplica el filtro
                       de frescura: se descartan los setups cuyo precio de entrada
                       (cierre de la barra del setup) ya esté a más de
                       max_dist_atr × ATR del precio actual, porque esa operación
                       ya no es ejecutable (el precio no volverá al nivel de entrada).
        max_dist_atr:  Múltiplo de ATR de distancia máxima entry↔precio actual.

    El filtro de frescura corrige dos problemas a la vez:
      1. Señales con entrada inalcanzable (p.ej. SHORT @ 7580 con precio en 7523).
      2. Sesgo de dirección: un setup obsoleto de alto score (típicamente un
         short de reversión en un máximo previo) dejaba de tapar a los setups
         frescos —long o short— cercanos al precio actual.
    """
    if df_setups.empty:
        return None
    cutoff = df_setups["ts"].max() - pd.Timedelta(hours=4 * n_last_bars)
    recent = df_setups[
        (df_setups["ts"] >= cutoff) &
        (df_setups.get("ml_approved", True) == True)
    ].copy()
    if recent.empty:
        return None

    # --- Filtro de frescura: la entrada debe seguir cerca del precio actual ---
    if current_price is not None and max_dist_atr is not None and "atr" in recent.columns:
        atr_safe   = recent["atr"].clip(lower=1e-9)
        dist_atr   = (recent["close"] - current_price).abs() / atr_safe
        executable = dist_atr <= max_dist_atr
        n_stale    = int((~executable).sum())
        if n_stale:
            logger.info(
                f"Frescura: {n_stale}/{len(recent)} setups descartados por entrada "
                f"lejana al precio actual ({current_price:.4f}, máx {max_dist_atr}×ATR)."
            )
        recent = recent[executable]
        if recent.empty:
            return None

    return recent.loc[recent["ml_score"].idxmax()]
