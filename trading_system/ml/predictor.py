"""
Inferencia: califica setups activos con el modelo entrenado.
"""
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import ML_PARAMS

# ml.trainer / ml.features se importan solo si el activo usa ML: lightgbm y
# scikit-learn ocupan ~85 MB de RAM y con use_ml=False (todos los activos) no se
# usan. Importa en Render Starter (512 MB compartidos entre el worker y el Shell).


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

    from ml.trainer  import load_model
    from ml.features import compute_base_features, build_setup_feature_row
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
                   max_dist_atr: float = None,
                   min_setup_ts=None,
                   max_setup_ts=None,
                   exclude_keys=None) -> pd.Series | None:
    """
    Retorna el mejor setup reciente aprobado por el ML.
    None si no hay setups válidos.

    Args:
        n_last_bars:   (modo heredado) ventana en barras de 4h contada desde el
                       último setup. Solo se usa si no se pasa min_setup_ts.
        current_price: Último precio conocido (cierre de la última vela CERRADA).
                       Si se provee, se aplica el filtro de frescura: se
                       descartan los setups cuyo precio de entrada (cierre de la
                       barra del setup) ya esté a más de max_dist_atr × ATR del
                       precio actual, porque esa operación ya no es ejecutable.
        max_dist_atr:  Múltiplo de ATR de distancia máxima entry↔precio actual.
        min_setup_ts:  Ventana anclada a AHORA (Fase 0): solo setups con
                       ts >= min_setup_ts (p. ej. la penúltima vela cerrada).
        max_setup_ts:  Solo setups con ts <= max_setup_ts (la última vela cerrada).
        exclude_keys:  Conjunto de (ts, direction) ya emitidos: se saltan para
                       que nunca se re-emita el mismo setup y para que, si el
                       mejor ya salió, pueda salir otro aprobado de la ventana.

    Por qué la ventana anclada (Fase 0, 2026-11): en el modo heredado el cutoff
    era `df_setups["ts"].max() - 4h×n`, es decir, contado desde el ÚLTIMO SETUP
    del histórico y no desde ahora. Si no aparecía un setup nuevo, el último
    seguía "vigente" durante semanas y se re-emitía cada vez que el precio volvía
    cerca de su entrada (8–9 de las 27 señales reales desde el freeze).

    El filtro de frescura corrige además:
      1. Señales con entrada inalcanzable (p.ej. SHORT @ 7580 con precio en 7523).
      2. Sesgo de dirección: un setup obsoleto de alto score dejaba de tapar a
         los setups frescos cercanos al precio actual.
    """
    if df_setups.empty:
        return None

    def _as_ts(value):
        t = pd.Timestamp(value)
        tz_series = getattr(df_setups["ts"].dt, "tz", None)
        if tz_series is not None and t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t

    if min_setup_ts is not None:
        in_window = df_setups["ts"] >= _as_ts(min_setup_ts)
        if max_setup_ts is not None:
            in_window &= df_setups["ts"] <= _as_ts(max_setup_ts)
    else:
        cutoff = df_setups["ts"].max() - pd.Timedelta(hours=4 * n_last_bars)
        in_window = df_setups["ts"] >= cutoff

    recent = df_setups[
        in_window &
        (df_setups.get("ml_approved", True) == True)
    ].copy()
    if recent.empty:
        return None

    if exclude_keys:
        keys = {(_as_ts(t), d) for t, d in exclude_keys}
        emitted = [(_as_ts(t), d) in keys
                   for t, d in zip(recent["ts"], recent["direction"])]
        recent = recent[[not e for e in emitted]]
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
