"""
Entrenamiento del modelo ML de priorización de setups.
Usa LightGBM con validación walk-forward temporal.
"""
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV
from loguru import logger

from config.settings import ML_PARAMS, WALK_FORWARD, BASE_DIR

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _get_feature_cols(df: pd.DataFrame) -> list:
    """Retorna las columnas que son features (excluye target, ts, etc.)."""
    exclude = {"target", "ts", "symbol", "setup_type", "direction"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]


def train_model(
    df_train: pd.DataFrame,
    symbol:   str = "ALL",
    save:     bool = True,
) -> tuple:
    """
    Entrena el modelo sobre el dataset de setups etiquetados.

    Args:
        df_train: DataFrame con features + columna 'target'
        symbol:   Nombre del activo (para guardar el modelo)
        save:     Si True, guarda el modelo en disco

    Returns:
        (model, feature_cols, metrics_dict)
    """
    p = ML_PARAMS
    feature_cols = _get_feature_cols(df_train)

    if len(df_train) < p["min_train_samples"]:
        logger.warning(
            f"Datos insuficientes para entrenar: {len(df_train)} "
            f"(mínimo {p['min_train_samples']})"
        )
        return None, [], {}

    X = df_train[feature_cols].copy()
    y = df_train["target"].copy()

    # Imputar NaN con mediana (los NaN vienen de indicadores con lookback)
    medians = X.median()
    X = X.fillna(medians)

    logger.info(f"Entrenando modelo | {len(X)} muestras | {len(feature_cols)} features")

    # --- Walk-forward para métricas honestas ---
    wf_metrics = _walk_forward_eval(X, y)

    # --- Entrenamiento final sobre todos los datos ---
    model = lgb.LGBMClassifier(
        n_estimators     = p["n_estimators"],
        learning_rate    = p["learning_rate"],
        num_leaves       = p["num_leaves"],
        max_depth        = p["max_depth"],
        min_child_samples= p["min_child_samples"],
        subsample        = p["subsample"],
        colsample_bytree = p["colsample_bytree"],
        class_weight     = "balanced",
        random_state     = 42,
        verbose          = -1,
    )

    # Calibrar probabilidades (Platt scaling) para umbrales más fiables
    calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
    calibrated.fit(X, y)

    if save:
        model_path = MODEL_DIR / f"model_{symbol}.pkl"
        meta_path  = MODEL_DIR / f"model_{symbol}_meta.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(calibrated, f)
        with open(meta_path, "wb") as f:
            pickle.dump({
                "feature_cols": feature_cols,
                "medians":      medians.to_dict(),
                "trained_at":   datetime.utcnow().isoformat(),
                "n_samples":    len(X),
                "wf_metrics":   wf_metrics,
            }, f)
        logger.success(f"Modelo guardado: {model_path}")

    logger.info(
        f"Walk-Forward → AUC: {wf_metrics['auc']:.3f} | "
        f"Acc: {wf_metrics['accuracy']:.3f} | "
        f"Precision@thresh: {wf_metrics['precision']:.3f}"
    )
    return calibrated, feature_cols, wf_metrics


def _walk_forward_eval(X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluación walk-forward con TimeSeriesSplit."""
    p = ML_PARAMS
    tss = TimeSeriesSplit(n_splits=WALK_FORWARD["n_splits"])

    aucs, accs, precs, recs = [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(tss.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        if len(X_tr) < 30 or len(X_te) < 5:
            continue
        if y_tr.nunique() < 2:
            continue

        m = lgb.LGBMClassifier(
            n_estimators     = p["n_estimators"],
            learning_rate    = p["learning_rate"],
            num_leaves       = p["num_leaves"],
            max_depth        = p["max_depth"],
            min_child_samples= p["min_child_samples"],
            subsample        = p["subsample"],
            colsample_bytree = p["colsample_bytree"],
            class_weight     = "balanced",
            random_state     = 42,
            verbose          = -1,
        )
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        pred  = (proba >= p["prob_threshold"]).astype(int)

        try:
            aucs.append(roc_auc_score(y_te, proba))
        except Exception:
            pass
        accs.append(accuracy_score(y_te, pred))
        precs.append(precision_score(y_te, pred, zero_division=0))
        recs.append(recall_score(y_te, pred, zero_division=0))

    return {
        "auc":       round(np.mean(aucs)  if aucs  else 0.0, 4),
        "accuracy":  round(np.mean(accs)  if accs  else 0.0, 4),
        "precision": round(np.mean(precs) if precs else 0.0, 4),
        "recall":    round(np.mean(recs)  if recs  else 0.0, 4),
        "n_folds":   len(accs),
    }


def load_model(symbol: str) -> tuple:
    """
    Carga modelo y metadata desde disco.
    Returns: (model, feature_cols, medians_dict) o (None, [], {})
    """
    model_path = MODEL_DIR / f"model_{symbol}.pkl"
    meta_path  = MODEL_DIR / f"model_{symbol}_meta.pkl"

    if not model_path.exists():
        logger.warning(f"Modelo no encontrado para {symbol}. Hay que entrenarlo primero.")
        return None, [], {}

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    logger.info(
        f"Modelo cargado: {symbol} | "
        f"Entrenado: {meta.get('trained_at', '?')} | "
        f"Muestras: {meta.get('n_samples', '?')}"
    )
    return model, meta["feature_cols"], meta["medians"]


def get_model_importance(symbol: str, top_n: int = 20) -> pd.Series:
    """Retorna feature importances del modelo guardado."""
    model_path = MODEL_DIR / f"model_{symbol}.pkl"
    if not model_path.exists():
        return pd.Series(dtype=float)
    with open(model_path, "rb") as f:
        calibrated = pickle.load(f)
    # El modelo base está dentro del CalibratedClassifierCV
    base = calibrated.calibrated_classifiers_[0].estimator
    meta_path = MODEL_DIR / f"model_{symbol}_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    imp = pd.Series(
        base.feature_importances_,
        index=meta["feature_cols"]
    ).sort_values(ascending=False)
    return imp.head(top_n)
