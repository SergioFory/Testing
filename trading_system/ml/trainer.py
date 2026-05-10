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


def _select_top_features(X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> list:
    """
    Selecciona las top_n features por importancia usando un modelo ligero.
    Reduce overfitting cuando hay pocas muestras y muchas features.
    """
    selector = lgb.LGBMClassifier(
        n_estimators=80, learning_rate=0.1, num_leaves=8,
        max_depth=3, min_child_samples=15,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    selector.fit(X, y)
    imp = pd.Series(selector.feature_importances_, index=X.columns)
    top = imp.nlargest(top_n).index.tolist()
    logger.info(f"Feature selection: {len(X.columns)} → {len(top)} features")
    return top


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

    # Filtrar a ventana de entrenamiento reciente (evita regímenes de mercado obsoletos).
    # Permite override por activo: ASSETS[symbol]["train_window_days"] tiene prioridad
    # sobre ML_PARAMS["train_window_days"]. 0 = sin filtro (usar todo el histórico).
    from config.settings import ASSETS
    asset_window = ASSETS.get(symbol, {}).get("train_window_days")
    train_window = asset_window if asset_window is not None else p.get("train_window_days", 0)
    if train_window and isinstance(df_train.index, pd.DatetimeIndex):
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=train_window)
        if df_train.index.tz is None:
            cutoff = cutoff.tz_localize(None)
        recent = df_train[df_train.index >= cutoff]
        if len(recent) >= p["min_train_samples"]:
            df_train = recent
            logger.info(f"Ventana de entrenamiento: últimos {train_window} días → {len(df_train)} muestras")

    X = df_train[feature_cols].copy()
    y = df_train["target"].copy()

    # Imputar NaN con mediana (los NaN vienen de indicadores con lookback)
    medians = X.median()
    pct_null = X.isna().mean()
    high_null = pct_null[pct_null > 0.3]
    if not high_null.empty:
        logger.warning(f"Features con >30% NaN (se imputan con mediana): {high_null.round(2).to_dict()}")
    X = X.fillna(medians)

    # Descartar features con >50% NaN: son ruido puro (la imputación no aporta señal)
    cols_high_null = pct_null[pct_null > 0.5].index.tolist()
    if cols_high_null:
        logger.warning(f"Descartando {len(cols_high_null)} features con >50% NaN: {cols_high_null}")
        X = X.drop(columns=cols_high_null)
        feature_cols = [c for c in feature_cols if c not in cols_high_null]

    top_n = p.get("top_features", 20)

    # --- Walk-forward con feature selection por fold (sin data leakage) ---
    wf_metrics = _walk_forward_eval(X, y, top_n=top_n if len(feature_cols) > top_n else 0)
    wf_metrics["n_samples"] = len(X)
    wf_metrics["threshold"] = p.get("prob_threshold", 0.5)

    # --- Selección de features para modelo final (correcto: usa todos los datos) ---
    if len(feature_cols) > top_n:
        feature_cols = _select_top_features(X, y, top_n=top_n)
        X = X[feature_cols]

    logger.info(f"Entrenando modelo | {len(X)} muestras | {len(feature_cols)} features")

    # --- Entrenamiento final sobre todos los datos ---
    model = lgb.LGBMClassifier(
        n_estimators     = p["n_estimators"],
        learning_rate    = p["learning_rate"],
        num_leaves       = p["num_leaves"],
        max_depth        = p["max_depth"],
        min_child_samples= p["min_child_samples"],
        subsample        = p["subsample"],
        colsample_bytree = p["colsample_bytree"],
        reg_alpha        = p.get("lambda_l1", 0.1),
        reg_lambda       = p.get("lambda_l2", 0.1),
        class_weight     = "balanced",
        random_state     = 42,
        verbose          = -1,
    )

    # Calibrar probabilidades (Platt scaling) para umbrales más fiables
    calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
    calibrated.fit(X, y)

    # Threshold adaptativo: percentil 85 de los scores en training.
    # Se ajusta solo cuando el modelo mejora; garantiza ~15% de setups aprobados.
    train_scores = calibrated.predict_proba(X)[:, 1]
    adaptive_threshold = round(float(np.percentile(train_scores, 85)), 4)
    wf_metrics["adaptive_threshold"] = adaptive_threshold
    logger.info(f"Threshold adaptativo (p85 training): {adaptive_threshold:.4f}")

    if save:
        model_path = MODEL_DIR / f"model_{symbol}.pkl"
        meta_path  = MODEL_DIR / f"model_{symbol}_meta.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(calibrated, f)
        with open(meta_path, "wb") as f:
            pickle.dump({
                "feature_cols":       feature_cols,
                "medians":            medians.to_dict(),
                "trained_at":         datetime.utcnow().isoformat(),
                "n_samples":          len(X),
                "wf_metrics":         wf_metrics,
                "adaptive_threshold": adaptive_threshold,
            }, f)
        logger.success(f"Modelo guardado: {model_path}")

    logger.info(
        f"Walk-Forward → AUC: {wf_metrics['auc']:.3f} | "
        f"Acc: {wf_metrics['accuracy']:.3f} | "
        f"Precision@thresh: {wf_metrics['precision']:.3f}"
    )
    return calibrated, feature_cols, wf_metrics


def _walk_forward_eval(X: pd.DataFrame, y: pd.Series, top_n: int = 0) -> dict:
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

        # Feature selection sólo sobre datos de train de este fold (evita data leakage)
        if top_n and len(X_tr.columns) > top_n:
            fold_features = _select_top_features(X_tr, y_tr, top_n=top_n)
            X_tr = X_tr[fold_features]
            X_te = X_te[fold_features]

        m = lgb.LGBMClassifier(
            n_estimators     = p["n_estimators"],
            learning_rate    = p["learning_rate"],
            num_leaves       = p["num_leaves"],
            max_depth        = p["max_depth"],
            min_child_samples= p["min_child_samples"],
            subsample        = p["subsample"],
            colsample_bytree = p["colsample_bytree"],
            reg_alpha        = p.get("lambda_l1", 0.1),
            reg_lambda       = p.get("lambda_l2", 0.1),
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
        return None, [], {}, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    logger.info(
        f"Modelo cargado: {symbol} | "
        f"Entrenado: {meta.get('trained_at', '?')} | "
        f"Muestras: {meta.get('n_samples', '?')}"
    )
    return model, meta["feature_cols"], meta["medians"], meta.get("adaptive_threshold")


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
