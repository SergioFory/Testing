"""
Puerta de validación (Fase 2, punto 2.1): `python main.py validate --protocol`.

Un examen FIJO, escrito de antemano, que cualquier cambio futuro de la
estrategia (umbral, filtro, activo, dirección) debe aprobar antes de llegar al
vivo. Comando OFFLINE (local o Shell de Render), nunca en el worker. Las reglas
están en la especificación de la Fase 2 y en REGLAS_DE_DECISION.md; aquí no se
deciden umbrales nuevos.

Qué hace este módulo (orquestación, informe y registro):
  1. Precondición: "vivo = validador" (backtest/live_emulation.compare_with_validator)
     con la configuración ACTUAL. Si falla, se detiene con un mensaje claro.
  2. Detecta y puntúa los setups UNA vez por activo (sin el filtro de dirección
     de la config, para poder evaluar "solo long" / "solo short") y simula cada
     setup UNA vez a precios medios con la convención de la Fase 1
     (validate._simulate_setup). Las variantes solo FILTRAN por umbral/dirección.
  3. Universo = setups APROBADOS por la configuración evaluada, uno por
     (vela, dirección) (como el validador y check-live; no "una posición a la vez").
  4. Reserva INTOCABLE: settings.PROTOCOL_RESERVE_START. La selección, la CPCV,
     el PBO y el DSR usan solo setups ANTERIORES (y desde el inicio del primer
     fold). Solo --final la evalúa y cada acceso se registra.
  5. Costes por escenario (backtest/costs.py): XM con swap 0/5/10 % y la
     referencia "binance" del validador. Tabla de E[R] con IC90 (bootstrap) y
     folds temporales fijos.
  6. Registro de pruebas append-only (trials_registry.jsonl): una línea por
     variante NUEVA evaluada (hash estable) y una por cada acceso --final. Si no
     se puede escribir, el comando lo dice y NO da veredicto.
  7. Holdout OKX (cripto, 2019-12 → 2022-04), solo con --final: de la tabla
     ohlcv (source='okx') si hay suficiente o, con --okx, descargado en memoria.

Robustez y veredicto (segunda mitad del 2.1):
  - analyze_symbol(...) devuelve, además de las cifras, las claves privadas
    "_trades" (operaciones de la selección, todas las direcciones), "_masks" (id
    de variante → máscara sobre "_trades"), "_R" (escenario → R netos), "_bounds"
    (velas [p0, p1) de la selección) y "_fwd" (horizonte). robustness_for() arma
    con ellas la rejilla para backtest/robustness.py: CPCV purgada con
    re-selección interna, PBO por CSCV (S=16) y DSR (con N = total_trials() del
    registro, que incluye la semilla de 400). Se calcula DESPUÉS de escribir el
    registro, para que N cuente también las variantes de esta ejecución.
  - evaluate_criteria() aplica los criterios de APROBADO a)–e) fijados el
    2026-11-16 (constantes CRIT_*; texto en REGLAS_DE_DECISION.md). Sin --final
    un candidato nunca sale "APROBADO" (falta e).

Rendimiento medido (local, TS_CACHE=1): ver LOG de la Fase 2.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import cost_breakdown_R, cost_scenarios, net_returns

# ---------------------------------------------------------------------------
# Constantes del protocolo (fijadas en la especificación de la Fase 2)
# ---------------------------------------------------------------------------
REGISTRY_PATH = Path(__file__).parent / "trials_registry.jsonl"
# Solo para pruebas/desarrollo: apunta el registro a otro archivo.
REGISTRY_ENV = "TRIALS_REGISTRY_PATH"
SEED_ENTRY = {"kind": "seed", "historical_trials": 400,
              "note": "variantes probadas antes de 2026-11 (informe final §4.1)"}

# Rejilla: umbral actual + desplazamientos × conjuntos de dirección = 15 variantes.
THRESHOLD_OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)
DIRECTION_SETS = (("ambas", ("long", "short")),
                  ("solo long", ("long",)),
                  ("solo short", ("short",)))
# Mínimo de operaciones para que una variante pueda ser elegida (CPCV interna).
MIN_TRADES_SELECT = 10

# Folds temporales fijos [inicio, fin) en UTC. El primer inicio marca también el
# comienzo de la muestra de selección (antes, en cripto, es el holdout OKX).
FOLDS = {
    "crypto": (("2022-05→2023-12", "2022-05-01", "2024-01-01"),
               ("2024", "2024-01-01", "2025-01-01"),
               ("2025-01→2025-11-14", "2025-01-01", "2025-11-15")),
    "daily": (("2006–2014", "2006-01-01", "2015-01-01"),
              ("2015–2020", "2015-01-01", "2021-01-01"),
              ("2021→2025-11-14", "2021-01-01", "2025-11-15")),
}

# Holdout OKX (cripto): setups en [inicio, fin). Los perpetuos USDT de OKX
# empiezan el 2019-12-16 (BTC) / 2019-12-25 (ETH); Gate.io, en 2022-04/05.
OKX_HOLDOUT_START = "2019-12-01"
OKX_HOLDOUT_END = "2022-05-01"
# Cobertura mínima en la tabla ohlcv para no tener que descargar.
OKX_TABLE_MIN_FIRST = "2020-01-01"     # la serie debe empezar antes de esto
OKX_TABLE_MIN_LAST = "2022-04-26"      # ... y llegar al menos hasta aquí
OKX_TABLE_MIN_COVERAGE = 0.95          # velas presentes / velas esperadas

# Bootstrap del IC90 (determinista).
BOOT_N = 10_000
BOOT_SEED = 20261116
CI_LEVEL = 0.90

# Claves de ASSETS que NO son estrategia (no entran en la huella de la variante).
_NON_STRATEGY_KEYS = {"label", "enabled", "notify", "exchange", "train_window_days",
                      "days_history", "ml_threshold", "directions"}


class ProtocolError(RuntimeError):
    """Error que detiene el protocolo con un mensaje para el usuario."""


class RegistryError(RuntimeError):
    """El registro de pruebas no se puede leer o escribir: no hay veredicto."""


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def _settings():
    from config import settings
    return settings


@contextlib.contextmanager
def config_override(symbol: str, overrides: dict | None = None,
                    params_override: dict | None = None):
    """
    Aplica TEMPORALMENTE cambios de configuración a un activo (en memoria):
    overrides sobre ASSETS[symbol] (valor None = quitar la clave) y
    params_override sobre ASSET_PARAMS_OVERRIDE[symbol] ({setup: {param: valor}}).
    Siempre restaura la configuración original al salir.
    """
    s = _settings()
    a0 = copy.deepcopy(s.ASSETS.get(symbol, {}))
    had_p = symbol in s.ASSET_PARAMS_OVERRIDE
    p0 = copy.deepcopy(s.ASSET_PARAMS_OVERRIDE.get(symbol, {}))
    try:
        for k, v in (overrides or {}).items():
            if v is None:
                s.ASSETS[symbol].pop(k, None)
            else:
                s.ASSETS[symbol][k] = copy.deepcopy(v)
        if params_override:
            s.ASSET_PARAMS_OVERRIDE.setdefault(symbol, {})
            for setup, kv in params_override.items():
                s.ASSET_PARAMS_OVERRIDE[symbol].setdefault(setup, {}).update(kv)
        yield
    finally:
        s.ASSETS[symbol] = a0
        if had_p:
            s.ASSET_PARAMS_OVERRIDE[symbol] = p0
        else:
            s.ASSET_PARAMS_OVERRIDE.pop(symbol, None)


def current_variant(symbol: str) -> dict:
    """Umbral y direcciones de la configuración ACTUAL del activo."""
    cfg = _settings().ASSETS[symbol]
    return {"ml_threshold": round(float(cfg.get("ml_threshold", 0.55)), 4),
            "directions": tuple(sorted(cfg.get("directions") or ("long", "short")))}


def fold_kind(symbol: str) -> str:
    """'daily' para activos de sesión con velas diarias (yfinance); si no, 'crypto'."""
    return "daily" if _settings().ASSETS[symbol].get("source") == "yfinance" else "crypto"


def strategy_fingerprint(symbol: str) -> str:
    """
    Huella (sha256, 12 hex) de todo lo que define la estrategia del activo salvo
    el umbral y la dirección (que son ejes de la rejilla): ASSETS[symbol] sin las
    claves operativas, ASSET_PARAMS_OVERRIDE[symbol] y los parámetros globales de
    los detectores. Si cambia cualquier cosa de la estrategia, cambian los hashes
    de todas sus variantes y cuentan como pruebas NUEVAS (conservador).
    """
    s = _settings()
    cfg = {k: v for k, v in s.ASSETS[symbol].items() if k not in _NON_STRATEGY_KEYS}
    blob = {"asset": cfg, "params": s.ASSET_PARAMS_OVERRIDE.get(symbol, {}),
            "breakout": s.BREAKOUT_PARAMS, "pullback": s.PULLBACK_PARAMS,
            "reversal": s.REVERSAL_PARAMS, "risk": {k: s.RISK[k] for k in
                                                    ("atr_stop_mult", "atr_target_mult")}}
    raw = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Registro de pruebas (append-only)
# ---------------------------------------------------------------------------

def registry_path() -> Path:
    """Ruta del registro: TRIALS_REGISTRY_PATH si está definida; si no, la del repo."""
    env = os.environ.get(REGISTRY_ENV)
    return Path(env) if env else REGISTRY_PATH


def variant_params(symbol: str, ml_threshold: float, directions, extra: dict | None = None,
                   fingerprint: str | None = None) -> dict:
    """
    Parámetros canónicos de una variante (lo que se hashea). FIJO: cambiar este
    formato cambia todos los hashes y duplicaría pruebas en el registro.
        {"symbol", "ml_threshold" (4 decimales), "directions" (ordenadas),
         "strategy" (strategy_fingerprint), "extra" (overrides del candidato)}
    """
    return {"symbol": symbol,
            "ml_threshold": round(float(ml_threshold), 4),
            "directions": sorted(directions),
            "strategy": fingerprint or strategy_fingerprint(symbol),
            "extra": extra or {}}


def variant_hash(params: dict) -> str:
    """Hash estable (sha256, 16 hex) de variant_params(): JSON con claves ordenadas."""
    raw = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def read_registry(path: Path) -> list[dict]:
    """
    Lee el registro completo. Lanza RegistryError si no existe, si una línea no
    es JSON válido o si la primera línea no es la semilla.
    """
    path = Path(path)
    try:
        text_ = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"no se puede leer {path}: {exc}") from exc
    entries = []
    for n, line in enumerate(text_.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{path} dañado en la línea {n}: {exc}") from exc
    if not entries or entries[0].get("kind") != "seed":
        raise RegistryError(f"{path}: la primera línea debe ser la semilla {{\"kind\": \"seed\", ...}}")
    return entries


def _append_lines(path: Path, entries: list[dict]) -> None:
    """Añade líneas JSON al final en UNA escritura con flush + fsync."""
    if not entries:
        return
    payload = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    try:
        needs_nl = path.exists() and path.stat().st_size > 0 and \
            path.read_bytes()[-1:] != b"\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(("\n" if needs_nl else "") + payload)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        raise RegistryError(f"no se puede escribir en {path}: {exc}") from exc


def ensure_registry(path: Path) -> list[dict]:
    """Crea el registro con la semilla si no existe y devuelve sus entradas."""
    path = Path(path)
    if not path.exists():
        _append_lines(path, [SEED_ENTRY])
    return read_registry(path)


def registered_hashes(entries: list[dict]) -> set:
    return {e["hash"] for e in entries if e.get("kind") == "variant" and "hash" in e}


def total_trials(entries: list[dict]) -> int:
    """Pruebas para el DSR: semilla histórica + variantes distintas registradas."""
    seed = sum(int(e.get("historical_trials", 0)) for e in entries if e.get("kind") == "seed")
    return seed + len(registered_hashes(entries))


def final_accesses(entries: list[dict], vhash: str) -> list[dict]:
    """Accesos --final anteriores de la variante con ese hash."""
    return [e for e in entries if e.get("kind") == "final_access" and e.get("hash") == vhash]


def new_variant_entries(entries: list[dict], variants: list[dict], run_id: str) -> list[dict]:
    """Líneas a añadir: una por variante cuyo hash aún no está en el registro."""
    seen = registered_hashes(entries)
    out = []
    for v in variants:
        if v["hash"] in seen:
            continue
        seen.add(v["hash"])
        out.append({"kind": "variant", "hash": v["hash"], "symbol": v["params"]["symbol"],
                    "params": v["params"], "label": v["label"], "run": run_id})
    return out


def final_access_entry(variant: dict, run_id: str, okx: bool, role: str | None = None) -> dict:
    """Línea de acceso a la reserva (y al holdout OKX en cripto). role: "candidate",
    "reference" (configuración actual enseñada junto a un candidato) o "current"
    (configuración actual evaluada sola). Las líneas antiguas no tienen role."""
    e = {"kind": "final_access", "hash": variant["hash"], "symbol": variant["params"]["symbol"],
         "label": variant["label"], "run": run_id, "okx": bool(okx)}
    if role:
        e["role"] = role
    return e


# ---------------------------------------------------------------------------
# Detección y simulación (una vez por activo)
# ---------------------------------------------------------------------------

def detect_universe(symbol: str, df_4h: pd.DataFrame, df_daily: pd.DataFrame = None,
                    overrides: dict | None = None, params_override: dict | None = None) -> pd.DataFrame:
    """
    Setups del activo en TODAS las direcciones (se quita "directions" de la
    config) con su raw_score; el resto de filtros de la config (tipos de setup,
    tendencia, sesión) se respetan. overrides/params_override permiten evaluar
    un candidato que cambie otros parámetros (config_override).

    Solo activos con use_ml=False (el score es raw_score): con ML el umbral no
    sería comparable y el protocolo se detiene.
    """
    from signals.detector import detect_all_setups
    ov = dict(overrides or {})
    ov["directions"] = None
    with config_override(symbol, ov, params_override):
        if _settings().ASSETS[symbol].get("use_ml", True):
            raise ProtocolError(f"{symbol}: el protocolo solo evalúa activos con use_ml=False (raw_score).")
        setups = detect_all_setups(df_4h, df_daily, symbol)
    if setups is None or setups.empty:
        return pd.DataFrame(columns=["ts", "direction", "setup_type", "raw_score", "atr"])
    return setups.reset_index(drop=True)


def simulate_trades(df: pd.DataFrame, setups: pd.DataFrame, atr_stop: float,
                    atr_target: float, forward_bars: int) -> pd.DataFrame:
    """
    Simula cada setup UNA vez a precios medios con la convención de la Fase 1
    (idéntica a validate._simulate_setup): entrada a la apertura de i+1, SL/TP
    desde i+1 hasta i+forward_bars (si una vela toca ambos, cuenta el SL),
    salida por tiempo al cierre de i+forward_bars.

    Devuelve una fila por operación simulable con: ts, direction, setup_type,
    raw_score, pos, entry_pos, exit_pos, entry_mid, exit_mid, exit_kind
    ('sl'/'tp'/'time'), risk (atr_stop × ATR), complete (la vela i+forward_bars ya
    está en los datos, aunque la operación saliera antes por SL/TP), d_in (apertura de la vela de entrada) y d_out
    (mitad de la vela de salida), para el swap. Los R por escenario de costes se
    calculan con costs.net_returns.
    """
    cols = ["ts", "direction", "setup_type", "raw_score", "pos", "entry_pos", "exit_pos",
            "entry_mid", "exit_mid", "exit_kind", "risk", "complete", "d_in", "d_out"]
    if setups is None or setups.empty:
        return pd.DataFrame(columns=cols)
    idx = df.index
    n = len(df)
    O = df["open"].to_numpy(dtype=float)
    H = df["high"].to_numpy(dtype=float)
    L = df["low"].to_numpy(dtype=float)
    C = df["close"].to_numpy(dtype=float)
    bar = idx.to_series().diff().median() if n > 1 else pd.Timedelta(hours=4)
    rows = []
    for s in setups.itertuples(index=False):
        pos = int(idx.searchsorted(s.ts))
        atr = float(getattr(s, "atr", 0) or 0)
        entry_pos = pos + 1
        if pos >= n or entry_pos >= n or not atr > 0:
            continue
        raw_entry = O[entry_pos]
        if raw_entry <= 0:
            continue
        risk = atr_stop * atr
        if s.direction == "long":
            sl, tp = raw_entry - atr_stop * atr, raw_entry + atr_target * atr
        else:
            sl, tp = raw_entry + atr_stop * atr, raw_entry - atr_target * atr
        last_pos = min(pos + forward_bars, n - 1)
        exit_mid, exit_pos, kind = None, last_pos, "time"
        for j in range(entry_pos, last_pos + 1):
            if s.direction == "long":
                hit_sl, hit_tp = L[j] <= sl, H[j] >= tp
            else:
                hit_sl, hit_tp = H[j] >= sl, L[j] <= tp
            if hit_sl:                       # también si toca ambos: SL primero
                exit_mid, exit_pos, kind = sl, j, "sl"
                break
            if hit_tp:
                exit_mid, exit_pos, kind = tp, j, "tp"
                break
        if exit_mid is None:
            exit_mid = C[last_pos]
        # Horizonte completo = la vela i+forward_bars ya está en los datos, SEA CUAL SEA
        # la salida. No vale "salió pronto por SL/TP": aceptar las resueltas pronto y
        # excluir las abiertas es seleccionar por el resultado (el SL está más cerca que
        # el TP → sesgo hacia pérdidas). Se usa en la reserva y en el holdout OKX.
        complete = (pos + forward_bars) <= n - 1
        rows.append({"ts": pd.Timestamp(s.ts), "direction": s.direction,
                     "setup_type": s.setup_type, "raw_score": float(s.raw_score),
                     "pos": pos, "entry_pos": entry_pos, "exit_pos": exit_pos,
                     "entry_mid": float(raw_entry), "exit_mid": float(exit_mid),
                     "exit_kind": kind, "risk": float(risk), "complete": bool(complete),
                     "d_in": idx[entry_pos], "d_out": idx[exit_pos] + bar / 2})
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Rejilla de variantes, reserva y folds
# ---------------------------------------------------------------------------

def _thr_label(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def build_grid(symbol: str, candidate: dict | None = None) -> list[dict]:
    """
    Rejilla fija del activo: umbral actual + {−0,10, −0,05, 0, +0,05, +0,10} ×
    {ambas, solo long, solo short} = 15 variantes; el candidato se añade si no
    coincide con ninguna. Cada variante:
        {"id", "label", "ml_threshold", "directions" (tupla), "is_current",
         "is_candidate", "extra" (overrides del candidato), "params", "hash"}
    Los umbrales negativos se dejan tal cual (con raw_score >= 0 equivalen a "sin umbral").
    """
    cur = current_variant(symbol)
    fp = strategy_fingerprint(symbol)
    grid = []
    for off in THRESHOLD_OFFSETS:
        thr = round(cur["ml_threshold"] + off, 4)
        for dname, dirs in DIRECTION_SETS:
            params = variant_params(symbol, thr, dirs, fingerprint=fp)
            grid.append({"id": f"thr{thr:.2f}_{'-'.join(sorted(dirs))}",
                         "label": f"umbral {_thr_label(thr)} · {dname}",
                         "ml_threshold": thr, "directions": tuple(sorted(dirs)),
                         "is_current": thr == cur["ml_threshold"] and tuple(sorted(dirs)) == cur["directions"],
                         "is_candidate": False, "extra": {},
                         "params": params, "hash": variant_hash(params)})
    if candidate:
        cv = candidate_variant(symbol, candidate)
        same = next((v for v in grid if v["hash"] == cv["hash"]), None)
        if same is not None:
            same["is_candidate"] = True
            if candidate.get("label"):
                same["label"] += f" (candidato: {candidate['label']})"
        else:
            grid.append(cv)
    return grid


def load_candidate(path: str | Path) -> dict:
    """
    Lee un candidato JSON. Formato:
        {"symbol": "ETHUSDT", "label": "ETH umbral 0,45",
         "ml_threshold": 0.45,                    (opcional; por defecto el actual)
         "directions": ["long", "short"],         (opcional; por defecto las actuales)
         "overrides": {...},                      (opcional: otras claves de ASSETS)
         "params_override": {...}}                (opcional: parámetros de detectores)
    Lanza ProtocolError si falta el símbolo o no existe.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"No se pudo leer el candidato {path}: {exc}") from exc
    sym = data.get("symbol")
    if not sym or sym not in _settings().ASSETS:
        raise ProtocolError(f"El candidato {path} debe indicar un 'symbol' válido (tiene {sym!r}).")
    dirs = data.get("directions")
    if dirs is not None and (not dirs or set(dirs) - {"long", "short"}):
        raise ProtocolError(f"'directions' del candidato debe ser una lista con long/short (tiene {dirs!r}).")
    for k in ("overrides", "params_override"):
        if k in data and not isinstance(data[k], dict):
            raise ProtocolError(f"'{k}' del candidato debe ser un objeto JSON.")
    return data


def candidate_extra(candidate: dict) -> dict:
    """Cambios del candidato que NO son umbral/dirección (requieren re-detectar)."""
    extra = {}
    if candidate.get("overrides"):
        extra["overrides"] = candidate["overrides"]
    if candidate.get("params_override"):
        extra["params_override"] = candidate["params_override"]
    return extra


def candidate_variant(symbol: str, candidate: dict) -> dict:
    """Variante (mismo formato que build_grid) que representa al candidato."""
    cur = current_variant(symbol)
    thr = round(float(candidate.get("ml_threshold", cur["ml_threshold"])), 4)
    dirs = tuple(sorted(candidate.get("directions") or cur["directions"]))
    extra = candidate_extra(candidate)
    params = variant_params(symbol, thr, dirs, extra=extra)
    return {"id": "candidato", "label": f"candidato: {candidate.get('label') or f'umbral {_thr_label(thr)}'}",
            "ml_threshold": thr, "directions": dirs, "is_current": False, "is_candidate": True,
            "extra": extra, "params": params, "hash": variant_hash(params)}


def variant_mask(trades: pd.DataFrame, variant: dict) -> np.ndarray:
    """Operaciones aprobadas por la variante: raw_score >= umbral y dirección permitida
    (misma comparación que ml.predictor.score_setups con use_ml=False)."""
    if trades is None or len(trades) == 0:
        return np.zeros(0, dtype=bool)
    return ((trades["raw_score"].to_numpy(dtype=float) >= variant["ml_threshold"])
            & trades["direction"].isin(variant["directions"]).to_numpy())


def reserve_start() -> pd.Timestamp:
    return pd.Timestamp(_settings().PROTOCOL_RESERVE_START, tz="UTC")


def selection_start(symbol: str) -> pd.Timestamp:
    """Inicio de la muestra de selección = inicio del primer fold."""
    return pd.Timestamp(FOLDS[fold_kind(symbol)][0][1], tz="UTC")


def split_reserve(trades: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (selección, reserva). Selección = setups con selection_start <= ts < reserva
    (todos con su horizonte completo). Reserva = setups con ts >= reserva cuyo
    horizonte COMPLETO (i+forward_bars) ya está en los datos, haya salido o no
    antes por SL/TP (ni las abiertas ni las cerradas pronto de esas últimas velas
    cuentan: así no se selecciona por el resultado). La regla es por fecha del
    SETUP (especificación).
    """
    if trades is None or len(trades) == 0:
        return trades, trades
    ts = pd.to_datetime(trades["ts"], utc=True)
    res = reserve_start()
    sel = trades[(ts >= selection_start(symbol)) & (ts < res)]
    rsv = trades[(ts >= res) & trades["complete"].astype(bool)]
    return sel.reset_index(drop=True), rsv.reset_index(drop=True)


def assign_folds(ts, symbol: str) -> np.ndarray:
    """Nombre del fold temporal de cada fecha (None si cae fuera de los folds)."""
    t = pd.to_datetime(pd.Series(ts), utc=True)
    out = np.full(len(t), None, dtype=object)
    for name, a, b in FOLDS[fold_kind(symbol)]:
        m = ((t >= pd.Timestamp(a, tz="UTC")) & (t < pd.Timestamp(b, tz="UTC"))).to_numpy()
        out[m] = name
    return out


# ---------------------------------------------------------------------------
# Estadística básica (la robustez —CPCV, PBO, DSR— está en backtest/robustness.py)
# ---------------------------------------------------------------------------

def bootstrap_ci(x, level: float = CI_LEVEL, n_boot: int = BOOT_N,
                 seed: int = BOOT_SEED) -> tuple[float, float]:
    """IC de la media por bootstrap percentil, determinista (semilla fija).
    Con menos de 3 valores devuelve (nan, nan)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, 2_000_000 // len(x))
    for a in range(0, n_boot, chunk):
        b = min(n_boot, a + chunk)
        means[a:b] = x[rng.integers(0, len(x), size=(b - a, len(x)))].mean(axis=1)
    q = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [q, 100 - q])
    return (float(lo), float(hi))


def summarize(R) -> dict:
    """n, E[R], IC90, % ganadas, R total y profit factor de una lista de R."""
    x = np.asarray(R, dtype=float)
    if len(x) == 0:
        return {"n": 0, "er": None, "ci_lo": None, "ci_hi": None,
                "win_rate": None, "total_R": 0.0, "pf": None}
    lo, hi = bootstrap_ci(x)
    loss = abs(x[x <= 0].sum())
    return {"n": int(len(x)), "er": float(x.mean()),
            "ci_lo": None if np.isnan(lo) else lo, "ci_hi": None if np.isnan(hi) else hi,
            "win_rate": float((x > 0).mean()), "total_R": float(x.sum()),
            "pf": float(x[x > 0].sum() / loss) if loss > 0 else None}


def cost_table(trades: pd.DataFrame, symbol: str, with_folds: bool = True) -> list[dict]:
    """
    Tabla de E[R] por escenario de costes de un conjunto de operaciones (ya
    filtrado por la variante): una fila por escenario con summarize(), el
    desglose del coste medio en R y, si with_folds, E[R] y N por fold temporal.
    """
    folds = assign_folds(trades["ts"], symbol) if with_folds and len(trades) else None
    rows = []
    for sc in cost_scenarios(symbol):
        R = net_returns(trades, sc)
        row = {"scenario": sc["name"], "label": sc["label"], "decision": sc["decision"],
               **summarize(R), "cost_R": cost_breakdown_R(trades, sc)}
        if with_folds:
            row["folds"] = []
            for name, _, _ in FOLDS[fold_kind(symbol)]:
                r = R[folds == name] if folds is not None else np.zeros(0)
                row["folds"].append({"fold": name, "n": int(len(r)),
                                     "er": float(r.mean()) if len(r) else None})
        rows.append(row)
    return rows


def decision_row(table: list[dict]) -> dict | None:
    """Fila del escenario de decisión (XM swap 5 %) de una cost_table."""
    return next((r for r in table if r.get("decision")), None)


# ---------------------------------------------------------------------------
# Holdout OKX (solo --final)
# ---------------------------------------------------------------------------

def resample_daily_utc(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Diario reconstruido desde el 4H en días UTC (00:00–24:00), como el diario de
    Gate.io. El 1D nativo de OKX abre a las 16:00 UTC (00:00 de Hong Kong): por eso
    no se usa. Solo días con sus 6 velas 4H (se descartan el primero y el último
    si están incompletos).
    """
    if df_4h is None or df_4h.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    g = df_4h.resample("1D", label="left", closed="left")
    d = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    d = d[g["close"].count() == 6]
    return d.dropna()


def _frame_from_rows(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
    df.index.name = None
    return df.astype(float)


def _okx_from_table(symbol: str, engine=None) -> tuple[pd.DataFrame | None, str]:
    """Velas 4H de OKX del holdout guardadas en la tabla ohlcv (source='okx')."""
    from sqlalchemy import text
    from backfill_history import _ohlcv_columns
    from data.database import get_engine, init_db
    if engine is None:
        init_db()
        engine = get_engine()
    if "source" not in _ohlcv_columns(engine):
        return None, "la tabla ohlcv no tiene columna source (aún no se ejecutó backfill_history.py)"
    lo = int(pd.Timestamp(OKX_HOLDOUT_START, tz="UTC").timestamp())
    hi = int(pd.Timestamp(OKX_HOLDOUT_END, tz="UTC").timestamp())
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=:s AND timeframe='4h' "
            "AND source='okx' AND ts >= :lo AND ts < :hi ORDER BY ts"),
            {"s": symbol, "lo": lo, "hi": hi}).fetchall()
    if not rows:
        return None, "la tabla ohlcv no tiene velas 4H de OKX en 2019-12 → 2022-04"
    df = _frame_from_rows([tuple(r) for r in rows])
    first, last = df.index[0], df.index[-1]
    expected = (last - first) / pd.Timedelta(hours=4) + 1
    if (first > pd.Timestamp(OKX_TABLE_MIN_FIRST, tz="UTC")
            or last < pd.Timestamp(OKX_TABLE_MIN_LAST, tz="UTC")
            or len(df) < OKX_TABLE_MIN_COVERAGE * expected):
        return None, (f"la tabla ohlcv solo tiene {len(df)} velas OKX ({first:%Y-%m-%d} → {last:%Y-%m-%d}); "
                      "hacen falta 2019-12 → 2022-04 completos")
    return df, "tabla ohlcv (source='okx')"


def _okx_download(symbol: str, exchange=None) -> pd.DataFrame:
    """Descarga EN MEMORIA el 4H de OKX del holdout (no escribe en la BD)."""
    import backfill_history as bh
    ex = exchange or bh.make_exchange("okx")
    lo = int(pd.Timestamp(OKX_HOLDOUT_START, tz="UTC").timestamp())
    hi = int(pd.Timestamp(OKX_HOLDOUT_END, tz="UTC").timestamp())
    rows = []
    for page, _invalid in bh.iter_closed_pages(ex, bh.SYMBOLS[symbol], "4h", lo, hi,
                                               bh.PAGE_LIMIT["okx"], int(time.time())):
        rows.extend(page)
    if not rows:
        raise ProtocolError(f"OKX no devolvió velas 4H de {symbol} para 2019-12 → 2022-04.")
    return _frame_from_rows(rows)


def load_okx_holdout(symbol: str, download: bool = False, exchange=None, engine=None) -> dict:
    """
    Datos del holdout OKX de un activo cripto:
        {"df_4h", "df_daily", "origin", "error"}
    Primero la tabla ohlcv (source='okx') si tiene el periodo completo; si no y
    download=True (--okx), lo descarga en memoria con ccxt okx. Si no hay datos,
    df_4h es None y "error" explica qué hacer.
    """
    import backfill_history as bh
    if symbol not in bh.SYMBOLS:
        return {"df_4h": None, "df_daily": None, "origin": None,
                "error": f"{symbol} no es cripto: no tiene holdout OKX"}
    try:
        df, why = _okx_from_table(symbol, engine)
    except Exception as exc:  # noqa: BLE001 — la BD es opcional aquí
        df, why = None, f"no se pudo leer la tabla ohlcv: {exc}"
    origin = why
    if df is None:
        if not download:
            return {"df_4h": None, "df_daily": None, "origin": None,
                    "error": f"{why}. Usa --okx para descargarlo en memoria."}
        try:
            df = _okx_download(symbol, exchange)
            origin = "descarga OKX en memoria (--okx)"
        except Exception as exc:  # noqa: BLE001
            return {"df_4h": None, "df_daily": None, "origin": None,
                    "error": f"no se pudo descargar OKX: {exc}"}
    return {"df_4h": df, "df_daily": resample_daily_utc(df), "origin": origin, "error": None}


# ---------------------------------------------------------------------------
# Análisis de un activo
# ---------------------------------------------------------------------------

def _sim_params(symbol: str) -> tuple[float, float, int]:
    s = _settings()
    cfg = s.ASSETS[symbol]
    return (cfg.get("atr_stop_mult", s.RISK["atr_stop_mult"]),
            cfg.get("atr_target_mult", s.RISK["atr_target_mult"]),
            int(cfg.get("forward_bars", 24)))


def universe_trades(symbol: str, df_4h: pd.DataFrame, df_daily: pd.DataFrame = None,
                    extra: dict | None = None) -> pd.DataFrame:
    """Detecta (todas las direcciones) y simula una vez. extra = overrides del candidato."""
    extra = extra or {}
    ov, po = extra.get("overrides"), extra.get("params_override")
    setups = detect_universe(symbol, df_4h, df_daily, ov, po)
    with config_override(symbol, ov, po):
        st, tg, fwd = _sim_params(symbol)
    return simulate_trades(df_4h, setups, st, tg, fwd)


def analyze_symbol(symbol: str, df_4h: pd.DataFrame, df_daily: pd.DataFrame = None,
                   candidate: dict | None = None, final: bool = False,
                   okx: dict | None = None) -> dict:
    """
    Cifras del protocolo para un activo (sin precondición ni registro):
      "grid": variantes con su resumen en la SELECCIÓN (escenario de decisión);
      "baseline": cost_table de la configuración actual en la selección;
      "candidate": ídem del candidato (si lo hay);
      "reserve" / "okx": solo con final=True (tablas en la reserva y en OKX);
      "robustness": None (la rellena run_protocol con robustness_for, tras el registro).
    Claves privadas: "_trades", "_masks", "_R", "_dec", "_bounds", "_fwd" (ver
    docstring del módulo) y "_cand_trades"/"_cand_R" si el candidato necesitó
    re-detección (su máscara se calcula con variant_mask).
    """
    trades = universe_trades(symbol, df_4h, df_daily)
    sel, rsv = split_reserve(trades, symbol)
    grid = build_grid(symbol, candidate)
    scen = cost_scenarios(symbol)
    dec = next(sc for sc in scen if sc["decision"])
    R_sel = {sc["name"]: net_returns(sel, sc) for sc in scen}
    masks = {}

    # Candidato con cambios que no son umbral/dirección: universo propio.
    cand = next((v for v in grid if v["is_candidate"]), None)
    cand_sel = cand_rsv = None
    if cand is not None and cand["extra"]:
        cand_all = universe_trades(symbol, df_4h, df_daily, cand["extra"])
        cand_sel, cand_rsv = split_reserve(cand_all, symbol)

    for v in grid:
        base = cand_sel if (v["is_candidate"] and cand_sel is not None) else sel
        m = variant_mask(base, v)
        if base is sel:
            masks[v["id"]] = m
            r = R_sel[dec["name"]][m]
        else:
            r = net_returns(base[m], dec)
        v["selection"] = {"n": int(m.sum()), "er": float(r.mean()) if len(r) else None,
                          "eligible": int(m.sum()) >= MIN_TRADES_SELECT}

    cur = next(v for v in grid if v["is_current"])
    _, _, fwd = _sim_params(symbol)
    out = {"symbol": symbol, "fold_kind": fold_kind(symbol),
           "selection_period": [str(selection_start(symbol).date()), str(reserve_start().date())],
           "n_setups_selection": int(len(sel)), "n_setups_reserve": int(len(rsv)),
           "current": _public_variant(cur), "grid": [_public_variant(v) for v in grid],
           "baseline": cost_table(sel[variant_mask(sel, cur)], symbol),
           "candidate": None, "reserve": None, "okx": None, "robustness": None,
           "_trades": sel, "_masks": masks, "_R": R_sel, "_dec": dec["name"],
           # Velas [p0, p1) de la muestra de selección y horizonte (CPCV/PBO).
           "_bounds": _selection_bounds(df_4h, symbol), "_fwd": int(fwd)}
    if cand is not None:
        base = cand_sel if cand_sel is not None else sel
        out["candidate_variant"] = _public_variant(cand)
        out["candidate"] = cost_table(base[variant_mask(base, cand)], symbol)
        if cand_sel is not None:
            out["_cand_trades"] = cand_sel
            out["_cand_R"] = net_returns(cand_sel, dec)

    if final:
        res = {"current": cost_table(rsv[variant_mask(rsv, cur)], symbol, with_folds=False)}
        if cand is not None:
            base = cand_rsv if cand_rsv is not None else rsv
            res["candidate"] = cost_table(base[variant_mask(base, cand)], symbol, with_folds=False)
        out["reserve"] = res
        if okx is not None:
            out["okx"] = _analyze_okx(symbol, okx, cur, cand)
    return out


def _analyze_okx(symbol: str, okx: dict, cur: dict, cand: dict | None) -> dict:
    """Tablas en el holdout OKX (operaciones con horizonte completo)."""
    if okx.get("df_4h") is None:
        return {"error": okx.get("error") or "sin datos OKX"}
    d4, d1 = okx["df_4h"], okx["df_daily"]
    t = universe_trades(symbol, d4, d1)
    t = t[t["complete"].astype(bool) & (pd.to_datetime(t["ts"], utc=True)
                                        < pd.Timestamp(OKX_HOLDOUT_END, tz="UTC"))]
    out = {"origin": okx.get("origin"), "bars": int(len(d4)),
           "period": [str(d4.index[0]), str(d4.index[-1])],
           "current": cost_table(t[variant_mask(t, cur)], symbol, with_folds=False)}
    if cand is not None:
        tc = universe_trades(symbol, d4, d1, cand["extra"]) if cand["extra"] else t
        if cand["extra"]:
            tc = tc[tc["complete"].astype(bool)]
        out["candidate"] = cost_table(tc[variant_mask(tc, cand)], symbol, with_folds=False)
    return out


def _public_variant(v: dict) -> dict:
    return {k: (list(val) if isinstance(val, tuple) else val) for k, val in v.items()}


def _selection_bounds(df: pd.DataFrame, symbol: str) -> tuple[int, int]:
    """Posiciones [p0, p1) de las velas de la muestra de selección en df."""
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return (int(idx.searchsorted(selection_start(symbol))),
            int(idx.searchsorted(reserve_start())))


# ---------------------------------------------------------------------------
# Robustez (CPCV purgada con re-selección, PBO, DSR): backtest/robustness.py
# ---------------------------------------------------------------------------

def robustness_variants(res: dict) -> tuple[list[dict], int, int | None]:
    """
    Variantes de la rejilla (en su orden) como {"id", "label", "pos", "r"} con los R
    del escenario de decisión en la SELECCIÓN, más los índices de la configuración
    actual y del candidato. Si el candidato re-detectó, usa su universo propio.
    """
    sel, masks, R = res["_trades"], res["_masks"], res["_R"][res["_dec"]]
    pos_all = sel["pos"].to_numpy(dtype=int) if len(sel) else np.zeros(0, dtype=int)
    out, cur_i, cand_i = [], None, None
    for g in res["grid"]:
        if g["id"] in masks:
            m = masks[g["id"]]
            pos, r = pos_all[m], np.asarray(R)[m]
        else:                                   # candidato con universo propio
            ct = res["_cand_trades"]
            m = variant_mask(ct, g)
            pos, r = ct["pos"].to_numpy(dtype=int)[m], np.asarray(res["_cand_R"])[m]
        if g["is_current"]:
            cur_i = len(out)
        if g["is_candidate"]:
            cand_i = len(out)
        out.append({"id": g["id"], "label": g["label"], "pos": pos, "r": r})
    return out, cur_i, cand_i


def robustness_for(res: dict, n_trials: int) -> dict:
    """CPCV purgada con re-selección, PBO (S=16) y DSR de un activo analizado."""
    from backtest.robustness import compute_robustness
    variants, cur_i, cand_i = robustness_variants(res)
    p0, p1 = res["_bounds"]
    rob = compute_robustness(variants, cur_i, cand_i, p0, p1, res["_fwd"], n_trials,
                             min_trades=MIN_TRADES_SELECT)
    rob["labels"] = {v["id"]: v["label"] for v in variants}
    return rob


# ---------------------------------------------------------------------------
# Veredicto: criterios de APROBADO a)–e) (fijados el 2026-11-16; REGLAS_DE_DECISION.md)
# ---------------------------------------------------------------------------
CRIT_MIN_IMPROVEMENT = 0.10   # b) mejora mínima del E[R] (R) en la mediana de la CPCV y por fold
CRIT_MIN_FOLDS = 2            # b) folds (de 3) con mejora >= CRIT_MIN_IMPROVEMENT
CRIT_MAX_PBO = 0.20           # c) PBO de la rejilla con el candidato < 0,20
CRIT_MIN_DSR = 0.95           # d) DSR del candidato > 0,95
CRIT_MIN_FINAL_DELTA = 0.0    # e) candidato − actual en la reserva y en OKX >= 0
FOLD_SMALL_N = 10             # solo informe: folds con menos operaciones se marcan "(!)"

# Avisos de lectura de la línea base (sin candidato): las cifras de la config
# actual son DENTRO DE MUESTRA y los umbrales c)/d) son para candidatos.
BASELINE_IN_SAMPLE = ("Cifras DENTRO DE MUESTRA: la configuración actual se eligió mirando estos mismos "
                      "datos; no son una expectativa de lo que va a dar (ver REGLAS_DE_DECISION.md).")


def _baseline_notes(results: list[dict]) -> list[str]:
    """Avisos de lectura de la línea base (sin candidato)."""
    fails = []
    for r in results:
        rob = r.get("robustness") or {}
        if not rob or rob.get("error"):
            continue
        pbo = (rob.get("pbo") or {}).get("pbo")
        dsr = (rob.get("dsr_current") or {}).get("dsr")
        bad = []
        if pbo is None or not pbo < CRIT_MAX_PBO:
            bad.append(f"PBO {_f(pbo, 2, False)}")
        if dsr is None or not dsr > CRIT_MIN_DSR:
            bad.append(f"DSR {_f(dsr, 2, False)}")
        if bad:
            fails.append(f"{r['symbol']}: {', '.join(bad)}")
    notes = [BASELINE_IN_SAMPLE]
    if fails:
        notes.append(f"Los umbrales de c) PBO < {CRIT_MAX_PBO:.2f} y d) DSR > {CRIT_MIN_DSR:.2f} se aplican a "
                     f"CANDIDATOS. La configuración actual tampoco los pasa ({'; '.join(fails)}): es lo "
                     "esperable con 30–55 operaciones por activo y cientos de pruebas registradas, y no "
                     "cambia nada del vivo.")
    return notes


def _dec_er(table) -> float | None:
    row = decision_row(table) if table else None
    return None if row is None else row.get("er")


def _crit(ok, text: str) -> dict:
    return {"ok": ok, "text": text}


def criteria_for_symbol(res: dict, final: bool) -> dict:
    """Criterios a)–e) de un activo con candidato: {letra: {"ok": True/False/None, "text"}}.
    ok=None = pendiente (solo e) sin --final)."""
    c = {}
    lv = res.get("candidate_live_check") or {}
    c["a"] = _crit(bool(lv.get("match")),
                   f"vivo = validador con la config del candidato: "
                   f"{'OK' if lv.get('match') else 'FALLA'} ({lv.get('approved', '—')} aprobados, "
                   f"{lv.get('emitted', '—')} emitidos)" if lv else
                   "vivo = validador con la config del candidato: no comprobado")
    rob = res.get("robustness") or {}
    if rob.get("error") or not rob:
        why = rob.get("error", "no calculada")
        for k in ("b", "c", "d"):
            c[k] = _crit(False, f"robustez no evaluable ({why})")
    else:
        d = (rob["cpcv"].get("delta_cand_cur") or {})
        med = d.get("median")
        cur_f = (decision_row(res["baseline"]) or {}).get("folds") or []
        cand_f = (decision_row(res["candidate"]) or {}).get("folds") or []
        deltas = []
        for fc, fk in zip(cur_f, cand_f):
            deltas.append(None if fc["er"] is None or fk["er"] is None else fk["er"] - fc["er"])
        good = sum(1 for x in deltas if x is not None and x >= CRIT_MIN_IMPROVEMENT - 1e-12)
        ok_b = med is not None and med >= CRIT_MIN_IMPROVEMENT - 1e-12 and good >= CRIT_MIN_FOLDS
        c["b"] = _crit(ok_b, f"mejora del E[R] (XM swap 5 %) frente a la actual: mediana CPCV "
                       f"{_f(med, 2)}R (pide ≥ +{CRIT_MIN_IMPROVEMENT:.2f}R); folds "
                       + " | ".join(_f(x, 2) for x in deltas)
                       + f" → {good} de {len(deltas)} con ≥ +{CRIT_MIN_IMPROVEMENT:.2f}R "
                       f"(pide {CRIT_MIN_FOLDS})")
        pbo = rob["pbo"].get("pbo")
        c["c"] = _crit(pbo is not None and pbo < CRIT_MAX_PBO,
                       f"PBO de la rejilla con el candidato = {_f(pbo, 2, False)} (pide < {CRIT_MAX_PBO:.2f})")
        dsr = (rob.get("dsr_candidate") or {}).get("dsr")
        c["d"] = _crit(dsr is not None and dsr > CRIT_MIN_DSR,
                       f"DSR del candidato = {_f(dsr, 2, False)} (pide > {CRIT_MIN_DSR:.2f}; "
                       f"{rob['n_trials']} pruebas registradas)")
    if not final:
        c["e"] = _crit(None, "reserva y OKX: pendiente (solo con --final, una única vez)")
    else:
        parts, ok_e = [], True
        rsv = res.get("reserve") or {}
        dr = None
        if rsv.get("current") is not None and rsv.get("candidate") is not None:
            a, b = _dec_er(rsv["candidate"]), _dec_er(rsv["current"])
            dr = None if a is None or b is None else a - b
        parts.append(f"reserva {_f(dr, 2)}R" if dr is not None else "reserva no evaluable")
        ok_e &= dr is not None and dr >= CRIT_MIN_FINAL_DELTA
        if res.get("fold_kind") == "crypto":
            ok = res.get("okx") or {}
            do = None
            if not ok.get("error") and ok.get("current") is not None and ok.get("candidate") is not None:
                a, b = _dec_er(ok["candidate"]), _dec_er(ok["current"])
                do = None if a is None or b is None else a - b
            parts.append(f"OKX {_f(do, 2)}R" if do is not None else
                         f"OKX no evaluable ({ok.get('error') or 'sin datos'})")
            ok_e &= do is not None and do >= CRIT_MIN_FINAL_DELTA
        c["e"] = _crit(bool(ok_e), "candidato − actual (XM swap 5 %): " + "; ".join(parts)
                       + " (pide que no sea negativa)")
    return c


def evaluate_criteria(results: list[dict], candidate: dict | None, final: bool = False) -> dict:
    """
    Veredicto del protocolo: {"status", "failed", "criteria", "text"}.
      - Sin candidato → "baseline": las cifras quedan como línea base.
      - Con candidato → criterios a)–e) (ver criteria_for_symbol y REGLAS_DE_DECISION.md):
          "rejected"      si falla cualquiera ("NO APROBADO: falla b, d");
          "pending_final" si pasa a)–d) pero falta --final (aún NO se adopta);
          "approved"      solo si pasa a)–e).
    """
    if not candidate:
        return {"status": "baseline", "failed": [], "criteria": None,
                "text": "LÍNEA BASE (sin candidato): estas cifras quedan como referencia; no se adopta nada.",
                "notes": _baseline_notes(results)}
    # El candidato es de un solo activo; si hubiera varios, cada criterio se combina
    # con "Y": falla si falla en alguno, pendiente si está pendiente en alguno.
    per = [criteria_for_symbol(r, final) for r in results]
    crit = {}
    for k in "abcde":
        oks = [p[k]["ok"] for p in per]
        ok = False if (False in oks or not oks) else (None if None in oks else True)
        crit[k] = {"ok": ok, "text": " · ".join(p[k]["text"] for p in per)}
    failed = [k for k in "abcde" if crit.get(k, {}).get("ok") is False]
    if failed:
        return {"status": "rejected", "failed": failed, "criteria": crit,
                "text": f"NO APROBADO: falla {', '.join(failed)}. El candidato NO se adopta."}
    if crit.get("e", {}).get("ok") is None:
        return {"status": "pending_final", "failed": [], "criteria": crit,
                "text": "PASA a)–d); FALTA e) (examen final con --final, una sola vez). Todavía NO se adopta."}
    return {"status": "approved", "failed": [], "criteria": crit,
            "text": "APROBADO: cumple a)–e). Adoptarlo es decisión tuya (ver REGLAS_DE_DECISION.md)."}


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def _f(x, nd: int = 3, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _table_lines(table: list[dict], indent: str = "  ", folds: bool = True) -> list[str]:
    lines = []
    head = f"{indent}{'Escenario de costes':56} {'N':>4} {'E[R]':>7}  {'IC90':^17} {'Ganadas':>7}"
    if folds and table and table[0].get("folds"):
        head += "   " + " | ".join(f"{f['fold']}" for f in table[0]["folds"])
    lines.append(head)
    small = False
    for r in table:
        lab = r["label"] + ("  ← decisión" if r["decision"] else "")
        ci = f"[{_f(r['ci_lo'], 2)}, {_f(r['ci_hi'], 2)}]" if r["n"] >= 3 else "(N<3)"
        wr = f"{r['win_rate'] * 100:5.1f} %" if r["win_rate"] is not None else "—"
        line = f"{indent}{lab:56} {r['n']:>4} {_f(r['er']):>7}  {ci:^17} {wr:>7}"
        if folds and r.get("folds"):
            parts = []
            for f in r["folds"]:
                flag = 0 < f["n"] < FOLD_SMALL_N
                small |= flag
                parts.append(f"{_f(f['er'], 2)} (N={f['n']}{' (!)' if flag else ''})")
            line += "   " + " | ".join(parts)
        lines.append(line)
    if small:
        lines.append(f"{indent}(!) = menos de {FOLD_SMALL_N} operaciones en ese tramo: esa cifra es casi ruido.")
    return lines


def _robustness_lines(rob: dict | None) -> list[str]:
    """Resumen de la robustez en español sencillo (bloque de cada activo)."""
    if not rob:
        return []
    if rob.get("error"):
        return [f"  Robustez: no se pudo calcular ({rob['error']})."]
    cp, pb, dc = rob["cpcv"], rob["pbo"], rob["dsr_current"]
    pr = cp["procedure"]
    L = ["  Robustez (solo datos anteriores a la reserva):"]
    if pr["n"]:
        sin_op = cp["n_comb"] - pr["n"]
        L.append(f"    · CPCV ({cp['n_comb']} pruebas): si se elige \"la mejor variante de la rejilla\" con 4/6 "
                 f"de la historia y se prueba en los otros 2/6, da de mediana {_f(pr['median'], 2)}R por "
                 f"operación (en 8 de cada 10 pruebas entre {_f(pr['p10'], 2)} y {_f(pr['p90'], 2)}; "
                 f"peor {_f(pr['min'], 2)}, mejor {_f(pr['max'], 2)}; por debajo de 0 en "
                 f"{round(pr['frac_neg'] * pr['n'])} de {pr['n']}"
                 + (f"; {sin_op} prueba(s) sin operaciones, no cuenta(n)" if sin_op > 0 else "") + ").")
    cf = cp["current_fixed"]
    L.append(f"      La configuración actual habría sido la elegida {cp['times_current']} de {cp['n_comb']} veces; "
             f"fija, en esas mismas pruebas, da de mediana {_f(cf['median'], 2)}R.")
    if cp.get("delta_cand_cur"):
        dd = cp["delta_cand_cur"]
        L.append(f"      Candidato − actual en las pruebas: mediana {_f(dd['median'], 2)}R "
                 f"(el candidato habría sido elegido {cp['times_candidate']} de {cp['n_comb']} veces).")
    L.append(f"    · PBO = {_f(pb['pbo'], 2, False)}: probabilidad de que \"la mejor\" de una mitad de la historia "
             f"quede en la mitad de abajo en la otra mitad (0,5 = como elegir al azar; el examen pide "
             f"< {CRIT_MAX_PBO:.2f}).")
    L.append(f"    · DSR = {_f(dc['dsr'], 2, False)} (actual): probabilidad de que la ventaja por operación sea "
             f"real descontando las {rob['n_trials']} pruebas hechas (el examen pide > {CRIT_MIN_DSR:.2f}). "
             f"Sin descontarlas: {_f(dc['psr0'], 2, False)}.")
    if rob.get("dsr_candidate"):
        L.append(f"      DSR del candidato = {_f(rob['dsr_candidate']['dsr'], 2, False)} "
                 f"(sin descontar: {_f(rob['dsr_candidate']['psr0'], 2, False)}).")
    return L


def _robustness_detail(rob: dict | None) -> list[str]:
    """Detalle técnico de CPCV, PBO y DSR."""
    if not rob or rob.get("error"):
        return [f"  CPCV / PBO / DSR: {'error: ' + rob['error'] if rob else 'no calculados'}"]
    cp, pb = rob["cpcv"], rob["pbo"]
    lab = rob.get("labels", {})
    pr = cp["procedure"]
    L = [f"  CPCV: {cp['n_groups']} grupos de velas, {cp['k_test']} de prueba ({cp['n_comb']} combinaciones), "
         f"purga {cp['purge_bars']} velas a cada lado y embargo {cp['embargo_bars']} velas; elegida = máx E[R] "
         f"de entrenamiento con N ≥ {MIN_TRADES_SELECT} (si ninguna, la actual).",
         f"    Procedimiento: media {_f(pr['mean'])}  mediana {_f(pr['median'])}  mín {_f(pr['min'])}  "
         f"máx {_f(pr['max'])}  agregado ΣR/N {_f(pr.get('pooled_er'))}",
         "    Elegidas: " + "; ".join(f"{lab.get(k, k)} ×{n}" for k, n in cp["chosen_counts"].items())]
    L.append("    Por combinación (grupos de prueba: elegida → E[R] prueba (N) | actual (N)"
             + (" | candidato (N)" if cp.get("delta_cand_cur") else "") + "):")
    for c in cp["combos"]:
        s = (f"      {c['test_groups']}: {lab.get(c['chosen'], c['chosen'])[:30]:30} → {_f(c['er_test'], 2)} "
             f"({c['n_test']}) | {_f(c['cur_er_test'], 2)} ({c['cur_n_test']})")
        if "cand_er_test" in c:
            s += f" | {_f(c['cand_er_test'], 2)} ({c['cand_n_test']})"
        L.append(s)
    L.append(f"  PBO (CSCV): S={pb['S']} bloques, {pb['n_comb']} particiones ({pb['n_valid']} válidas), "
             f"{pb['n_variants']} variantes; λ mediana {_f(pb['lambda_median'], 2)}; E[R] de la mejor: dentro "
             f"{_f(pb['is_best_er_median'], 2)} → fuera {_f(pb['oos_best_er_median'], 2)} (medianas); "
             f"P(E[R] fuera < 0) = {_f(pb['p_oos_loss'], 2, False)}.")
    for name, d in (("actual", rob["dsr_current"]), ("candidato", rob.get("dsr_candidate"))):
        if d:
            L.append(f"  DSR {name}: N={d['n']} SR/operación {_f(d['sr'])}  asimetría {_f(d['skew'], 2)}  "
                     f"curtosis {_f(d['kurt'], 2, False)}  pruebas {d['n_trials']}  sd(SR rejilla) "
                     f"{_f(d['sd_trials_sr'], 3, False)} ({rob['grid_sr_n']} variantes con N ≥ {MIN_TRADES_SELECT})  "
                     f"E[máx Z] {_f(d['e_max_z'], 2, False)}  SR0 {_f(d['sr0'], 3, False)}  "
                     f"PSR(0) {_f(d['psr0'], 3, False)}  DSR {_f(d['dsr'], 3, False)}")
    return L


def _cost_lines(table: list[dict] | None, indent: str = "      ") -> list[str]:
    """Coste medio por operación del escenario de decisión (XM swap 5 %) de una tabla."""
    dec = decision_row(table) if table else None
    c = (dec or {}).get("cost_R") or {}
    if not dec or not dec.get("n") or not c:
        return []
    return [f"{indent}coste medio por operación (XM swap 5 %): spread {_f(c.get('spread'), 3, False)}R, "
            f"swap {_f(c.get('swap'), 3, False)}R"]


def _reserve_note_lines(prior: dict) -> list[str]:
    """Nota de independencia de la reserva para la configuración ACTUAL (y el
    candidato), según los accesos --final que ya constaban en el registro."""
    L = ["    Nota: la configuración actual se eligió en sep-2026 mirando datos hasta entonces; "
         "para ella solo es independiente el tramo posterior al 18-sep-2026"
         + ("." if not prior.get("current") else ",")]
    if prior.get("current"):
        L.append(f"    y además YA VIO esta reserva en {len(prior['current'])} --final anterior(es) "
                 f"(el primero {prior['current'][0]}): para ella ningún tramo de la reserva es ya independiente.")
    if prior.get("candidate") and prior.get("candidate") is not prior.get("current"):
        L.append(f"    El candidato también YA VIO esta reserva (el primero {prior['candidate'][0]}).")
    return L


def format_report(report: dict) -> str:
    """Texto del informe en español: veredicto, tablas por escenario y detalle técnico."""
    L = []
    L.append("=" * 100)
    L.append(f"PUERTA DE VALIDACIÓN (validate --protocol) — {report['run']}")
    L.append("=" * 100)
    v = report.get("verdict")
    if v:
        L.append(f"VEREDICTO: {v['text']}")
        for k, c in (v.get("criteria") or {}).items():
            st = {True: "OK", False: "FALLA", None: "PENDIENTE"}[c["ok"]]
            L.append(f"  {k}) [{st:^9}] {c['text']}")
        for n in v.get("notes") or []:
            L.append(f"  · {n}")
    else:
        L.append("SIN VEREDICTO: " + (report.get("registry_error") or "no se pudo completar el protocolo."))
    for w in report.get("warnings", []):
        L.append("")
        L.append("!" * 100)
        L.append(f"!!  {w}")
        L.append("!" * 100)
    reg = report.get("registry", {})
    if reg:
        L.append(f"Registro de pruebas: {reg['total_trials']} pruebas en total "
                 f"({reg['seed']} históricas + {reg['variants']} variantes registradas; "
                 f"{reg['new']} nuevas en esta ejecución). Archivo: {reg['path']}")
        if reg.get("new"):
            L.append("  Recuerda guardar el registro en git (en Render el disco se borra en cada despliegue).")
    L.append(f"Reserva intocable: setups desde {report['reserve_start']} (solo --final la mira).")
    for res in report["symbols"]:
        cur = res["current"]
        L.append("")
        L.append("-" * 100)
        L.append(f"[{res['symbol']}] configuración actual: {cur['label']}")
        lv = res.get("live_check")
        if lv:
            L.append(f"  vivo = validador: OK ({lv['approved']} aprobados = {lv['emitted']} emitidos)")
        L.append(f"  Selección: setups {res['selection_period'][0]} → {res['selection_period'][1]} "
                 f"(sin la reserva). Operaciones aprobadas: {decision_row(res['baseline'])['n']}")
        L.extend(_table_lines(res["baseline"]))
        if res.get("candidate"):
            L.append(f"  {res['candidate_variant']['label']}:")
            L.extend(_table_lines(res["candidate"]))
        L.extend(_robustness_lines(res.get("robustness")))
        if res.get("reserve"):
            L.append(f"  RESERVA (setups ≥ {report['reserve_start']}, solo operaciones cuyo horizonte "
                     "completo ya cabe en los datos):")
            L.extend(_reserve_note_lines(res.get("prior_final_access") or {}))
            L.append("    Actual:")
            L.extend(_table_lines(res["reserve"]["current"], "    ", folds=False))
            L.extend(_cost_lines(res["reserve"]["current"]))
            if "candidate" in res["reserve"]:
                L.append("    Candidato:")
                L.extend(_table_lines(res["reserve"]["candidate"], "    ", folds=False))
                L.extend(_cost_lines(res["reserve"]["candidate"]))
        if res.get("okx"):
            ok = res["okx"]
            if ok.get("error"):
                L.append(f"  HOLDOUT OKX: no disponible — {ok['error']}")
            else:
                L.append(f"  HOLDOUT OKX ({ok['origin']}; {ok['bars']} velas 4H, "
                         f"{ok['period'][0][:10]} → {ok['period'][1][:10]}; diario reconstruido en días UTC; "
                         "solo operaciones cuyo horizonte completo cabe en esos datos):")
                L.append("    Actual:")
                L.extend(_table_lines(ok["current"], "    ", folds=False))
                L.extend(_cost_lines(ok["current"]))
                if "candidate" in ok:
                    L.append("    Candidato:")
                    L.extend(_table_lines(ok["candidate"], "    ", folds=False))
                    L.extend(_cost_lines(ok["candidate"]))
                L.append("    Ojo: el spread de XM va en USD fijos (los de hoy); con el precio bajo de 2020–2021 "
                         "pesa mucho más en R que hoy (supuesto conservador). Compara con la fila del validador.")
    L.append("")
    L.append("-" * 100)
    L.append("DETALLE TÉCNICO")
    for res in report["symbols"]:
        dec = decision_row(res["baseline"])
        c = dec.get("cost_R", {}) if dec else {}
        L.append(f"[{res['symbol']}] coste medio por operación (XM swap 5 %): spread {_f(c.get('spread'), 3, False)}R, "
                 f"swap {_f(c.get('swap'), 3, False)}R ({_f(c.get('nights_mean'), 1, False)} noches ponderadas de media)")
        L.append("  Rejilla (selección, E[R] XM swap 5 %; * = actual, N<"
                 f"{MIN_TRADES_SELECT} no elegible):")
        for g in res["grid"]:
            mark = "*" if g["is_current"] else ("C" if g["is_candidate"] else " ")
            s = g["selection"]
            L.append(f"   {mark} {g['label']:42} N={s['n']:>4}  E[R]={_f(s['er'])}"
                     + ("" if s["eligible"] else "  (no elegible)"))
        L.extend(_robustness_detail(res.get("robustness")))
    L.append("Costes: spread de XM medio a la entrada y medio a la salida (1 por operación), en unidades de "
             "precio FIJAS (las de hoy) en toda la historia: en años de precio bajo pesa mucho más en R "
             "(conservador); swap siempre en contra por noche (cripto 7/semana; oro y SPX 5 con triple el "
             "miércoles), rollover a las 00:00 UTC.")
    L.append("=" * 100)
    return "\n".join(L)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (set, tuple, np.ndarray)):
        return list(o)
    return str(o)


def public_report(report: dict) -> dict:
    """Copia del informe sin las claves privadas (las que empiezan por '_')."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if not str(k).startswith("_")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return strip(report)


# ---------------------------------------------------------------------------
# Orquestación (CLI)
# ---------------------------------------------------------------------------

def _default_live_check(symbol, df_4h, df_daily):
    from backtest.live_emulation import compare_with_validator
    return compare_with_validator(symbol, df_4h, df_daily)


def _default_load_data(symbol):
    from main import _load_data
    d4, d1, _ = _load_data(symbol)
    return d4, d1


def _okx_shown(res: dict) -> bool:
    """True si el informe enseña cifras del holdout OKX de este activo (sin error)."""
    ok = res.get("okx")
    return ok is not None and not ok.get("error")


def _final_access_lines(res: dict, entries: list[dict], run_id: str, warnings: list[str]) -> list[dict]:
    """
    Líneas final_access de un activo en un --final: una por CADA configuración cuyas
    cifras de la reserva (y de OKX) enseña el informe, es decir, el candidato (si lo
    hay; role="candidate") y SIEMPRE la configuración actual (role="reference" si va
    junto a un candidato, "current" si se evalúa sola). Si el candidato coincide con
    la actual (mismo hash) se escribe una sola línea.

    Antes de añadirlas comprueba en el registro si alguna de ellas YA vio la
    reserva: añade el aviso en grande a `warnings` y guarda en
    res["prior_final_access"] ({"current"/"candidate": lista de "run" anteriores})
    lo que necesita el informe para la nota de independencia de la reserva.
    """
    cand = next((g for g in res["grid"] if g["is_candidate"]), None)
    cur = res["current"]
    shown = []
    if cand is not None:
        shown.append((cand, "candidate"))
    if cand is None or cand["hash"] != cur["hash"]:
        shown.append((cur, "reference" if cand is not None else "current"))
    okx = _okx_shown(res)
    out, prior = [], {}
    for v, role in shown:
        prev = final_accesses(entries, v["hash"])
        if prev:
            runs = [e.get("run") for e in prev]
            if v["hash"] == cur["hash"]:
                prior["current"] = runs
            if role == "candidate":
                prior["candidate"] = runs
            quien = "ESTE CANDIDATO" if role == "candidate" else "LA CONFIGURACIÓN ACTUAL"
            vio = "LA RESERVA" + (" Y OKX" if any(e.get("okx") for e in prev) else "")
            warnings.append(
                f"{quien} YA VIO {vio} ({res['symbol']}: {v['label']}; {len(prev)} acceso(s) previo(s), "
                f"el primero {prev[0].get('run')}): el resultado ya no es independiente.")
        out.append(final_access_entry(v, run_id, okx=okx, role=role))
    res["prior_final_access"] = prior
    return out


def run_protocol(symbol: str | None = None, candidate_path: str | None = None,
                 final: bool = False, okx: bool = False, out: str | None = None,
                 registry: str | Path | None = None, load_data=None, live_check=None,
                 okx_loader=None, echo=print) -> tuple[int, dict]:
    """
    Ejecuta el protocolo completo y devuelve (código de salida, informe).
      0 = informe completo (con veredicto o línea base);
      1 = detenido (precondición vivo = validador, datos o candidato inválido);
      2 = el registro de pruebas no se pudo leer/escribir → SIN veredicto.
    load_data(symbol) → (df_4h, df_daily); live_check(symbol, df_4h, df_daily) →
    dict de compare_with_validator; okx_loader(symbol, download) → dict de
    load_okx_holdout. Son inyectables para las pruebas.
    """
    load_data = load_data or _default_load_data
    live_check = live_check or _default_live_check
    okx_loader = okx_loader or (lambda s, d: load_okx_holdout(s, download=d))
    path = Path(registry) if registry else registry_path()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {"run": run_id, "reserve_start": str(reserve_start().date()),
              "final": bool(final), "symbols": [], "warnings": [], "verdict": None}

    try:
        entries = ensure_registry(path)
    except RegistryError as exc:
        echo(f"ERROR del registro de pruebas: {exc}\nSIN VEREDICTO: sin registro no se evalúa nada.")
        report["registry_error"] = str(exc)
        return 2, report

    try:
        candidate = load_candidate(candidate_path) if candidate_path else None
    except ProtocolError as exc:
        echo(f"ERROR: {exc}")
        return 1, report
    s = _settings()
    if candidate:
        if symbol and symbol != candidate["symbol"]:
            echo(f"ERROR: el candidato es de {candidate['symbol']} y se pidió --symbol {symbol}.")
            return 1, report
        symbols = [candidate["symbol"]]
    else:
        symbols = [symbol] if symbol else s.active_assets()

    for sym in symbols:
        d4, d1 = load_data(sym)
        if d4 is None or len(d4) == 0:
            echo(f"ERROR: sin datos para {sym}; el protocolo se detiene.")
            return 1, report
        lv = live_check(sym, d4, d1)
        if not lv.get("match"):
            echo(f"\n[{sym}] PROTOCOLO DETENIDO: el vivo NO emite lo mismo que valida el validador con la "
                 "configuración actual (vivo = validador falla).\n  Sin esa igualdad, las cifras del "
                 "backtest no describen lo que recibes por Telegram. Revisa `python main.py check-live "
                 f"--symbol {sym}`.")
            try:
                from backtest.live_emulation import print_comparison
                print_comparison(lv)
            except Exception:  # noqa: BLE001
                pass
            report["stopped"] = {"symbol": sym, "reason": "vivo != validador"}
            return 1, report
        okx_data = okx_loader(sym, okx) if (final and fold_kind(sym) == "crypto") else None
        try:
            res = analyze_symbol(sym, d4, d1, candidate=candidate if candidate else None,
                                 final=final, okx=okx_data)
        except ProtocolError as exc:
            echo(f"ERROR: {exc}")
            return 1, report
        res["live_check"] = {"approved": lv.get("approved"), "emitted": lv.get("emitted")}
        if candidate:
            # Criterio a): vivo = validador también con la configuración del candidato.
            cv = res["candidate_variant"]
            ov = dict(candidate.get("overrides") or {})
            ov.update({"ml_threshold": cv["ml_threshold"], "directions": list(cv["directions"])})
            try:
                with config_override(sym, ov, candidate.get("params_override")):
                    lvc = live_check(sym, d4, d1)
            except Exception as exc:  # noqa: BLE001 — sin comprobación, a) falla
                lvc = {"match": False, "error": str(exc)}
            res["candidate_live_check"] = {k: lvc.get(k) for k in
                                           ("match", "approved", "emitted", "repeated", "error")}
        report["symbols"].append(res)
        del d4, d1

    # Registro: variantes nuevas y accesos a la reserva, ANTES de cualquier veredicto.
    variants = [g for r in report["symbols"] for g in r["grid"]]
    new = new_variant_entries(entries, variants, run_id)
    finals = []
    if final:
        for r in report["symbols"]:
            finals.extend(_final_access_lines(r, entries, run_id, report["warnings"]))
    try:
        _append_lines(path, new + finals)
        entries = entries + new + finals
    except RegistryError as exc:
        report["registry_error"] = f"no se pudo escribir el registro de pruebas ({exc})"
        for r in report["symbols"]:          # sin registro no se enseña la reserva
            r["reserve"] = None
            r["okx"] = None
        echo(format_report(report))
        echo(f"\nERROR: {report['registry_error']}. SIN VEREDICTO.")
        return 2, report

    seed = sum(int(e.get("historical_trials", 0)) for e in entries if e.get("kind") == "seed")
    report["registry"] = {"path": str(path), "seed": seed,
                          "variants": len(registered_hashes(entries)),
                          "total_trials": total_trials(entries), "new": len(new),
                          "final_accesses_logged": len(finals)}
    # Robustez con el número de pruebas YA registrado (incluye las de esta ejecución).
    for r in report["symbols"]:
        try:
            r["robustness"] = robustness_for(r, report["registry"]["total_trials"])
        except Exception as exc:  # noqa: BLE001 — sin robustez, b)–d) no se aprueban
            r["robustness"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["verdict"] = evaluate_criteria(report["symbols"], candidate, final=final)
    echo(format_report(report))
    if out:
        try:
            Path(out).write_text(json.dumps(public_report(report), ensure_ascii=False, indent=2,
                                            default=_json_default), encoding="utf-8")
            echo(f"\nJSON guardado en {out}")
        except OSError as exc:
            echo(f"\nAVISO: no se pudo guardar el JSON en {out}: {exc}")
    return 0, report
