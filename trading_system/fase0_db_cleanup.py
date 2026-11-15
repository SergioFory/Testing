"""
Limpieza de la tabla `signals` — Fase 0 (2026-11). Se ejecuta UNA vez.

Qué hace:
  1. Copia de seguridad: CREATE TABLE signals_backup_fase0 AS SELECT * FROM signals
     (dentro de la misma base de datos; si ya existe, no se toca).
  2. Migra columnas nuevas (setup_ts, mode, pnl_r, exit_price, quality_flag...).
  3. Marca en `quality_flag` las alertas ANTIGUAS (sin setup_ts) que NO debieron
     enviarse (no se borra nada; las señales nuevas de la Fase 0 no se tocan):
       - 'stale_reemission' : su entrada, SL y TP son idénticos a los de una alerta
                              ANTERIOR del mismo activo y dirección (setup viejo
                              re-enviado, F01).
       - 'intrabar_repeat'  : llega menos de 4 h (cripto) / 24 h (diarios) después
                              de otra alerta del mismo activo y dirección (vela aún
                              abierta, F03).
     Las estadísticas del dashboard excluyen las filas marcadas.
  4. Re-resuelve las señales con la lógica nueva (velas cerradas, salida por
     tiempo con PnL, misma fuente que el vivo: Gate.io / GC=F / ^GSPC):
       - recalcula resultado, pnl_pct, exit_price, pnl_r y pnl_usd (= tamaño × %);
       - las 'expired' sin PnL (caducadas antes de tiempo, F04) pasan a tener su
         salida por tiempo real;
       - se respetan los resultados manuales: 'be' y los 'win'/'loss' puestos a
         mano desde el dashboard (sin pnl_pct).
  5. Borra de la tabla `ohlcv` las velas que se desvían más de un 2 % de la fuente
     actual (velas guardadas desde GLD×10 / SPY×10 o velas a medio formar).

Uso (en el Shell de Render, desde la raíz del repo):
    python trading_system/fase0_db_cleanup.py            # SOLO muestra lo que haría
    python trading_system/fase0_db_cleanup.py --apply    # aplica los cambios

Funciona en PostgreSQL (producción) y SQLite (desarrollo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from loguru import logger
from sqlalchemy import text

logger.remove()
logger.add(sys.stderr, level="WARNING")

from config.settings import ASSETS                      # noqa: E402
from data.database import get_engine, init_db            # noqa: E402

BACKUP_TABLE = "signals_backup_fase0"
REPEAT_WINDOW_H = {"yfinance": 24}                       # resto (cripto 4H): 4 h
OHLCV_MAX_DEV = 0.02


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _ts(value) -> pd.Timestamp:
    if isinstance(value, (int, float)):
        return pd.Timestamp(int(value), unit="s", tz="UTC")
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tz is None else t


def _table_exists(conn, name: str, dialect: str) -> bool:
    if dialect == "sqlite":
        q = "SELECT name FROM sqlite_master WHERE type='table' AND name=:n"
    else:
        q = "SELECT table_name FROM information_schema.tables WHERE table_name=:n"
    return conn.execute(text(q), {"n": name}).fetchone() is not None


def load_all_signals(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql_query(text("SELECT * FROM signals ORDER BY ts ASC, id ASC"), conn)
    df["ts_dt"] = df["ts"].map(_ts)
    return df


# ---------------------------------------------------------------------------
# 3. Clasificación de repeticiones
# ---------------------------------------------------------------------------

def classify_repeats(df: pd.DataFrame) -> pd.Series:
    """
    quality_flag propuesto por fila (None = alerta original).

    Regla (la misma que se dio al usuario para operar a mano):
      (a) entrada, SL y TP idénticos a los de una alerta anterior del mismo
          activo y dirección (el mismo setup re-enviado) → 'stale_reemission';
      (b) si no, llega a menos de 4 h (cripto) / 24 h (diarios) de la alerta
          anterior del mismo activo y dirección → 'intrabar_repeat'.

    Solo se clasifican las filas ANTIGUAS (setup_ts NULL). Las señales emitidas
    con la Fase 0 ya llevan su antiduplicado por setup y conservan su quality_flag.
    """
    flags = df["quality_flag"].copy() if "quality_flag" in df.columns else \
        pd.Series([None] * len(df), index=df.index, dtype=object)
    flags = flags.astype(object)
    legacy = df["setup_ts"].isna() if "setup_ts" in df.columns else pd.Series(True, index=df.index)

    def same(a, b):
        return abs(a - b) <= max(abs(a) * 1e-6, 1e-9)

    for (sym, direction), g in df[legacy].sort_values(["ts_dt", "id"]).groupby(["symbol", "direction"]):
        src = ASSETS.get(sym, {}).get("source", "binance")
        window = pd.Timedelta(hours=REPEAT_WINDOW_H.get(src, 4))
        seen: list[tuple] = []
        prev_ts = None
        for i, row in g.iterrows():
            lv = (float(row["entry_price"]), float(row["stop_loss"]), float(row["take_profit"]))
            if any(all(same(x, y) for x, y in zip(lv, s_)) for s_ in seen):
                flags[i] = "stale_reemission"
            elif prev_ts is not None and row["ts_dt"] - prev_ts < window:
                flags[i] = "intrabar_repeat"
            else:
                flags[i] = None
            seen.append(lv)
            prev_ts = row["ts_dt"]
    return flags


# ---------------------------------------------------------------------------
# 4. Re-resolución
# ---------------------------------------------------------------------------

def load_history(symbol: str, since: pd.Timestamp) -> pd.DataFrame:
    """Velas CERRADAS desde antes de la señal más antigua (misma fuente que el vivo)."""
    from data.bars import drop_unclosed_bars
    src = ASSETS.get(symbol, {}).get("source", "binance")
    days = int((pd.Timestamp.now(tz="UTC") - since).days) + 15
    try:
        if src == "yfinance":
            from data.fetchers.gold_fetcher import fetch_yfinance_asset
            return drop_unclosed_bars(fetch_yfinance_asset(symbol, days=days), "1d", session=True)
        if src == "binance":
            from data.fetchers.binance_fetcher import fetch_ohlcv
            return drop_unclosed_bars(fetch_ohlcv(symbol, timeframe="4h", days=days), "4h")
    except Exception as exc:
        print(f"   {symbol}: no se pudieron cargar datos ({exc})")
        return pd.DataFrame()
    # Forex (Twelve Data, apagado): no se re-resuelve; solo se corrige pnl_usd.
    return pd.DataFrame()


def re_resolve(df: pd.DataFrame, histories: dict) -> pd.DataFrame:
    """Calcula el resultado con la lógica nueva para cada señal no manual."""
    from data.outcome_tracker import check_outcome_detail, compute_result_fields
    out = []
    for _, r in df.iterrows():
        rec = {"id": r["id"], "new_outcome": None, "new_exit": None, "new_pnl_pct": None,
               "new_pnl_r": None, "new_pnl_usd": None, "new_resolved_ts": None, "note": ""}
        if r["resultado_real"] not in (None, "win", "loss", "expired") and not pd.isna(r["resultado_real"]):
            rec["note"] = f"manual '{r['resultado_real']}': se respeta"
            out.append(rec)
            continue
        if r["resultado_real"] in ("win", "loss") and pd.isna(r.get("pnl_pct")):
            rec["note"] = f"'{r['resultado_real']}' puesto a mano (sin pnl_pct): se respeta"
            rec["manual"] = True
            out.append(rec)
            continue
        hist = histories.get(r["symbol"])
        if hist is None or hist.empty:
            rec["note"] = "sin datos de la fuente"
            out.append(rec)
            continue
        sts = r.get("setup_ts")
        sts = pd.Timestamp(int(sts), unit="s", tz="UTC") if sts is not None and not pd.isna(sts) else None
        res = check_outcome_detail(r["symbol"], r["ts_dt"], r["direction"], float(r["entry_price"]),
                                   float(r["stop_loss"]), float(r["take_profit"]), df=hist,
                                   setup_ts=sts)
        if res is None:
            rec["note"] = "aún abierta / sin velas suficientes"
            out.append(rec)
            continue
        f = compute_result_fields(r["direction"], float(r["entry_price"]), float(r["stop_loss"]),
                                  res["exit_price"], r.get("position_size"))
        rec.update(new_outcome=res["outcome"], new_exit=res["exit_price"], new_pnl_pct=f["pnl_pct"],
                   new_pnl_r=f["pnl_r"], new_pnl_usd=f["pnl_usd"],
                   new_resolved_ts=int(pd.Timestamp(res["exit_ts"]).timestamp()))
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 5. OHLCV con precios sustitutos o velas a medio formar
# ---------------------------------------------------------------------------

def bad_ohlcv_rows(engine, symbol: str, timeframe: str, fresh: pd.DataFrame) -> list:
    if fresh is None or fresh.empty:
        return []
    with engine.connect() as conn:
        stored = pd.read_sql_query(text(
            "SELECT id, ts, close FROM ohlcv WHERE symbol=:s AND timeframe=:tf"
        ), conn, params={"s": symbol, "tf": timeframe})
    if stored.empty:
        return []
    ref = fresh["close"].copy()
    ref.index = [int(pd.Timestamp(t).timestamp()) for t in ref.index]
    stored["ref"] = stored["ts"].map(ref)
    stored = stored.dropna(subset=["ref"])
    dev = (stored["close"] / stored["ref"] - 1).abs()
    return stored.loc[dev > OHLCV_MAX_DEV, "id"].astype(int).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(apply: bool) -> None:
    engine = get_engine()
    dialect = engine.dialect.name
    print(f"Base de datos: {dialect}. Modo: {'APLICAR' if apply else 'SOLO VISTA PREVIA'}\n")

    # 1. Backup
    with engine.connect() as conn:
        has_backup = _table_exists(conn, BACKUP_TABLE, dialect)
    if apply and not has_backup:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE TABLE {BACKUP_TABLE} AS SELECT * FROM signals"))
        print(f"1. Copia de seguridad creada: {BACKUP_TABLE}")
    else:
        print(f"1. Copia de seguridad: {'ya existe ' + BACKUP_TABLE if has_backup else 'se creará ' + BACKUP_TABLE}")

    # 2. Migración (solo si se aplica: en vista previa no se modifica nada)
    if apply:
        init_db(force=True)
        print("2. Columnas nuevas verificadas.")
    else:
        print("2. Columnas nuevas: se añadirán al aplicar (si faltan).")

    df = load_all_signals(engine)
    if df.empty:
        print("No hay señales. Nada que hacer.")
        return
    for col in ("quality_flag", "pnl_r", "exit_price", "resolved_ts", "mode", "setup_ts"):
        if col not in df.columns:
            df[col] = None

    # 3. Repeticiones
    flags = classify_repeats(df)
    df["new_flag"] = flags
    n_stale = int((flags == "stale_reemission").sum())
    n_intra = int((flags == "intrabar_repeat").sum())
    print(f"\n3. Repeticiones: {len(df)} señales → {len(df) - n_stale - n_intra} originales, "
          f"{n_stale} re-emisiones de setups viejos, {n_intra} re-alertas de vela abierta.")
    by_sym = df.assign(f=flags.fillna("original")).groupby(["symbol", "f"]).size().unstack(fill_value=0)
    print(by_sym.to_string())

    # 4. Re-resolución
    histories = {}
    for sym in sorted(df["symbol"].unique()):
        if sym not in ASSETS:
            continue
        since = df.loc[df["symbol"] == sym, "ts_dt"].min() - pd.Timedelta(days=3)
        histories[sym] = load_history(sym, since)
        h = histories[sym]
        print(f"   Datos {sym}: {len(h) if h is not None else 0} velas "
              f"({h.attrs.get('source', '?') if h is not None and not h.empty else 'sin datos'})")
    rr = re_resolve(df, histories)
    m = df.merge(rr, on="id", how="left")

    changed_outcome = m[m["new_outcome"].notna() & (m["new_outcome"] != m["resultado_real"])]
    filled_expired = m[(m["resultado_real"] == "expired") & m["pnl_pct"].isna() & m["new_pnl_pct"].notna()]
    print(f"\n4. Re-resolución con la lógica nueva:")
    print(f"   resultados que cambian: {len(changed_outcome)}")
    if not changed_outcome.empty:
        cols = ["id", "symbol", "direction", "resultado_real", "new_outcome", "new_pnl_r", "note"]
        print(changed_outcome[cols].to_string(index=False))
    print(f"   'expired' sin PnL que reciben su salida por tiempo: {len(filled_expired)}")
    pending = m[m["new_outcome"].isna()]
    if not pending.empty:
        print(f"   sin cambio de resultado ({len(pending)}): " +
              "; ".join(f"#{i} {n}" for i, n in zip(pending["id"], pending["note"]) if n)[:600])

    # Resumen en R de lo que cuenta (originales, modo live) tras la limpieza
    ok = m[m["new_flag"].isna() & (m["mode"].fillna("live") == "live") & m["new_pnl_r"].notna()]
    if not ok.empty:
        print(f"\n   Señales originales resueltas: {len(ok)} | suma {ok['new_pnl_r'].sum():+.2f}R | "
              f"media {ok['new_pnl_r'].mean():+.3f}R por operación")

    # 5. OHLCV
    bad = {}
    for sym, h in histories.items():
        tf = "1d" if ASSETS.get(sym, {}).get("source") == "yfinance" else "4h"
        ids = bad_ohlcv_rows(engine, sym, tf, h)
        if tf == "1d":
            ids += bad_ohlcv_rows(engine, sym, "4h", h)   # en diarios se guardan ambos
        bad[sym] = ids
    print(f"\n5. Velas de la tabla ohlcv con desvío > {OHLCV_MAX_DEV:.0%}: " +
          ", ".join(f"{s}: {len(v)}" for s, v in bad.items()))

    if not apply:
        print("\nVista previa terminada. Nada se ha modificado. Para aplicar: --apply")
        return

    with engine.begin() as conn:
        for _, r in m.iterrows():
            flag = r["new_flag"]
            params = {"id": int(r["id"]), "flag": None if flag is None or pd.isna(flag) else flag}
            sets = ["quality_flag=:flag"]
            if r.get("manual") is True:
                conn.execute(text(f"UPDATE signals SET {', '.join(sets)} WHERE id=:id"), params)
                continue
            if r["new_outcome"] is not None and not pd.isna(r["new_outcome"]):
                sets += ["resultado_real=:o", "pnl_pct=:p", "pnl_usd=:u", "pnl_r=:r",
                         "exit_price=:x", "resolved_ts=:rts"]
                params.update(o=r["new_outcome"], p=float(r["new_pnl_pct"]),
                              u=None if pd.isna(r["new_pnl_usd"]) else float(r["new_pnl_usd"]),
                              r=None if pd.isna(r["new_pnl_r"]) else float(r["new_pnl_r"]),
                              x=float(r["new_exit"]), rts=int(r["new_resolved_ts"]))
            elif r["resultado_real"] in ("win", "loss", "expired") and not pd.isna(r.get("pnl_pct")):
                # Sin datos para re-resolver (p. ej. forex apagado): corregir pnl_usd
                # con el tamaño y deducir el R y el precio de salida del pnl_pct.
                entry, sl, pct = float(r["entry_price"]), float(r["stop_loss"]), float(r["pnl_pct"])
                sign = 1.0 if r["direction"] == "long" else -1.0
                risk = abs(entry - sl)
                sets += ["pnl_usd=:u", "pnl_r=:r", "exit_price=:x"]
                params.update(u=round(float(r["position_size"]) * pct, 2),
                              r=round(pct * entry / risk, 4) if risk > 0 else None,
                              x=entry * (1 + sign * pct))
            conn.execute(text(f"UPDATE signals SET {', '.join(sets)} WHERE id=:id"), params)
        for sym, ids in bad.items():
            for i in ids:
                conn.execute(text("DELETE FROM ohlcv WHERE id=:id"), {"id": int(i)})
    print("\nCambios aplicados. La copia original sigue en la tabla", BACKUP_TABLE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Aplicar los cambios (por defecto solo vista previa)")
    main(ap.parse_args().apply)
