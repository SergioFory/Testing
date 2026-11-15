"""
Pruebas de la Fase 1 — copia propia del histórico (punto 1.11):
  - backfill_history.py: Gate.io (máximo disponible) + OKX solo para fechas
    anteriores, columna ohlcv.source, vista previa por defecto (no escribe nada),
    --apply sin sobrescribir, --refresh (OKX nunca pisa Gate.io), etiquetado de
    filas sin fuente idénticas, velas cerradas, página a página;
  - la última vela de cada página se vuelve a pedir (Gate.io la devuelve plana
    y con volumen 0);
  - migración idempotente de ohlcv.source y save_ohlcv con la fuente;
  - python main.py history-stats (cobertura por símbolo/timeframe/fuente y huecos).

    cd trading_system && python -m pytest tests -q

Sin red: los exchanges son dobles deterministas y la BD es SQLite temporal.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

H4 = 4 * 3600
D1 = 86400
NOW = 1_763_200_000 - (1_763_200_000 % H4) + 2 * 3600    # a mitad de una vela 4H


# ---------------------------------------------------------------------------
# Dobles
# ---------------------------------------------------------------------------

def _price(ts: int, salt: float = 0.0) -> tuple:
    base = 100.0 + (ts // H4) % 97 + salt
    return base, base + 2.0, base - 2.0, base + 0.5, 1000.0 + (ts // H4) % 13


class FakeExchange:
    """Sirve velas de paso fijo entre first_ts y el 'ahora' (incluida la vela en
    formación), por ventanas [since, since + (limit-1)·paso] como Gate.io, y con
    la ÚLTIMA vela de cada página plana y con volumen 0 (rareza real de Gate.io)."""

    def __init__(self, first_ts: dict, now: int, salt: float = 0.0, flat_last=True,
                 fail_from: int | None = None):
        self.first_ts = first_ts           # timeframe -> primera vela disponible
        self.now = now
        self.salt = salt
        self.flat_last = flat_last
        self.fail_from = fail_from
        self.calls = []

    def fetch_ohlcv(self, market, timeframe, since=None, limit=None):
        self.calls.append((market, timeframe, since, limit))
        step = {"4h": H4, "1d": D1}[timeframe]
        s = since // 1000
        if self.fail_from is not None and s >= self.fail_from:
            raise RuntimeError("HTTP 502")
        first = self.first_ts[timeframe]
        start = max(s, first)
        start += (-start) % step
        end = min(s + (limit - 1) * step, self.now - self.now % step)   # hasta la vela en formación
        out = []
        t = start
        while t <= end:
            o, h, l, c, v = _price(t, self.salt)
            out.append([t * 1000, o, h, l, c, v])
            t += step
        if out and self.flat_last:
            o = out[-1][1]
            out[-1] = [out[-1][0], o, o, o, o, 0.0]
        return out


def _factory(gate, okx):
    return lambda source: {"gateio": gate, "okx": okx}[source]


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    import data.database as db
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    yield db, tmp_path / "test.db"
    if db._engine is not None:
        db._engine.dispose()


def _old_schema(path):
    """Tabla ohlcv como en producción antes de este cambio (sin source)."""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE ohlcv (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL, ts INTEGER NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
        low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL, UNIQUE(symbol, timeframe, ts))""")
    con.commit()
    return con


def _rows(db, where="1=1"):
    with db.get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(
            f"SELECT * FROM ohlcv WHERE {where} ORDER BY symbol, timeframe, ts")).mappings().fetchall()]


GATE_FIRST = {"4h": NOW - 300 * H4 - (NOW % H4), "1d": NOW - 200 * D1 - (NOW % D1)}
OKX_FIRST = {"4h": GATE_FIRST["4h"] - 500 * H4, "1d": GATE_FIRST["1d"] - 100 * D1}
SINCE = "2018-01-01"


def _run(bh, gate, okx, **kw):
    kw.setdefault("since", SINCE)
    return bh.run_backfill(["BTCUSDT"], ["4h", "1d"], exchange_factory=_factory(gate, okx),
                           now_s=NOW, sleep=lambda s: None, log=lambda m: None, **kw)


# ---------------------------------------------------------------------------
# Paginación
# ---------------------------------------------------------------------------

def test_pages_drop_flat_last_candle_and_unclosed_and_skip_empty_windows():
    import backfill_history as bh
    ex = FakeExchange({"4h": GATE_FIRST["4h"]}, NOW)
    start = GATE_FIRST["4h"] - 50 * H4 * 10        # ventanas vacías antes del contrato
    pages = list(bh.iter_closed_pages(ex, "BTC/USDT:USDT", "4h", start, NOW, 100, NOW))
    rows = [r for p, _ in pages for r in p]
    ts = [r[0] for r in rows]
    assert ts[0] == GATE_FIRST["4h"]
    assert ts == list(range(GATE_FIRST["4h"], ts[-1] + H4, H4))     # sin huecos ni repetidas
    assert ts[-1] + H4 <= NOW < ts[-1] + 2 * H4                    # solo cerradas, hasta la última
    assert not any(r[1] == r[2] == r[3] == r[4] and r[5] == 0 for r in rows)   # ninguna plana
    assert all(len(p) <= 100 for p, _ in pages)                    # nunca más de una página
    assert all(c[3] == 100 for c in ex.calls)


def test_pages_drop_invalid_candles():
    import backfill_history as bh

    class Bad(FakeExchange):
        def fetch_ohlcv(self, *a, **k):
            out = super().fetch_ohlcv(*a, **k)
            if len(out) > 3:
                out[1] = [out[1][0], 10.0, 9.0, 11.0, 10.0, 5.0]     # high < low
                out[2] = [out[2][0], float("nan"), 1, 1, 1, 1]
            return out
    ex = Bad({"4h": GATE_FIRST["4h"]}, NOW)
    pages = list(bh.iter_closed_pages(ex, "X", "4h", GATE_FIRST["4h"], GATE_FIRST["4h"] + 20 * H4, 1000, NOW))
    assert sum(inv for _, inv in pages) == 2
    assert len([r for p, _ in pages for r in p]) == 18


def test_page_error_after_retries():
    import backfill_history as bh
    ex = FakeExchange({"4h": GATE_FIRST["4h"]}, NOW, fail_from=0)
    waits = []
    with pytest.raises(bh.PageError):
        list(bh.iter_closed_pages(ex, "X", "4h", GATE_FIRST["4h"], NOW, 1000, NOW, sleep=waits.append))
    assert len(ex.calls) == bh.PAGE_RETRIES and waits == [2, 4]


# ---------------------------------------------------------------------------
# Vista previa / apply / refresh
# ---------------------------------------------------------------------------

def test_preview_writes_nothing_even_without_source_column(temp_db):
    import backfill_history as bh
    db, path = temp_db
    con = _old_schema(path)
    con.execute("INSERT INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume) VALUES "
                "('BTCUSDT','4h',?,1,2,0.5,1.5,10)", (GATE_FIRST["4h"],))
    con.commit()
    before = con.execute("SELECT * FROM ohlcv").fetchall()
    con.close()
    reps = _run(bh, FakeExchange(GATE_FIRST, NOW), FakeExchange(OKX_FIRST, NOW, salt=0.3))
    con = sqlite3.connect(path)
    assert con.execute("SELECT * FROM ohlcv").fetchall() == before
    assert "source" not in {r[1] for r in con.execute("PRAGMA table_info(ohlcv)")}
    g4 = next(r for r in reps if r.source == "gateio" and r.timeframe == "4h")
    assert g4.different == 1 and g4.new == g4.downloaded - 1
    text_ = bh.format_report(reps, apply=False, refresh=False)
    assert "VISTA PREVIA" in text_ and "--refresh" in text_


def test_apply_fills_gate_then_okx_before_it_with_sources(temp_db):
    import backfill_history as bh
    db, _ = temp_db
    gate, okx = FakeExchange(GATE_FIRST, NOW), FakeExchange(OKX_FIRST, NOW, salt=0.3)
    reps = _run(bh, gate, okx, apply=True)
    assert not any(r.error for r in reps)
    for tf, step in (("4h", H4), ("1d", D1)):
        rows = _rows(db, f"timeframe='{tf}'")
        ts = [r["ts"] for r in rows]
        assert ts[0] == OKX_FIRST[tf]
        assert ts == list(range(ts[0], ts[-1] + step, step))          # continua, sin huecos
        assert ts[-1] + step <= NOW                                  # solo velas cerradas
        src = {r["ts"]: r["source"] for r in rows}
        assert all(src[t] == "okx" for t in ts if t < GATE_FIRST[tf])
        assert all(src[t] == "gateio" for t in ts if t >= GATE_FIRST[tf])
        assert not any(r["volume"] == 0 for r in rows)               # ninguna vela plana
    # OKX solo se pide para fechas anteriores a Gate.io
    assert all(c[2] // 1000 < GATE_FIRST[c[1]] for c in okx.calls)
    # Segunda pasada: nada nuevo (idempotente)
    again = _run(bh, FakeExchange(GATE_FIRST, NOW), FakeExchange(OKX_FIRST, NOW, salt=0.3), apply=True)
    assert sum(r.new for r in again) == 0
    assert sum(r.identical for r in again) == sum(r.downloaded for r in again)


def test_existing_rows_label_keep_and_refresh_rules(temp_db):
    import backfill_history as bh
    db, path = temp_db
    con = _old_schema(path)
    t_same, t_diff = GATE_FIRST["4h"] + 10 * H4, GATE_FIRST["4h"] + 11 * H4
    t_okx_other = OKX_FIRST["4h"] + 5 * H4                      # en el rango de OKX, de otra fuente
    o, h, l, c, v = _price(t_same)
    con.execute("INSERT INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume) VALUES ('BTCUSDT','4h',?,?,?,?,?,?)",
                (t_same, o, h, l, c, v))
    con.execute("INSERT INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume) VALUES ('BTCUSDT','4h',?,1,1,1,1,0)",
                (t_diff,))                                          # vela plana guardada antes
    con.commit()
    con.close()
    db.init_db()                                                    # migra: añade source
    with db.get_engine().begin() as conn:
        conn.execute(text("INSERT INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume,source) "
                          "VALUES ('BTCUSDT','4h',:t,5,6,4,5,1,'binance')"), {"t": t_okx_other})

    reps = _run(bh, FakeExchange(GATE_FIRST, NOW), FakeExchange(OKX_FIRST, NOW, salt=0.3), apply=True)
    by = {r["ts"]: r for r in _rows(db, "timeframe='4h'")}
    assert by[t_same]["source"] == "gateio"                         # idéntica sin fuente → etiquetada
    assert by[t_diff]["close"] == 1 and by[t_diff]["source"] is None  # distinta → no se toca
    g4 = next(r for r in reps if r.source == "gateio" and r.timeframe == "4h")
    assert (g4.labelled, g4.different, g4.refreshed) == (1, 1, 0)

    reps = _run(bh, FakeExchange(GATE_FIRST, NOW), FakeExchange(OKX_FIRST, NOW, salt=0.3),
                apply=True, refresh=True)
    by = {r["ts"]: r for r in _rows(db, "timeframe='4h'")}
    assert by[t_diff]["close"] == _price(t_diff)[3] and by[t_diff]["source"] == "gateio"
    assert by[t_okx_other]["close"] == 5 and by[t_okx_other]["source"] == "binance"  # OKX solo pisa NULL/okx
    o4 = next(r for r in reps if r.source == "okx" and r.timeframe == "4h")
    assert o4.protected == 1 and o4.refreshed == 0


def test_okx_stops_at_oldest_gateio_row_already_saved(temp_db):
    """La ventana de Gate.io se mueve: lo que Gate.io sirvió en una pasada anterior
    (y ya no sirve) marca el límite de OKX."""
    import backfill_history as bh
    db, _ = temp_db
    db.init_db()
    old = OKX_FIRST["4h"] + 100 * H4
    with db.get_engine().begin() as conn:
        conn.execute(text("INSERT INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume,source) "
                          "VALUES ('BTCUSDT','4h',:t,5,6,4,5,1,'gateio')"), {"t": old})
    reps = _run(bh, FakeExchange(GATE_FIRST, NOW), FakeExchange(OKX_FIRST, NOW, salt=0.3), apply=True)
    o4 = next(r for r in reps if r.source == "okx" and r.timeframe == "4h")
    assert o4.end_s == old and o4.last_ts == old - H4


def test_gate_failure_skips_okx_and_main_returns_error(temp_db, monkeypatch, capsys):
    import backfill_history as bh
    db, _ = temp_db
    gate = FakeExchange(GATE_FIRST, NOW, fail_from=0)
    okx = FakeExchange(OKX_FIRST, NOW)
    reps = _run(bh, gate, okx, apply=True)
    assert all(r.error for r in reps if r.source == "gateio")
    assert all(r.skipped for r in reps if r.source == "okx") and okx.calls == []
    assert "ERROR" in bh.format_report(reps, apply=True, refresh=False)
    monkeypatch.setattr(bh, "run_backfill", lambda *a, **k: reps)
    monkeypatch.setattr(bh, "format_history_stats", lambda *a, **k: "")
    assert bh.main(["--apply"]) == 1


def test_okx_without_data_before_gate_is_reported(temp_db):
    import backfill_history as bh
    db, _ = temp_db
    late_okx = {"4h": GATE_FIRST["4h"] + 10 * H4, "1d": GATE_FIRST["1d"] + 10 * D1}
    reps = _run(bh, FakeExchange(GATE_FIRST, NOW), FakeExchange(late_okx, NOW), apply=True)
    okx = [r for r in reps if r.source == "okx"]
    assert all(r.downloaded == 0 and r.error is None for r in okx)
    assert "la fuente no tiene velas" in bh.format_report(reps, apply=True, refresh=False)


def test_rejects_unknown_symbol_or_timeframe(temp_db):
    import backfill_history as bh
    with pytest.raises(ValueError):
        bh.run_backfill(["XAUUSD"], ["4h"], exchange_factory=_factory(None, None), now_s=NOW)
    with pytest.raises(ValueError):
        bh.run_backfill(["BTCUSDT"], ["1h"], exchange_factory=_factory(None, None), now_s=NOW)


# ---------------------------------------------------------------------------
# BD: migración, save_ohlcv, history-stats
# ---------------------------------------------------------------------------

def test_migration_adds_ohlcv_source_idempotently(temp_db):
    db, path = temp_db
    _old_schema(path).close()
    db.init_db()
    db.init_db(force=True)
    con = sqlite3.connect(path)
    assert "source" in {r[1] for r in con.execute("PRAGMA table_info(ohlcv)")}


def test_save_ohlcv_records_source_from_attrs(temp_db):
    db, _ = temp_db
    idx = pd.date_range("2026-01-01", periods=3, freq="4h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx)
    df.attrs["source"] = "gateio"
    assert db.save_ohlcv(df.tail(2).copy(), "BTCUSDT", "4h") == 2      # attrs sobreviven a tail/copy
    assert db.save_ohlcv(df, "BTCUSDT", "4h", source="otra") == 1      # no sobrescribe
    assert [r["source"] for r in _rows(db)] == ["otra", "gateio", "gateio"]


def test_history_stats_coverage_sources_and_gaps(temp_db):
    import backfill_history as bh
    db, _ = temp_db
    db.init_db()
    t0 = GATE_FIRST["4h"]
    rows = [(t0 + i * H4, "okx" if i < 5 else "gateio") for i in range(10) if i not in (6, 7)]
    with db.get_engine().begin() as conn:
        for t, s in rows:
            conn.execute(text("INSERT INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume,source) "
                              "VALUES ('BTCUSDT','4h',:t,1,2,0.5,1.5,1,:s)"), {"t": t, "s": s})
        conn.execute(text("INSERT INTO ohlcv (symbol,timeframe,ts,open,high,low,close,volume) "
                          "VALUES ('XAUUSD','1d',:t,1,2,0.5,1.5,1)"), {"t": t0})
    cov = db.ohlcv_coverage()
    assert {(r["symbol"], r["source"], r["n"]) for r in cov} == {
        ("BTCUSDT", "okx", 5), ("BTCUSDT", "gateio", 3), ("XAUUSD", None, 1)}
    assert db.ohlcv_gaps("BTCUSDT", "4h", H4) == {"gaps": 1, "missing": 2, "max_gap_s": 3 * H4}
    out = bh.format_history_stats(now_s=NOW)
    assert "okx" in out and "gateio" in out and "(sin fuente)" in out
    assert "huecos: 1 (faltan 2 velas" in out
    assert out.count("huecos") == 1                                   # el oro (diario de sesión) no


def test_main_has_history_stats_command(temp_db, capsys, monkeypatch):
    import main
    args = main.build_parser().parse_args(["history-stats"])
    assert args.command == "history-stats"
    db, _ = temp_db
    db.init_db()
    monkeypatch.setattr(sys, "argv", ["main.py", "history-stats"])
    main.main()
    assert "vacía" in capsys.readouterr().out
