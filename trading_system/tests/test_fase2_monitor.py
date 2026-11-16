"""
Pruebas de la Fase 2 — monitor de datos y deriva (punto 2.3,
monitoring/data_monitor.py):
  - comprobaciones por ciclo (velas planas, duplicadas, desordenadas, fin de
    semana, huecos, precio congelado, saltos absurdos, cambio de fuente);
  - los DOS fallos reales de esta semana: la vela plana de Gate.io en el corte de
    página y la fila de GC=F fechada en domingo;
  - tabla data_quality (UNIQUE), avisos graves con throttling de 6 h, y que nada
    de esto cambia ni tumba la señal;
  - nocturno: contraste OKX vs Gate.io y deriva frente a DRIFT_BASELINE;
  - bloque "Calidad de datos" del resumen nocturno.

    cd trading_system && python -m pytest tests/test_fase2_monitor.py -q

Sin red: requests y los exchanges de ccxt se sustituyen; BD SQLite temporal.
"""
import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

U = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())   # noqa: E731


# ---------------------------------------------------------------------------
# Fixtures y datos sintéticos
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    import data.database as db
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    db.init_db()
    yield db
    if db._engine is not None:
        db._engine.dispose()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)
    import monitoring.data_monitor as dm

    def _no_ex():
        raise AssertionError("intento de crear un exchange real en una prueba")
    monkeypatch.setattr(dm, "_okx_exchange", _no_ex)
    monkeypatch.setattr(dm, "_gate_exchange", _no_ex)
    # Estado de avisos en memoria limpio en cada prueba
    monkeypatch.setattr(dm, "_alert_state", {})
    monkeypatch.setattr(dm, "_alerted_keys", set())


def _ohlcv(index, start=60000.0, seed=0, vol=100.0, step=0.004):
    """Serie OHLCV realista (paseo aleatorio, sin velas planas ni saltos)."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, step, len(index))))
    open_ = np.r_[start, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.003, len(index)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.003, len(index)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(vol * 0.5, vol * 1.5, len(index))}, index=index)


def _utc(t):
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def _crypto_4h(end="2026-06-16 16:00", n=300, seed=1):
    idx = pd.date_range(end=_utc(end), periods=n, freq="4h")
    df = _ohlcv(idx, seed=seed)
    df.attrs["source"] = "gateio"
    return df


def _crypto_1d(end="2026-06-16", n=120, seed=2):
    idx = pd.date_range(end=_utc(end), periods=n, freq="1D")
    df = _ohlcv(idx, seed=seed, step=0.02)
    df.attrs["source"] = "gateio"
    return df


def _gold_daily(end="2026-11-13", n=120, seed=3):
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n, tz="UTC")
    df = _ohlcv(idx, start=4000.0, seed=seed, vol=200000, step=0.01)
    df.attrs["source"] = "GC=F"
    return df


# Vela plana REAL de Gate.io (BTC 4H, snapshot del 2026-11-15; una de las 8 por
# serie que dejaba el corte de página del fetcher antiguo).
REAL_FLAT_TS = pd.Timestamp("2026-06-16 16:00", tz="UTC")
REAL_FLAT_PRICE = 65835.3


def _set_flat(df, ts, price):
    df.loc[ts, ["open", "high", "low", "close"]] = price
    df.loc[ts, "volume"] = 0.0
    return df


# ---------------------------------------------------------------------------
# Comprobaciones por ciclo
# ---------------------------------------------------------------------------

def test_datos_limpios_sin_incidencias():
    from monitoring.data_monitor import cycle_incidents
    assert cycle_incidents("BTCUSDT", _crypto_4h(), _crypto_1d()) == []
    assert cycle_incidents("XAUUSD", _gold_daily(), None) == []


def test_vela_plana_gateio_real_en_ultima_vela_es_grave():
    from monitoring.data_monitor import cycle_incidents
    df = _set_flat(_crypto_4h(end=REAL_FLAT_TS), REAL_FLAT_TS, REAL_FLAT_PRICE)
    inc = cycle_incidents("BTCUSDT", df, _crypto_1d())
    assert [(i["check"], i["ts"], i["grave"]) for i in inc] == [("flat", REAL_FLAT_TS, True)]
    assert "65835.3" in inc[0]["detalle"] and "volumen 0" in inc[0]["detalle"]
    # Penúltima vela: también grave; más atrás en la ventana: se registra, sin aviso.
    df2 = _crypto_4h(end=REAL_FLAT_TS)
    _set_flat(df2, df2.index[-2], 65000.0)
    _set_flat(df2, df2.index[-10], 64000.0)
    got = {(i["ts"], i["grave"]) for i in cycle_incidents("BTCUSDT", df2, None)}
    assert got == {(df2.index[-2], True), (df2.index[-10], False)}
    # Fuera de las 60 velas revisadas: no se mira (coste fijo por ciclo).
    df3 = _crypto_4h(end=REAL_FLAT_TS)
    _set_flat(df3, df3.index[-61], 64000.0)
    assert cycle_incidents("BTCUSDT", df3, None) == []
    # Plana CON volumen (mercado parado de verdad) no cuenta.
    df4 = _crypto_4h(end=REAL_FLAT_TS)
    _set_flat(df4, df4.index[-1], 65000.0)
    df4.loc[df4.index[-1], "volume"] = 3.0
    assert cycle_incidents("BTCUSDT", df4, None) == []


class _PagedExchange:
    """Exchange ccxt falso con páginas: imita a Gate.io."""
    rateLimit = 0
    id = "gate"

    def __init__(self, df, page=50, flat_end_of_page=True, flat_inside=None):
        self.df, self.page = df, page
        self.flat_end_of_page, self.flat_inside = flat_end_of_page, flat_inside

    def parse_timeframe(self, tf):
        return 4 * 3600

    def fetch_ohlcv(self, symbol, tf, since=None, limit=1000):
        ms = (self.df.index.as_unit("ms").asi8)
        sel = self.df[ms >= since].iloc[:self.page]
        rows = [[int(t.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume]
                for t, r in sel.iterrows()]
        if rows and self.flat_end_of_page and len(sel) == self.page:
            c = rows[-1][4]          # la vela del corte llega PLANA y con volumen 0
            rows[-1] = [rows[-1][0], c, c, c, c, 0.0]
        if self.flat_inside is not None:
            for r in rows:
                if r[0] == int(self.flat_inside.timestamp() * 1000):
                    r[1:] = [r[4]] * 4 + [0.0]
        return rows


def test_fetcher_actual_quita_la_plana_y_si_volviera_el_monitor_la_ve(monkeypatch):
    """Fallo real 1 de esta semana (Gate.io): con el fetcher corregido la vela del
    corte de página no llega; si Gate.io cambiara y la plana apareciera DENTRO de
    una página (el fetcher ya no la puede quitar), el monitor la detecta."""
    import data.fetchers.binance_fetcher as bf
    from data.bars import drop_unclosed_bars
    from monitoring.data_monitor import cycle_incidents
    monkeypatch.setattr(bf.time, "sleep", lambda s: None)
    now = pd.Timestamp("2026-06-16 21:00", tz="UTC")
    src = _crypto_4h(end="2026-06-16 20:00", n=200)          # la última está en curso
    monkeypatch.setattr(bf, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: now.to_pydatetime())}))
    df = bf._fetch_ohlcv_from(_PagedExchange(src), "BTC/USDT:USDT", "4h", days=40)
    df.attrs["source"] = "gateio"
    df = drop_unclosed_bars(df, "4h", now=now)
    flat = (df.open == df.high) & (df.high == df.low) & (df.low == df.close) & (df.volume == 0)
    assert len(df) > 100 and not flat.any()
    assert cycle_incidents("BTCUSDT", df, None) == []
    # Regresión simulada: la plana llega dentro de la página (última vela cerrada).
    ex = _PagedExchange(src, flat_end_of_page=False, flat_inside=REAL_FLAT_TS)
    bad = drop_unclosed_bars(bf._fetch_ohlcv_from(ex, "BTC/USDT:USDT", "4h", days=40), "4h", now=now)
    assert bad.index[-1] == REAL_FLAT_TS
    inc = cycle_incidents("BTCUSDT", bad, None)
    assert [(i["check"], i["ts"], i["grave"]) for i in inc] == [("flat", REAL_FLAT_TS, True)]


def test_fila_gcf_en_domingo_filtro_actual_y_regresion(monkeypatch):
    """Fallo real 2 de esta semana (GC=F): Yahoo devuelve la sesión en curso como
    una fila fechada DOMINGO 2026-11-15. Con la corrección no llega al ciclo (y el
    monitor no da falsa alarma); si la corrección fallara, el monitor la marca
    como grave."""
    import data.bars as bars
    from monitoring.data_monitor import cycle_incidents
    raw = _gold_daily(end="2026-11-13")
    sunday = pd.Timestamp("2026-11-15", tz="UTC")
    raw.loc[sunday] = [4080.0, 4085.0, 4078.0, 4083.0, 1500.0]   # unas horas de la sesión del lunes
    raw.attrs["source"] = "GC=F"
    now = pd.Timestamp("2026-11-15 23:30", tz="UTC")             # domingo 18:30 ET
    ok = bars.drop_unclosed_bars(raw, "1d", session=True, now=now)
    assert ok.index[-1] == pd.Timestamp("2026-11-13", tz="UTC")
    assert cycle_incidents("XAUUSD", ok, None) == []
    # Regresión: sin _session_rows_trustworthy la fila del domingo pasa como cerrada.
    monkeypatch.setattr(bars, "_session_rows_trustworthy", lambda df, now_utc: np.ones(len(df), bool))
    bad = bars.drop_unclosed_bars(raw, "1d", session=True, now=now)
    assert bad.index[-1] == sunday
    inc = cycle_incidents("XAUUSD", bad, None)
    assert [(i["check"], i["ts"], i["grave"]) for i in inc] == [("weekend", sunday, True)]
    assert "domingo" in inc[0]["detalle"]
    # En cripto el fin de semana es normal.
    assert cycle_incidents("BTCUSDT", _crypto_4h(end="2026-11-15 16:00"), None) == []


def test_duplicadas_desordenadas_huecos_congelado_saltos_fuente():
    from monitoring.data_monitor import check_frame, cycle_incidents
    base = _crypto_4h(end="2026-11-10 12:00", n=80)
    # Duplicada en la última vela (grave) y desordenada.
    dup = pd.concat([base, base.iloc[[-1]]])
    assert [(i["check"], i["grave"]) for i in check_frame(dup, "4h", expected="gateio")] == \
        [("duplicate", True)]
    uns = base.iloc[list(range(len(base) - 12)) + [-11, -12] + list(range(-10, 0))]
    inc = check_frame(uns, "4h", expected="gateio")
    assert [(i["check"], i["ts"], i["grave"]) for i in inc] == [("unsorted", base.index[-12], False)]
    # Hueco: faltan 2 velas 4H (no es grave: solo se registra).
    gap = base.drop(base.index[[-20, -19]])
    inc = check_frame(gap, "4h", expected="gateio")
    assert [(i["check"], i["ts"], i["grave"]) for i in inc] == [("gap", base.index[-20], False)]
    assert "faltan 2 vela(s)" in inc[0]["detalle"]
    # En activos de sesión los huecos (fines de semana, festivos) NO cuentan.
    assert check_frame(_gold_daily(), "1d", session=True, expected="GC=F") == []
    # Precio congelado: 6 cierres idénticos seguidos (5 no basta). Fechado en la 1.ª.
    fr = base.copy()
    fr.iloc[-6:, fr.columns.get_loc("close")] = 70000.0
    inc = check_frame(fr, "4h", expected="gateio")
    assert [(i["check"], i["ts"]) for i in inc] == [("frozen", base.index[-6])]
    fr5 = base.copy()
    fr5.iloc[-5:, fr5.columns.get_loc("close")] = 70000.0
    assert check_frame(fr5, "4h", expected="gateio") == []
    # Saltos: > 25 % en cripto (grave si es la última vela); 20 % no salta.
    j = base.copy()
    j.iloc[-1, j.columns.get_loc("close")] = j["close"].iloc[-2] * 1.30
    inc = check_frame(j, "4h", expected="gateio")
    assert [(i["check"], i["grave"]) for i in inc] == [("jump", True)]
    j2 = base.copy()
    j2.iloc[-1, j2.columns.get_loc("close")] = j2["close"].iloc[-2] * 1.20
    assert check_frame(j2, "4h", expected="gateio") == []
    # ... y > 10 % en oro/SPX (el −12 % del SPX del 16-mar-2020 saltaría).
    g = _gold_daily()
    g.iloc[-1, g.columns.get_loc("close")] = g["close"].iloc[-2] * 0.88
    assert [(i["check"], i["grave"]) for i in check_frame(g, "1d", session=True, expected="GC=F")] == \
        [("jump", True)]
    # Cambio de fuente (no grave); sin attrs no se juzga.
    s = base.copy()
    s.attrs["source"] = "okx"
    inc = cycle_incidents("BTCUSDT", s, None)
    assert [(i["check"], i["grave"]) for i in inc] == [("source", False)]
    s.attrs = {}
    assert cycle_incidents("BTCUSDT", s, None) == []
    # Diario de cripto: mismo examen con sufijo ':1d' (no choca con la 4H).
    d = _crypto_1d(end="2026-11-10")
    _set_flat(d, d.index[-1], 70000.0)
    inc = cycle_incidents("BTCUSDT", base, d)
    assert [(i["check"], i["grave"]) for i in inc] == [("flat:1d", True)]


# ---------------------------------------------------------------------------
# Tabla data_quality, avisos y ciclo
# ---------------------------------------------------------------------------

def test_data_quality_unique_y_migracion_idempotente(temp_db):
    from monitoring.data_monitor import record_incidents
    db = temp_db
    inc = [{"check": "flat", "ts": REAL_FLAT_TS, "detalle": "x", "grave": True},
           {"check": "gap", "ts": REAL_FLAT_TS, "detalle": "y", "grave": False}]
    assert record_incidents("BTCUSDT", inc) == {("flat", U("2026-06-16 16:00")), ("gap", U("2026-06-16 16:00"))}
    assert record_incidents("BTCUSDT", inc) == set()            # mismo incidente: 1 fila
    assert record_incidents("ETHUSDT", inc[:1]) == {("flat", U("2026-06-16 16:00"))}
    db.init_db(force=True)
    db.init_db(force=True)                                       # idempotente
    with db.get_engine().connect() as c:
        rows = c.execute(text("SELECT symbol, check_name, grave FROM data_quality ORDER BY id")).fetchall()
    assert [tuple(r) for r in rows] == [("BTCUSDT", "flat", 1), ("BTCUSDT", "gap", 0), ("ETHUSDT", "flat", 1)]


def test_avisos_graves_throttling_6h(temp_db, monkeypatch):
    import main
    import monitoring.data_monitor as dm
    import notifications.telegram as tg
    assert dm.ALERT_EVERY == main._DATA_ALERT_EVERY == timedelta(hours=6)
    sent = []
    monkeypatch.setattr(tg, "send_data_quality_alert", lambda s, items: sent.append((s, list(items))) or True)
    t0 = pd.Timestamp("2026-06-16 20:05", tz="UTC")
    df = _set_flat(_crypto_4h(end=REAL_FLAT_TS), REAL_FLAT_TS, REAL_FLAT_PRICE)
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0)
    assert len(sent) == 1 and "vela plana" in sent[0][1][0] and "65835.3" in sent[0][1][0]
    # El mismo incidente en ciclos siguientes: ni aviso ni fila nueva.
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0 + pd.Timedelta(hours=1))
    assert len(sent) == 1
    # Tras un reinicio del worker (memoria vacía), lo que ya está en la BD no se repite.
    monkeypatch.setattr(dm, "_alert_state", {})
    monkeypatch.setattr(dm, "_alerted_keys", set())
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0 + pd.Timedelta(hours=7))
    assert len(sent) == 1
    monkeypatch.setattr(dm, "_alert_state", {"BTCUSDT": t0})
    # Otro grave nuevo antes de 6 h: queda en la tabla (resumen) pero sin aviso.
    df2 = _set_flat(_crypto_4h(end="2026-06-17 00:00"), pd.Timestamp("2026-06-17 00:00", tz="UTC"), 66000.0)
    dm.run_cycle_checks("BTCUSDT", df2, None, notify=True, now=t0 + pd.Timedelta(hours=4))
    assert len(sent) == 1
    # Pasadas 6 h, un grave nuevo sí avisa.
    df3 = _set_flat(_crypto_4h(end="2026-06-17 04:00"), pd.Timestamp("2026-06-17 04:00", tz="UTC"), 66100.0)
    dm.run_cycle_checks("BTCUSDT", df3, None, notify=True, now=t0 + pd.Timedelta(hours=7))
    assert len(sent) == 2
    # Los no graves no avisan nunca; notify=False (modo sombra) tampoco.
    gap = _crypto_4h(end="2026-06-18 00:00").drop(pd.Timestamp("2026-06-17 12:00", tz="UTC"))
    dm.run_cycle_checks("BTCUSDT", gap, None, notify=True, now=t0 + pd.Timedelta(hours=20))
    df4 = _crypto_4h(end="2026-06-19 00:00")
    _set_flat(df4, df4.index[-1], float(df4["close"].iloc[-2]))
    dm.run_cycle_checks("ETHUSDT", df4, None, notify=False, now=t0 + pd.Timedelta(hours=30))
    assert len(sent) == 2
    with temp_db.get_engine().connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM data_quality")).scalar() == 5


def test_aviso_si_la_bd_falla_y_nunca_lanza(monkeypatch):
    import monitoring.data_monitor as dm
    import notifications.telegram as tg
    sent = []
    monkeypatch.setattr(tg, "send_data_quality_alert", lambda s, items: sent.append(s) or True)
    monkeypatch.setattr(dm, "record_incidents", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("BD caída")))
    df = _set_flat(_crypto_4h(end=REAL_FLAT_TS), REAL_FLAT_TS, REAL_FLAT_PRICE)
    t0 = pd.Timestamp("2026-06-16 20:05", tz="UTC")
    assert len(dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0)) == 1
    assert len(dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0 + pd.Timedelta(hours=7))) == 1
    assert sent == ["BTCUSDT"]                  # la memoria del proceso evita repetirlo
    monkeypatch.setattr(dm, "cycle_incidents", lambda *a: 1 / 0)
    assert dm.run_cycle_checks("BTCUSDT", df, None, notify=True) == []


def _patch_cycle(monkeypatch, frames, seen):
    """cmd_signal con datos inyectados; se detiene en la detección (sin setups)."""
    import main
    import data.bars as bars
    import signals.detector as det
    monkeypatch.setattr(main, "_load_data", lambda sym, closed_only=True, history="full": frames[sym])
    monkeypatch.setattr(bars, "last_bar_age_hours", lambda *a, **k: 1.0)

    def _detect(df_4h, df_daily, symbol):
        seen.append((symbol, df_4h.copy(), None if df_daily is None else df_daily.copy()))
        return pd.DataFrame()
    monkeypatch.setattr(det, "detect_all_setups", _detect)


def test_cmd_signal_registra_avisa_y_no_cambia_la_senal(temp_db, monkeypatch):
    import main
    import notifications.telegram as tg
    sent = []
    monkeypatch.setattr(tg, "send_data_quality_alert", lambda s, items: sent.append((s, items)) or True)
    btc = _set_flat(_crypto_4h(end=REAL_FLAT_TS), REAL_FLAT_TS, REAL_FLAT_PRICE)
    gold = _gold_daily(end="2026-11-13")
    gold.loc[pd.Timestamp("2026-11-15", tz="UTC")] = [4080.0, 4085.0, 4078.0, 4083.0, 1500.0]
    spx = _gold_daily()
    spx.attrs["source"] = "^GSPC"
    _set_flat(spx, spx.index[-1], float(spx["close"].iloc[-2]))     # sombra: se registra, no avisa
    frames = {"BTCUSDT": (btc, _crypto_1d(), None), "XAUUSD": (gold, gold.copy(), None),
              "SPX500": (spx, spx.copy(), None)}
    seen = []
    _patch_cycle(monkeypatch, frames, seen)
    btc0, gold0 = btc.copy(), gold.copy()
    for sym in ("BTCUSDT", "XAUUSD", "SPX500"):
        main.cmd_signal(sym, notify=True)
    # La detección recibe EXACTAMENTE los mismos datos (el monitor no toca nada).
    assert [s for s, _, _ in seen] == ["BTCUSDT", "XAUUSD", "SPX500"]
    pd.testing.assert_frame_equal(seen[0][1], btc0)
    pd.testing.assert_frame_equal(seen[1][1], gold0)
    # Avisos: BTC (plana) y oro (domingo); SPX está en modo sombra → no avisa.
    assert [s for s, _ in sent] == ["BTCUSDT", "XAUUSD"]
    assert "fin de semana" in sent[1][1][0] and "domingo" in sent[1][1][0]
    with temp_db.get_engine().connect() as c:
        rows = c.execute(text("SELECT symbol, check_name, ts, grave FROM data_quality ORDER BY id")).fetchall()
    assert [tuple(r) for r in rows] == [("BTCUSDT", "flat", U("2026-06-16 16:00"), 1),
                                        ("XAUUSD", "weekend", U("2026-11-15"), 1),
                                        ("SPX500", "flat", U("2026-11-13"), 1)]
    # Si el monitor revienta, la señal sigue su camino igual.
    import monitoring.data_monitor as dm
    monkeypatch.setattr(dm, "cycle_incidents", lambda *a: 1 / 0)
    main.cmd_signal("BTCUSDT", notify=True)
    monkeypatch.setattr(dm, "run_cycle_checks", lambda *a, **k: 1 / 0)
    main.cmd_signal("BTCUSDT", notify=True)
    assert [s for s, _, _ in seen][-2:] == ["BTCUSDT", "BTCUSDT"]


def test_coste_por_ciclo_menor_de_50_ms():
    """Tamaño real del ciclo vivo: 4H de 300 días (1.800 velas) + diario de 900."""
    from monitoring.data_monitor import cycle_incidents
    df4 = _crypto_4h(n=1800)
    d1 = _crypto_1d(n=900)
    cycle_incidents("BTCUSDT", df4, d1)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        cycle_incidents("BTCUSDT", df4, d1)
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) < 0.050


def test_monitor_no_importa_ml_ni_scipy():
    code = ("import sys; sys.path.insert(0, %r); import monitoring.data_monitor, monitoring.daily_summary; "
            "print(sorted(m for m in ('lightgbm', 'sklearn', 'scipy') if m in sys.modules))" % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stderr


# ---------------------------------------------------------------------------
# Nocturno: contraste OKX vs Gate.io
# ---------------------------------------------------------------------------

def test_compare_closes_umbral_mediana_y_maximo():
    from monitoring.data_monitor import compare_closes
    idx = pd.date_range("2026-11-14", periods=12, freq="4h", tz="UTC")
    gate = pd.Series(np.linspace(90000, 91000, 12), index=idx)
    assert compare_closes(gate * 1.0003, gate)["status"] == "ok"
    r = compare_closes(gate * 1.006, gate)                       # mediana 0,6 % > 0,5 %
    assert r["status"] == "alert" and abs(r["median"] - 0.006) < 1e-9
    one = gate.copy()
    one.iloc[5] *= 1.025                                         # una vela > 2 %
    r = compare_closes(one, gate)
    assert r["status"] == "alert" and r["max_ts"] == int(idx[5].timestamp()) and r["median"] == 0
    assert compare_closes(gate.iloc[:5], gate)["status"] == "insufficient"


class _FakeOKX:
    def __init__(self, closes, unconfirmed_last=True):
        self.closes, self.unconfirmed_last, self.calls = closes, unconfirmed_last, []

    def publicGetMarketCandles(self, params):
        self.calls.append(params)
        rows = []
        items = list(self.closes.items())
        for k, (ts, c) in enumerate(reversed(items)):            # OKX: más reciente primero
            conf = "0" if (k == 0 and self.unconfirmed_last) else "1"
            rows.append([str(int(ts.timestamp() * 1000)), str(c), str(c), str(c), str(c), "1", "1", "1", conf])
        return {"code": "0", "data": rows}


def test_contraste_con_gateio_de_la_tabla_y_okx_falso(temp_db, monkeypatch):
    import monitoring.data_monitor as dm
    now = pd.Timestamp("2026-11-16 03:00", tz="UTC")
    idx = pd.date_range(end=pd.Timestamp("2026-11-16 00:00", tz="UTC"), periods=15, freq="4h")
    gate = _ohlcv(idx, start=90000.0, seed=5)
    temp_db.save_ohlcv(gate, "BTCUSDT", "4h", source="gateio")
    temp_db.save_ohlcv(gate * 1.5, "BTCUSDT", "4h", source="okx")   # filas OKX no cuentan (UNIQUE)
    okx_closes = pd.Series(gate["close"].to_numpy() * 1.0004, index=idx)
    okx_closes.iloc[-1] = 1.0                                       # 00:00 sin confirmar: se ignora
    fake = _FakeOKX(okx_closes)
    monkeypatch.setattr(dm, "_okx_exchange", lambda: fake)
    # 04:00:20: la vela de las 00:00 ya terminó por reloj pero OKX aún no la confirma.
    r = dm.cross_source_check("BTCUSDT", now=pd.Timestamp("2026-11-16 04:00:20", tz="UTC"))
    assert r["status"] == "ok" and r["n"] == 12
    fake.calls.clear()
    r = dm.cross_source_check("BTCUSDT", now=now)
    assert r["status"] == "ok" and r["n"] == 12 and r["gate_from"] == "tabla ohlcv"
    assert abs(r["median"] - 0.0004) < 1e-6
    assert fake.calls == [{"instId": "BTC-USDT-SWAP", "bar": "4H", "limit": "14"}]   # UNA petición
    # Gate.io con un precio distinto → alerta.
    okx_bad = okx_closes.copy()
    okx_bad.iloc[-3] *= 1.03
    monkeypatch.setattr(dm, "_okx_exchange", lambda: _FakeOKX(okx_bad))
    assert dm.cross_source_check("BTCUSDT", now=now)["status"] == "alert"
    # OKX caído → status error, sin lanzar.
    def _down():
        raise TimeoutError("okx timeout 10000 ms")
    monkeypatch.setattr(dm, "_okx_exchange", _down)
    r = dm.cross_source_check("BTCUSDT", now=now)
    assert r["status"] == "error" and "TimeoutError" in r["error"]


def test_contraste_sin_tabla_usa_una_peticion_a_gateio(temp_db, monkeypatch):
    import monitoring.data_monitor as dm
    now = pd.Timestamp("2026-11-16 03:00", tz="UTC")
    idx = pd.date_range(end=pd.Timestamp("2026-11-16 00:00", tz="UTC"), periods=15, freq="4h")
    closes = pd.Series(np.linspace(3000, 3100, 15), index=idx)

    class _FakeGate:
        calls = []

        def publicFuturesGetSettleCandlesticks(self, params):
            self.calls.append(params)
            return [{"t": int(t.timestamp()), "c": str(c)} for t, c in closes.items()]
    g = _FakeGate()
    monkeypatch.setattr(dm, "_gate_exchange", lambda: g)
    monkeypatch.setattr(dm, "_okx_exchange", lambda: _FakeOKX(closes.iloc[:-1]))
    r = dm.cross_source_check("ETHUSDT", now=now)
    assert r["gate_from"] == "petición a Gate.io" and r["status"] == "ok" and r["n"] == 12
    assert len(g.calls) == 1 and g.calls[0]["contract"] == "ETH_USDT"


def test_exchanges_reales_con_timeout_10s(monkeypatch):
    """Sin red: solo se instancian (ccxt no conecta hasta pedir algo)."""
    import importlib
    import monitoring.data_monitor as dm
    real = importlib.reload(dm)
    try:
        assert real._okx_exchange().timeout == 10000
        assert real._gate_exchange().timeout == 10000
    finally:
        importlib.reload(dm)


# ---------------------------------------------------------------------------
# Nocturno: deriva del detector
# ---------------------------------------------------------------------------

def _shadow_rows(db, symbol, n, approved, start, created):
    with db.get_engine().begin() as c:
        for k in range(n):
            c.execute(text(
                "INSERT INTO shadow_setups (symbol, setup_ts, direction, approved, motivo, entry, sl, tp, "
                "created_ts) VALUES (:s, :t, 'long', :a, 'below_threshold', 1, 0.9, 1.1, :c)"),
                {"s": symbol, "t": start + k * 14400, "a": int(k < approved), "c": created})


def test_deriva_niveles_y_sin_datos_suficientes(temp_db):
    from config.settings import DRIFT_BASELINE, active_assets
    from monitoring.data_monitor import drift_status
    for sym in active_assets():
        for m in ("setups", "approved"):
            p = DRIFT_BASELINE[sym][m]
            assert p["p1"] <= p["p5"] <= p["p50"] <= p["p95"] <= p["p99"]
    now = pd.Timestamp("2026-12-20 03:00", tz="UTC")
    # Menos de 30 días de registro sombra → sin datos suficientes.
    _shadow_rows(temp_db, "BTCUSDT", 4, 1, U("2026-12-10"), created=U("2026-12-01"))
    assert drift_status("BTCUSDT", now=now)["status"] == "insufficient"
    # Con 30+ días: 4 setups y 1 aprobado = normal (p50 de BTC).
    _shadow_rows(temp_db, "ETHUSDT", 4, 1, U("2026-12-10"), created=U("2026-11-01"))
    d = drift_status("ETHUSDT", now=now)
    assert (d["status"], d["setups"], d["approved"]) == ("ok", 4, 1)
    # Setups viejos (> 30 días) no cuentan; 10 setups (p95 9 < 10 <= p99 11) → nota.
    _shadow_rows(temp_db, "ETHUSDT", 3, 0, U("2026-11-02"), created=U("2026-11-01"))
    _shadow_rows(temp_db, "ETHUSDT", 6, 0, U("2026-12-15"), created=U("2026-11-01"))
    d = drift_status("ETHUSDT", now=now)
    assert (d["status"], d["setups"], d["level_setups"]) == ("note", 10, "note")
    # 4 aprobados en 30 días (p99 de ETH = 3) → alerta.
    _shadow_rows(temp_db, "ETHUSDT", 3, 3, U("2026-12-18"), created=U("2026-11-01"))
    d = drift_status("ETHUSDT", now=now)
    assert d["status"] == "alert" and d["approved"] == 4 and d["level_approved"] == "alert"
    assert drift_status("XAGUSD", now=now)["status"] == "no_baseline"


def test_nocturno_guarda_contraste_y_avisa_lo_grave(temp_db, monkeypatch):
    import monitoring.data_monitor as dm
    import notifications.telegram as tg
    from data.database import kv_get
    sent = []
    monkeypatch.setattr(tg, "send_data_quality_alert", lambda s, items: sent.append((s, items)) or True)
    now = pd.Timestamp("2026-12-20 03:00", tz="UTC")
    monkeypatch.setattr(dm, "cross_source_check", lambda sym, now=None: {
        "symbol": sym, "status": "alert" if sym == "BTCUSDT" else "ok", "n": 12,
        "median": 0.007 if sym == "BTCUSDT" else 0.0002, "max": 0.01, "checked_ts": int(now.timestamp())})
    _shadow_rows(temp_db, "XAUUSD", 6, 3, U("2026-12-01"), created=U("2026-10-01"))   # > p99 (5 y 2)
    _shadow_rows(temp_db, "SPX500", 9, 3, U("2026-12-01"), created=U("2026-10-01"))   # sombra: no avisa
    rep = dm.run_nightly_checks(notify=True, now=now)
    assert json.loads(kv_get(dm.KV_CROSS_PREFIX + "BTCUSDT"))["status"] == "alert"
    assert json.loads(kv_get(dm.KV_CROSS_PREFIX + "ETHUSDT"))["status"] == "ok"
    assert rep["drift"]["XAUUSD"]["status"] == "alert" and rep["drift"]["SPX500"]["status"] == "alert"
    assert rep["drift"]["BTCUSDT"]["status"] == "insufficient"
    assert sorted(s for s, _ in sent) == ["BTCUSDT", "XAUUSD"]
    assert "no coinciden" in dict(sent)["BTCUSDT"][0] and "0,70 %" in dict(sent)["BTCUSDT"][0]
    # Purga de data_quality: > 90 días fuera.
    with temp_db.get_engine().begin() as c:
        c.execute(text("INSERT INTO data_quality (symbol, ts, check_name, created_ts) VALUES "
                       "('BTCUSDT', 1, 'flat', :t)"), {"t": U("2026-08-01")})
    dm.run_nightly_checks(notify=False, now=now)
    with temp_db.get_engine().connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM data_quality")).scalar() == 0


def test_nocturno_nunca_lanza(temp_db, monkeypatch):
    import monitoring.data_monitor as dm
    monkeypatch.setattr(dm, "cross_source_check", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(dm, "drift_status", lambda *a, **k: 1 / 0)
    rep = dm.run_nightly_checks(notify=True)
    assert rep["alerts"] == []


# ---------------------------------------------------------------------------
# Resumen nocturno: bloque "Calidad de datos"
# ---------------------------------------------------------------------------

def test_bloque_calidad_de_datos_en_el_resumen(temp_db, monkeypatch):
    import monitoring.data_monitor as dm
    from monitoring.daily_summary import build_daily_summary
    now = pd.Timestamp("2026-12-20 03:10", tz="UTC")
    dm.record_incidents("BTCUSDT", [
        {"check": "flat", "ts": pd.Timestamp("2026-12-19 16:00", tz="UTC"), "detalle": "x", "grave": True},
        {"check": "gap", "ts": pd.Timestamp("2026-12-19 08:00", tz="UTC"), "detalle": "y", "grave": False}],
        now=now - pd.Timedelta(hours=3))
    dm.record_incidents("ETHUSDT", [{"check": "flat", "ts": 1, "detalle": "vieja", "grave": True}],
                        now=now - pd.Timedelta(days=3))           # fuera de las 24 h
    temp_db.kv_set(dm.KV_CROSS_PREFIX + "BTCUSDT", json.dumps(
        {"status": "ok", "n": 12, "median": 0.00021, "max": 0.0006, "checked_ts": int(now.timestamp()) - 300}))
    temp_db.kv_set(dm.KV_CROSS_PREFIX + "ETHUSDT", json.dumps(
        {"status": "alert", "n": 12, "median": 0.0071, "max": 0.03, "checked_ts": int(now.timestamp()) - 300}))
    _shadow_rows(temp_db, "BTCUSDT", 4, 1, U("2026-12-10"), created=U("2026-11-01"))
    msg = build_daily_summary(now=now, symbols=["BTCUSDT", "ETHUSDT", "XAUUSD", "SPX500"])
    blk = msg.split("<b>Calidad de datos</b>")[1].split("<b>Resultados")[0]
    assert "• BTCUSDT: ⚠️ 2 incidencia(s) en 24 h: vela plana, hueco (1 grave(s))" in blk
    assert "OKX vs Gate.io 0,021 % (máx 0,060 %, 12 velas) ✓" in blk
    assert "setups 30 d: 4 (normal 1–10), aprobados 1 (normal 0–4) ✓" in blk
    assert "• ETHUSDT: sin incidencias en 24 h; OKX vs Gate.io 0,71 % (máx 3,00 %, 12 velas) ⚠️ NO COINCIDEN" in blk
    assert "sin datos suficientes (&lt; 30 días de registro sombra)" in blk
    assert "Oro y SPX: sin fuente gratuita alternativa fiable" in blk
    assert len(msg) < 4000
    # Contraste viejo (> 36 h) o ausente.
    temp_db.kv_set(dm.KV_CROSS_PREFIX + "BTCUSDT", json.dumps({"status": "ok", "checked_ts": 1}))
    msg = build_daily_summary(now=now, symbols=["BTCUSDT"])
    assert "sin contraste reciente" in msg
    # Si el bloque falla, el resto del resumen sale igual.
    monkeypatch.setattr(dm, "quality_data", lambda *a, **k: 1 / 0)
    msg = build_daily_summary(now=now, symbols=["BTCUSDT"])
    assert "<b>Calidad de datos</b>: no disponible (ZeroDivisionError)" in msg and "Resultados" in msg
