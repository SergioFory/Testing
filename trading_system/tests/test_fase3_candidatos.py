"""
Pruebas de la Fase 3, punto 3.3 — candidatos en sombra y evaluación con la puerta:
  - signals/context_features.py: sma200_with SIN look-ahead (cripto 4H usa el diario
    del día anterior; oro/SPX el del propio setup), mapeo sobre el índice completo,
    tsmom, atr_rank, dist_atr_now con signo, rv30/vrp, null si falta algo;
  - filtro candidato "setup_filter": "sma200_with" en detect_all_setups (y filtro
    desconocido = error);
  - data/fetchers/deribit_fetcher.py: timeout 5 s, fail-silent, caché del ciclo,
    sin reintentar la otra moneda si Deribit no responde, valores absurdos = null;
  - columna `features` de shadow_setups: migración idempotente (también sobre una
    tabla antigua), se guarda, no se sobrescribe y se rellena si faltaba;
  - el ciclo vivo registra las features y, si su cálculo falla, la fila se guarda
    igual y la señal no cambia;
  - puerta: "filter" en el candidato (validación, hash estable que lo incluye, el
    candidato sin filtro conserva su hash, re-detección con setup_filter y criterio
    a) con el filtro aplicado).

    cd trading_system && python -m pytest tests/test_fase3_candidatos.py -q

Sin red: requests.get/post y Session.request fallan si alguien intenta usarlos;
Deribit se sustituye por dobles; Telegram doble; BD SQLite temporal.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signals import context_features as CF          # noqa: E402
from data.fetchers import deribit_fetcher as DF     # noqa: E402

UTC = "UTC"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(requests.Session, "request", _boom)
    DF.clear_cycle_cache()
    yield
    DF.clear_cycle_cache()


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


def _rows(db, table, where="1=1"):
    with db.get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(f"SELECT * FROM {table} WHERE {where} ORDER BY id"))
                .mappings().fetchall()]


def _daily(closes, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1D", tz=UTC)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0},
                        index=idx)


def _intraday_from_daily(daily):
    """4H con 6 velas por día al precio de cierre de ese día."""
    idx = pd.date_range(daily.index[0], daily.index[-1] + pd.Timedelta(hours=20), freq="4h", tz=UTC)
    c = daily["close"].reindex(idx.floor("D")).to_numpy()
    rng = np.random.default_rng(1)
    c = c * (1 + rng.normal(0, 0.002, len(idx)))
    return pd.DataFrame({"open": c, "high": c * 1.004, "low": c * 0.996, "close": c, "volume": 1.0},
                        index=idx)


# ---------------------------------------------------------------------------
# sma200_with: dirección, None sin SMA200 y sin look-ahead
# ---------------------------------------------------------------------------

def test_sma200_with_direccion_y_none_sin_historia():
    d = _daily(np.linspace(100, 200, 260))           # subida: cierre > SMA200 desde el día 200
    d4 = _intraday_from_daily(d)
    ts_ok = d4.index[-3]
    ts_early = d4.index[6 * 50]                      # día 50: aún no hay SMA200
    s = pd.DataFrame({"ts": [ts_ok, ts_ok, ts_early], "direction": ["long", "short", "long"]})
    assert CF.sma200_with(s, d4, d) == [True, False, None]
    # El filtro quita solo el short en contra; conserva el que no tiene SMA200.
    kept = CF.apply_setup_filter(s, "sma200_with", d4, d)
    assert list(kept["direction"]) == ["long", "long"]
    with pytest.raises(ValueError):
        CF.apply_setup_filter(s, "sma50_with", d4, d)


def test_sma200_cripto_usa_el_diario_del_dia_anterior():
    """Vela 4H del día D: cambiar el cierre diario de D NO cambia nada (no ha cerrado);
    cambiar el de D−1 sí puede cambiarlo."""
    base = np.full(260, 100.0)
    base[-2] = 50.0                                  # D−1 muy por debajo de la SMA200
    d = _daily(base)
    d4 = _intraday_from_daily(d)
    ts = d4.index[-2]                                # vela 4H del último día D
    s = pd.DataFrame({"ts": [ts], "direction": ["short"]})
    assert CF.sma200_with(s, d4, d) == [True]        # usa D−1 (50 < SMA200)
    d2 = d.copy()
    d2.iloc[-1, d2.columns.get_loc("close")] = 1e6   # el diario de D (futuro para esa vela)
    assert CF.sma200_with(s, d4, d2) == [True]
    d3 = d.copy()
    d3.iloc[-2, d3.columns.get_loc("close")] = 1e6   # D−1 por encima → ya no está a favor del short
    assert CF.sma200_with(s, d4, d3) == [False]


def test_sma200_diarios_usa_el_mismo_dia_y_mapea_sobre_el_indice_completo():
    """Oro/SPX: la vela del setup es el diario (cerrado al evaluarse) → mismo día.
    Cripto con setups SUELTOS (cada varios días) sigue siendo intradía: el mapeo se
    hace sobre el índice completo de velas, no sobre los setups."""
    base = np.full(260, 100.0)
    base[-1] = 50.0
    d = _daily(base)
    s = pd.DataFrame({"ts": [d.index[-1]], "direction": ["short"]})
    assert CF.sma200_with(s, d, d) == [True]         # base diaria: el propio día
    d4 = _intraday_from_daily(_daily(np.full(260, 100.0)))
    d1 = _daily(base)
    sparse = pd.DataFrame({"ts": [d4.index[6 * 230], d4.index[6 * 245], d4.index[-1]],
                           "direction": ["short"] * 3})
    # La última vela 4H del día D no ve el 50 del diario D (aún abierto): no está a favor.
    assert CF.sma200_with(sparse, d4, d1)[-1] is False


# ---------------------------------------------------------------------------
# compute_setup_features
# ---------------------------------------------------------------------------

def test_features_contenido_y_signos():
    d = _daily(np.linspace(100, 300, 400))
    d4 = _intraday_from_daily(d)
    ts = d4.index[-1]
    entry, atr = float(d4["close"].iloc[-1]), 2.0
    w = pd.DataFrame({"ts": [ts, ts], "direction": ["long", "short"],
                      "close": [entry, entry], "atr": [atr, atr]})
    f = CF.compute_setup_features(w, d4, d, current_price=entry + 3.0, dvol=60.0, with_vol=True)
    fl = json.loads(f[(ts, "long")])
    fs = json.loads(f[(ts, "short")])
    assert fl["sma200_with"] is True and fs["sma200_with"] is False
    assert fl["tsmom"] == [1, 1, 1] and fs["tsmom"] == [1, 1, 1]      # signo crudo, no alineado
    assert fl["dist_atr_now"] == pytest.approx(1.5) and fs["dist_atr_now"] == pytest.approx(-1.5)
    assert 0 < fl["atr_rank"] <= 1
    lr = np.log(d4["close"]).diff()
    rv = lr.iloc[-180:].std() * np.sqrt(6 * 365)
    assert fl["rv30"] == pytest.approx(rv, rel=1e-3)
    assert fl["vrp"] == pytest.approx(0.60 / rv, rel=1e-3) and fl["dvol"] == 60.0
    # Sin DVOL (Deribit caído): dvol y vrp null, lo demás igual.
    f2 = json.loads(CF.compute_setup_features(w, d4, d, entry, dvol=None, with_vol=True)[(ts, "long")])
    assert f2["dvol"] is None and f2["vrp"] is None and f2["rv30"] is not None
    # Oro/SPX: sin claves de volatilidad implícita.
    f3 = json.loads(CF.compute_setup_features(w, d4, d, entry, with_vol=False)[(ts, "long")])
    assert set(f3) == {"sma200_with", "tsmom", "atr_rank", "dist_atr_now"}
    assert "," in f[(ts, "long")] and ", " not in f[(ts, "long")]      # JSON compacto


def test_features_una_variable_rota_no_tumba_las_demas(monkeypatch):
    d = _daily(np.linspace(100, 300, 400))
    d4 = _intraday_from_daily(d)
    ts = d4.index[-1]
    w = pd.DataFrame({"ts": [ts], "direction": ["long"], "close": [1.0], "atr": [None]})
    monkeypatch.setattr(CF, "_atr_rank_on_bars", lambda *_: (_ for _ in ()).throw(RuntimeError("x")))
    f = json.loads(CF.compute_setup_features(w, d4, d, 1.0)[(ts, "long")])
    assert f["atr_rank"] is None and f["dist_atr_now"] is None and f["sma200_with"] is True
    assert CF.compute_setup_features(w.iloc[0:0], d4, d, 1.0) == {}


# ---------------------------------------------------------------------------
# Filtro candidato en el detector
# ---------------------------------------------------------------------------

def test_detector_aplica_setup_filter_solo_si_esta_configurado(monkeypatch):
    import signals.detector as det
    from backtest.protocol import config_override
    d = _daily(np.linspace(100, 300, 400))
    d4 = _intraday_from_daily(d)
    marks = d4.index[[-30, -20, -10]]

    def _fake_breakouts(df, params=None):
        out = df.copy()
        out["atr"] = 1.0
        out["breakout_long"] = out.index.isin(marks[:2])
        out["breakout_short"] = out.index == marks[2]
        out["breakout_dir"] = np.where(out["breakout_long"], "long",
                                       np.where(out["breakout_short"], "short", None))
        out["breakout_score"] = 0.9
        return out
    monkeypatch.setattr(det, "detect_breakouts", _fake_breakouts)
    s0 = det.detect_all_setups(d4, d, "BTCUSDT")
    assert len(s0) == 3                                   # sin filtro: como hoy
    with config_override("BTCUSDT", {"setup_filter": "sma200_with"}):
        s1 = det.detect_all_setups(d4, d, "BTCUSDT")
    assert list(s1["direction"]) == ["long", "long"]      # el short va contra la SMA200
    from config.settings import ASSETS
    assert "setup_filter" not in ASSETS["BTCUSDT"]       # se restaura


# ---------------------------------------------------------------------------
# Deribit
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload


def _ok_get(calls, value=45.5):
    def _get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return _Resp({"result": {"data": [[1, 40, 41, 39, 40.0], [2, 40, 46, 40, value]]}})
    return _get


def test_deribit_valor_timeout_y_cache_del_ciclo():
    calls = []
    assert DF.fetch_dvol("BTC", http_get=_ok_get(calls)) == 45.5
    assert calls[0]["timeout"] == 5.0
    assert calls[0]["params"]["currency"] == "BTC" and calls[0]["params"]["resolution"] == "3600"
    assert "get_volatility_index_data" in calls[0]["url"]
    assert DF.fetch_dvol("BTC", http_get=_ok_get(calls)) == 45.5
    assert len(calls) == 1                                  # una petición por moneda y ciclo
    assert DF.dvol_for("ETHUSDT", http_get=_ok_get(calls, 60.0)) == 60.0 and len(calls) == 2
    assert DF.dvol_for("XAUUSD", http_get=_ok_get(calls)) is None and len(calls) == 2
    DF.clear_cycle_cache()
    DF.fetch_dvol("BTC", http_get=_ok_get(calls))
    assert len(calls) == 3                                  # ciclo nuevo → petición nueva


def test_deribit_fail_silent_y_circuito_del_ciclo():
    import requests
    calls = []

    def _timeout(url, params=None, timeout=None):
        calls.append(params["currency"])
        raise requests.exceptions.ConnectTimeout("sin respuesta")
    assert DF.fetch_dvol("BTC", http_get=_timeout) is None
    assert DF.fetch_dvol("ETH", http_get=_timeout) is None
    assert calls == ["BTC"]                                # no se reintenta la otra moneda
    DF.clear_cycle_cache()

    def _proxy(url, params=None, timeout=None):
        calls.append(params["currency"])
        raise requests.exceptions.ProxyError("502")
    calls.clear()
    DF.fetch_dvol("BTC", http_get=_proxy)
    DF.fetch_dvol("ETH", http_get=_proxy)
    assert calls == ["BTC"]
    DF.clear_cycle_cache()
    # Respuestas raras: HTTP 500, sin datos, JSON roto, valor absurdo → None sin lanzar.
    assert DF.fetch_dvol("BTC", http_get=lambda *a, **k: _Resp({}, status=500)) is None
    DF.clear_cycle_cache()
    assert DF.fetch_dvol("BTC", http_get=lambda *a, **k: _Resp({"result": {"data": []}})) is None
    DF.clear_cycle_cache()
    assert DF.fetch_dvol("BTC", http_get=lambda *a, **k: _Resp({"result": None})) is None
    DF.clear_cycle_cache()
    assert DF.fetch_dvol("BTC", http_get=_ok_get([], value=0.0)) is None
    DF.clear_cycle_cache()
    # Sin doble: requests.get está bloqueado en las pruebas → None, sin excepción.
    assert DF.fetch_dvol("BTC") is None


def test_deribit_limite_de_tiempo_total(monkeypatch):
    """Revisión adversarial Fase 3: el timeout de requests es por lectura; una
    respuesta que llega byte a byte podía tardar minutos. Ahora la petición entera
    tiene un límite total (timeout_s) y, si vence, la otra moneda no se pide."""
    import time
    import config.settings as st
    monkeypatch.setitem(st.DERIBIT_DVOL, "timeout_s", 0.3)
    calls = []

    class _SlowBody(_Resp):
        def json(self):                       # el cuerpo llega muy despacio
            time.sleep(3.0)
            return {"result": {"data": [[1, 2, 3, 4, 55.5]]}}

    def _drip(url, params=None, timeout=None):
        calls.append((params["currency"], timeout))
        return _SlowBody({})
    t0 = time.monotonic()
    assert DF.fetch_dvol("BTC", http_get=_drip) is None
    assert DF.fetch_dvol("ETH", http_get=_drip) is None
    took = time.monotonic() - t0
    assert took < 1.5, took                   # antes: ~3 s por moneda (aquí) / 150 s (servidor real)
    assert calls == [("BTC", 0.3)]            # la segunda moneda no se pide en este ciclo
    DF.clear_cycle_cache()

    # Conexión colgada (la llamada misma no vuelve): mismo límite.
    def _hang(url, params=None, timeout=None):
        calls.append((params["currency"], timeout))
        time.sleep(3.0)
        return _Resp({"result": {"data": [[1, 2, 3, 4, 55.5]]}})
    calls.clear()
    t0 = time.monotonic()
    assert DF.dvol_for("ETHUSDT", http_get=_hang) is None
    assert time.monotonic() - t0 < 1.0
    DF.clear_cycle_cache()

    # Una respuesta normal dentro del límite sigue funcionando (y por el hilo).
    assert DF.fetch_dvol("BTC", http_get=_ok_get([])) == 45.5


def test_deribit_apagado_no_pide_nada(monkeypatch):
    import config.settings as st
    monkeypatch.setitem(st.DERIBIT_DVOL, "enabled", False)
    calls = []
    assert DF.fetch_dvol("BTC", http_get=_ok_get(calls)) is None and calls == []


# ---------------------------------------------------------------------------
# Columna features de shadow_setups
# ---------------------------------------------------------------------------

def _row(ts, **kw):
    base = {"setup_ts": ts, "direction": "long", "setup_type": "breakout", "raw_score": 0.6,
            "approved": 1, "motivo": "not_best", "entry": 100.0, "sl": 97.0, "tp": 104.5,
            "atr": 2.0, "keep_existing": False}
    base.update(kw)
    return base


def test_features_migracion_guardado_y_no_sobrescritura(temp_db):
    from data.shadow import record_shadow_setups
    with temp_db.get_engine().connect() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(shadow_setups)")).fetchall()}
    assert "features" in cols
    temp_db.init_db(force=True)
    temp_db.init_db(force=True)
    ts = pd.Timestamp("2026-11-15 04:00", tz=UTC)
    record_shadow_setups("BTCUSDT", [_row(ts, features='{"dvol":40.0}')])
    record_shadow_setups("BTCUSDT", [_row(ts, motivo="emitted", features='{"dvol":99.0}')])
    r = _rows(temp_db, "shadow_setups")
    assert len(r) == 1 and r[0]["features"] == '{"dvol":40.0}' and r[0]["motivo"] == "emitted"
    # Una fila sin features (cálculo fallido) se rellena en el ciclo siguiente.
    ts2 = ts + pd.Timedelta(hours=4)
    record_shadow_setups("BTCUSDT", [_row(ts2)])
    record_shadow_setups("BTCUSDT", [_row(ts2, features='{"dvol":41.0}')])
    assert _rows(temp_db, "shadow_setups", "setup_ts = %d" % int(ts2.timestamp()))[0]["features"] \
        == '{"dvol":41.0}'


def test_features_migracion_sobre_tabla_sombra_antigua(tmp_path, monkeypatch):
    import sqlalchemy
    url = f"sqlite:///{tmp_path / 'old.db'}"
    eng = sqlalchemy.create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE shadow_setups (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                          "symbol TEXT NOT NULL, setup_ts INTEGER NOT NULL, direction TEXT NOT NULL, "
                          "setup_type TEXT, raw_score REAL, approved INTEGER, motivo TEXT NOT NULL, "
                          "entry REAL, sl REAL, tp REAL, atr REAL, created_ts INTEGER, resultado TEXT, "
                          "pnl_r REAL, exit_price REAL, resolved_ts INTEGER, portfolio_flag TEXT, "
                          "UNIQUE(symbol, setup_ts, direction))"))
        conn.execute(text("INSERT INTO shadow_setups (symbol, setup_ts, direction, motivo) "
                          "VALUES ('BTCUSDT', 1, 'long', 'emitted')"))
    eng.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    import data.database as db
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    db.init_db()
    rows = _rows(db, "shadow_setups")
    assert len(rows) == 1 and rows[0]["features"] is None
    db._engine.dispose()


def test_classify_window_pasa_features_por_clave():
    from data.shadow import classify_window
    ts = pd.Timestamp("2026-11-15 04:00", tz=UTC)
    w = pd.DataFrame([{"ts": ts, "direction": "long", "setup_type": "breakout", "raw_score": 0.4,
                       "ml_approved": False, "close": 100.0, "stop_loss": 97.0,
                       "take_profit": 104.5, "atr": 2.0},
                      {"ts": ts, "direction": "short", "setup_type": "breakout", "raw_score": 0.4,
                       "ml_approved": False, "close": 100.0, "stop_loss": 103.0,
                       "take_profit": 95.5, "atr": 2.0}])
    rows = classify_window(w, set(), 100.0, 1.0, features={(ts, "long"): '{"a":1}'})
    assert [r["features"] for r in rows] == ['{"a":1}', None]


# ---------------------------------------------------------------------------
# Ciclo vivo: registra features; si fallan, todo sigue igual
# ---------------------------------------------------------------------------

def _cycle(monkeypatch, feature_fn=None, dvol=50.0):
    """cmd_signal con datos y detección simulados (como en test_fase3_cartera)."""
    import main
    import signals.detector as det
    import ml.predictor as pred
    import data.bars as bars
    import monitoring.data_monitor as dm
    import notifications.telegram as tg
    d1 = _daily(np.linspace(100, 300, 400))
    df = _intraday_from_daily(d1)
    df.attrs["source"] = "test"
    last = df.index[-1]
    c = float(df["close"].iloc[-1])
    setups = pd.DataFrame([{"ts": last, "symbol": "BTCUSDT", "setup_type": "breakout",
                            "direction": "long", "raw_score": 0.7, "atr": c * 0.01, "close": c,
                            "stop_loss": c * 0.985, "take_profit": c * 1.0225}])
    monkeypatch.setattr(main, "_load_data", lambda s, closed_only=True, history="full": (df, d1, None))
    monkeypatch.setattr(bars, "last_bar_age_hours", lambda *a, **k: 1.0)
    monkeypatch.setattr(det, "detect_all_setups", lambda *a, **k: setups.copy())
    monkeypatch.setattr(pred, "score_setups",
                        lambda d, *a, **k: d.assign(ml_score=d["raw_score"],
                                                    ml_approved=d["raw_score"] >= 0.55))
    monkeypatch.setattr(dm, "run_cycle_checks", lambda *a, **k: [])
    sent = []
    monkeypatch.setattr(tg, "_send", lambda text, parse_mode="HTML", retries=3: sent.append(text) or True)
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "TEST-TOKEN")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "4242")
    monkeypatch.setattr(DF, "dvol_for", lambda symbol, http_get=None: dvol)
    if feature_fn is not None:
        monkeypatch.setattr(CF, "compute_setup_features", feature_fn)
    main.cmd_signal("BTCUSDT", notify=True)
    return sent


def test_ciclo_registra_features_con_dvol(temp_db, monkeypatch):
    sent = _cycle(monkeypatch)
    rows = _rows(temp_db, "shadow_setups")
    assert len(rows) == 1 and rows[0]["motivo"] == "emitted"
    f = json.loads(rows[0]["features"])
    assert f["sma200_with"] is True and f["dvol"] == 50.0 and f["vrp"] > 0
    assert f["dist_atr_now"] == 0.0                         # edad 0: precio actual = entrada
    assert "-0.0" not in rows[0]["features"]
    assert len(sent) == 1 and "dvol" not in sent[0].lower()  # el mensaje no cambia


def test_ciclo_si_fallan_las_features_la_fila_y_la_senal_siguen(temp_db, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("features rotas")
    sent = _cycle(monkeypatch, feature_fn=_boom)
    rows = _rows(temp_db, "shadow_setups")
    assert len(rows) == 1 and rows[0]["motivo"] == "emitted" and rows[0]["features"] is None
    assert len(sent) == 1
    assert len(_rows(temp_db, "signals")) == 1


def test_ciclo_con_deribit_caido_guarda_null(temp_db, monkeypatch):
    _cycle(monkeypatch, dvol=None)
    f = json.loads(_rows(temp_db, "shadow_setups")[0]["features"])
    assert f["dvol"] is None and f["vrp"] is None and f["sma200_with"] is True


# ---------------------------------------------------------------------------
# Puerta: "filter" en el candidato
# ---------------------------------------------------------------------------

def test_candidato_filter_validacion_y_hash(tmp_path):
    from backtest import protocol as P
    good = tmp_path / "c.json"
    good.write_text(json.dumps({"symbol": "BTCUSDT", "filter": "sma200_with"}), encoding="utf-8")
    cand = P.load_candidate(good)
    cv = P.candidate_variant("BTCUSDT", cand)
    assert cv["extra"] == {"filter": "sma200_with"} and cv["params"]["extra"] == {"filter": "sma200_with"}
    assert "SMA200" in cv["label"]
    assert P.candidate_variant("BTCUSDT", cand)["hash"] == cv["hash"]          # estable
    plain = P.candidate_variant("BTCUSDT", {"symbol": "BTCUSDT"})
    cur = next(v for v in P.build_grid("BTCUSDT") if v["is_current"])
    assert plain["hash"] == cur["hash"] != cv["hash"]       # sin filtro: el hash de siempre
    # El filtro en la rejilla: el candidato se añade como variante nueva.
    grid = P.build_grid("BTCUSDT", cand)
    assert len(grid) == 16 and grid[-1]["is_candidate"] and grid[-1]["hash"] == cv["hash"]
    assert P.extra_config(cv["extra"]) == ({"setup_filter": "sma200_with"}, None)
    assert P.extra_config({}) == (None, None)
    for bad in ({"symbol": "BTCUSDT", "filter": "sma50"},
                {"symbol": "BTCUSDT", "filter": "sma200_with", "overrides": {"setup_filter": "x"}}):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(P.ProtocolError):
            P.load_candidate(p)


def test_protocolo_con_filtro_redetecta_y_comprueba_a_con_el_filtro(tmp_path, monkeypatch):
    """El candidato con filtro re-detecta con setup_filter y el criterio a) se comprueba
    con setup_filter puesto; la configuración actual, sin él."""
    from backtest import protocol as P
    from config import settings as st
    seen = {"detect": [], "live": []}
    idx = pd.date_range("2022-04-27", "2026-11-15", freq="4h", tz=UTC, inclusive="left")
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    df = pd.DataFrame({"open": close, "high": close * 1.006, "low": close * 0.994,
                       "close": close, "volume": 1000.0}, index=idx)

    def _detect(symbol, df_4h, df_daily=None, overrides=None, params_override=None):
        flt = (overrides or {}).get("setup_filter")
        seen["detect"].append(flt)
        pos = np.arange(30, len(df_4h), 37 if not flt else 74)
        return pd.DataFrame({"ts": df_4h.index[pos],
                             "direction": np.where(np.arange(len(pos)) % 2 == 0, "long", "short"),
                             "setup_type": "breakout", "raw_score": 0.7,
                             "atr": (df_4h["high"] - df_4h["low"]).to_numpy()[pos]})

    def _live(symbol, d4, d1):
        seen["live"].append(st.ASSETS[symbol].get("setup_filter"))
        return {"match": True, "approved": 5, "emitted": 5, "repeated": 0}
    monkeypatch.setattr(P, "detect_universe", _detect)
    cpath = tmp_path / "c.json"
    cpath.write_text(json.dumps({"symbol": "BTCUSDT", "label": "BTC SMA200", "filter": "sma200_with"}),
                     encoding="utf-8")
    code, rep = P.run_protocol(candidate_path=str(cpath), registry=tmp_path / "reg.jsonl",
                               load_data=lambda s: (df, P.resample_daily_utc(df)),
                               live_check=_live, echo=lambda *_: None)
    assert code == 0
    assert seen["live"] == [None, "sma200_with"]            # actual, luego candidato
    assert seen["detect"] == [None, "sma200_with"]
    r = rep["symbols"][0]
    assert "_cand_trades" in r and len(r["_cand_trades"]) < len(r["_trades"])
    assert rep["verdict"]["criteria"]["a"]["ok"] is True
    assert rep["verdict"]["status"] in ("rejected", "pending_final")      # nunca "approved" sin --final
    reg = P.read_registry(tmp_path / "reg.jsonl")
    assert any(e.get("params", {}).get("extra") == {"filter": "sma200_with"} for e in reg)
    assert "setup_filter" not in st.ASSETS["BTCUSDT"]
