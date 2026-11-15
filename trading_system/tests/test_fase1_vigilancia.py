"""
Pruebas de la Fase 1 — vigilancia (puntos 1.12 y 1.4):
  - SPRT (0 frente a +0,25R; σ=1,3; α=0,05; β=0,20) y CUSUM de deterioro
    (k=0,125; h=20R) de monitoring/sequential.py;
  - universo de señales live (quality_flag NULL, setup_ts NOT NULL, resueltas,
    en orden de resolución) y versión sobre shadow_setups aprobados;
  - resumen nocturno desde la BD (monitoring/daily_summary.py);
  - latido HEARTBEAT_URL solo al final de un ciclo correcto (monitoring/heartbeat.py).

    cd trading_system && python -m pytest tests -q

Sin red: requests.get/post fallan si alguien intenta usarlos; BD SQLite temporal.
"""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture()
def no_network(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)


NOW = pd.Timestamp("2026-11-20 03:00", tz="UTC")
U = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())   # noqa: E731


def _signal(db, **kw):
    """Inserta una fila en signals con valores por defecto razonables."""
    row = {"symbol": "BTCUSDT", "ts": U("2026-11-18 08:00"), "setup_type": "breakout",
           "direction": "long", "ml_score": 0.6, "entry_price": 100.0, "stop_loss": 98.0,
           "take_profit": 103.0, "position_size": 1000.0, "atr": 1.3,
           "setup_ts": U("2026-11-18 04:00"), "mode": "live", "resultado_real": None,
           "pnl_r": None, "exit_price": None, "resolved_ts": None, "quality_flag": None}
    row.update(kw)
    cols = ", ".join(row)
    with db.get_engine().begin() as conn:
        conn.execute(text(f"INSERT INTO signals ({cols}) VALUES ({', '.join(':' + c for c in row)})"), row)
        return conn.execute(text("SELECT MAX(id) FROM signals")).scalar()


def _shadow(db, **kw):
    row = {"symbol": "BTCUSDT", "setup_ts": U("2026-11-18 04:00"), "direction": "long",
           "setup_type": "breakout", "raw_score": 0.6, "approved": 1, "motivo": "emitted",
           "entry": 100.0, "sl": 98.0, "tp": 103.0, "atr": 1.3, "created_ts": U("2026-11-19 10:00"),
           "resultado": None, "pnl_r": None, "exit_price": None, "resolved_ts": None}
    row.update(kw)
    cols = ", ".join(row)
    with db.get_engine().begin() as conn:
        conn.execute(text(f"INSERT INTO shadow_setups ({cols}) VALUES ({', '.join(':' + c for c in row)})"), row)


# ---------------------------------------------------------------------------
# SPRT y CUSUM (aritmética)
# ---------------------------------------------------------------------------

def test_sprt_parametros_preregistrados():
    from monitoring.sequential import SPRT_PARAMS, CUSUM_PARAMS, FREEZE_MAX_SIGNALS, sprt_bounds
    assert SPRT_PARAMS == {"mu0": 0.0, "mu1": 0.25, "sigma": 1.3, "alpha": 0.05, "beta": 0.20}
    assert CUSUM_PARAMS == {"k": 0.125, "h": 20.0}
    assert FREEZE_MAX_SIGNALS == 150
    lo, up = sprt_bounds()
    assert lo == pytest.approx(math.log(0.20 / 0.95))   # −1.558
    assert up == pytest.approx(math.log(0.80 / 0.05))   # +2.773


def test_sprt_incremento_normal():
    from monitoring.sequential import sprt_increment
    assert sprt_increment(0.125) == pytest.approx(0.0)
    assert sprt_increment(1.125) == pytest.approx(0.25 / 1.69)
    assert sprt_increment(-1.0) == pytest.approx(0.25 / 1.69 * (-1.125))


def test_sprt_decide_edge_y_es_definitivo():
    from monitoring.sequential import sprt
    inc = 0.25 / 1.69 * (1.5 - 0.125)
    n_needed = math.ceil(math.log(16) / inc)
    res = sprt([1.5] * n_needed)
    assert res.decision == "edge_aceptado" and res.decided_at == n_needed
    assert res.n == n_needed and res.mean_r == pytest.approx(1.5)
    assert sprt([1.5] * (n_needed - 1)).decision == "seguir"
    # Señales malas DESPUÉS del cruce no deshacen la decisión (pero el log-LR baja).
    later = sprt([1.5] * n_needed + [-1.0] * 40)
    assert later.decision == "edge_aceptado" and later.decided_at == n_needed
    assert later.llr < later.lower


def test_sprt_decide_sin_edge():
    from monitoring.sequential import sprt
    inc = 0.25 / 1.69 * (-1.0 - 0.125)
    n_needed = math.ceil(math.log(0.2 / 0.95) / inc)
    res = sprt([-1.0] * n_needed)
    assert res.decision == "sin_edge" and res.decided_at == n_needed
    empty = sprt([])
    assert empty.decision == "seguir" and empty.n == 0 and empty.mean_r is None


def test_cusum_deterioro():
    from monitoring.sequential import cusum
    # Cada −1R suma 1.125 → alarma en la señal ceil(20/1.125) = 18.
    res = cusum([-1.0] * 30)
    assert res.alarm and res.alarm_at == math.ceil(20 / 1.125)
    # Las buenas lo devuelven a 0 (nunca negativo).
    res2 = cusum([-1.0, -1.0, 2.0, 2.0])
    assert res2.path == pytest.approx([1.125, 2.25, 0.375, 0.0])
    assert not res2.alarm and res2.s == 0.0 and res2.max_s == pytest.approx(2.25)
    assert cusum([]).n == 0


# ---------------------------------------------------------------------------
# Universo desde la BD
# ---------------------------------------------------------------------------

def test_universo_live_filtros_y_orden(temp_db):
    from monitoring.sequential import load_live_r, load_shadow_approved_r
    db = temp_db
    # Cuentan, en orden de RESOLUCIÓN (no de id):
    _signal(db, resultado_real="win", pnl_r=1.5, resolved_ts=U("2026-11-19 12:00"))
    _signal(db, resultado_real="loss", pnl_r=-1.0, resolved_ts=U("2026-11-18 20:00"),
            setup_ts=U("2026-11-17 04:00"))
    _signal(db, resultado_real="be", pnl_r=None, resolved_ts=U("2026-11-19 16:00"),
            setup_ts=U("2026-11-16 04:00"))                                    # be sin R → 0
    # No cuentan:
    _signal(db, resultado_real="win", pnl_r=2.0, quality_flag="reemision")    # marca de calidad
    _signal(db, resultado_real="win", pnl_r=2.0, mode="shadow", symbol="SPX500")  # modo sombra
    _signal(db, resultado_real=None)                                           # abierta
    _signal(db, resultado_real="expired", pnl_r=None, resolved_ts=U("2026-11-19 18:00"))  # sin R
    _signal(db, resultado_real="loss", pnl_r=-1.0, setup_ts=None)             # anterior a la Fase 0
    assert load_live_r() == [-1.0, 1.5, 0.0]
    assert load_live_r(phase0=False) == [-1.0]
    assert load_live_r(since_ts=U("2026-11-19 00:00")) == [1.5, 0.0]

    _shadow(db, resultado="win", pnl_r=1.5, resolved_ts=U("2026-11-19 12:00"))
    _shadow(db, setup_ts=U("2026-11-17 04:00"), motivo="not_fresh", resultado="loss", pnl_r=-1.0,
            resolved_ts=U("2026-11-18 12:00"))
    _shadow(db, setup_ts=U("2026-11-16 04:00"), approved=0, motivo="below_threshold",
            resultado="win", pnl_r=1.5, resolved_ts=U("2026-11-17 12:00"))      # no aprobado
    _shadow(db, symbol="SPX500", motivo="shadow_asset", resultado="win", pnl_r=1.67,
            resolved_ts=U("2026-11-18 13:00"))                                  # activo en sombra
    _shadow(db, setup_ts=U("2026-11-15 04:00"), motivo="duplicate")             # sin resolver
    assert load_shadow_approved_r() == [-1.0, 1.5]
    assert load_shadow_approved_r(exclude_motivos=()) == [-1.0, 1.67, 1.5]


def test_estado_vigilancia_y_reinicios(temp_db):
    from monitoring.sequential import monitoring_state, format_monitoring, KV_SINCE, KV_CUSUM_SINCE
    db = temp_db
    for k in range(5):
        _signal(db, resultado_real="loss", pnl_r=-1.0, resolved_ts=U("2026-11-10") + k * 3600,
                setup_ts=U("2026-11-09") + k * 14400)
    _signal(db, resultado_real="win", pnl_r=1.5, resolved_ts=U("2026-11-19"), setup_ts=U("2026-11-18"))
    st = monitoring_state()
    assert st["n_live"] == 6 and st["sprt_live"].n == 6 and st["cusum_live"].n == 6
    txt = format_monitoring(st)
    assert "SPRT señales live" in txt and "CUSUM" in txt and "6/150" in txt
    # Reinicio solo del CUSUM tras una revisión
    db.kv_set(KV_CUSUM_SINCE, U("2026-11-15"))
    st = monitoring_state()
    assert st["sprt_live"].n == 6 and st["cusum_live"].n == 1
    assert "CUSUM reiniciado el 2026-11-15" in format_monitoring(st)
    # Reinicio de todo tras un cambio de configuración
    db.kv_set(KV_SINCE, U("2026-11-18"))
    st = monitoring_state()
    assert st["sprt_live"].n == 1 and st["cusum_live"].n == 1


def test_formato_alarma_y_decisiones():
    from monitoring.sequential import sprt, cusum, format_monitoring
    st = {"sprt_live": sprt([1.5] * 20), "sprt_shadow": sprt([]), "cusum_live": cusum([-1.0] * 18),
          "n_live": 20, "freeze_max": 150, "since": None, "cusum_since": None,
          "pre_phase0": {"n": 8, "mean_r": -0.1}}
    txt = format_monitoring(st)
    assert "EDGE ACEPTADO" in txt and "ALARMA" in txt and "riesgo a la mitad" in txt
    assert "aún sin señales resueltas" in txt and "anteriores a la Fase 0" in txt


# ---------------------------------------------------------------------------
# Resumen nocturno
# ---------------------------------------------------------------------------

def test_resumen_diario_desde_bd(temp_db, monkeypatch):
    from monitoring.daily_summary import build_daily_summary, record_data_status, KV_LAST_CYCLE_OK
    db = temp_db
    record_data_status("BTCUSDT", "gateio", pd.Timestamp("2026-11-19 20:00", tz="UTC"),
                       pd.Timestamp("2026-11-20 00:00", tz="UTC"), now=NOW - pd.Timedelta(minutes=30))
    db.kv_set(KV_LAST_CYCLE_OK, int(NOW.timestamp()) - 1800)
    # 24 h: 3 setups vistos (1 emitido, 2 bajo umbral) + uno viejo que no cuenta
    _shadow(db, setup_ts=U("2026-11-19 12:00"), motivo="emitted", created_ts=U("2026-11-19 16:05"))
    _shadow(db, setup_ts=U("2026-11-19 12:00"), direction="short", approved=0, motivo="below_threshold",
            created_ts=U("2026-11-19 16:05"))
    _shadow(db, setup_ts=U("2026-11-19 16:00"), approved=0, motivo="below_threshold",
            created_ts=U("2026-11-19 20:05"))
    _shadow(db, setup_ts=U("2026-11-10 16:00"), approved=0, motivo="not_fresh",
            created_ts=U("2026-11-10 20:05"))
    _shadow(db, setup_ts=U("2026-11-15 16:00"), motivo="not_fresh", resultado="win", pnl_r=1.5,
            created_ts=U("2026-11-15 20:05"), resolved_ts=U("2026-11-19 08:00"))
    # señales: una emitida hoy y abierta, una resuelta hoy (con fill), una resuelta antes
    open_id = _signal(db, ts=U("2026-11-19 16:05"), setup_ts=U("2026-11-19 12:00"))
    win_id = _signal(db, resultado_real="win", pnl_r=1.5, exit_price=103.0,
                     resolved_ts=U("2026-11-19 22:00"), setup_ts=U("2026-11-17 00:00"),
                     ts=U("2026-11-17 04:05"))
    _signal(db, resultado_real="loss", pnl_r=-1.0, exit_price=98.0, resolved_ts=U("2026-11-15 12:00"),
            setup_ts=U("2026-11-14 00:00"), ts=U("2026-11-14 04:05"))
    db.save_fill(win_id, "fill", price=100.5, ts=U("2026-11-17 05:00"), update_id=1)
    db.save_fill(open_id, "skip", note="tarde", ts=U("2026-11-19 17:00"), update_id=2)

    msg = build_daily_summary(now=NOW, symbols=["BTCUSDT", "SPX500"])
    assert "Resumen diario" in msg and "Último ciclo correcto: hace 30 min" in msg
    assert "gateio" in msg and "última vela 2026-11-19 20:00 UTC" in msg and "cerró hace 3.0 h" in msg
    assert "24 h: 3 setups vistos, 1 señal emitida" in msg
    assert "score bajo el umbral (2 de 2)" in msg
    assert f"Abiertas: 1 — #{open_id} long (cierre por tiempo 11-23 16:00 UTC)" in msg
    assert f"#{win_id} BTCUSDT long: ganada (TP) +1.50R" in msg
    assert "Total del día: <b>+1.50R</b> en 1 señal" in msg
    assert "Aprobados NO emitidos (registro sombra): 1 resuelto(s), +1.50R" in msg
    assert "<b>SPX500</b> <i>(modo sombra: no envía)</i>" in msg and "aún sin registro de ciclos" in msg
    assert "SPRT señales live: <b>seguir</b>. 2 señales" in msg
    # fill a 100.5 (entrada 100, SL 98, salida 103): R ejecutado 1.25 vs 1.5 → −0.25
    assert "ejecutadas (/fill): 1; no ejecutadas (/skip): 1" in msg
    assert "R ejecutado − R teórico (entrada): -0.250R de media en 1 señal(es) resuelta(s)" in msg
    assert len(msg) < 4000


def test_resumen_sin_datos_y_escapado(temp_db):
    from monitoring.daily_summary import build_daily_summary, record_data_status
    record_data_status("BTCUSDT", "gate<io>", problem="la fuente <b>falló</b> & no hay velas")
    msg = build_daily_summary(now=NOW, symbols=["BTCUSDT"])
    assert "gate&lt;io&gt;" in msg and "la fuente &lt;b&gt;falló&lt;/b&gt; &amp; no hay velas" in msg
    assert "Ninguna señal live resuelta" in msg and "aún sin señales resueltas" in msg
    assert "Sin fills registrados" in msg


def test_estado_datos_conserva_ultima_vela_si_falla(temp_db):
    import json
    from monitoring.daily_summary import record_data_status, KV_DATA_PREFIX
    record_data_status("ETHUSDT", "gateio", pd.Timestamp("2026-11-19 20:00", tz="UTC"),
                       pd.Timestamp("2026-11-20 00:00", tz="UTC"))
    record_data_status("ETHUSDT", None, problem="sin datos")
    st = json.loads(temp_db.kv_get(KV_DATA_PREFIX + "ETHUSDT"))
    assert st["source"] == "gateio" and st["problem"] == "sin datos"
    assert st["last_bar_ts"] == U("2026-11-19 20:00") and st["last_bar_close"] == U("2026-11-20 00:00")


def test_cmd_signal_registra_estado_de_datos(temp_db, monkeypatch, no_network):
    import json
    import main
    from monitoring.daily_summary import KV_DATA_PREFIX
    monkeypatch.setattr(main, "_load_data", lambda s, closed_only=True, history="full": (None, None, None))
    main._cycle_fresh.clear()
    main.cmd_signal("BTCUSDT", notify=False)
    st = json.loads(temp_db.kv_get(KV_DATA_PREFIX + "BTCUSDT"))
    assert "no devolvió datos" in st["problem"] and "BTCUSDT" not in main._cycle_fresh


def test_send_daily_report_recorta_y_no_lanza(temp_db, monkeypatch):
    import monitoring.daily_summary as ds
    import notifications.telegram as tg
    sent = []
    monkeypatch.setattr(tg, "send_text", lambda t: sent.append(t) or True)
    monkeypatch.setattr(ds, "build_daily_summary", lambda now=None: "\n".join(["línea <b>x</b>"] * 600))
    assert ds.send_daily_report() is True
    assert len(sent[0]) <= ds.TELEGRAM_MAX_CHARS and sent[0].endswith("python main.py summary)")
    assert sent[0].count("<b>") == sent[0].count("</b>")       # HTML válido: corta por líneas

    def _boom(now=None):
        raise RuntimeError("BD caída")
    monkeypatch.setattr(ds, "build_daily_summary", _boom)
    assert ds.send_daily_report() is False


# ---------------------------------------------------------------------------
# Latido
# ---------------------------------------------------------------------------

def test_heartbeat_sin_url_no_hace_nada(monkeypatch, no_network):
    import config.settings as cs
    from monitoring.heartbeat import ping_heartbeat
    monkeypatch.setattr(cs, "HEARTBEAT_URL", "")
    monkeypatch.delenv("HEARTBEAT_URL", raising=False)
    assert ping_heartbeat() is False


def test_heartbeat_get_con_timeout_y_fail_silent(monkeypatch):
    import requests
    import config.settings as cs
    from monitoring.heartbeat import ping_heartbeat
    calls = []

    class _Resp:
        status_code = 200
    monkeypatch.setattr(cs, "HEARTBEAT_URL", "https://hc.example/ping/abc")
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: calls.append((url, timeout)) or _Resp())
    assert ping_heartbeat() is True
    assert calls == [("https://hc.example/ping/abc", 5.0)]

    def _fail(url, timeout=None):
        raise requests.Timeout("lento")
    monkeypatch.setattr(requests, "get", _fail)
    assert ping_heartbeat() is False


class _StopLoop(Exception):
    pass


def _run_one_cycle(monkeypatch, fake_cmd_signal):
    """Ejecuta el ciclo inicial de cmd_schedule (sin Telegram) y corta el bucle."""
    import time
    import main
    import data.outcome_tracker as ot
    import data.shadow as sh
    import monitoring.heartbeat as hb
    pings = []
    monkeypatch.setattr(main, "cmd_signal", fake_cmd_signal)
    monkeypatch.setattr(main, "active_assets", lambda: ["BTCUSDT", "ETHUSDT"])
    monkeypatch.setattr(ot, "resolve_open_signals", lambda ohlcv_cache=None: 0)
    monkeypatch.setattr(sh, "resolve_shadow_setups", lambda ohlcv_cache=None: 0)
    monkeypatch.setattr(hb, "ping_heartbeat", lambda *a, **k: pings.append(1) or True)

    def _stop(_s):
        raise _StopLoop()
    monkeypatch.setattr(time, "sleep", _stop)
    import config.settings as cs
    monkeypatch.setattr(cs, "TELEGRAM_TOKEN", "")
    monkeypatch.setattr(cs, "TELEGRAM_CHAT_ID", "")
    with pytest.raises(_StopLoop):
        main.cmd_schedule(60, "03:00")
    import schedule
    schedule.clear()
    return pings


def test_latido_solo_en_ciclo_correcto(temp_db, monkeypatch, no_network):
    import main
    from monitoring.daily_summary import KV_LAST_CYCLE_OK

    def ok(symbol, notify=False, ohlcv_cache=None):
        main._cycle_fresh.add(symbol)
    assert _run_one_cycle(monkeypatch, ok) == [1]
    assert temp_db.kv_get(KV_LAST_CYCLE_OK) is not None

    def falla_uno(symbol, notify=False, ohlcv_cache=None):
        main._cycle_fresh.add(symbol)
        if symbol == "ETHUSDT":
            raise RuntimeError("bug")
    assert _run_one_cycle(monkeypatch, falla_uno) == []

    def sin_datos(symbol, notify=False, ohlcv_cache=None):
        return None   # p. ej. todas las fuentes caídas: ningún activo con datos frescos
    assert _run_one_cycle(monkeypatch, sin_datos) == []


def test_cli_summary_existe():
    import main
    args = main.build_parser().parse_args(["summary", "--send"])
    assert args.command == "summary" and args.send is True
