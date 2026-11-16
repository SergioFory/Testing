"""
Pruebas de la Fase 4 — grupo A: ciclo alineado a hora fija UTC + comandos cada 2 min.

  - Horas del ciclo (cycle_slot_times / extra_cycle_times): cada hora en :01 más
    22:20 UTC (DAILY_BAR_CLOSE_UTC + 5 min); otros --interval.
  - El bucle REAL de main.cmd_schedule con un reloj falso (schedule y main leen
    la hora de un datetime falso; time.sleep avanza el reloj): ciclos a las
    HH:01 (± la precisión del bucle), ciclo extra de las 22:20, ejecución
    inmediata al arrancar, tarea nocturna de las 03:00 antes del ciclo de las
    03:01, sondeo de comandos cada 2 min sin solapes, un ciclo largo solo retrasa
    (sin repeticiones), compatibilidad con --interval.
  - Oro: con las horas nuevas la vela diaria se evalúa a las 22:20 en invierno y
    a las 04:01 en verano (sin el ciclo extra, en invierno sería a las 05:01).
  - Reintentos de señales no entregadas con los ciclos alineados.
  - Sondeo de comandos: offset bien guardado, sin dobles respuestas, límite de
    tiempo TOTAL de getUpdates y como mucho un hilo colgado.

    cd trading_system && python -m pytest tests -q

Sin red: requests.get/post y Session.request fallan si alguien intenta usarlos;
Telegram se sustituye por un doble y la BD es SQLite temporal.
"""
import datetime as _dt
import sys
import threading
import time as _time
import types
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UTC = _dt.timezone.utc


# ---------------------------------------------------------------------------
# Fixtures y utilidades
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
    monkeypatch.setattr(requests.Session, "request", _boom)


@pytest.fixture()
def fake_telegram(monkeypatch, no_network):
    import notifications.telegram as tg
    import notifications.telegram_commands as tc
    state = {"ok": True, "sent": []}

    def _fake_send(text_, parse_mode="HTML", retries=3):
        state["sent"].append(text_)
        return state["ok"]
    monkeypatch.setattr(tg, "_send", _fake_send)
    monkeypatch.setattr(tg, "_stale_logged", set())
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "TEST-TOKEN")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "4242")
    monkeypatch.setattr(tc, "_update_failures", {})
    monkeypatch.setattr(tc, "_inflight", {"thread": None})
    return state


@pytest.fixture(autouse=True)
def _clean_schedule():
    import schedule
    schedule.clear()
    yield
    schedule.clear()


class _StopLoop(Exception):
    pass


class _Clock:
    """Reloj falso (UTC, con zona). schedule y main lo leen a través de un
    datetime falso; time.sleep y los dobles del ciclo lo hacen avanzar."""

    def __init__(self, start):
        self.now = pd.Timestamp(start, tz="UTC").to_pydatetime()

    def advance(self, seconds):
        self.now = self.now + _dt.timedelta(seconds=float(seconds))

    def ts(self):
        return pd.Timestamp(self.now)


def _install_clock(monkeypatch, clock):
    import schedule
    import main

    class FakeDT(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:      # como datetime.now(): hora local sin zona
                return clock.now.astimezone().replace(tzinfo=None)
            return clock.now.astimezone(tz)

    fake_mod = types.SimpleNamespace(datetime=FakeDT, time=_dt.time, timedelta=_dt.timedelta,
                                     date=_dt.date, timezone=_dt.timezone)
    monkeypatch.setattr(schedule, "datetime", fake_mod)
    monkeypatch.setattr(main, "datetime", FakeDT)


def _run_daemon(monkeypatch, start, end, interval=60, cycle_seconds=3.0, long_cycles=None,
                notify=True, assets=("BTCUSDT", "XAUUSD")):
    """Ejecuta main.cmd_schedule de verdad con el reloj falso hasta `end`.
    long_cycles: {"HH:MM": segundos} → el ciclo que EMPIECE en ese minuto dura eso.
    Devuelve la lista de eventos (tipo, Timestamp, extra)."""
    import time
    import main
    import config.settings as cs
    import data.outcome_tracker as ot
    import data.shadow as sh
    import monitoring.heartbeat as hb
    import monitoring.data_monitor as dm
    import monitoring.daily_summary as ds
    import notifications.telegram as tg
    import notifications.telegram_commands as tc

    clock = _Clock(start)
    end_ts = pd.Timestamp(end, tz="UTC").to_pydatetime()
    _install_clock(monkeypatch, clock)
    ev = []
    long_cycles = dict(long_cycles or {})

    def _resolve(ohlcv_cache=None):
        ev.append(("start", clock.ts(), None))
        return 0

    def _cmd_signal(symbol, notify=False, ohlcv_cache=None):
        main._cycle_fresh.add(symbol)
        if symbol == assets[0]:
            hhmm = clock.ts().strftime("%H:%M")
            clock.advance(long_cycles.pop(hhmm, cycle_seconds))

    def _flush():
        ev.append(("end", clock.ts(), None))
        return 0

    def _poll(*a, **k):
        ev.append(("poll", clock.ts(), dict(k)))
        return {"updates": 0, "commands": 0, "ignored": 0, "errors": 0}

    def _sleep(s):
        ev.append(("sleep", clock.ts(), s))
        clock.advance(s)
        if clock.now >= end_ts:
            raise _StopLoop()

    monkeypatch.setattr(main, "cmd_signal", _cmd_signal)
    monkeypatch.setattr(main, "active_assets", lambda: list(assets))
    monkeypatch.setattr(main, "_flush_ohlcv_copies", _flush)
    monkeypatch.setattr(ot, "resolve_open_signals", _resolve)
    monkeypatch.setattr(sh, "resolve_shadow_setups", lambda ohlcv_cache=None: 0)
    monkeypatch.setattr(hb, "ping_heartbeat", lambda *a, **k: ev.append(("ping", clock.ts(), None)) or True)
    monkeypatch.setattr(dm, "run_nightly_checks", lambda notify=False: ev.append(("nightly", clock.ts(), None)))
    monkeypatch.setattr(ds, "send_daily_report", lambda *a, **k: ev.append(("report", clock.ts(), None)) or True)
    monkeypatch.setattr(tg, "retry_failed_signal_alerts", lambda *a, **k: {})
    monkeypatch.setattr(tg, "test_connection", lambda *a, **k: True)
    monkeypatch.setattr(tc, "poll_commands", _poll)
    monkeypatch.setattr(cs, "TELEGRAM_TOKEN", "T" if notify else "")
    monkeypatch.setattr(cs, "TELEGRAM_CHAT_ID", "4242" if notify else "")
    monkeypatch.setattr(time, "sleep", _sleep)
    with pytest.raises(_StopLoop):
        main.cmd_schedule(interval, "03:00")
    return ev


def _starts(ev):
    return [t for k, t, _ in ev if k == "start"]


# ---------------------------------------------------------------------------
# 1. Horas del ciclo
# ---------------------------------------------------------------------------

def test_horas_del_ciclo_por_defecto():
    import main
    from config.settings import CYCLE_SCHEDULE, TELEGRAM_POLL_MINUTES, DAILY_BAR_CLOSE_UTC
    assert CYCLE_SCHEDULE["minute"] == 1 and TELEGRAM_POLL_MINUTES == 2
    assert DAILY_BAR_CLOSE_UTC == (22, 15)
    assert main.extra_cycle_times() == ["22:20"]
    slots = main.cycle_slot_times(60)
    assert len(slots) == 25
    assert slots[:3] == ["00:01", "01:01", "02:01"] and "22:20" in slots and slots[-1] == "23:01"
    assert slots == sorted(slots) and len(set(slots)) == 25


def test_horas_con_otros_intervalos_y_sin_repetir():
    import main
    assert main.cycle_slot_times(240) == ["00:01", "04:01", "08:01", "12:01", "16:01", "20:01", "22:20"]
    assert main.cycle_slot_times(30)[:3] == ["00:01", "00:31", "01:01"]
    assert len(main.cycle_slot_times(30)) == 49
    # Intervalos que no dividen el día: modo antiguo (None).
    assert main.cycle_slot_times(50) is None and main.cycle_slot_times(70) is None
    assert main.cycle_slot_times(0) is None
    # Si el extra coincide con una hora del ciclo no se duplica.
    assert main.cycle_slot_times(60, minute=20).count("22:20") == 1
    assert len(main.cycle_slot_times(60, minute=20)) == 24


def test_ciclo_extra_sigue_a_daily_bar_close(monkeypatch):
    import main
    import config.settings as cs
    monkeypatch.setattr(cs, "DAILY_BAR_CLOSE_UTC", (21, 58))
    assert main.extra_cycle_times() == ["22:03"]
    monkeypatch.setattr(cs, "DAILY_BAR_CLOSE_UTC", (23, 58))
    assert main.extra_cycle_times() == ["00:03"]
    monkeypatch.setitem(cs.CYCLE_SCHEDULE, "after_daily_close_min", None)
    assert main.extra_cycle_times() == []


# ---------------------------------------------------------------------------
# 2. El bucle real de cmd_schedule con reloj falso
# ---------------------------------------------------------------------------

def test_ciclos_en_hora_fija_extra_2220_y_nocturna(temp_db, monkeypatch):
    ev = _run_daemon(monkeypatch, "2026-01-14 10:37:20", "2026-01-15 05:30:00")
    starts = _starts(ev)
    # Ejecución inmediata al arrancar.
    assert starts[0] == pd.Timestamp("2026-01-14 10:37:20", tz="UTC")
    rest = starts[1:]
    expected = [pd.Timestamp(f"2026-01-14 {h:02d}:01", tz="UTC") for h in range(11, 23)]
    expected.append(pd.Timestamp("2026-01-14 22:20", tz="UTC"))
    expected += [pd.Timestamp("2026-01-14 23:01", tz="UTC")]
    expected += [pd.Timestamp(f"2026-01-15 {h:02d}:01", tz="UTC") for h in range(0, 6)]
    assert len(rest) == len(expected) == 20
    for got, want in zip(rest, expected):
        # Precisión: el bucle duerme 5 s; la tarea nocturna (03:00) va antes de las 03:01.
        assert want <= got <= want + pd.Timedelta(seconds=6), (got, want)
    # Tarea nocturna: una vez, a las 03:00, ANTES del ciclo de las 03:01.
    nightly = [t for k, t, _ in ev if k == "nightly"]
    assert len(nightly) == 1 and pd.Timestamp("2026-01-15 03:00", tz="UTC") <= nightly[0] \
        < pd.Timestamp("2026-01-15 03:00:06", tz="UTC")
    c0301 = [t for t in rest if t.strftime("%H:%M") == "03:01"]
    assert len(c0301) == 1 and c0301[0] > nightly[0]
    # Un latido por ciclo correcto; ninguno del sondeo de comandos.
    assert sum(1 for k, *_ in ev if k == "ping") == len(starts)
    # El bucle duerme como mucho 5 s.
    assert max(s for k, _, s in ev if k == "sleep") <= 5


def test_sondeo_de_comandos_cada_2_min_sin_solapes(temp_db, monkeypatch):
    ev = _run_daemon(monkeypatch, "2026-01-14 10:00:30", "2026-01-14 13:30:00", cycle_seconds=40)
    polls = [(t, k) for kind, t, k in ev if kind == "poll"]
    job_polls = [t for t, k in polls if k.get("log_failures") is False]
    cycle_polls = [t for t, k in polls if not k]
    # El ciclo también lee los comandos al empezar (con avisos en el log).
    assert len(cycle_polls) == len(_starts(ev))
    # Trabajo ligero: cada ~2 min (+ ≤ 5 s del bucle; tras un ciclo de 40 s, algo más).
    gaps = [(b - a).total_seconds() for a, b in zip(job_polls, job_polls[1:])]
    assert len(job_polls) >= 95 and min(gaps) >= 120 and max(gaps) <= 120 + 5 + 40 + 1
    # Nunca durante un ciclo: entre "start" y "end" no hay sondeos del trabajo.
    inside = False
    for kind, t, k in ev:
        if kind == "start":
            inside = True
        elif kind == "end":
            inside = False
        elif kind == "poll" and k.get("log_failures") is False:
            assert not inside, t


def test_sin_telegram_no_hay_sondeo(temp_db, monkeypatch):
    ev = _run_daemon(monkeypatch, "2026-01-14 10:00:30", "2026-01-14 11:30:00", notify=False)
    assert not [1 for k, *_ in ev if k == "poll"]
    assert [t.strftime("%H:%M") for t in _starts(ev)] == ["10:00", "10:01", "11:01"]


def test_ciclo_largo_solo_retrasa_sin_repetir(temp_db, monkeypatch):
    # El ciclo de las 12:01 tarda 75 min: el de las 13:01 corre al terminar (13:16),
    # una sola vez; el sondeo se retrasa y corre UNA vez (no se acumulan 37).
    ev = _run_daemon(monkeypatch, "2026-01-14 11:59:00", "2026-01-14 14:30:00",
                     long_cycles={"12:01": 75 * 60})
    starts = _starts(ev)[1:]
    hhmm = [t.strftime("%H:%M") for t in starts]
    assert hhmm == ["12:01", "13:16", "14:01"], hhmm
    job_polls = [t for kind, t, k in ev if kind == "poll" and k.get("log_failures") is False]
    after_long = [t for t in job_polls if pd.Timestamp("2026-01-14 13:16", tz="UTC") <= t
                  < pd.Timestamp("2026-01-14 13:18", tz="UTC")]
    assert len(after_long) == 1


def test_bloqueo_de_varias_horas_no_repite_ciclos(temp_db, monkeypatch):
    # Un ciclo bloqueado 3 h 10 min deja pendientes las 13:01, 14:01 y 15:01: corre
    # UNO al terminar y los otros se omiten (ya vieron las mismas velas).
    ev = _run_daemon(monkeypatch, "2026-01-14 11:59:00", "2026-01-14 16:30:00",
                     long_cycles={"12:01": 190 * 60})
    hhmm = [t.strftime("%H:%M") for t in _starts(ev)[1:]]
    assert hhmm == ["12:01", "15:11", "16:01"], hhmm


def test_compatibilidad_interval(temp_db, monkeypatch):
    import schedule
    # --interval 240: alineado a las 00:01, 04:01... + 22:20.
    ev = _run_daemon(monkeypatch, "2026-01-14 00:00:30", "2026-01-15 00:00:00", interval=240)
    hhmm = [t.strftime("%H:%M") for t in _starts(ev)[1:]]
    assert hhmm == ["00:01", "04:01", "08:01", "12:01", "16:01", "20:01", "22:20"]
    schedule.clear()
    # --interval 50 (no divide el día): cada 50 min desde el arranque + 22:20.
    ev = _run_daemon(monkeypatch, "2026-01-14 20:10:00", "2026-01-14 23:30:00", interval=50)
    hhmm = [t.strftime("%H:%M") for t in _starts(ev)]
    assert hhmm == ["20:10", "21:00", "21:50", "22:20", "22:40", "23:30"][:len(hhmm)]
    assert "22:20" in hhmm and "21:00" in hhmm


def test_trabajos_registrados_en_utc(temp_db, monkeypatch):
    import schedule
    _run_daemon(monkeypatch, "2026-07-14 10:00:30", "2026-07-14 10:01:00")
    jobs = schedule.get_jobs()
    days = [j for j in jobs if j.unit == "days"]
    assert len(days) == 26                    # 25 horas del ciclo + tarea nocturna
    assert all(str(j.at_time_zone) == "UTC" for j in days)
    assert sorted(str(j.at_time)[:5] for j in days if j.job_func.func.__name__ == "_run_nightly") == ["03:00"]
    minutes = [j for j in jobs if j.unit == "minutes"]
    assert len(minutes) == 1 and minutes[0].interval == 2


# ---------------------------------------------------------------------------
# 3. Oro: en qué ciclo se evalúa la vela diaria
# ---------------------------------------------------------------------------

def _first_slot_with_day(day, slots):
    """Primer instante de `slots` (desde las 22:00 del día `day`) en que la vela
    diaria de `day` está cerrada y se acepta (data.bars.drop_unclosed_bars)."""
    from data.bars import drop_unclosed_bars
    idx = pd.date_range(pd.Timestamp(day) - pd.Timedelta(days=6), pd.Timestamp(day) + pd.Timedelta(days=1),
                        freq="D", tz="UTC")
    df = pd.DataFrame({"close": range(len(idx))}, index=idx)
    t = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=22)
    stop = t + pd.Timedelta(hours=12)
    while t < stop:
        if t.strftime("%H:%M") in slots:
            out = drop_unclosed_bars(df, "1d", session=True, now=t)
            if out.index[-1] == pd.Timestamp(day, tz="UTC"):
                return t
        t += pd.Timedelta(minutes=1)
    return None


def test_oro_invierno_2220_y_verano_0401():
    import main
    slots = set(main.cycle_slot_times(60))
    # Invierno (miércoles 14-ene-2026, 18:00 NY = 23:00 UTC): esa misma noche.
    assert _first_slot_with_day("2026-01-14", slots) == pd.Timestamp("2026-01-14 22:20", tz="UTC")
    # Verano (miércoles 15-jul-2026, 18:00 NY = 22:00 UTC): a las 04:01 UTC.
    assert _first_slot_with_day("2026-07-15", slots) == pd.Timestamp("2026-07-16 04:01", tz="UTC")
    # Sin el ciclo extra, en invierno pasaría a las 05:01 del día siguiente.
    only_01 = set(main.cycle_slot_times(60, extra=[]))
    assert _first_slot_with_day("2026-01-14", only_01) == pd.Timestamp("2026-01-15 05:01", tz="UTC")
    # Viernes de invierno: también a las 22:20.
    assert _first_slot_with_day("2026-01-16", slots) == pd.Timestamp("2026-01-16 22:20", tz="UTC")


# ---------------------------------------------------------------------------
# 4. Reintentos con los ciclos alineados
# ---------------------------------------------------------------------------

class _Trade:
    def __init__(self, symbol="BTCUSDT", direction="long", entry=100.0, sl=97.0, tp=104.5):
        self.symbol, self.direction, self.setup_type = symbol, direction, "breakout"
        self.entry_price, self.stop_loss, self.take_profit = entry, sl, tp
        self.position_size, self.atr, self.ml_score = 1000.0, 2.0, 0.6
        self.rr_ratio, self.risk_usd, self.reward_usd = 1.5, 30.0, 45.0


def _U(s):
    return int(pd.Timestamp(s, tz="UTC").timestamp())


def _new_failed_signal(db, emitted):
    from data.outcome_tracker import save_signal_from_trade
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-01-14 00:00", tz="UTC"),
                                 mode="live", notified=False, alert_text="SEÑAL #X")
    with db.get_engine().begin() as conn:
        conn.execute(text("UPDATE signals SET ts=:t WHERE id=:i"), {"t": _U(emitted), "i": sid})
    return sid


def test_reintentos_con_ciclo_alineado(temp_db, fake_telegram):
    import notifications.telegram as tg
    # Señal guardada a las 04:01:40 (Telegram caído): reintentos a las 05:01 y 06:01.
    _new_failed_signal(temp_db, "2026-01-14 04:01:40")
    fake_telegram["ok"] = False
    r1 = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-01-14 05:01:05", tz="UTC"))
    r2 = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-01-14 06:01:05", tz="UTC"))
    assert r1["failed"] == 1 and r2["failed"] == 1
    r3 = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-01-14 07:01:05", tz="UTC"))
    assert r3.get("failed", 0) == 0 and r3["stale"] == 1


def test_reintentos_senal_del_ciclo_2220(temp_db, fake_telegram):
    import notifications.telegram as tg
    _new_failed_signal(temp_db, "2026-01-14 22:20:30")
    fake_telegram["ok"] = False
    assert tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-01-14 23:01:05", tz="UTC"))["failed"] == 1
    assert tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-01-15 00:01:05", tz="UTC"))["failed"] == 1
    r = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-01-15 01:01:05", tz="UTC"))
    assert r.get("failed", 0) == 0 and r["stale"] == 1


# ---------------------------------------------------------------------------
# 5. Sondeo de comandos: offset, dobles respuestas, límite de tiempo, /help
# ---------------------------------------------------------------------------

class _R:
    status_code = 200

    def __init__(self, data):
        self._d = data
        self.text = str(data)

    def json(self):
        return self._d


def _upd(uid, text_, chat_id=4242):
    return {"update_id": uid, "message": {"date": 1768400000, "chat": {"id": chat_id}, "text": text_}}


def _telegram_server(monkeypatch, inbox):
    """getUpdates como Telegram: devuelve las actualizaciones con update_id ≥ offset."""
    import requests
    seen = []

    def _get(url, params=None, timeout=None):
        assert "getUpdates" in url and params["timeout"] == 0
        seen.append(params.get("offset"))
        off = params.get("offset") or 0
        return _R({"ok": True, "result": [u for u in inbox if u["update_id"] >= off]})
    monkeypatch.setattr(requests, "get", _get)
    return seen


def test_sondeos_seguidos_sin_dobles_respuestas(temp_db, fake_telegram, monkeypatch):
    from notifications.telegram_commands import poll_commands, KV_OFFSET
    inbox = [_upd(500, "/help")]
    seen = _telegram_server(monkeypatch, inbox)
    a = poll_commands(log_failures=False)          # sondeo de cada 2 min
    b = poll_commands()                            # sondeo al inicio del ciclo
    c = poll_commands(log_failures=False)
    assert a["commands"] == 1 and b["updates"] == 0 and c["updates"] == 0
    assert len(fake_telegram["sent"]) == 1 and "/fill ID PRECIO" in fake_telegram["sent"][0]
    assert temp_db.kv_get(KV_OFFSET) == "501" and seen == [None, 501, 501]
    inbox.append(_upd(501, "/help"))
    assert poll_commands(log_failures=False)["commands"] == 1
    assert len(fake_telegram["sent"]) == 2 and temp_db.kv_get(KV_OFFSET) == "502"


def test_getupdates_con_limite_total_y_un_solo_hilo(temp_db, fake_telegram, monkeypatch):
    import requests
    import notifications.telegram_commands as tc
    monkeypatch.setattr(tc, "GETUPDATES_DEADLINE_S", 0.3)
    release = threading.Event()
    calls = []

    def _hung(url, params=None, timeout=None):
        calls.append(1)
        release.wait(5)          # Telegram "colgado" (goteo) hasta que se libere
        return _R({"ok": True, "result": [_upd(900, "/help")]})
    monkeypatch.setattr(requests, "get", _hung)
    try:
        t0 = _time.perf_counter()
        out = tc.poll_commands(log_failures=False)
        assert _time.perf_counter() - t0 < 1.5
        assert out["updates"] == 0 and fake_telegram["sent"] == []
        assert temp_db.kv_get(tc.KV_OFFSET) is None
        # Siguiente sondeo con el hilo anterior aún colgado: no se lanza otro.
        out = tc.poll_commands(log_failures=False)
        assert out["updates"] == 0 and len(calls) == 1
    finally:
        release.set()
    tc._inflight["thread"].join(2)
    # Telegram vuelve: el comando se procesa una vez.
    out = tc.poll_commands(log_failures=False)
    assert out["commands"] == 1 and len(calls) == 2 and len(fake_telegram["sent"]) == 1


def test_sondeo_silencioso_no_avisa_en_el_log(temp_db, fake_telegram, monkeypatch):
    import requests
    from loguru import logger
    import notifications.telegram_commands as tc

    def _down(*a, **k):
        raise requests.ConnectionError("sin red")
    monkeypatch.setattr(requests, "get", _down)
    msgs = []
    hid = logger.add(lambda m: msgs.append(m.record["level"].name), level="DEBUG")
    try:
        tc.poll_commands(log_failures=False)
        assert "WARNING" not in msgs and "DEBUG" in msgs
        msgs.clear()
        tc.poll_commands()
        assert "WARNING" in msgs
    finally:
        logger.remove(hid)


def test_help_dice_cada_2_minutos(monkeypatch):
    import notifications.telegram_commands as tc
    import config.settings as cs
    assert "cada 2 minutos" in tc.HELP_TEXT and "hasta una hora" not in tc.HELP_TEXT
    monkeypatch.setattr(cs, "TELEGRAM_POLL_MINUTES", 0)
    assert "cada hora" in tc._when_read_text()


def test_reglas_documentan_las_horas():
    reglas = (ROOT / "REGLAS_DE_DECISION.md").read_text(encoding="utf-8")
    i = reglas.index("## Cuándo llegan las señales (Fase 4")
    sec = reglas[i:]
    for frag in ("HH:01 UTC", "22:20 UTC", "04:01 UTC", "cada 2 minutos", "--interval 60",
                 "13 de 13 señales iguales"):
        assert frag in sec, frag
