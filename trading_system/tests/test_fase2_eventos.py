"""
Pruebas de la Fase 2, punto 2.4 — eventos macro (FOMC, CPI, NFP):
  - calendario propio data/econ_events.csv: formato, sin duplicados, ET→UTC con
    America/New_York (CPI/NFP 08:30 ET, FOMC 14:00 ET salvo las 3 extraordinarias de
    2020), sin huecos absurdos (12 CPI y 12 NFP por año, 8 FOMC + extraordinarias;
    excepciones documentadas: cierre del gobierno de EE. UU. en 2025) y cobertura
    hasta 2027;
  - econ_calendar: el CSV es la fuente principal (sin red), el feed solo complemento,
    fail-open, y las ventanas (2 h antes, 1 h después) y a quién aplica NO cambian;
  - el filtro actual nunca coincide con un FOMC/CPI/NFP a las horas a las que el vivo
    evalúa el oro y el SPX (hallazgo de 2.4.2, fijado como regresión);
  - mensaje de la señal: línea "⚠️ Evento de alto impacto dentro del plazo de la
    operación" (máx. 3, HTML escapado) y resumen nocturno con los eventos de 48 h.

    cd trading_system && python -m pytest tests/test_fase2_eventos.py -q

Sin red (requests.get/post fallan si alguien los usa); BD SQLite temporal.
"""
import csv
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CSV_PATH = ROOT / "data" / "econ_events.csv"
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
EXTRAORDINARIAS = {"2020-03-03", "2020-03-15", "2020-03-23"}


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
    import data.fetchers.econ_calendar as cal
    # Feed: sin red y sin caché heredada de otra prueba → fail-open (lista vacía).
    monkeypatch.setattr(cal, "_CACHE", {"ts": 0.0, "events": None})


def _rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# El CSV
# ---------------------------------------------------------------------------

def test_csv_formato_sin_duplicados_y_ordenado():
    rows = _rows()
    assert list(rows[0].keys()) == ["date_utc", "time_utc", "event", "source"]
    keys = [(r["date_utc"], r["time_utc"], r["event"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys)
    assert {r["event"] for r in rows} == {"FOMC", "CPI", "NFP"}
    assert all(r["source"].strip() for r in rows)
    assert all(("federalreserve.gov" in r["source"]) == (r["event"] == "FOMC") for r in rows)
    assert all(("bls.gov" in r["source"]) == (r["event"] != "FOMC") for r in rows)


def test_csv_horas_et_a_utc_con_cambio_de_hora():
    """Cada fila, devuelta a hora de Nueva York, es 08:30 (CPI/NFP) o 14:00 (FOMC),
    con horario de verano/invierno; solo las 3 extraordinarias de 2020 difieren."""
    for r in _rows():
        local = _utc(f"{r['date_utc']} {r['time_utc']}").astimezone(NY)
        assert local.weekday() < 5 or r["date_utc"] == "2020-03-15"   # domingo: recorte de urgencia
        if r["event"] == "FOMC" and r["date_utc"] in EXTRAORDINARIAS:
            continue
        esperado = (14, 0) if r["event"] == "FOMC" else (8, 30)
        assert (local.hour, local.minute) == esperado, r
    rows = {(r["date_utc"], r["event"]): r["time_utc"] for r in _rows()}
    assert rows[("2024-11-13", "CPI")] == "13:30"      # EST
    assert rows[("2024-06-12", "CPI")] == "12:30"      # EDT
    assert rows[("2024-06-12", "FOMC")] == "18:00"
    assert rows[("2024-12-18", "FOMC")] == "19:00"
    assert rows[("2020-03-03", "FOMC")] == "15:00"     # 10:00 EST, extraordinaria
    assert rows[("2020-03-15", "FOMC")] == "21:00"     # 17:00 EDT, extraordinaria


def test_csv_sin_huecos_absurdos_y_cobertura():
    rows = _rows()
    c = Counter((r["date_utc"][:4], r["event"]) for r in rows)
    for y in range(2015, 2028):
        y = str(y)
        cierre_2025 = y == "2025"   # oct-2025: ni CPI ni empleo (cierre del gobierno; bls.gov)
        assert c[(y, "CPI")] == (11 if cierre_2025 else 12), y
        assert c[(y, "NFP")] == (11 if cierre_2025 else 12), y
        assert c[(y, "FOMC")] == (10 if y == "2020" else 8), y   # 2020: 3 extraordinarias, 1 cancelada
    # Entre dos publicaciones seguidas nunca pasan más de 45 días, salvo el cierre de 2025.
    for ev in ("CPI", "NFP"):
        d = sorted(_utc(r["date_utc"]) for r in rows if r["event"] == ev)
        gaps = [(a, b) for a, b in zip(d, d[1:]) if (b - a).days > 45]
        assert all(a.year == 2025 and a.month in (9, 10) for a, _ in gaps), (ev, gaps)
    d = sorted(_utc(r["date_utc"]) for r in rows if r["event"] == "FOMC")
    assert max((b - a).days for a, b in zip(d, d[1:])) <= 63
    assert rows[0]["date_utc"].startswith("2015-01")
    assert max(r["date_utc"] for r in rows) >= "2027-12-01"


# ---------------------------------------------------------------------------
# econ_calendar: CSV principal, feed complemento, ventanas sin cambios
# ---------------------------------------------------------------------------

def test_carga_del_csv_sin_red_cache_y_fail_open(tmp_path):
    import data.fetchers.econ_calendar as cal
    evs = cal.load_events_csv()
    assert len(evs) == len(_rows())
    assert evs[0]["time"].tzinfo is not None and evs == sorted(evs, key=lambda e: e["time"])
    e = next(x for x in evs if x["time"] == _utc("2026-12-10 13:30"))
    assert (e["currency"], e["impact"], e["event"]) == ("USD", "High", "CPI")
    assert cal.load_events_csv() is evs                       # caché mientras no cambie el archivo
    assert cal.load_events_csv(tmp_path / "no_existe.csv") == []
    bad = tmp_path / "malo.csv"
    bad.write_text("date_utc,time_utc,event,source\nbasura,xx,CPI,x\n2026-01-02,13:30,NFP,x\n")
    assert [x["event"] for x in cal.load_events_csv(bad)] == ["NFP"]
    assert cal.format_event_short(e) == "CPI 10-dic 13:30 UTC"


def test_blackout_con_el_csv_ventanas_y_activos_sin_cambios(monkeypatch):
    # Fase 3 (punto 0): EVENT_FILTER desapareció de settings y el vivo ya no bloquea
    # (tests/test_fase3_eventos_riesgo.py); is_event_blackout queda como utilidad.
    import data.fetchers.econ_calendar as cal
    assert cal.ASSET_CURRENCIES["BTCUSDT"] == set() and cal.ASSET_CURRENCIES["ETHUSDT"] == set()
    cpi = _utc("2026-12-10 13:30")
    kw = dict(hours_before=2.0, hours_after=1.0)
    # Bloquea si el evento está entre 1 h antes de "ahora" y 2 h después: ahora ∈ [CPI−2h, CPI+1h].
    for dt_min, esperado in [(-120, True), (-121, False), (0, True), (60, True), (61, False)]:
        b, motivo = cal.is_event_blackout("XAUUSD", now_utc=cpi + timedelta(minutes=dt_min), **kw)
        assert b is esperado, (dt_min, motivo)
    b, motivo = cal.is_event_blackout("SPX500", now_utc=cpi - timedelta(minutes=30), **kw)
    assert b and "CPI" in motivo and "en 30min" in motivo
    assert cal.is_event_blackout("BTCUSDT", now_utc=cpi, **kw) == (False, "")   # cripto exento
    # Sin CSV y sin feed: fail-open.
    monkeypatch.setattr(cal, "EVENTS_CSV", Path("/no/existe.csv"))
    assert cal.is_event_blackout("XAUUSD", now_utc=cpi, **kw) == (False, "")


def test_feed_como_complemento(monkeypatch):
    import data.fetchers.econ_calendar as cal
    t = _utc("2026-11-17 13:30")   # ventas minoristas: no está en el CSV
    feed = [{"currency": "USD", "impact": "High", "time": t, "title": "Retail Sales m/m"},
            {"currency": "CAD", "impact": "High", "time": t, "title": "CPI m/m"},
            {"currency": "USD", "impact": "Medium", "time": t, "title": "Algo"}]
    monkeypatch.setattr(cal, "fetch_calendar", lambda force=False: feed)
    b, motivo = cal.is_event_blackout("XAUUSD", now_utc=t - timedelta(hours=1))
    assert b and "Retail Sales" in motivo
    monkeypatch.setattr(cal, "fetch_calendar", lambda force=False: feed[1:])
    assert cal.is_event_blackout("XAUUSD", now_utc=t - timedelta(hours=1)) == (False, "")
    # Si el feed lanza, el CSV sigue funcionando.
    monkeypatch.setattr(cal, "fetch_calendar", lambda force=False: 1 / 0)
    assert cal.is_event_blackout("XAUUSD", now_utc=_utc("2026-12-10 13:00"))[0] is True


def test_el_filtro_nunca_coincide_con_las_horas_de_evaluacion_del_vivo():
    """2.4.2: el vivo evalúa el oro y el SPX tras el cierre diario (22:15–23:00 UTC en
    invierno, o desde las 00:00 ET por la corrección de la vela en curso: 04:00–06:59 UTC
    contando hasta 2 emisiones más en los ciclos siguientes). Con FOMC/CPI/NFP del
    CSV, ninguna de esas horas cae en la ventana del filtro en 2015–2027."""
    import data.fetchers.econ_calendar as cal
    d = datetime(2015, 1, 1, tzinfo=UTC)
    n = 0
    while d.year < 2028:
        if d.weekday() < 5:
            for hh, mm in [(22, 15), (22, 59), (4, 0), (5, 30), (6, 59)]:
                t = d.replace(hour=hh, minute=mm)
                assert not cal.events_between(t - timedelta(hours=1), t + timedelta(hours=2)), t
                n += 1
        d += timedelta(days=1)
    assert n > 15000


# ---------------------------------------------------------------------------
# Mensaje de la señal y resumen nocturno
# ---------------------------------------------------------------------------

def _trade(symbol="XAUUSD", direction="long", entry=2650.0, sl=2610.0, tp=2703.3):
    from risk.position_sizer import build_trade_setup
    row = pd.Series({"close": entry, "stop_loss": sl, "take_profit": tp,
                     "atr": abs(entry - sl) / 1.5, "direction": direction,
                     "symbol": symbol, "setup_type": "breakout"})
    return build_trade_setup(row, 1000.0, 0.31)


def test_linea_de_eventos_en_el_mensaje():
    from notifications.telegram import format_signal_alert, event_warning_line, EVENT_LINE_NOTE as NOTE
    # Oro: setup del 2026-11-30, plazo de 20 sesiones → incluye NFP 4-dic, CPI 10-dic, FOMC 9-dic…
    msg = format_signal_alert(_trade(), setup_ts=pd.Timestamp("2026-11-30", tz="UTC"),
                              current_price=2650.0, now=pd.Timestamp("2026-11-30 22:30", tz="UTC"))
    line = [ln for ln in msg.splitlines() if "Evento de alto impacto" in ln]
    assert len(line) == 1
    assert line[0] == ("⚠️ <b>Evento de alto impacto dentro del plazo de la operación:</b> "
                       "NFP 04-dic 13:30 UTC, FOMC 09-dic 19:00 UTC, CPI 10-dic 13:30 UTC "
                       "<i>(solo aviso: no es motivo para no entrar)</i>")
    # Más de 3 eventos: los 3 primeros y cuántos más.
    assert event_warning_line(_utc("2026-12-01 00:00"), _utc("2027-01-31 00:00")).endswith(
        "NFP 04-dic 13:30 UTC, FOMC 09-dic 19:00 UTC, CPI 10-dic 13:30 UTC (y 3 más) " + NOTE)   # ene-2027: NFP 8, CPI 13, FOMC 27
    assert msg.index("Cerrar a mercado") < msg.index("Evento de alto impacto")
    # BTC: plazo de 4 días sin eventos → sin línea.
    msg = format_signal_alert(_trade("BTCUSDT", "short", 90000.0, 91500.0, 87750.0),
                              setup_ts=pd.Timestamp("2026-11-12 00:00", tz="UTC"),
                              now=pd.Timestamp("2026-11-12 04:10", tz="UTC"))
    assert "Evento de alto impacto" not in msg
    # BTC con CPI dentro del plazo: aparece (aviso para todos los activos; no bloquea).
    msg = format_signal_alert(_trade("BTCUSDT", "short", 90000.0, 91500.0, 87750.0),
                              setup_ts=pd.Timestamp("2026-12-08 00:00", tz="UTC"),
                              now=pd.Timestamp("2026-12-08 04:10", tz="UTC"))
    assert "FOMC 09-dic 19:00 UTC, CPI 10-dic 13:30 UTC</" not in msg   # el texto va escapado y sin etiqueta dentro
    assert "Evento de alto impacto dentro del plazo de la operación:</b> FOMC 09-dic 19:00 UTC, CPI 10-dic 13:30 UTC" in msg
    # Exactamente los eventos entre la entrada (ahora) y la fecha límite, ambos incluidos.
    assert event_warning_line(_utc("2026-12-10 13:30"), _utc("2026-12-10 13:30")).endswith("CPI 10-dic 13:30 UTC " + NOTE)
    assert event_warning_line(_utc("2026-12-10 13:31"), _utc("2026-12-30 00:00")) is None


def test_linea_de_eventos_escapa_html_y_nunca_lanza(monkeypatch):
    import data.fetchers.econ_calendar as cal
    from notifications.telegram import event_warning_line
    t = _utc("2026-12-10 13:30")
    monkeypatch.setattr(cal, "events_between", lambda a, b: [{"event": "<X&Y>", "time": t}])
    line = event_warning_line(t, t)
    assert "&lt;X&amp;Y&gt; 10-dic 13:30 UTC" in line and "<X" not in line
    monkeypatch.setattr(cal, "events_between", lambda a, b: 1 / 0)
    assert event_warning_line(t, t) is None


def test_bloque_de_eventos_del_resumen(temp_db, tmp_path):
    from monitoring.daily_summary import events_block, build_daily_summary
    now = int(pd.Timestamp("2026-12-09 03:10", tz="UTC").timestamp())
    blk = events_block(now)
    assert blk[0] == "📅 <b>Eventos macro (próximas 48 h)</b>"
    assert "  • FOMC 09-dic 19:00 UTC — FOMC (decisión de tipos de la Fed)" in blk
    assert "  • CPI 10-dic 13:30 UTC — CPI (inflación de EE. UU.)" in blk
    assert not any("NFP" in x for x in blk) and not any("⚠️" in x for x in blk)
    blk = events_block(int(pd.Timestamp("2026-12-20 03:10", tz="UTC").timestamp()))
    assert "  Ninguno (FOMC, CPI, NFP)." in blk
    # Calendario que se acaba pronto → aviso para actualizarlo; ilegible → aviso.
    short = tmp_path / "c.csv"
    short.write_text("date_utc,time_utc,event,source\n2026-12-10,13:30,CPI,x\n")
    assert any("solo llega hasta el 2026-12-10" in x for x in events_block(now, path=short))
    assert any("ilegible" in x for x in events_block(now, path=tmp_path / "no.csv"))
    msg = build_daily_summary(now=pd.Timestamp(now, unit="s", tz="UTC"), symbols=["BTCUSDT"])
    assert "📅 <b>Eventos macro (próximas 48 h)</b>" in msg and "CPI 10-dic 13:30 UTC" in msg
    assert msg.index("Eventos macro") < msg.index("Resultados")
