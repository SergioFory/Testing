"""
Pruebas de la Fase 4 — correcciones de la revisión adversarial (no cambia qué
señales salen):

  1. Comandos de Telegram: con el sondeo cada 2 min, un /fill que falla por un
     error pasajero de la BD ya no se da por perdido en ~4 min: se abandona solo
     tras MAX_UPDATE_ATTEMPTS intentos Y 2 h desde el primer fallo (como antes).
  2. restore: si la BD se recreó y el worker ya emitió, no duplica señales (mismo
     setup con otro id) ni cuelga /fill de señales equivocadas.
  3. Dashboard: "Resultado realizado hoy" con el signo natural (sin "-0.00 %"); SL y
     TP con su signo real en los cortos; el oro aparte en las ejecuciones.
  4. Copia mensual: el día 7 (último intento) el resumen no dice "se reintenta mañana".

    cd trading_system && python -m pytest tests/test_fase4_revision.py -q

Sin red: requests.get/post y Session.request fallan; Telegram es un doble y la BD
es SQLite temporal.
"""
import datetime as _dt
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures y utilidades
# ---------------------------------------------------------------------------

def _use_db(monkeypatch, path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    import data.database as db
    if db._engine is not None:
        db._engine.dispose()
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    return db


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db = _use_db(monkeypatch, tmp_path / "test.db")
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
    monkeypatch.setattr(requests.Session, "request", _boom)


@pytest.fixture(autouse=True)
def clean_health():
    from monitoring import cycle_health
    cycle_health.reset()
    yield
    cycle_health.reset()


@pytest.fixture()
def fake_telegram(monkeypatch):
    import notifications.telegram as tg
    import notifications.telegram_commands as tc
    state = {"sent": []}

    def _fake_send(text_, parse_mode="HTML", retries=3):
        state["sent"].append(text_)
        return True
    monkeypatch.setattr(tg, "_send", _fake_send)
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "TEST-TOKEN")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "4242")
    monkeypatch.setattr(tc, "_update_failures", {})
    return state


T0 = 1_790_000_000


def _sig(db, symbol, direction, setup_ts, entry, sid=None, **kw):
    row = {"symbol": symbol, "ts": setup_ts + 3660, "setup_type": "breakout", "direction": direction,
           "ml_score": 0.7, "entry_price": entry,
           "stop_loss": entry * (0.98 if direction == "long" else 1.02),
           "take_profit": entry * (1.03 if direction == "long" else 0.97),
           "position_size": 0.01, "atr": 1.0, "setup_ts": setup_ts, "mode": "live", "notified": 1,
           "data_source": "gateio"}
    if sid is not None:
        row["id"] = sid
    row.update(kw)
    cols = ", ".join(row)
    with db.get_engine().begin() as c:
        c.execute(text(f"INSERT INTO signals ({cols}) VALUES ({', '.join(':' + k for k in row)})"), row)
        return c.execute(text("SELECT MAX(id) FROM signals")).scalar()


def _q(db, sql):
    with db.get_engine().connect() as c:
        return c.execute(text(sql)).fetchall()


# ---------------------------------------------------------------------------
# 1. Reintento de comandos medido por tiempo
# ---------------------------------------------------------------------------

def _fake_updates(monkeypatch, batch):
    import requests

    class _R:
        status_code = 200

        def __init__(self, data):
            self._d, self.text = data, str(data)

        def json(self):
            return self._d

    def _get(url, params=None, timeout=None):
        off = (params or {}).get("offset")
        return _R({"ok": True, "result": [u for u in batch if off is None or u["update_id"] >= int(off)]})
    monkeypatch.setattr(requests, "get", _get)


def _upd(uid, text_):
    return {"update_id": uid, "message": {"date": T0, "chat": {"id": 4242}, "text": text_}}


def _clock(monkeypatch, step_s):
    import notifications.telegram_commands as tc
    clock = {"t": float(T0)}

    def _tick():
        clock["t"] += step_s
        return clock["t"]
    monkeypatch.setattr(tc, "_clock", _tick)
    return clock


def test_skip_con_bd_inestable_10_min_no_se_pierde_con_sondeo_de_2_min(temp_db, fake_telegram, monkeypatch):
    """Escenario 'dbflaky' de la revisión: la escritura falla 10 min y la BD vuelve."""
    import data.database as db
    from notifications.telegram_commands import poll_commands, KV_OFFSET
    sid = _sig(temp_db, "BTCUSDT", "long", T0, 100.0)
    _fake_updates(monkeypatch, [_upd(700, f"/skip {sid} tarde")])
    real = db.save_fill
    monkeypatch.setattr(db, "save_fill", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("BD caída")))
    _clock(monkeypatch, 120)                       # un sondeo cada 2 min
    for _ in range(5):                             # 10 min de fallos (5 sondeos ≥ 3 intentos)
        poll_commands(log_failures=False)
        assert temp_db.kv_get(KV_OFFSET) in (None, "700")
    assert fake_telegram["sent"] == []             # ni "No pude registrar" ni respuesta
    monkeypatch.setattr(db, "save_fill", real)     # la BD vuelve
    poll_commands(log_failures=False)
    assert temp_db.kv_get(KV_OFFSET) == "701"
    assert [r[0] for r in _q(temp_db, "SELECT kind FROM fills")] == ["skip"]
    assert len(fake_telegram["sent"]) == 1 and "No pude registrar" not in fake_telegram["sent"][0]
    import notifications.telegram_commands as tc
    assert tc._update_failures == {}               # sin restos en memoria


def test_fallo_permanente_se_abandona_tras_3_intentos_y_2_horas(temp_db, fake_telegram, monkeypatch):
    import data.database as db
    import notifications.telegram_commands as tc
    from notifications.telegram_commands import poll_commands, KV_OFFSET
    sid = _sig(temp_db, "BTCUSDT", "long", T0, 100.0)
    _fake_updates(monkeypatch, [_upd(800, f"/fill {sid} 100.5")])
    monkeypatch.setattr(db, "save_fill", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("caída")))
    clock = _clock(monkeypatch, 120)
    polls = 0
    while temp_db.kv_get(KV_OFFSET) != "801":
        poll_commands(log_failures=False)
        polls += 1
        assert polls < 200
    # Primer fallo en T0+120; se abandona en el primer sondeo con ≥ 2 h desde él.
    waited = clock["t"] - (T0 + 120)
    assert tc.UPDATE_RETRY_WINDOW_S <= waited < tc.UPDATE_RETRY_WINDOW_S + 120
    assert polls == tc.UPDATE_RETRY_WINDOW_S // 120 + 1
    assert len(fake_telegram["sent"]) == 1 and "No pude registrar" in fake_telegram["sent"][0]
    assert tc._update_failures == {}
    # Con sondeos muy espaciados (> 2 h) siguen haciendo falta los 3 intentos.
    monkeypatch.setattr(tc, "_update_failures", {})
    _fake_updates(monkeypatch, [_upd(801, f"/fill {sid} 100.5")])
    _clock(monkeypatch, 5 * 3600)
    poll_commands()
    poll_commands()
    assert temp_db.kv_get(KV_OFFSET) == "801" and len(fake_telegram["sent"]) == 1
    poll_commands()
    assert temp_db.kv_get(KV_OFFSET) == "802" and len(fake_telegram["sent"]) == 2


# ---------------------------------------------------------------------------
# 2. restore: no duplica señales ni cuelga /fill de señales equivocadas
# ---------------------------------------------------------------------------

def _old_db_and_backup(monkeypatch, tmp_path):
    from data import backup as B
    old = _use_db(monkeypatch, tmp_path / "old.db")
    old.init_db()
    for i in range(5):                                   # ids 1..5, SOL long
        _sig(old, "SOLUSDT", "long", T0 + i * 14400, 50.0 + i)
    old.save_fill(2, "fill", price=51.3, update_id=555)
    old.save_fill(4, "fill", price=53.1, update_id=556)
    f = B.export_backup(tmp_path / "b.json.gz")["path"]
    return f


def test_restore_bd_recreada_con_senales_del_worker_no_duplica_ni_cruza_fills(tmp_path, monkeypatch):
    """Escenario de la revisión: el worker guardó en la BD nueva id 1 = el setup de la
    antigua id 5 e id 2 = otra señal (XRP short)."""
    from data import backup as B
    from risk.portfolio import load_open_positions
    f = _old_db_and_backup(monkeypatch, tmp_path)
    new = _use_db(monkeypatch, tmp_path / "new.db")
    new.init_db()
    _sig(new, "SOLUSDT", "long", T0 + 4 * 14400, 54.0)     # id 1 = setup de la antigua 5
    _sig(new, "XRPUSDT", "short", T0 + 9 * 14400, 900.0)   # id 2 = otra señal
    prev = B.restore_backup(f)
    txt = B.format_restore(prev)
    assert "ESE SETUP YA ESTÁ" in txt and "NO hagas --apply" in txt and "worker PARADO" in txt
    assert "de señales en conflicto u omitidas" in txt
    rep = B.restore_backup(f, apply=True)
    plans = {p["table"]: p for p in rep["plans"]}
    assert plans["signals"]["conflict"] == 2 and plans["signals"]["logical"] == 1
    assert plans["signals"]["logical_ids"] == [5]
    assert rep["inserted"]["signals"] == 2                  # solo las antiguas 3 y 4
    assert plans["fills"]["orphan"] == 1 and plans["fills"]["orphan_ids"] == [2]
    assert rep["inserted"]["fills"] == 1                    # el /fill de la 4 sí
    # (a) sin señales duplicadas por setup
    assert _q(new, "SELECT symbol, direction, setup_ts FROM signals GROUP BY 1, 2, 3 "
                   "HAVING COUNT(*) > 1") == []
    # (b) ningún /fill pegado a otra señal: el único es el de la antigua 4 (SOL 53)
    assert _q(new, "SELECT f.signal_id, f.price, s.symbol, s.entry_price FROM fills f "
                   "JOIN signals s ON s.id = f.signal_id") == [(4, 53.1, "SOLUSDT", 53.0)]
    # La cartera no cuenta dos veces el mismo setup.
    ids = sorted(p["id"] for p in load_open_positions())
    assert ids == [1, 2, 3, 4]


def test_restore_misma_senal_con_otro_estado_conserva_sus_fills(tmp_path, monkeypatch):
    """Conflicto de la MISMA señal (p. ej. ya resuelta en la BD): sus /fill sí valen."""
    from data import backup as B
    f = _old_db_and_backup(monkeypatch, tmp_path)
    new = _use_db(monkeypatch, tmp_path / "same.db")
    new.init_db()
    _sig(new, "SOLUSDT", "long", T0 + 14400, 51.0, sid=2, resultado_real="win", pnl_r=1.5)
    rep = B.restore_backup(f, apply=True)
    plans = {p["table"]: p for p in rep["plans"]}
    assert plans["signals"]["conflict"] == 1 and plans["signals"]["logical"] == 0
    assert plans["fills"]["orphan"] == 0 and rep["inserted"]["fills"] == 2
    assert _q(new, "SELECT resultado_real FROM signals WHERE id = 2") == [("win",)]   # la BD manda
    assert sorted(_q(new, "SELECT signal_id, price FROM fills")) == [(2, 51.3), (4, 53.1)]


def test_restore_bd_vacia_reproduce_tambien_duplicados_de_origen(tmp_path, monkeypatch):
    """La clave lógica solo se compara con la BD de destino: si la BD de origen tenía
    dos filas con el mismo setup, una restauración en BD vacía las reproduce tal cual."""
    from data import backup as B
    src = _use_db(monkeypatch, tmp_path / "src.db")
    src.init_db()
    _sig(src, "BTCUSDT", "long", T0, 100.0)
    _sig(src, "BTCUSDT", "long", T0, 100.0, mode="shadow")
    e = _sig(src, "ETHUSDT", "short", T0, 3000.0)
    with src.get_engine().begin() as c:                     # señal antigua sin setup_ts
        c.execute(text("UPDATE signals SET setup_ts = NULL WHERE id = :i"), {"i": e})
    src.save_fill(2, "fill", price=100.2, update_id=9)
    before = sorted(_q(src, "SELECT * FROM signals")), sorted(_q(src, "SELECT * FROM fills"))
    f = B.export_backup(tmp_path / "s.json.gz")["path"]
    dst = _use_db(monkeypatch, tmp_path / "dst.db")
    dst.init_db()
    rep = B.restore_backup(f, apply=True)
    assert rep["inserted"]["signals"] == 3 and rep["inserted"]["fills"] == 1
    assert (sorted(_q(dst, "SELECT * FROM signals")), sorted(_q(dst, "SELECT * FROM fills"))) == before
    assert "NO hagas --apply" not in B.format_restore(rep)
    rep2 = B.restore_backup(f, apply=True)                  # idempotente
    assert sum(rep2["inserted"].values()) == 0
    assert all(p.get("logical", 0) == 0 and p.get("orphan", 0) == 0 for p in rep2["plans"])


def test_reglas_restaurar_con_worker_parado_y_bd_vacia():
    txt = (ROOT / "REGLAS_DE_DECISION.md").read_text(encoding="utf-8")
    sec = txt[txt.index("**Restaurar**"):txt.index("**Histórico propio de velas")]
    for frag in ("worker PARADO", "BD VACÍA", "NO hagas", "mismo setup", "no se insertan"):
        assert frag in sec, frag


# ---------------------------------------------------------------------------
# 3. Dashboard
# ---------------------------------------------------------------------------

def test_resultado_realizado_hoy_signo_natural_y_sin_menos_cero(temp_db, monkeypatch):
    import risk.portfolio as rp
    from dashboard import views
    assert views.signed_pct(-0.0) == "+0.00 %" and views.signed_pct(-0.001) == "+0.00 %"
    assert views.signed_pct(0.375) == "+0.38 %" and views.signed_pct(-1.25) == "-1.25 %"
    p = views.portfolio_view()                            # BD vacía
    assert views.signed_pct(p["day_result_pct"]) == "+0.00 %"
    assert str(p["day_loss_pct"]) == "0.0"                # sin "-0.0"
    for loss, shown in ((-0.00375, "+0.38 %"), (0.02, "-2.00 %"), (-0.0013, "+0.13 %")):
        monkeypatch.setattr(rp, "realized_loss_today", lambda now=None, capital=None, v=loss: v)
        p = views.portfolio_view()
        assert views.signed_pct(p["day_result_pct"]) == shown
        assert p["day_loss_pct"] == pytest.approx(loss * 100)   # la convención de R4 no cambia
    assert "3 %" in views.DAY_RESULT_HELP and "R4" in views.DAY_RESULT_HELP


def test_resultado_realizado_hoy_con_senales_reales(temp_db, monkeypatch):
    import config.settings as cs
    from dashboard import views
    monkeypatch.setattr(cs, "TRADING_CAPITAL", 1000.0)
    now = pd.Timestamp("2026-11-17 12:00", tz="UTC")
    r = int(now.timestamp()) - 60
    # Riesgo 0,25 %: 166,67 USD de nominal × 1,5 % de stop = 2,5 USD de 1.000.
    _sig(temp_db, "BTCUSDT", "long", T0, 100000.0, stop_loss=98500.0, take_profit=102250.0,
         position_size=166.67, resultado_real="win", pnl_r=1.5, resolved_ts=r, resolved_by="auto")
    p = views.portfolio_view(now=now)
    assert views.signed_pct(p["day_result_pct"]) == "+0.38 %"


def test_sl_y_tp_con_su_signo_real_en_largos_y_cortos():
    from dashboard import views
    assert views.level_deltas(4168.40, 4323.68, 3961.36) == ("+3.73%", "-4.97%")   # corto de oro
    assert views.level_deltas(100.0, 97.0, 104.5) == ("-3.00%", "+4.50%")          # largo
    src = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert 'delta_color="inverse"' not in src and "views.level_deltas(" in src
    assert "Pérdida realizada hoy" not in src and "Resultado realizado hoy (UTC)" in src
    assert "views.DAY_RESULT_HELP" in src


def _resolved(db, symbol, direction, entry, sl, tp, exit_price, pnl_r, res):
    return _sig(db, symbol, direction, T0, entry, stop_loss=sl, take_profit=tp, exit_price=exit_price,
                pnl_r=pnl_r, resultado_real=res, resolved_ts=T0 + 86400, resolved_by="auto")


def test_oro_aparte_en_ejecuciones_y_resumen(temp_db):
    from dashboard import views
    from monitoring.daily_summary import fills_vs_theory, _fills_block
    # Oro SHORT de la revisión: cerrado en el SL; /fill al precio de XM (0,44 % abajo).
    g = _resolved(temp_db, "XAUUSD", "short", 4168.40, 4323.68, 3961.36, 4323.68, -1.0, "loss")
    temp_db.save_fill(g, "fill", price=4150.0, update_id=1)
    b = _resolved(temp_db, "BTCUSDT", "long", 100.0, 97.0, 104.5, 104.5, 1.5, "win")
    temp_db.save_fill(b, "fill", price=101.0, update_id=2)
    fv = fills_vs_theory()
    d_gold = -(4323.68 - 4150.0) / (4323.68 - 4168.40) + 1.0
    d_btc = (104.5 - 101.0) / 3.0 - 1.5
    assert fv["n_resolved"] == 2 and fv["mean_diff_r"] == pytest.approx((d_gold + d_btc) / 2)
    assert fv["gold"]["n"] == 1 and fv["gold"]["mean_diff_r"] == pytest.approx(d_gold)
    assert fv["crypto"]["n"] == 1 and fv["crypto"]["mean_diff_r"] == pytest.approx(d_btc)
    block = "\n".join(_fills_block())
    assert "BTC/ETH: -0.333R en 1; oro: -0.118R en 1" in block
    assert "futuro GC=F / contado de XM" in block and "no es coste" in block
    t = views.executions()["table"].set_index("señal")
    assert "no es coste" in t.loc[g, "aviso"] and t.loc[b, "aviso"] == ""
    src = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "media BTC/ETH" in src and "GC=F" in src


def test_regla6_solo_con_btc_eth_y_sin_oro_no_hay_linea_extra(temp_db):
    from monitoring.daily_summary import _fills_block
    for i in range(20):
        sid = _resolved(temp_db, "ETHUSDT", "long", 100.0, 98.0, 103.0, 103.0, 1.5, "win")
        temp_db.save_fill(sid, "fill", price=100.5, update_id=100 + i)      # −0,25R cada una
    block = "\n".join(_fills_block())
    assert "Regla 6: con ≥20 fills de BTC/ETH" in block and "oro:" not in block


def test_regla6_no_salta_solo_por_el_oro(temp_db):
    from monitoring.daily_summary import _fills_block
    for i in range(25):
        sid = _resolved(temp_db, "XAUUSD", "short", 4168.40, 4323.68, 3961.36, 4323.68, -1.0, "loss")
        temp_db.save_fill(sid, "fill", price=4150.0, update_id=200 + i)
    block = "\n".join(_fills_block())
    assert "oro: -0.118R en 25" in block and "⚠️ Regla 6" not in block


# ---------------------------------------------------------------------------
# 4. Copia mensual: el día 7 no promete reintento
# ---------------------------------------------------------------------------

def _utc(s):
    return _dt.datetime.fromisoformat(s).replace(tzinfo=_dt.timezone.utc)


def test_copia_mensual_dia_7_ultimo_intento(temp_db):
    from data import backup as B
    fail = lambda *a, **k: (False, "Telegram no respondió en 10 s")   # noqa: E731
    B.run_monthly_backup(now=_utc("2026-12-02 03:00"), send_document=fail)
    line = B.summary_line(int(_utc("2026-12-02 03:01").timestamp()))
    assert "Se reintenta mañana" in line
    B.run_monthly_backup(now=_utc("2026-12-07 03:00"), send_document=fail)
    line = B.summary_line(int(_utc("2026-12-07 03:01").timestamp()))
    assert "NO enviada" in line and "Se reintenta mañana" not in line
    assert "último intento del mes" in line and "backup --send" in line
    assert B.monthly_backup_due(_utc("2026-12-08 03:00")) is False          # en efecto, no hay más


def test_reglas_reintento_de_comandos_y_dashboard():
    txt = (ROOT / "REGLAS_DE_DECISION.md").read_text(encoding="utf-8")
    assert "3 intentos **y** 2 horas desde el" in txt
    assert "Resultado\n    realizado hoy (UTC)" in txt and "solo con BTC/ETH" in txt
    assert "con signo real" in txt and "último intento" in txt
