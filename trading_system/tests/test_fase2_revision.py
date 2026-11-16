"""
Pruebas de las correcciones de la revisión adversarial de la Fase 2:
  1. --final registra el acceso de TODAS las configuraciones que enseña (candidato
     y configuración actual) y avisa si la actual ya vio la reserva;
  2. censura de la reserva/OKX: solo operaciones con el horizonte completo en los
     datos, sea cual sea la salida (no se aceptan las resueltas pronto por SL/TP);
  3. monitor: si el envío por Telegram de un aviso grave falla, se reintenta en el
     ciclo siguiente (respetando el throttling) y caduca a las 24 h;
  4. informe: rango p10–p90 de la CPCV explicado, con peor/mejor; pruebas sin
     operaciones; avisos de la línea base (dentro de muestra, c/d son para
     candidatos); folds con N < 10 marcados; costes en la reserva y en OKX;
  5. mensaje de señal: la línea de eventos dice que es solo un aviso.

    cd trading_system && python -m pytest tests/test_fase2_revision.py -q

Sin red: datos sintéticos, detector sustituido, Telegram sustituido, SQLite temporal.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import protocol as P         # noqa: E402

UTC = "UTC"


# ---------------------------------------------------------------------------
# Datos sintéticos (mismo esquema que test_fase2_costes_registro.py)
# ---------------------------------------------------------------------------

def _ohlcv(start, end, freq="4h", seed=7):
    idx = pd.date_range(start, end, freq=freq, tz=UTC, inclusive="left")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(idx))))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1000.0}, index=idx)


def _setups(df, every=37, seed=3):
    rng = np.random.default_rng(seed)
    pos = np.arange(30, len(df), every)
    return pd.DataFrame({
        "ts": df.index[pos],
        "direction": np.where(np.arange(len(pos)) % 2 == 0, "long", "short"),
        "setup_type": "breakout",
        "raw_score": np.round(rng.uniform(0.3, 0.8, len(pos)), 3),
        "atr": (df["high"] - df["low"]).rolling(14, min_periods=1).mean().to_numpy()[pos],
    })


@pytest.fixture()
def fake_detector(monkeypatch):
    monkeypatch.setattr(P, "detect_universe",
                        lambda symbol, df_4h, df_daily=None, overrides=None, params_override=None:
                        _setups(df_4h))


@pytest.fixture()
def registry(tmp_path):
    return tmp_path / "trials_registry.jsonl"


def _live_ok(symbol, df_4h, df_daily):
    return {"symbol": symbol, "match": True, "approved": 5, "emitted": 5, "repeated": 0,
            "missing": [], "extra": [], "pct_age0": 1.0, "pct_age_le1": 1.0}


def _loader(symbol):
    df = _ohlcv("2022-04-27", "2026-11-15")
    return df, P.resample_daily_utc(df)


_OKX_DF = _ohlcv("2019-12-16 04:00", "2022-05-01", seed=11)


def _okx_ok(symbol, download):
    return {"df_4h": _OKX_DF, "df_daily": P.resample_daily_utc(_OKX_DF), "origin": "prueba", "error": None}


def _okx_err(symbol, download):
    return {"df_4h": None, "df_daily": None, "origin": None, "error": "sin red"}


def _finals(registry):
    return [e for e in P.read_registry(registry) if e["kind"] == "final_access"]


# ---------------------------------------------------------------------------
# 1. Accesos --final de la configuración actual
# ---------------------------------------------------------------------------

def test_final_sin_candidato_tras_final_con_candidato_avisa_de_la_actual(fake_detector, registry):
    cand = registry.parent / "cand.json"
    cand.write_text(json.dumps({"symbol": "ETHUSDT", "ml_threshold": 0.45, "label": "ETH 0,45"}),
                    encoding="utf-8")
    lines = []
    code, rep = P.run_protocol(None, candidate_path=str(cand), final=True, registry=registry,
                               load_data=_loader, live_check=_live_ok, okx_loader=_okx_ok,
                               echo=lines.append)
    assert code == 0
    cur_hash = rep["symbols"][0]["current"]["hash"]
    assert "YA VIO" not in "\n".join(lines)
    fa = _finals(registry)
    assert [(e["role"], e["okx"]) for e in fa] == [("candidate", True), ("reference", True)]
    assert fa[1]["hash"] == cur_hash
    # --final SIN candidato: la configuración actual ya vio la reserva y OKX en el anterior.
    lines = []
    code, rep = P.run_protocol("ETHUSDT", final=True, registry=registry, load_data=_loader,
                               live_check=_live_ok, okx_loader=_okx_err, echo=lines.append)
    txt = "\n".join(lines)
    assert code == 0
    assert any("LA CONFIGURACIÓN ACTUAL YA VIO LA RESERVA Y OKX" in w for w in rep["warnings"])
    assert "ESTE CANDIDATO" not in txt
    # La nota de independencia de la reserva lo dice (no solo "posterior al 18-sep-2026").
    assert "YA VIO esta reserva en 1 --final anterior(es)" in txt
    assert rep["symbols"][0]["prior_final_access"]["current"] == [fa[1]["run"]]
    fa = _finals(registry)
    assert len(fa) == 3 and fa[2]["hash"] == cur_hash and fa[2]["role"] == "current"
    assert fa[2]["okx"] is False                 # OKX con error: no se enseñó, no se registra como visto


def test_final_candidato_igual_a_la_actual_una_sola_linea(fake_detector, registry):
    cur = P.current_variant("ETHUSDT")
    cand = registry.parent / "cand.json"
    cand.write_text(json.dumps({"symbol": "ETHUSDT", "ml_threshold": cur["ml_threshold"]}), encoding="utf-8")
    code, rep = P.run_protocol(None, candidate_path=str(cand), final=True, registry=registry,
                               load_data=_loader, live_check=_live_ok, okx_loader=_okx_ok,
                               echo=lambda *_: None)
    assert code == 0
    fa = _finals(registry)
    assert len(fa) == 1 and fa[0]["role"] == "candidate" and fa[0]["hash"] == rep["symbols"][0]["current"]["hash"]
    # Una evaluación posterior de la actual sola ve ese acceso.
    code, rep = P.run_protocol("ETHUSDT", final=True, registry=registry, load_data=_loader,
                               live_check=_live_ok, okx_loader=_okx_ok, echo=lambda *_: None)
    assert any("LA CONFIGURACIÓN ACTUAL YA VIO" in w for w in rep["warnings"])


def test_registro_del_repo_tiene_el_acceso_de_la_eth_actual():
    """El --final de ETH 0,45 (2026-11-16T00:57:10Z) enseñó también la reserva y OKX de la
    ETH actual: su acceso se añadió a posteriori (sin reescribir nada)."""
    entries = P.read_registry(P.REGISTRY_PATH)
    fa = [e for e in entries if e.get("kind") == "final_access"]
    assert fa[0]["hash"] == "95198bb3a9fcd374" and "role" not in fa[0]   # línea original intacta
    ref = [e for e in fa if e["hash"] == "ee31ab019358db9e"]
    assert ref and ref[0]["run"] == "2026-11-16T00:57:10Z" and ref[0]["okx"] is True
    assert ref[0]["role"] == "reference" and ref[0]["symbol"] == "ETHUSDT"
    # Es la ETH actual de settings (si cambia la estrategia, cambia el hash y esto avisa).
    assert P.current_variant("ETHUSDT") and next(
        g for g in P.build_grid("ETHUSDT") if g["is_current"])["hash"] == "ee31ab019358db9e"


# ---------------------------------------------------------------------------
# 2. Censura: horizonte completo sea cual sea la salida
# ---------------------------------------------------------------------------

def test_censura_horizonte_completo_sin_mirar_la_salida():
    df = _ohlcv("2025-10-01", "2025-12-01")
    n = len(df)
    fwd = 24
    # Setup con el horizonte sin cumplir y un TP que se toca enseguida (ATR minúsculo):
    # antes contaba como "completo" por haber salido pronto; ahora no.
    pos_early = n - 10
    pos_ok = n - 1 - fwd                        # horizonte justo dentro de los datos
    s = pd.DataFrame({"ts": df.index[[pos_ok, pos_early]], "direction": ["long", "long"],
                      "setup_type": "breakout", "raw_score": 0.9, "atr": [1e-4, 1e-4]})
    t = P.simulate_trades(df, s, 1.5, 2.25, fwd)
    assert list(t["exit_kind"]) != ["time", "time"]           # salen pronto por SL/TP
    assert t["exit_pos"].iloc[1] < n - 1
    assert list(t["complete"]) == [True, False]
    _, rsv = P.split_reserve(t, "BTCUSDT")
    assert list(rsv["pos"]) == [pos_ok]


# ---------------------------------------------------------------------------
# 3. Monitor: reintento de avisos graves que no se pudieron enviar
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    import data.database as db
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    db.init_db()
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)
    import monitoring.data_monitor as dm
    monkeypatch.setattr(dm, "_alert_state", {})
    monkeypatch.setattr(dm, "_alerted_keys", set())
    monkeypatch.setattr(dm, "_pending", {})
    yield db
    if db._engine is not None:
        db._engine.dispose()


def _flat_btc(end="2026-06-16 16:00", price=65835.3):
    idx = pd.date_range(end=pd.Timestamp(end, tz=UTC), periods=300, freq="4h")
    rng = np.random.default_rng(1)
    close = 60000 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
    open_ = np.r_[60000, close[:-1]]
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                       "low": np.minimum(open_, close) * 0.998, "close": close,
                       "volume": 100.0}, index=idx)
    df.attrs["source"] = "gateio"
    df.loc[idx[-1], ["open", "high", "low", "close"]] = price
    df.loc[idx[-1], "volume"] = 0.0
    return df


def test_monitor_reintenta_aviso_grave_si_falla_el_envio(temp_db, monkeypatch):
    import monitoring.data_monitor as dm
    import notifications.telegram as tg
    calls, results = [], [False, True]
    monkeypatch.setattr(tg, "send_data_quality_alert",
                        lambda s, items: calls.append((s, list(items))) or results.pop(0))
    t0 = pd.Timestamp("2026-06-16 20:05", tz=UTC)
    df = _flat_btc()
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0)
    assert len(calls) == 1 and dm._pending                 # falló: queda pendiente
    # Ciclo siguiente, misma vela (ya NO es nueva en la BD): segundo intento, que sale.
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0 + pd.Timedelta(hours=1))
    assert len(calls) == 2 and "65835.3" in calls[1][1][0]
    assert not dm._pending
    # Enviado: no se repite.
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0 + pd.Timedelta(hours=2))
    assert len(calls) == 2


def test_monitor_excepcion_al_enviar_deja_el_aviso_pendiente(temp_db, monkeypatch):
    import monitoring.data_monitor as dm
    import notifications.telegram as tg
    calls = []

    def _send(s, items):
        calls.append(s)
        if len(calls) == 1:
            raise RuntimeError("Telegram caído")
        return True
    monkeypatch.setattr(tg, "send_data_quality_alert", _send)
    t0 = pd.Timestamp("2026-06-16 20:05", tz=UTC)
    df = _flat_btc()
    assert dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0)      # no lanza
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0 + pd.Timedelta(hours=1))
    assert len(calls) == 2 and not dm._pending


def test_monitor_pendiente_se_reintenta_aunque_ya_no_sea_grave_y_caduca(temp_db, monkeypatch):
    import monitoring.data_monitor as dm
    import notifications.telegram as tg
    calls = []
    monkeypatch.setattr(tg, "send_data_quality_alert", lambda s, items: calls.append(s) and False)
    t0 = pd.Timestamp("2026-06-16 20:05", tz=UTC)
    dm.run_cycle_checks("BTCUSDT", _flat_btc(), None, notify=True, now=t0)
    assert len(calls) == 1
    # 12 h después la vela plana ya no está entre las 2 últimas (no es grave), pero el
    # aviso pendiente se reintenta.
    later = _flat_btc(end="2026-06-17 04:00")
    later.loc[later.index[-1], ["open", "high", "low", "close", "volume"]] = [66000, 66100, 65900, 66050, 10]
    dm.run_cycle_checks("BTCUSDT", later, None, notify=True, now=t0 + pd.Timedelta(hours=12))
    assert len(calls) == 2
    # Una excepción del envío cuenta como fallo (sigue pendiente, el ciclo no se cae).
    def _raise(s, items):
        calls.append(s)
        raise RuntimeError("Telegram caído")
    monkeypatch.setattr(tg, "send_data_quality_alert", _raise)
    dm.run_cycle_checks("BTCUSDT", later, None, notify=True, now=t0 + pd.Timedelta(hours=13))
    assert len(calls) == 3 and dm._pending
    # Pasadas 24 h desde el primer fallo, caduca (queda solo en el resumen nocturno).
    dm.run_cycle_checks("BTCUSDT", later, None, notify=True, now=t0 + pd.Timedelta(hours=25))
    assert len(calls) == 3 and not dm._pending


# ---------------------------------------------------------------------------
# 4. Informe
# ---------------------------------------------------------------------------

def _rob(n=14, n_comb=15, pbo=0.56, dsr=0.36):
    return {"cpcv": {"n_comb": n_comb, "times_current": 4, "times_candidate": 0,
                     "procedure": {"n": n, "median": 0.16, "p10": -0.08, "p90": 1.26, "min": -1.018,
                                   "max": 1.6, "frac_neg": 2 / n, "mean": 0.2},
                     "current_fixed": {"median": 0.45}},
            "pbo": {"pbo": pbo}, "dsr_current": {"dsr": dsr, "psr0": 0.93}, "n_trials": 460}


def test_resumen_cpcv_explica_el_rango_y_las_pruebas_sin_operaciones():
    txt = "\n".join(P._robustness_lines(_rob()))
    assert "en 8 de cada 10 pruebas entre -0.08 y +1.26" in txt
    assert "peor -1.02, mejor +1.60" in txt
    assert "por debajo de 0 en 2 de 14; 1 prueba(s) sin operaciones, no cuenta(n)" in txt
    assert "sin operaciones" not in "\n".join(P._robustness_lines(_rob(n=15)))


def test_linea_base_avisa_dentro_de_muestra_y_umbrales_de_candidato():
    res = [{"symbol": "BTCUSDT", "robustness": _rob()},
           {"symbol": "SPX500", "robustness": _rob(pbo=0.15, dsr=0.02)},
           {"symbol": "OKUSD", "robustness": _rob(pbo=0.10, dsr=0.99)}]
    v = P.evaluate_criteria(res, None)
    assert v["status"] == "baseline"
    assert "DENTRO DE MUESTRA" in v["notes"][0]
    assert "tampoco los pasa (BTCUSDT: PBO 0.56, DSR 0.36; SPX500: DSR 0.02)" in v["notes"][1]
    assert "OKUSD" not in v["notes"][1]
    # Si todas los pasaran, no se dice "tampoco".
    assert len(P.evaluate_criteria(res[2:], None)["notes"]) == 1


def test_folds_con_pocas_operaciones_marcados():
    row = {"label": "XM, swap 5 %", "decision": True, "n": 33, "er": 0.3, "ci_lo": 0.0, "ci_hi": 0.6,
           "win_rate": 0.5, "folds": [{"fold": "a", "n": 3, "er": 0.51}, {"fold": "b", "n": 21, "er": 0.3},
                                      {"fold": "c", "n": 0, "er": None}]}
    txt = "\n".join(P._table_lines([row]))
    assert "+0.51 (N=3 (!))" in txt and "(N=21)" in txt and "(N=0)" in txt and "(N=0 (!))" not in txt
    assert "(!) = menos de 10 operaciones" in txt


def test_informe_final_costes_en_reserva_y_okx_y_avisos_de_linea_base(fake_detector, registry):
    lines = []
    code, _ = P.run_protocol("BTCUSDT", final=True, registry=registry, load_data=_loader,
                             live_check=_live_ok, okx_loader=_okx_ok, echo=lines.append)
    txt = "\n".join(lines)
    assert code == 0
    rsv = txt[txt.index("RESERVA ("):txt.index("HOLDOUT OKX")]
    okx = txt[txt.index("HOLDOUT OKX"):txt.index("DETALLE TÉCNICO")]
    assert "coste medio por operación (XM swap 5 %): spread" in rsv
    assert "coste medio por operación (XM swap 5 %): spread" in okx
    assert "USD fijos" in okx
    assert "DENTRO DE MUESTRA" in txt[:txt.index("[BTCUSDT]")]
    assert "horizonte completo" in rsv


# ---------------------------------------------------------------------------
# 5. Mensaje de señal: la línea de eventos es solo un aviso
# ---------------------------------------------------------------------------

def test_linea_de_eventos_dice_que_es_solo_aviso():
    from notifications.telegram import event_warning_line
    line = event_warning_line(pd.Timestamp("2026-12-10 13:00", tz=UTC), pd.Timestamp("2026-12-10 14:00", tz=UTC))
    assert line.startswith("⚠️ <b>Evento de alto impacto dentro del plazo de la operación:</b> CPI 10-dic")
    assert line.endswith("<i>(solo aviso: no es motivo para no entrar)</i>")
