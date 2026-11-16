"""
Pruebas de la Fase 2, punto 2.1 (primera mitad) — puerta de validación:
  - backtest/costs.py: escenarios XM swap 0/5/10 % + "binance", noches de swap
    (cripto 7/semana; oro/SPX 5 con triple el miércoles), spread 1 por operación;
  - el escenario "binance" reproduce EXACTAMENTE validate._simulate_setup;
  - registro trials_registry.jsonl: semilla, hash estable, solo se añade, accesos
    --final y avisos, fallos de lectura/escritura → sin veredicto;
  - reserva intocable (PROTOCOL_RESERVE_START), folds temporales, rejilla de 15
    variantes (+ candidato);
  - precondición vivo = validador (si falla, el protocolo se detiene);
  - holdout OKX: tabla ohlcv (source='okx') o descarga en memoria, diario UTC;
  - CLI `validate --protocol`.

    cd trading_system && python -m pytest tests/test_fase2_costes_registro.py -q

Sin red: datos sintéticos, detector sustituido por un doble, BD SQLite temporal.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import costs as C            # noqa: E402
from backtest import protocol as P         # noqa: E402

UTC = "UTC"


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

def _ohlcv(start: str, end: str, freq: str = "4h", seed: int = 7) -> pd.DataFrame:
    """Paseo aleatorio OHLCV determinista."""
    idx = pd.date_range(start, end, freq=freq, tz=UTC, inclusive="left")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(idx))))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1000.0}, index=idx)


def _setups(df: pd.DataFrame, every: int = 37, seed: int = 3) -> pd.DataFrame:
    """Setups sintéticos: uno cada `every` velas, dirección y raw_score alternos."""
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
    """Sustituye la detección por setups sintéticos (rápido y sin depender del detector)."""
    def _detect(symbol, df_4h, df_daily=None, overrides=None, params_override=None):
        return _setups(df_4h)
    monkeypatch.setattr(P, "detect_universe", _detect)
    return _detect


@pytest.fixture()
def registry(tmp_path):
    return tmp_path / "trials_registry.jsonl"


def _live_ok(symbol, df_4h, df_daily):
    return {"symbol": symbol, "match": True, "approved": 5, "emitted": 5, "repeated": 0,
            "missing": [], "extra": [], "pct_age0": 1.0, "pct_age_le1": 1.0}


def _loader(symbol):
    if symbol in ("XAUUSD", "SPX500"):
        df = _ohlcv("2006-11-20", "2026-11-14", "1D")
        df = df[df.index.dayofweek < 5]
        return df, df
    df = _ohlcv("2022-04-27", "2026-11-15")
    return df, P.resample_daily_utc(df)


# ---------------------------------------------------------------------------
# Costes
# ---------------------------------------------------------------------------

def test_noches_de_swap_por_calendario():
    mon = pd.Timestamp("2026-11-16", tz=UTC)          # lunes
    d = lambda k: mon + pd.Timedelta(days=k)
    # session: lunes→jueves = lun 1 + mar 1 + mié 3 = 5; viernes→lunes = 1; semana = 7
    s = C.rollover_nights([mon, d(4), mon, mon], [d(3), d(7), d(7), mon], "session")
    assert s.tolist() == [5.0, 1.0, 7.0, 0.0]
    # crypto: todas las medianoches
    c = C.rollover_nights([d(4), mon], [d(7), d(14)], "crypto")
    assert c.tolist() == [3.0, 14.0]
    # mismo día con horas distintas → 0 noches; cruzar medianoche UTC → 1
    a = pd.Timestamp("2026-11-16 04:00", tz=UTC)
    assert C.rollover_nights([a], [a + pd.Timedelta(hours=14)], "crypto").tolist() == [0.0]
    assert C.rollover_nights([a], [a + pd.Timedelta(hours=22)], "crypto").tolist() == [1.0]
    with pytest.raises(ValueError):
        C.rollover_nights([a], [a], "semana")


def test_escenarios_de_costes_por_activo(monkeypatch):
    names = [s["name"] for s in C.cost_scenarios("BTCUSDT")]
    assert names == ["xm_swap0", "xm_swap5", "xm_swap10", "binance"]
    dec = [s["name"] for s in C.cost_scenarios("XAUUSD") if s["decision"]]
    assert dec == ["xm_swap5"]
    spreads = {sym: C.cost_scenarios(sym)[0]["spread"] for sym in ("BTCUSDT", "ETHUSDT", "XAUUSD", "SPX500")}
    assert spreads == {"BTCUSDT": 5.0, "ETHUSDT": 5.0, "XAUUSD": 0.45, "SPX500": 1.0}
    assert C.cost_scenarios("SPX500")[0]["swap_nights"] == "session"
    assert C.cost_scenarios("ETHUSDT")[0]["swap_nights"] == "crypto"
    assert all(s["commission"] == 0.0 for s in C.cost_scenarios("BTCUSDT") if s["kind"] == "broker")
    # Si algún día se conoce el swap real, aparece un escenario más (la decisión no cambia).
    from config import settings
    xm = json.loads(json.dumps(settings.BROKER_COSTS))
    xm["xm"]["SPX500"]["swap_pct_year"] = 0.031
    monkeypatch.setattr(settings, "BROKER_COSTS", xm)
    sc = C.cost_scenarios("SPX500")
    assert [s["name"] for s in sc] == ["xm_swap0", "xm_swap5", "xm_swap10", "xm_real", "binance"]
    assert sc[3]["swap_pct_year"] == 0.031 and not sc[3]["decision"]


def test_net_returns_xm_spread_y_swap():
    t = pd.DataFrame({"direction": ["long", "short"], "entry_mid": [100.0, 200.0],
                      "exit_mid": [110.0, 190.0], "risk": [5.0, 10.0],
                      "d_in": [pd.Timestamp("2026-11-16", tz=UTC)] * 2,       # lunes
                      "d_out": [pd.Timestamp("2026-11-19 12:00", tz=UTC)] * 2})  # jueves
    base = {"kind": "broker", "spread": 1.0, "commission": 0.0, "swap_nights": "session"}
    r0 = C.net_returns(t, dict(base, swap_pct_year=0.0))
    assert np.allclose(r0, [(10 - 1) / 5, (10 - 1) / 10])
    r5 = C.net_returns(t, dict(base, swap_pct_year=0.05))
    noches = 5.0                                                      # lun 1 + mar 1 + mié 3
    assert np.allclose(r5, [(10 - 1 - 100 * 0.05 / 365 * noches) / 5,
                            (10 - 1 - 200 * 0.05 / 365 * noches) / 10])
    # El swap es SIEMPRE un coste (también en cortos) y crece con el %.
    r10 = C.net_returns(t, dict(base, swap_pct_year=0.10))
    assert (r10 < r5).all() and (r5 < r0).all()
    # Comisión por lado sobre entrada + salida
    rc = C.net_returns(t, dict(base, swap_pct_year=0.0, commission=0.001))
    assert np.allclose(rc, [(10 - 1 - 210 * 0.001) / 5, (10 - 1 - 390 * 0.001) / 10])
    bd = C.cost_breakdown_R(t, dict(base, swap_pct_year=0.05))
    assert bd["nights_mean"] == 5.0 and np.isclose(bd["spread"], (1 / 5 + 1 / 10) / 2)


def test_escenario_binance_igual_que_el_validador():
    """simulate_trades + net_returns('binance') == validate._simulate_setup, setup a setup."""
    from backtest.validate import _simulate_setup
    from config.settings import BACKTEST
    df = _ohlcv("2024-01-01", "2024-03-15")
    s = _setups(df, every=11)
    # incluir setups en las últimas velas (horizonte incompleto y sin vela de entrada)
    extra = pd.DataFrame({"ts": df.index[[-3, -1]], "direction": ["long", "short"],
                          "setup_type": "breakout", "raw_score": 0.6, "atr": [1.0, 1.0]})
    s = pd.concat([s, extra], ignore_index=True)
    st, tg, fwd = 1.5, 2.25, 24
    t = P.simulate_trades(df, s, st, tg, fwd)
    bn = next(x for x in C.cost_scenarios("BTCUSDT") if x["name"] == "binance")
    R = C.net_returns(t, bn)
    ref = []
    for r in s.itertuples():
        pos = int(df.index.searchsorted(r.ts))
        v = _simulate_setup(df, pos, r.direction, float(r.atr), st, tg, fwd,
                            BACKTEST["commission"], BACKTEST["slippage"])
        if v is not None:
            ref.append(v)
    assert len(t) == len(ref) == len(s) - 1          # la última vela no tiene entrada
    assert np.allclose(R, ref, rtol=0, atol=1e-12)
    assert set(t["exit_kind"]) <= {"sl", "tp", "time"}
    assert not t["complete"].iloc[-1] and t["complete"].iloc[0]


# ---------------------------------------------------------------------------
# Registro de pruebas
# ---------------------------------------------------------------------------

def test_registro_semilla_y_hash_estable(registry):
    entries = P.ensure_registry(registry)
    first = registry.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first) == {"kind": "seed", "historical_trials": 400,
                                 "note": "variantes probadas antes de 2026-11 (informe final §4.1)"}
    assert list(json.loads(first)) == ["kind", "historical_trials", "note"]
    assert P.total_trials(entries) == 400
    # Hash FIJO: si cambia, se duplicarían pruebas en el registro versionado.
    p = {"symbol": "ETHUSDT", "ml_threshold": 0.45, "directions": ["long", "short"],
         "strategy": "abc123", "extra": {}}
    assert P.variant_hash(p) == "8de089c94eb1da13"
    assert P.variant_hash(dict(reversed(list(p.items())))) == "8de089c94eb1da13"
    q = P.variant_params("ETHUSDT", 0.45000000001, ("short", "long"), fingerprint="abc123")
    assert P.variant_hash(q) == "8de089c94eb1da13"
    # El registro del repo empieza por la semilla y es JSON válido línea a línea.
    repo = P.read_registry(P.REGISTRY_PATH)
    assert repo[0]["kind"] == "seed" and repo[0]["historical_trials"] == 400


def test_registro_solo_anade_variantes_nuevas(registry):
    entries = P.ensure_registry(registry)
    grid = P.build_grid("BTCUSDT")
    new = P.new_variant_entries(entries, grid, "run1")
    assert len(new) == 15 and len({e["hash"] for e in new}) == 15
    P._append_lines(registry, new)
    entries = P.read_registry(registry)
    assert P.total_trials(entries) == 415
    assert P.new_variant_entries(entries, P.build_grid("BTCUSDT"), "run2") == []
    # Otro activo → 15 más; un cambio de estrategia cambia todos los hashes.
    assert len(P.new_variant_entries(entries, P.build_grid("ETHUSDT"), "run2")) == 15
    with P.config_override("BTCUSDT", {"atr_target_mult": 3.0}):
        assert len(P.new_variant_entries(entries, P.build_grid("BTCUSDT"), "run3")) == 15
    # Acceso final
    fa = P.final_access_entry(grid[6], "run4", okx=True)
    P._append_lines(registry, [fa])
    entries = P.read_registry(registry)
    assert len(P.final_accesses(entries, grid[6]["hash"])) == 1
    assert P.final_accesses(entries, grid[0]["hash"]) == []


def test_registro_danado_o_no_escribible(registry, tmp_path):
    registry.write_text('{"kind": "seed", "historical_trials": 400}\n{roto\n', encoding="utf-8")
    with pytest.raises(P.RegistryError, match="línea 2"):
        P.read_registry(registry)
    bad = tmp_path / "sin_semilla.jsonl"
    bad.write_text('{"kind": "variant", "hash": "x"}\n', encoding="utf-8")
    with pytest.raises(P.RegistryError, match="semilla"):
        P.read_registry(bad)
    with pytest.raises(P.RegistryError):
        P._append_lines(tmp_path, [{"kind": "variant"}])        # es un directorio
    # Sin salto de línea final: se añade uno antes de escribir (no se corrompe).
    ok = tmp_path / "ok.jsonl"
    ok.write_text('{"kind": "seed", "historical_trials": 400}', encoding="utf-8")
    P._append_lines(ok, [{"kind": "variant", "hash": "h1"}])
    assert len(P.read_registry(ok)) == 2


def test_registry_path_por_entorno(monkeypatch, tmp_path):
    monkeypatch.delenv(P.REGISTRY_ENV, raising=False)
    assert P.registry_path() == P.REGISTRY_PATH
    assert P.REGISTRY_PATH.name == "trials_registry.jsonl" and P.REGISTRY_PATH.parent.name == "backtest"
    monkeypatch.setenv(P.REGISTRY_ENV, str(tmp_path / "x.jsonl"))
    assert P.registry_path() == tmp_path / "x.jsonl"


# ---------------------------------------------------------------------------
# Reserva, folds, rejilla
# ---------------------------------------------------------------------------

def test_reserva_y_folds():
    from config import settings
    assert settings.PROTOCOL_RESERVE_START == "2025-11-15"
    ts = pd.to_datetime(["2022-04-30 20:00", "2022-05-01 00:00", "2023-12-31 20:00", "2024-06-01 00:00",
                         "2025-11-14 20:00", "2025-11-15 00:00", "2026-11-10 00:00"], utc=True)
    t = pd.DataFrame({"ts": ts, "complete": [True] * 6 + [False]})
    sel, rsv = P.split_reserve(t, "BTCUSDT")
    assert sel["ts"].tolist() == list(ts[1:5])           # sin OKX (< 2022-05) ni reserva
    assert rsv["ts"].tolist() == [ts[5]]                 # la incompleta no cuenta
    f = P.assign_folds(ts, "ETHUSDT")
    assert list(f) == [None, "2022-05→2023-12", "2022-05→2023-12", "2024",
                       "2025-01→2025-11-14", None, None]
    fd = P.assign_folds(pd.to_datetime(["2006-11-20", "2014-12-31", "2015-01-02", "2021-01-04",
                                        "2025-11-14", "2025-11-17"], utc=True), "XAUUSD")
    assert list(fd) == ["2006–2014", "2006–2014", "2015–2020", "2021→2025-11-14", "2021→2025-11-14", None]
    sel_d, _ = P.split_reserve(pd.DataFrame({"ts": pd.to_datetime(["2006-11-21"], utc=True),
                                             "complete": [True]}), "SPX500")
    assert len(sel_d) == 1


def test_rejilla_de_15_variantes_y_candidato():
    g = P.build_grid("BTCUSDT")
    assert len(g) == 15 and sum(v["is_current"] for v in g) == 1
    assert sorted({v["ml_threshold"] for v in g}) == [0.45, 0.5, 0.55, 0.6, 0.65]
    cur = next(v for v in g if v["is_current"])
    assert cur["ml_threshold"] == 0.55 and cur["directions"] == ("long", "short")
    spx = next(v for v in P.build_grid("SPX500") if v["is_current"])
    assert spx["ml_threshold"] == 0.12 and spx["directions"] == ("long",)
    # Candidato "ETH con umbral 0,45" = variante de la rejilla (0,55 − 0,10): no se duplica.
    ge = P.build_grid("ETHUSDT", {"symbol": "ETHUSDT", "ml_threshold": 0.45, "label": "ETH 0,45"})
    assert len(ge) == 15 and sum(v["is_candidate"] for v in ge) == 1
    assert next(v for v in ge if v["is_candidate"])["ml_threshold"] == 0.45
    g2 = P.build_grid("ETHUSDT", {"symbol": "ETHUSDT", "ml_threshold": 0.47})
    assert len(g2) == 16 and g2[-1]["id"] == "candidato"
    g3 = P.build_grid("ETHUSDT", {"symbol": "ETHUSDT", "overrides": {"adx_min_trend": 25}})
    assert len(g3) == 16 and g3[-1]["extra"] == {"overrides": {"adx_min_trend": 25}}
    # Máscaras: umbral >= y dirección
    t = pd.DataFrame({"raw_score": [0.55, 0.5499, 0.7], "direction": ["long", "long", "short"]})
    assert P.variant_mask(t, cur).tolist() == [True, False, True]
    assert P.variant_mask(t, next(v for v in g if v["id"] == "thr0.55_long")).tolist() == [True, False, False]


def test_config_override_restaura():
    from config import settings
    before = json.dumps(settings.ASSETS["ETHUSDT"], sort_keys=True)
    had = "ETHUSDT" in settings.ASSET_PARAMS_OVERRIDE
    with P.config_override("ETHUSDT", {"ml_threshold": 0.1, "setups": None},
                           {"breakout": {"adx_min_trend": 99}}):
        assert settings.ASSETS["ETHUSDT"]["ml_threshold"] == 0.1
        assert "setups" not in settings.ASSETS["ETHUSDT"]
    assert json.dumps(settings.ASSETS["ETHUSDT"], sort_keys=True) == before
    assert ("ETHUSDT" in settings.ASSET_PARAMS_OVERRIDE) == had


def test_bootstrap_determinista_y_resumen():
    x = np.random.default_rng(1).normal(0.2, 1.0, 60)
    assert P.bootstrap_ci(x) == P.bootstrap_ci(x)
    lo, hi = P.bootstrap_ci(x)
    assert lo < x.mean() < hi
    s = P.summarize([1.0, -1.0])
    assert s["n"] == 2 and s["ci_lo"] is None and s["er"] == 0.0
    assert P.summarize([])["n"] == 0


# ---------------------------------------------------------------------------
# Protocolo de punta a punta (con dobles)
# ---------------------------------------------------------------------------

def test_protocolo_linea_base(fake_detector, registry, tmp_path):
    out = tmp_path / "informe.json"
    lines = []
    code, rep = P.run_protocol("BTCUSDT", registry=registry, load_data=_loader,
                               live_check=_live_ok, out=str(out), echo=lines.append)
    text_ = "\n".join(lines)
    assert code == 0, text_
    assert rep["verdict"]["status"] == "baseline"
    assert text_.index("VEREDICTO") < text_.index("Escenario de costes") < text_.index("DETALLE TÉCNICO")
    assert "XM, swap 5 %  ← decisión" in text_
    r = rep["symbols"][0]
    assert [row["scenario"] for row in r["baseline"]] == ["xm_swap0", "xm_swap5", "xm_swap10", "binance"]
    assert [f["fold"] for f in r["baseline"][1]["folds"]] == ["2022-05→2023-12", "2024", "2025-01→2025-11-14"]
    assert r["reserve"] is None and r["okx"] is None            # sin --final no se mira
    assert pd.to_datetime(r["_trades"]["ts"]).max() < pd.Timestamp("2025-11-15", tz=UTC)
    assert set(r["_masks"]) == {v["id"] for v in r["grid"]} and len(r["_masks"]) == 15
    assert len(r["_R"]["xm_swap5"]) == len(r["_trades"])
    # Más swap → menos E[R]; el número de operaciones no cambia entre escenarios.
    ers = [row["er"] for row in r["baseline"][:3]]
    assert ers[0] > ers[1] > ers[2]
    assert len({row["n"] for row in r["baseline"]}) == 1
    # Registro: 15 variantes nuevas; una segunda ejecución no añade nada.
    assert rep["registry"]["new"] == 15 and rep["registry"]["total_trials"] == 415
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "_trades" not in json.dumps(data) and data["verdict"]["status"] == "baseline"
    code2, rep2 = P.run_protocol("BTCUSDT", registry=registry, load_data=_loader,
                                 live_check=_live_ok, echo=lambda *_: None)
    assert code2 == 0 and rep2["registry"]["new"] == 0
    assert len(P.read_registry(registry)) == 16


def test_protocolo_se_detiene_si_vivo_distinto_de_validador(fake_detector, registry):
    lines = []
    bad = lambda s, d4, d1: dict(_live_ok(s, d4, d1), match=False, emitted=7, repeated=2)
    code, rep = P.run_protocol("ETHUSDT", registry=registry, load_data=_loader,
                               live_check=bad, echo=lines.append)
    assert code == 1 and rep["verdict"] is None
    assert "PROTOCOLO DETENIDO" in "\n".join(lines)
    assert len(P.read_registry(registry)) == 1                 # nada registrado


def test_protocolo_sin_registro_no_da_veredicto(fake_detector, registry, monkeypatch):
    registry.write_text("no es json\n", encoding="utf-8")
    lines = []
    code, rep = P.run_protocol("BTCUSDT", registry=registry, load_data=_loader,
                               live_check=_live_ok, echo=lines.append)
    assert code == 2 and rep["verdict"] is None and "SIN VEREDICTO" in "\n".join(lines)
    # Registro legible pero no escribible: informe sin veredicto y SIN la reserva.
    registry.unlink()

    def _fail(path, entries):
        if entries and entries[0].get("kind") != "seed":
            raise P.RegistryError("disco lleno")
        return orig(path, entries)
    orig = P._append_lines
    monkeypatch.setattr(P, "_append_lines", _fail)
    lines = []
    okx = lambda s, d: {"df_4h": None, "df_daily": None, "origin": None, "error": "sin datos"}
    code, rep = P.run_protocol("BTCUSDT", final=True, registry=registry, load_data=_loader,
                               live_check=_live_ok, okx_loader=okx, echo=lines.append)
    txt = "\n".join(lines)
    assert code == 2 and rep["verdict"] is None
    assert "SIN VEREDICTO" in txt and "disco lleno" in txt and "RESERVA (" not in txt
    assert rep["symbols"][0]["reserve"] is None


def test_protocolo_final_registra_acceso_y_avisa(fake_detector, registry):
    okx_df = _ohlcv("2019-12-16 04:00", "2022-05-01", seed=11)
    okx = lambda s, d: {"df_4h": okx_df, "df_daily": P.resample_daily_utc(okx_df),
                        "origin": "prueba", "error": None}
    cand = registry.parent / "cand.json"
    cand.write_text(json.dumps({"symbol": "ETHUSDT", "ml_threshold": 0.45, "label": "ETH 0,45"}),
                    encoding="utf-8")
    lines = []
    code, rep = P.run_protocol(None, candidate_path=str(cand), final=True, registry=registry,
                               load_data=_loader, live_check=_live_ok, okx_loader=okx,
                               echo=lines.append)
    txt = "\n".join(lines)
    assert code == 0, txt
    assert [r["symbol"] for r in rep["symbols"]] == ["ETHUSDT"]   # solo el activo del candidato
    r = rep["symbols"][0]
    assert r["reserve"]["current"] and r["reserve"]["candidate"]
    assert r["okx"]["origin"] == "prueba" and r["okx"]["current"]
    assert "YA VIO LA RESERVA" not in txt
    assert rep["verdict"]["status"] in ("rejected", "approved")    # criterios a)–e) con --final
    assert rep["verdict"]["criteria"]["e"]["ok"] is not None
    entries = P.read_registry(registry)
    fa = [e for e in entries if e["kind"] == "final_access"]
    # Una línea por configuración enseñada: el candidato y la actual (referencia).
    assert [e["role"] for e in fa] == ["candidate", "reference"]
    assert all(e["symbol"] == "ETHUSDT" and e["okx"] is True for e in fa)
    assert fa[0]["hash"] == next(g for g in r["grid"] if g["is_candidate"])["hash"]
    assert fa[1]["hash"] == r["current"]["hash"]
    # Segundo acceso del mismo candidato → aviso en grande (candidato y actual).
    lines = []
    P.run_protocol(None, candidate_path=str(cand), final=True, registry=registry,
                   load_data=_loader, live_check=_live_ok, okx_loader=okx, echo=lines.append)
    assert "ESTE CANDIDATO YA VIO LA RESERVA" in "\n".join(lines)
    assert "LA CONFIGURACIÓN ACTUAL YA VIO LA RESERVA Y OKX" in "\n".join(lines)
    assert len([e for e in P.read_registry(registry) if e["kind"] == "final_access"]) == 4


def test_candidato_invalido(registry, tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"symbol": "NOPE"}), encoding="utf-8")
    with pytest.raises(P.ProtocolError):
        P.load_candidate(p)
    p.write_text(json.dumps({"symbol": "ETHUSDT", "directions": ["up"]}), encoding="utf-8")
    with pytest.raises(P.ProtocolError):
        P.load_candidate(p)
    p.write_text(json.dumps({"symbol": "ETHUSDT"}), encoding="utf-8")
    lines = []
    code, _ = P.run_protocol("BTCUSDT", candidate_path=str(p), registry=registry,
                             load_data=_loader, live_check=_live_ok, echo=lines.append)
    assert code == 1 and "ETHUSDT" in lines[-1]


# ---------------------------------------------------------------------------
# Holdout OKX
# ---------------------------------------------------------------------------

def test_diario_utc_desde_4h():
    df = _ohlcv("2020-01-01 08:00", "2020-01-04 12:00")
    d = P.resample_daily_utc(df)
    assert list(d.index) == list(pd.to_datetime(["2020-01-02", "2020-01-03"], utc=True))
    day = df.loc["2020-01-02"]
    assert d.iloc[0]["open"] == day["open"].iloc[0] and d.iloc[0]["close"] == day["close"].iloc[-1]
    assert d.iloc[0]["high"] == day["high"].max() and d.iloc[0]["volume"] == day["volume"].sum()


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


class _FakeOKX:
    """Sirve velas 4H de paso fijo por ventanas [since, since+(limit-1)·4h]."""
    def __init__(self, df):
        self.df = df
        self.calls = 0

    def fetch_ohlcv(self, market, timeframe, since=None, limit=None):
        self.calls += 1
        lo = pd.Timestamp(since, unit="ms", tz=UTC)
        part = self.df[(self.df.index >= lo) & (self.df.index < lo + pd.Timedelta(hours=4 * limit))]
        return [[int(ts.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume]
                for ts, r in part.iterrows()]


def test_okx_tabla_o_descarga_en_memoria(temp_db):
    df = _ohlcv("2019-12-16 04:00", "2022-05-01", seed=5)
    # Tabla vacía y sin --okx: no hay datos y se explica qué hacer.
    r = P.load_okx_holdout("BTCUSDT", download=False)
    assert r["df_4h"] is None and "--okx" in r["error"]
    # --okx: descarga en memoria (doble de ccxt) sin escribir en la BD.
    fake = _FakeOKX(df)
    r = P.load_okx_holdout("BTCUSDT", download=True, exchange=fake)
    assert r["error"] is None and fake.calls > 10 and r["origin"].startswith("descarga")
    assert len(r["df_4h"]) == len(df) and np.allclose(r["df_4h"]["close"], df["close"])
    assert r["df_daily"].index[0] == pd.Timestamp("2019-12-17", tz=UTC)
    from sqlalchemy import text
    with temp_db.get_engine().connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM ohlcv")).scalar() == 0
    # Tabla con OKX parcial (solo hasta 2021-10, como producción) → insuficiente.
    temp_db.save_ohlcv(df[df.index < "2021-10-01"], "BTCUSDT", "4h", source="okx")
    r = P.load_okx_holdout("BTCUSDT", download=False)
    assert r["df_4h"] is None and "2019-12 → 2022-04" in r["error"]
    # Velas de Gate.io en el mismo periodo NO cuentan como OKX.
    temp_db.save_ohlcv(df[df.index >= "2021-10-01"], "BTCUSDT", "4h", source="gateio")
    assert P.load_okx_holdout("BTCUSDT")["df_4h"] is None
    # Tabla completa de OKX → se usa sin descargar.
    temp_db.save_ohlcv(df, "ETHUSDT", "4h", source="okx")
    r = P.load_okx_holdout("ETHUSDT", download=True, exchange=None)
    assert r["error"] is None and r["origin"].startswith("tabla") and len(r["df_4h"]) == len(df)
    assert P.load_okx_holdout("XAUUSD")["df_4h"] is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_validate_protocol(monkeypatch):
    import main
    a = main.build_parser().parse_args(["validate", "--protocol", "--symbol", "ETHUSDT",
                                        "--candidate", "c.json", "--final", "--okx", "--out", "o.json"])
    assert (a.protocol, a.symbol, a.candidate, a.final, a.okx, a.out) == \
        (True, "ETHUSDT", "c.json", True, True, "o.json")
    seen = {}

    def fake_run(symbol, candidate_path=None, final=False, okx=False, out=None):
        seen.update(symbol=symbol, candidate=candidate_path, final=final, okx=okx, out=out)
        return 0, {}
    monkeypatch.setattr(P, "run_protocol", fake_run)
    monkeypatch.setattr(sys, "argv", ["main.py", "validate", "--protocol", "--symbol", "XAUUSD"])
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code == 0 and seen["symbol"] == "XAUUSD" and seen["final"] is False
    # --final sin --protocol es un error de uso (no corre el validador clásico).
    monkeypatch.setattr(sys, "argv", ["main.py", "validate", "--final"])
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code == 2
