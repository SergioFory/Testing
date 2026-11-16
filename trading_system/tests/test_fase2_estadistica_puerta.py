"""
Pruebas de la Fase 2, punto 2.1 (segunda mitad) — estadística de la puerta:
  - backtest/robustness.py: CPCV purgada (15 combinaciones, purga + embargo) con
    re-selección SOLO con el entrenamiento, PBO por CSCV (S=16) y DSR;
  - casos sintéticos de resultado conocido: rejilla de ruido → PBO ≈ 0,5 y DSR
    bajo; una variante con ventaja fuerte → PBO ≈ 0 y DSR alto; matrices hechas a
    mano con PBO = 1 y PBO = 0; E[máx Z] frente a su valor conocido;
  - criterios de aprobado a)–e) (backtest/protocol.evaluate_criteria) y su
    integración en run_protocol (vivo = validador con la config del candidato,
    la robustez no usa la reserva, un fallo de la robustez no da aprobado).

    cd trading_system && python -m pytest tests/test_fase2_estadistica_puerta.py -q

Sin red: datos sintéticos y detector sustituido por un doble.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import protocol as P          # noqa: E402
from backtest import robustness as RB       # noqa: E402
import test_fase2_costes_registro as TA     # noqa: E402  (datos sintéticos del grupo A)


# ---------------------------------------------------------------------------
# Rejillas sintéticas
# ---------------------------------------------------------------------------

def _noise_grid(V=15, n_bars=6000, n=300, seed=0, edge=None, mu=0.8):
    """V variantes independientes con R en {−1, +1,5} (E[R] = 0); `edge` suma mu."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(V):
        pos = np.sort(rng.integers(0, n_bars, n))
        r = rng.choice([-1.0, 1.5], n, p=[0.6, 0.4])
        if edge == k:
            r = r + mu
        out.append({"id": f"v{k}", "pos": pos, "r": r})
    return out


# ---------------------------------------------------------------------------
# CPCV purgada
# ---------------------------------------------------------------------------

def test_cpcv_combinaciones_purga_y_embargo():
    sp = RB.cpcv_splits(0, 6000, forward_bars=24)
    assert len(sp) == 15                                   # C(6, 2)
    veces = np.zeros(6, dtype=int)
    for s in sp:
        for g in s["test_groups"]:
            veces[g] += 1
    assert (veces == 5).all()                              # cada grupo en prueba 5 veces
    s = next(x for x in sp if x["test_groups"] == (1, 3))
    assert s["test"] == [(1000, 2000), (3000, 4000)] and s["embargo_bars"] == 60
    pos = np.array([975, 976, 1500, 2023, 2024, 2083, 2084, 2500, 4084, 5000])
    tr, te = RB.split_masks(pos, s)
    # Purga: 24 velas antes (976 ya solapa) y 24 después + 60 de embargo (hasta 2083).
    assert tr.tolist() == [True, False, False, False, False, False, True, True, True, True]
    assert te.tolist() == [False, False, True, False, False, False, False, False, False, False]
    assert not (tr & te).any()


def test_seleccion_minimo_de_operaciones_y_empates():
    assert RB.select_best(np.array([0.1, 0.5, 0.9]), np.array([20, 20, 5]), 0, 10) == 1
    assert RB.select_best(np.array([0.1, 0.5]), np.array([3, 4]), 0, 10) == 0        # ninguna llega
    assert RB.select_best(np.array([0.5, 0.5, 0.2]), np.array([20, 20, 20]), 1, 10) == 1  # empate → actual


def test_cpcv_reselecciona_solo_con_entrenamiento():
    """Una variante 'trampa' brilla SOLO en el grupo 0: cuando el grupo 0 es de prueba,
    su entrenamiento es malo y no puede ser elegida (no hay fuga de la prueba)."""
    rng = np.random.default_rng(1)
    base = _noise_grid(V=3, seed=2)
    pos = np.sort(rng.integers(0, 6000, 400))
    r = np.where(pos < 1000, 5.0, -1.0)
    variants = base + [{"id": "trampa", "pos": pos, "r": r}]
    res = RB.cpcv_reselection(variants, current=0, p0=0, p1=6000, forward_bars=24, candidate=3)
    for c in res["combos"]:
        if 0 in c["test_groups"]:
            assert c["chosen"] != "trampa"
            assert c["cand_er_test"] > 0                    # en la prueba sí brillaría
        else:
            assert c["chosen"] == "trampa"                  # con el grupo 0 entrenando, sí
    assert res["n_comb"] == 15 and res["times_candidate"] <= 10


def test_cpcv_ruido_mediana_cero_y_ventaja_elegida():
    meds = [RB.cpcv_reselection(_noise_grid(seed=s), 6, 0, 6000, 24)["procedure"]["median"]
            for s in range(8)]
    assert abs(np.mean(meds)) < 0.12                        # el procedimiento no gana nada
    res = RB.cpcv_reselection(_noise_grid(seed=3, edge=4), 6, 0, 6000, 24, candidate=4)
    assert res["times_candidate"] == 15 and res["times_current"] == 0
    assert res["delta_cand_cur"]["median"] > 0.5


# ---------------------------------------------------------------------------
# PBO por CSCV
# ---------------------------------------------------------------------------

def test_pbo_matrices_hechas_a_mano():
    cnt = np.full((4, 2), 10.0)
    # A buena donde B es mala y viceversa: la mejor "dentro" siempre es la peor "fuera".
    anti = np.array([[3, -3], [1, -1], [-1, 1], [-3, 3]], dtype=float)
    assert RB.pbo_cscv(anti, cnt, fallback=0, min_trades=1)["pbo"] == 1.0
    # A siempre mejor que B: nunca hay sobreajuste.
    pers = np.array([[1, -1]] * 4, dtype=float)
    out = RB.pbo_cscv(pers, cnt, fallback=1, min_trades=1)
    assert out["pbo"] == 0.0 and out["n_comb"] == 6 and out["times_fallback"] == 0


def test_pbo_ruido_cerca_de_medio_y_ventaja_cerca_de_cero():
    pb = [RB.pbo_for_variants(_noise_grid(seed=s), 0, 6000, current=6)["pbo"] for s in range(12)]
    assert 0.35 < np.mean(pb) < 0.70                         # teórico con 15 variantes: 8/15
    out = RB.pbo_for_variants(_noise_grid(seed=5, edge=2), 0, 6000, current=6)
    assert out["S"] == 16 and out["n_comb"] == 12870 and out["pbo"] < 0.05


# ---------------------------------------------------------------------------
# DSR
# ---------------------------------------------------------------------------

def test_esperanza_del_maximo_y_dsr():
    # E[máx] de 100 normales ≈ 2,51 (Monte Carlo); la aproximación de Bailey-LdP da 2,53.
    assert abs(RB.expected_max_z(100) - 2.531) < 0.01
    assert RB.expected_max_z(1) == 0.0
    r = np.random.default_rng(5).choice([-1.0, 1.5, 2.0], 60, p=[.5, .3, .2])
    # Con varianza 0 entre pruebas, DSR = PSR(0).
    d0 = RB.deflated_sharpe(r, 460, 0.0)
    assert d0["sr0"] == 0.0 and abs(d0["dsr"] - d0["psr0"]) < 1e-12
    # Cálculo a mano de la fórmula.
    T, sr = len(r), r.mean() / r.std(ddof=1)
    g3, g4 = RB.skew_kurt(r)
    sr0 = math.sqrt(0.02) * RB.expected_max_z(460)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    ref = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    d = RB.deflated_sharpe(r, 460, 0.02)
    assert abs(d["dsr"] - ref) < 1e-12 and abs(d["sr0"] - sr0) < 1e-12
    # Más pruebas → más exigente.
    assert RB.deflated_sharpe(r, 4600, 0.02)["dsr"] < d["dsr"]
    # Asimetría y curtosis de una normal ≈ 0 y 3.
    g3n, g4n = RB.skew_kurt(np.random.default_rng(0).standard_normal(200_000))
    assert abs(g3n) < 0.03 and abs(g4n - 3) < 0.05


def test_dsr_contra_scipy_si_esta():
    st = pytest.importorskip("scipy.stats")
    r = np.random.default_rng(9).choice([-1.0, 1.33, 2.0], 80, p=[.55, .3, .15])
    assert abs(RB.skew_kurt(r)[0] - st.skew(r)) < 1e-12
    assert abs(RB.skew_kurt(r)[1] - st.kurtosis(r, fisher=False)) < 1e-12


def test_robustez_ruido_dsr_bajo_y_ventaja_dsr_alto():
    rb = RB.compute_robustness(_noise_grid(seed=1), 6, None, 0, 6000, 24, n_trials=460)
    assert rb["dsr_current"]["dsr"] < 0.5 and rb["dsr_candidate"] is None
    rb = RB.compute_robustness(_noise_grid(seed=1, edge=3), 6, 3, 0, 6000, 24, n_trials=460)
    assert rb["dsr_candidate"]["dsr"] > 0.95 and rb["pbo"]["pbo"] < 0.05
    assert rb["n_trials"] == 460 and rb["grid_sr_n"] == 15


# ---------------------------------------------------------------------------
# Criterios a)–e)
# ---------------------------------------------------------------------------

def _row(er, folds=None, decision=True):
    r = {"scenario": "xm_swap5", "decision": decision, "er": er, "n": 20}
    if folds is not None:
        r["folds"] = [{"fold": f"f{i}", "n": 5, "er": e} for i, e in enumerate(folds)]
    return r


def _res(delta_med=0.2, pbo=0.1, dsr=0.97, live=True, cur_f=(0.1, 0.1, 0.1),
         cand_f=(0.3, 0.3, 0.0), kind="crypto", rsv=(0.3, 0.2), okx=(0.1, 0.1), okx_err=None):
    res = {"symbol": "ETHUSDT", "fold_kind": kind,
           "candidate_live_check": {"match": live, "approved": 9, "emitted": 9},
           "baseline": [_row(0.1, cur_f)], "candidate": [_row(0.2, cand_f)],
           "robustness": {"cpcv": {"delta_cand_cur": {"median": delta_med}}, "pbo": {"pbo": pbo},
                          "dsr_candidate": {"dsr": dsr}, "n_trials": 460},
           "reserve": {"current": [_row(rsv[1])], "candidate": [_row(rsv[0])]}}
    res["okx"] = ({"error": okx_err} if okx_err else
                  {"current": [_row(okx[1])], "candidate": [_row(okx[0])]})
    return res


def test_criterios_aprobado_pendiente_y_fallos():
    cand = {"symbol": "ETHUSDT"}
    assert P.evaluate_criteria([_res()], None)["status"] == "baseline"
    v = P.evaluate_criteria([_res()], cand, final=False)
    assert v["status"] == "pending_final" and v["criteria"]["e"]["ok"] is None
    assert "Todavía NO se adopta" in v["text"]
    v = P.evaluate_criteria([_res()], cand, final=True)
    assert v["status"] == "approved" and v["text"].startswith("APROBADO")
    v = P.evaluate_criteria([_res(delta_med=0.05, dsr=0.90)], cand, final=True)
    assert v["status"] == "rejected" and v["failed"] == ["b", "d"]
    assert v["text"].startswith("NO APROBADO: falla b, d")
    # b) folds: solo 1 de 3 con mejora >= +0,10R.
    v = P.evaluate_criteria([_res(cand_f=(0.3, 0.15, 0.1))], cand, final=True)
    assert v["failed"] == ["b"]
    # Exactamente +0,10R cuenta como mejora suficiente (≥).
    assert P.evaluate_criteria([_res(delta_med=0.10, cand_f=(0.2, 0.2, 0.0))], cand,
                               final=True)["status"] == "approved"
    # a) vivo != validador con la config del candidato; c) PBO límite (0,20 no vale).
    v = P.evaluate_criteria([_res(live=False, pbo=0.20)], cand, final=True)
    assert v["failed"] == ["a", "c"]
    # e) reserva negativa; OKX sin datos (cripto) también falla; diario no necesita OKX.
    assert P.evaluate_criteria([_res(rsv=(0.1, 0.2))], cand, final=True)["failed"] == ["e"]
    assert P.evaluate_criteria([_res(okx_err="sin datos")], cand, final=True)["failed"] == ["e"]
    assert P.evaluate_criteria([_res(kind="daily", okx_err="x")], cand, final=True)["status"] == "approved"
    # Sin robustez (error) → b, c y d no se aprueban.
    r = _res()
    r["robustness"] = {"error": "boom"}
    assert P.evaluate_criteria([r], cand, final=True)["failed"] == ["b", "c", "d"]


# ---------------------------------------------------------------------------
# Integración en run_protocol
# ---------------------------------------------------------------------------

@pytest.fixture()
def dense_detector(monkeypatch):
    def _detect(symbol, df_4h, df_daily=None, overrides=None, params_override=None):
        return TA._setups(df_4h, every=13)
    monkeypatch.setattr(P, "detect_universe", _detect)


def test_protocolo_linea_base_con_robustez(dense_detector, tmp_path):
    lines = []
    code, rep = P.run_protocol("BTCUSDT", registry=tmp_path / "r.jsonl", load_data=TA._loader,
                               live_check=TA._live_ok, out=str(tmp_path / "o.json"),
                               echo=lines.append)
    txt = "\n".join(lines)
    assert code == 0, txt
    rob = rep["symbols"][0]["robustness"]
    assert rob["cpcv"]["n_comb"] == 15 and rob["pbo"]["S"] == 16 and rob["pbo"]["n_variants"] == 15
    assert rob["n_trials"] == 415 and rob["dsr_candidate"] is None
    assert rob["cpcv"]["purge_bars"] == 24
    assert "Robustez (solo datos anteriores a la reserva)" in txt and "PBO = " in txt and "DSR = " in txt
    assert txt.index("VEREDICTO") < txt.index("Robustez") < txt.index("DETALLE TÉCNICO") < txt.index("PBO (CSCV)")
    data = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert data["symbols"][0]["robustness"]["pbo"]["pbo"] == rob["pbo"]["pbo"]


def test_robustez_no_mira_la_reserva(dense_detector, tmp_path):
    """Cambiar los precios bastante después del inicio de la reserva no cambia nada de la robustez."""
    d4, d1 = TA._loader("BTCUSDT")
    cut = pd.Timestamp("2025-12-20", tz="UTC")
    d4b = d4.copy()
    d4b.loc[d4b.index >= cut, ["open", "high", "low", "close"]] *= 1.37
    rob = []
    for df in (d4, d4b):
        res = P.analyze_symbol("BTCUSDT", df, P.resample_daily_utc(df))
        p0, p1 = res["_bounds"]
        assert df.index[p1] >= pd.Timestamp("2025-11-15", tz="UTC") > df.index[p1 - 1]
        variants, _, _ = P.robustness_variants(res)
        assert all((v["pos"] < p1).all() for v in variants)
        rob.append(P.robustness_for(res, 460))
    assert rob[0]["pbo"] == rob[1]["pbo"] and rob[0]["dsr_current"] == rob[1]["dsr_current"]
    assert rob[0]["cpcv"]["combos"] == rob[1]["cpcv"]["combos"]


def test_candidato_vivo_validador_con_su_config_y_veredicto(dense_detector, tmp_path):
    from config import settings
    seen = []

    def live(sym, d4, d1):
        cfg = settings.ASSETS[sym]
        seen.append((cfg.get("ml_threshold"), tuple(cfg.get("directions") or ())))
        return TA._live_ok(sym, d4, d1)

    cand = tmp_path / "c.json"
    cand.write_text(json.dumps({"symbol": "ETHUSDT", "ml_threshold": 0.45, "directions": ["short"],
                                "label": "ETH 0,45 cortos"}), encoding="utf-8")
    lines = []
    code, rep = P.run_protocol(None, candidate_path=str(cand), registry=tmp_path / "r.jsonl",
                               load_data=TA._loader, live_check=live, echo=lines.append)
    txt = "\n".join(lines)
    assert code == 0, txt
    assert seen[0] == (0.55, ()) and seen[1] == (0.45, ("short",))   # actual y luego candidato
    assert settings.ASSETS["ETHUSDT"]["ml_threshold"] == 0.55          # config restaurada
    assert "directions" not in settings.ASSETS["ETHUSDT"]
    v = rep["verdict"]
    assert v["status"] in ("rejected", "pending_final") and v["criteria"]["e"]["ok"] is None
    assert v["criteria"]["a"]["ok"] is True
    rob = rep["symbols"][0]["robustness"]
    assert rob["dsr_candidate"] is not None and rob["cpcv"]["delta_cand_cur"] is not None
    assert txt.splitlines()[3].startswith("VEREDICTO: ")
    assert " a) [" in txt and " e) [PENDIENTE]" in txt


def test_fallo_de_robustez_no_aprueba(dense_detector, tmp_path, monkeypatch):
    def boom(res, n):
        raise ValueError("sin velas")
    monkeypatch.setattr(P, "robustness_for", boom)
    cand = tmp_path / "c.json"
    cand.write_text(json.dumps({"symbol": "ETHUSDT", "ml_threshold": 0.45}), encoding="utf-8")
    lines = []
    code, rep = P.run_protocol(None, candidate_path=str(cand), registry=tmp_path / "r.jsonl",
                               load_data=TA._loader, live_check=TA._live_ok, echo=lines.append)
    assert code == 0
    assert rep["verdict"]["status"] == "rejected" and {"b", "c", "d"} <= set(rep["verdict"]["failed"])
    assert "Robustez: no se pudo calcular" in "\n".join(lines)
