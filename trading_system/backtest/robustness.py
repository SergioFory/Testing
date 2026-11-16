"""
Robustez estadística de la puerta de validación (Fase 2, punto 2.1, segunda mitad).

Tres medidas, todas SOLO con datos anteriores a la reserva (settings.PROTOCOL_RESERVE_START):

  1. CPCV purgada con RE-SELECCIÓN interna (López de Prado, "Advances in Financial
     Machine Learning", cap. 12). La muestra de selección se parte en N=6 grupos
     temporales consecutivos con el mismo número de velas; cada combinación usa k=2
     grupos de prueba y los 4 restantes de entrenamiento (C(6,2)=15 combinaciones).
     En cada combinación se elige, SOLO con el entrenamiento, la mejor variante de la
     rejilla (máximo E[R] neto en el escenario de decisión, con al menos
     MIN_TRADES_SELECT operaciones; si ninguna llega, la configuración actual) y se
     mide en la prueba. Así se mide el PROCEDIMIENTO "elegir la mejor de la rejilla",
     no una variante elegida mirando todos los datos.
       - Purga: se quitan del entrenamiento las operaciones cuyo horizonte (setup en
         la vela p, salida como tarde en p + forward_bars) se solapa con un bloque de
         prueba: setups con p en [inicio − forward_bars, fin + forward_bars).
       - Embargo: además, el 1 % de las velas de la muestra justo después de cada
         bloque de prueba (redondeado hacia arriba).
     Los grupos se definen por VELAS (observaciones), no por operaciones, para que
     sean los mismos en todas las variantes.

  2. PBO (probabilidad de sobreajuste del backtest) por CSCV (Bailey, Borwein,
     López de Prado y Zhu, 2015) con S=16 bloques temporales de igual número de
     velas. Para cada una de las C(16,8)=12.870 particiones en dos mitades: se elige
     la mejor variante en la mitad "dentro" (misma regla que la CPCV) y se mira su
     puesto en la mitad "fuera"; λ = logit(puesto relativo). PBO = fracción de
     particiones con λ <= 0 (la "mejor" queda en la mitad mala o justo en la mediana).
     Con variantes sin ninguna ventaja, PBO ≈ 0,5 (con 15 variantes, 8/15 ≈ 0,53).
     La matriz (bloque × variante) guarda la SUMA de R y el NÚMERO de operaciones, y
     el E[R] de una mitad es ΣR / N (no la media de medias: muchos bloques tienen 0–3
     operaciones).

  3. DSR (Deflated Sharpe Ratio, Bailey y López de Prado, 2014) del Sharpe POR
     OPERACIÓN de la configuración evaluada:
         SR0 = sqrt(V) · ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N·e)))
         DSR = Φ( (SR − SR0) · sqrt(T − 1) / sqrt(1 − g3·SR + (g4 − 1)/4 · SR²) )
     N = pruebas totales del registro (incluida la semilla histórica de 400),
     V = varianza (ddof=1) de los Sharpe por operación de las variantes ELEGIBLES de
     la rejilla (N >= MIN_TRADES_SELECT), T = operaciones, g3 = asimetría y g4 =
     curtosis (no exceso) de los R, ambas muestrales sesgadas (como scipy por
     defecto). γ = constante de Euler-Mascheroni. DSR es la probabilidad de que el
     Sharpe verdadero supere al máximo esperado por azar entre N pruebas.

Funciones puras: solo numpy y la librería estándar (statistics.NormalDist para Φ y
Φ⁻¹). Sin scipy, sin BD, sin red. Las variantes se pasan como listas de dicts
{"id", "pos" (vela del setup, entero), "r" (R netos)} sobre el MISMO índice de velas.
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np

# Parámetros fijados en la especificación de la Fase 2 (no cambiar a la vista de datos).
CPCV_GROUPS = 6
CPCV_TEST_GROUPS = 2
CPCV_EMBARGO_FRAC = 0.01
PBO_BLOCKS = 16
EULER_GAMMA = 0.5772156649015329

_N01 = NormalDist()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def equal_blocks(p0: int, p1: int, n: int) -> list[tuple[int, int]]:
    """Parte las velas [p0, p1) en n bloques consecutivos de (casi) igual tamaño."""
    if p1 - p0 < n:
        raise ValueError(f"hacen falta al menos {n} velas para {n} bloques (hay {p1 - p0})")
    edges = np.linspace(p0, p1, n + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def block_of(pos, blocks: list[tuple[int, int]]) -> np.ndarray:
    """Índice de bloque de cada vela (−1 si cae fuera de todos)."""
    pos = np.asarray(pos, dtype=int)
    out = np.full(len(pos), -1, dtype=int)
    for k, (a, b) in enumerate(blocks):
        out[(pos >= a) & (pos < b)] = k
    return out


def _mean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if len(x) else float("nan")


def _dist(values) -> dict:
    """Resumen de una distribución (ignora NaN)."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return {"n": 0, "median": None, "mean": None, "p10": None, "p90": None,
                "min": None, "max": None, "frac_neg": None}
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "min": float(v.min()), "max": float(v.max()), "frac_neg": float((v < 0).mean())}


def _as_arrays(variants: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.asarray(v["pos"], dtype=int), np.asarray(v["r"], dtype=float)) for v in variants]


# ---------------------------------------------------------------------------
# 1. CPCV purgada con re-selección
# ---------------------------------------------------------------------------

def cpcv_splits(p0: int, p1: int, forward_bars: int, n_groups: int = CPCV_GROUPS,
                k_test: int = CPCV_TEST_GROUPS,
                embargo_frac: float = CPCV_EMBARGO_FRAC) -> list[dict]:
    """
    Combinaciones de la CPCV sobre las velas [p0, p1). Cada una:
        {"test_groups": (i, j), "test": [(a, b), ...] rangos de prueba,
         "excluded": [(a, b), ...] rangos de setup que NO pueden entrenar
                     (prueba + purga a ambos lados + embargo detrás)}
    """
    groups = equal_blocks(p0, p1, n_groups)
    emb = int(math.ceil(embargo_frac * (p1 - p0)))
    out = []
    for comb in itertools.combinations(range(n_groups), k_test):
        test = [groups[g] for g in comb]
        excl = [(a - forward_bars, b + forward_bars + emb) for a, b in test]
        out.append({"test_groups": comb, "test": test, "excluded": excl,
                    "purge_bars": int(forward_bars), "embargo_bars": emb})
    return out


def split_masks(pos, split: dict) -> tuple[np.ndarray, np.ndarray]:
    """(entrenamiento, prueba) de unas operaciones según su vela de setup."""
    pos = np.asarray(pos, dtype=int)
    test = np.zeros(len(pos), dtype=bool)
    excl = np.zeros(len(pos), dtype=bool)
    for a, b in split["test"]:
        test |= (pos >= a) & (pos < b)
    for a, b in split["excluded"]:
        excl |= (pos >= a) & (pos < b)
    return ~excl, test


def select_best(er: np.ndarray, n: np.ndarray, fallback: int, min_trades: int) -> int:
    """
    Regla de selección de la rejilla (fija): la variante de mayor E[R] entre las que
    tienen al menos min_trades operaciones; si ninguna llega, `fallback` (la
    configuración actual). En empate gana `fallback` si está entre las empatadas; si
    no, la primera en el orden de la rejilla.
    """
    er = np.asarray(er, dtype=float)
    ok = (np.asarray(n) >= min_trades) & ~np.isnan(er)
    if not ok.any():
        return int(fallback)
    best = np.max(er[ok])
    if ok[fallback] and er[fallback] == best:
        return int(fallback)
    return int(np.flatnonzero(ok & (er == best))[0])


def cpcv_reselection(variants: list[dict], current: int, p0: int, p1: int,
                     forward_bars: int, candidate: int | None = None,
                     min_trades: int = 10, n_groups: int = CPCV_GROUPS,
                     k_test: int = CPCV_TEST_GROUPS,
                     embargo_frac: float = CPCV_EMBARGO_FRAC) -> dict:
    """
    CPCV purgada con re-selección interna sobre la rejilla.

    Args:
        variants: [{"id", "pos", "r"}, ...] (la rejilla, con el candidato si lo hay).
        current: índice de la configuración actual en variants.
        candidate: índice del candidato (o None).
        p0, p1: velas de la muestra de selección [p0, p1).
        forward_bars: horizonte máximo de una operación (purga).

    Returns:
        {"n_groups", "k_test", "n_comb", "purge_bars", "embargo_bars",
         "combos": [{"test_groups", "chosen", "n_train", "er_train", "n_test",
                     "er_test", "cur_n_test", "cur_er_test", "cand_n_test",
                     "cand_er_test"}],
         "procedure": _dist(E[R] de prueba de la variante elegida) + "pooled_er"
                      (ΣR / N de todas las pruebas juntas),
         "current_fixed": _dist(E[R] de prueba de la configuración actual),
         "times_current", "times_candidate", "chosen_counts" {id: veces},
         "delta_cand_cur": _dist(E[R] candidato − actual en la prueba) (si hay candidato)}
    """
    arrs = _as_arrays(variants)
    splits = cpcv_splits(p0, p1, forward_bars, n_groups, k_test, embargo_frac)
    combos = []
    for sp in splits:
        er_tr = np.empty(len(arrs))
        n_tr = np.empty(len(arrs), dtype=int)
        tests = []
        for k, (pos, r) in enumerate(arrs):
            tr, te = split_masks(pos, sp)
            n_tr[k] = int(tr.sum())
            er_tr[k] = _mean(r[tr])
            tests.append(r[te])
        ch = select_best(er_tr, n_tr, current, min_trades)
        row = {"test_groups": list(sp["test_groups"]), "chosen": variants[ch]["id"],
               "chosen_idx": ch, "n_train": int(n_tr[ch]),
               "er_train": None if np.isnan(er_tr[ch]) else float(er_tr[ch]),
               "n_test": int(len(tests[ch])), "er_test": _none(_mean(tests[ch])),
               "cur_n_test": int(len(tests[current])), "cur_er_test": _none(_mean(tests[current]))}
        if candidate is not None:
            row["cand_n_test"] = int(len(tests[candidate]))
            row["cand_er_test"] = _none(_mean(tests[candidate]))
        row["_test_sum"] = float(tests[ch].sum())
        combos.append(row)

    proc = _dist([c["er_test"] if c["er_test"] is not None else np.nan for c in combos])
    tot_n = sum(c["n_test"] for c in combos)
    proc["pooled_er"] = (sum(c["_test_sum"] for c in combos) / tot_n) if tot_n else None
    for c in combos:
        c.pop("_test_sum")
    counts: dict = {}
    for c in combos:
        counts[c["chosen"]] = counts.get(c["chosen"], 0) + 1
    out = {"n_groups": n_groups, "k_test": k_test, "n_comb": len(combos),
           "purge_bars": splits[0]["purge_bars"], "embargo_bars": splits[0]["embargo_bars"],
           "groups": equal_blocks(p0, p1, n_groups), "combos": combos, "procedure": proc,
           "current_fixed": _dist([c["cur_er_test"] if c["cur_er_test"] is not None else np.nan
                                   for c in combos]),
           "times_current": sum(c["chosen_idx"] == current for c in combos),
           "times_candidate": (sum(c["chosen_idx"] == candidate for c in combos)
                               if candidate is not None else None),
           "chosen_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
           "delta_cand_cur": None}
    if candidate is not None:
        d = [(c["cand_er_test"] - c["cur_er_test"])
             if c["cand_er_test"] is not None and c["cur_er_test"] is not None else np.nan
             for c in combos]
        out["delta_cand_cur"] = _dist(d)
    return out


def _none(x: float):
    return None if x is None or np.isnan(x) else float(x)


# ---------------------------------------------------------------------------
# 2. PBO por CSCV
# ---------------------------------------------------------------------------

def block_matrix(variants: list[dict], p0: int, p1: int,
                 n_blocks: int = PBO_BLOCKS) -> tuple[np.ndarray, np.ndarray]:
    """Matrices (bloque × variante) de suma de R y número de operaciones."""
    blocks = equal_blocks(p0, p1, n_blocks)
    arrs = _as_arrays(variants)
    sums = np.zeros((n_blocks, len(arrs)))
    cnts = np.zeros((n_blocks, len(arrs)))
    for k, (pos, r) in enumerate(arrs):
        b = block_of(pos, blocks)
        ok = b >= 0
        np.add.at(sums[:, k], b[ok], r[ok])
        np.add.at(cnts[:, k], b[ok], 1)
    return sums, cnts


def pbo_cscv(sums: np.ndarray, cnts: np.ndarray, fallback: int = 0,
             min_trades: int = 10) -> dict:
    """
    PBO por CSCV sobre matrices (bloque × variante) de ΣR y N.

    Para cada partición de los S bloques en dos mitades de S/2 ("dentro"/"fuera"):
      - E[R] de cada variante en cada mitad = ΣR / N;
      - elegida = select_best(E[R] dentro, N dentro, fallback, min_trades);
      - puesto de la elegida "fuera" entre las variantes con N fuera >= 1
        (1 = peor, empates promediados), ω = puesto / (V + 1), λ = ln(ω / (1 − ω)).
    Las particiones en que la elegida no tiene operaciones "fuera" no cuentan.

    Returns:
        {"S", "n_comb", "n_valid", "pbo", "lambda_median", "is_best_er_median",
         "oos_best_er_median", "p_oos_loss" (fracción con E[R] fuera de la
         elegida < 0), "times_fallback"}
    """
    S, V = sums.shape
    if S % 2:
        raise ValueError("S debe ser par")
    combs = np.array(list(itertools.combinations(range(S), S // 2)), dtype=int)
    sel = np.zeros((len(combs), S))
    sel[np.arange(len(combs))[:, None], combs] = 1.0
    is_sum, is_n = sel @ sums, sel @ cnts
    oos_sum, oos_n = sums.sum(0) - is_sum, cnts.sum(0) - is_n
    with np.errstate(invalid="ignore", divide="ignore"):
        is_er = np.where(is_n > 0, is_sum / np.where(is_n > 0, is_n, 1), np.nan)
        oos_er = np.where(oos_n > 0, oos_sum / np.where(oos_n > 0, oos_n, 1), np.nan)
    # Elegida en cada partición (vectorizado; misma regla que select_best).
    ok = (is_n >= min_trades) & ~np.isnan(is_er)
    masked = np.where(ok, is_er, -np.inf)
    best = masked.max(axis=1)
    any_ok = ok.any(axis=1)
    fb_wins = ok[:, fallback] & (is_er[:, fallback] == best)
    first = np.argmax(ok & (masked == best[:, None]), axis=1)
    chosen = np.where(~any_ok | fb_wins, fallback, first)
    rows = np.arange(len(combs))
    ch_oos = oos_er[rows, chosen]
    valid_cols = ~np.isnan(oos_er)
    nv = valid_cols.sum(axis=1)
    less = (valid_cols & (oos_er < ch_oos[:, None])).sum(axis=1)
    equal = (valid_cols & (oos_er == ch_oos[:, None])).sum(axis=1)
    rank = less + 0.5 * (equal - 1) + 1          # 1 = peor; empates promediados
    valid = ~np.isnan(ch_oos) & (nv >= 2)
    w = rank[valid] / (nv[valid] + 1)
    lam = np.log(w / (1 - w))
    return {"S": int(S), "n_comb": int(len(combs)), "n_valid": int(valid.sum()),
            "pbo": float((lam <= 0).mean()) if len(lam) else None,
            "lambda_median": float(np.median(lam)) if len(lam) else None,
            "is_best_er_median": float(np.median(is_er[rows, chosen][valid])) if len(lam) else None,
            "oos_best_er_median": float(np.median(ch_oos[valid])) if len(lam) else None,
            "p_oos_loss": float((ch_oos[valid] < 0).mean()) if len(lam) else None,
            "times_fallback": int((chosen == fallback).sum())}


def pbo_for_variants(variants: list[dict], p0: int, p1: int, current: int,
                     min_trades: int = 10, n_blocks: int = PBO_BLOCKS) -> dict:
    """PBO por CSCV de una rejilla (atajo: block_matrix + pbo_cscv)."""
    sums, cnts = block_matrix(variants, p0, p1, n_blocks)
    out = pbo_cscv(sums, cnts, fallback=current, min_trades=min_trades)
    out["n_variants"] = len(variants)
    return out


# ---------------------------------------------------------------------------
# 3. DSR
# ---------------------------------------------------------------------------

def sharpe(r) -> float:
    """Sharpe por operación: media / desviación típica (ddof=1). NaN si N<2 o sd=0."""
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def skew_kurt(r) -> tuple[float, float]:
    """Asimetría y curtosis (NO exceso: normal = 3), muestrales sesgadas."""
    r = np.asarray(r, dtype=float)
    d = r - r.mean()
    m2 = (d ** 2).mean()
    if m2 <= 0:
        return 0.0, 3.0
    return float((d ** 3).mean() / m2 ** 1.5), float((d ** 4).mean() / m2 ** 2)


def expected_max_z(n_trials: int) -> float:
    """E[máx] aproximado de n_trials normales estándar independientes (Bailey-LdP)."""
    if n_trials < 2:
        return 0.0
    return ((1 - EULER_GAMMA) * _N01.inv_cdf(1 - 1 / n_trials)
            + EULER_GAMMA * _N01.inv_cdf(1 - 1 / (n_trials * math.e)))


def grid_sharpe_variance(variants: list[dict], min_trades: int = 10) -> tuple[float, int]:
    """Varianza (ddof=1) de los Sharpe por operación de las variantes con N >= min_trades."""
    srs = [sharpe(v["r"]) for v in variants if len(v["r"]) >= min_trades]
    srs = [s for s in srs if not np.isnan(s)]
    if len(srs) < 2:
        return 0.0, len(srs)
    return float(np.var(srs, ddof=1)), len(srs)


def psr(r, sr0: float) -> float:
    """Probabilistic Sharpe Ratio: P(Sharpe verdadero > sr0) con asimetría y curtosis."""
    r = np.asarray(r, dtype=float)
    T = len(r)
    sr = sharpe(r)
    if T < 3 or np.isnan(sr):
        return float("nan")
    g3, g4 = skew_kurt(r)
    den = max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12)
    return float(_N01.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(den)))


def deflated_sharpe(r, n_trials: int, var_sr: float) -> dict:
    """
    Deflated Sharpe Ratio del Sharpe por operación de r.
    Returns: {"n", "sr", "skew", "kurt", "n_trials", "sd_trials_sr", "e_max_z",
              "sr0", "psr0", "dsr"}
    """
    r = np.asarray(r, dtype=float)
    g3, g4 = skew_kurt(r) if len(r) else (float("nan"), float("nan"))
    ez = expected_max_z(int(n_trials))
    sr0 = math.sqrt(max(var_sr, 0.0)) * ez
    return {"n": int(len(r)), "sr": _none(sharpe(r)), "skew": _none(g3), "kurt": _none(g4),
            "n_trials": int(n_trials), "sd_trials_sr": math.sqrt(max(var_sr, 0.0)),
            "e_max_z": ez, "sr0": sr0, "psr0": _none(psr(r, 0.0)), "dsr": _none(psr(r, sr0))}


# ---------------------------------------------------------------------------
# Conjunto
# ---------------------------------------------------------------------------

def compute_robustness(variants: list[dict], current: int, candidate: int | None,
                       p0: int, p1: int, forward_bars: int, n_trials: int,
                       min_trades: int = 10) -> dict:
    """
    CPCV purgada con re-selección, PBO (CSCV, S=16) y DSR de la configuración
    actual y, si lo hay, del candidato.
    """
    var_sr, n_sr = grid_sharpe_variance(variants, min_trades)
    out = {"cpcv": cpcv_reselection(variants, current, p0, p1, forward_bars,
                                    candidate=candidate, min_trades=min_trades),
           "pbo": pbo_for_variants(variants, p0, p1, current, min_trades),
           "dsr_current": deflated_sharpe(variants[current]["r"], n_trials, var_sr),
           "dsr_candidate": None, "grid_sr_var": var_sr, "grid_sr_n": n_sr,
           "n_trials": int(n_trials), "bars": [int(p0), int(p1)]}
    if candidate is not None:
        out["dsr_candidate"] = deflated_sharpe(variants[candidate]["r"], n_trials, var_sr)
    return out
