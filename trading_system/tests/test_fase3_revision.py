"""
Pruebas de la revisión adversarial de la Fase 3 — documentación para el usuario.

Las correcciones de código tienen sus pruebas junto a las del módulo:
  - cartera sin señales que no llegaron (notified = 0): tests/test_fase3_cartera.py
  - límite de tiempo total de Deribit: tests/test_fase3_candidatos.py
Aquí se comprueba que REGLAS_DE_DECISION.md lleva dentro del repo lo que hace
falta para decidir (sin remitir a ficheros temporales fuera del repo).

    cd trading_system && python -m pytest tests -q

Sin red ni BD: solo lee el Markdown.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGLAS = (ROOT / "REGLAS_DE_DECISION.md").read_text(encoding="utf-8")


def _section(title_start: str) -> str:
    """Texto de la sección '## <title_start>…' hasta la siguiente '## '."""
    i = REGLAS.index("## " + title_start)
    j = REGLAS.find("\n## ", i + 3)
    return REGLAS[i:] if j < 0 else REGLAS[i:j]


def test_control_de_cartera_lleva_la_regla_de_decision_con_numeros():
    sec = _section("Control de cartera")
    for frag in ("≤ 15 %", "no cae más de un 10 %", "p < 0,10", "5.000",
                 "p = proporción", "p = 1", "al menos 1"):
        assert frag in sec, frag
    assert "scratchpad" not in sec and "LOG.md" not in sec


def test_control_de_cartera_dice_que_las_no_entregadas_no_cuentan():
    sec = _section("Control de cartera")
    assert "te llegaron por Telegram" in sec and "/fill" in sec


def test_recuento_de_pruebas_registradas_de_los_candidatos():
    assert "460 → 464 (4 variantes nuevas" in REGLAS
    assert "461 → 464" not in REGLAS


def test_regla_5_del_vrp_remite_a_la_regla_de_activacion():
    i = REGLAS.index("5. **El filtro VRP")
    regla5 = REGLAS[i:REGLAS.index("6. **Costes reales", i)]
    for frag in ("Regla de activación del VRP (fijada el 16-nov-2026)", "desde el 16-nov-2026",
                 "E6", "mismo signo", "+0,10R", "60 %", "--final"):
        assert frag in regla5, frag
    assert "### Regla de activación del VRP (fijada el 16-nov-2026" in REGLAS
