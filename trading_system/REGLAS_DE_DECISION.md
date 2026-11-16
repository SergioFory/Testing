# Reglas de decisión (escritas de antemano)

**Fecha de registro:** 15-nov-2026 (Fase 1, punto 1.12). **Código:** `monitoring/sequential.py`.
**Dónde verlas cada día:** en el resumen nocturno de Telegram (03:00 UTC) o con
`python trading_system/main.py summary` (vista previa, no envía nada).

## Por qué existen

Un mes bueno o malo no dice casi nada: con ~26 señales al año, la suerte pesa más que el
sistema durante mucho tiempo. Si se decide "a ojo" mirando meses sueltos, se acaba
cambiando el sistema por ruido (y cada cambio añade más sobreajuste). Por eso las reglas
se fijan **antes** de ver los resultados y no se tocan a la vista de los datos. Si alguna
vez se cambia algo de este documento, se escribe aquí con fecha y motivo.

## Qué se cuenta

- **Señales live**: las que recibes por Telegram (no las del activo en modo sombra, hoy SPX),
  sin marca de calidad, emitidas desde la Fase 0 (tienen la vela del setup guardada),
  ya resueltas (ganada, perdida o cerrada por tiempo), **en el orden en que se resolvieron**.
  Las señales anteriores a la Fase 0 no cuentan; el resumen muestra aparte solo las
  **originales emitidas desde la congelación** (18-sep-2026 17:21 UTC) de los activos que hoy
  se notifican: las 8 del informe.
- El resultado de cada una se mide en **R** (1R = la distancia de la entrada al stop).
  Ganar el TP de BTC = +1,5R; tocar el stop = −1R; cierre por tiempo = lo que haya.
- Son los R del seguimiento automático: **sin costes del bróker** (spread, comisión, swap)
  y con la entrada al precio del mensaje. Los costes reales se vigilan con los fills (regla 6).
- **Versión con más datos (registro sombra)**: los setups APROBADOS que el sistema detectó
  aunque no se enviaran (por frescura, duplicado, etc.; "evento macro" solo en filas anteriores
  al 16-nov-2026, cuando se quitó ese filtro; "control de cartera" solo si algún día se activa
  su modo "enforce"), resueltos con la misma
  regla. Sirve para tener una lectura antes, pero **las decisiones de riesgo usan las señales live**.

## Las dos alarmas

### 1. SPRT — "¿hay ventaja o no?"

Compara dos hipótesis con cada señal resuelta:
- **Sin ventaja:** el R medio por señal es 0.
- **Con ventaja:** el R medio por señal es +0,25R.

Parámetros fijos: dispersión típica de un resultado σ = 1,3R; probabilidad de dar por buena
una ventaja que no existe α = 5 %; probabilidad de no ver una ventaja de +0,25R que sí existe
β = 20 %.

Cómo se lee: cada señal suma o resta a un **indicador** (empieza en 0). Una señal con
R mayor que +0,125 lo sube; una menor, lo baja (exactamente `0,148 × (R − 0,125)`).
- Si el indicador llega a **+2,77** → **"edge aceptado"**.
- Si baja a **−1,56** → **"sin edge"**.
- Mientras esté en medio → **"seguir"** (todavía no se sabe).
La decisión es la del primer cruce: lo que pase después no la deshace.

### 2. CUSUM — "¿se ha estropeado?"

Acumula cuánto se queda cada señal por debajo de lo esperado:
`S = máx(0, S anterior + 0,125 − R)`. Las señales buenas lo devuelven hacia 0; una racha
mala lo hace crecer. **Alarma si S llega a 20R.** Solo con las señales live.

## Las reglas

1. **Configuración congelada.** No se cambian umbrales, filtros, R:R ni activos hasta que el
   SPRT decida o haya **150 señales aprobadas** (live, resueltas). Todo cambio reinicia el SPRT
   (y suma otra comparación múltiple): se anota la fecha del cambio en la clave
   `vigilancia_desde` de la tabla `kv` (segundos Unix) y el SPRT y el CUSUM cuentan desde ahí. Después, las ideas acumuladas pasan juntas por la
   puerta de validación de la Fase 2 y solo se adopta lo que la pase.
2. **Si salta el CUSUM:** el riesgo por operación baja **a la mitad** y se hace una revisión
   (datos, ejecución, mercado). No se "arregla" el sistema retocando filtros sobre los mismos
   datos. Tras la revisión, el CUSUM puede reiniciarse desde esa fecha (clave `cusum_desde`
   de la tabla `kv`, en segundos Unix), dejándolo escrito aquí.
   **Cómo se aplica:** cambiando `RISK["risk_per_trade"]` de 0,0025 a 0,00125 en
   `config/settings.py` (desde la Fase 3 el tamaño lo fija de verdad ese valor: ya no hay tope
   de nominal que lo anule) o, mientras tanto, operando a mano **la mitad de las unidades (o
   lotes)** de la línea "Tamaño" del mensaje.
3. **El riesgo no sube del 0,25 %** por operación hasta que el SPRT con las señales live diga
   "edge aceptado". (En la práctica puede tardar años; esperar no cuesta rendimiento.)
   **Desde el 16-nov-2026 (Fase 3) el sistema lo cumple solo:** cada señal arriesga
   exactamente el 0,25 % del capital (2,50 USD con 1.000 USD; ver "Riesgo por operación").
   La línea "Para respetar el 0.25 %" del mensaje solo saldría si algún día el riesgo
   calculado lo superase (no debería ocurrir).
4. **SPX vuelve a notificarse solo si** el E[R] neto con los costes reales de XM es
   ≥ +0,10R en al menos 2 de 3 épocas **y** pasa la puerta de validación (Fase 2).
   Si no, sigue en sombra (se registra y se mide, sin coste).
5. **El filtro VRP (volatilidad implícita/realizada) solo se activa** si se cumple TODO lo de la
   sección "Regla de activación del VRP (fijada el 16-nov-2026)", más abajo (informe §7, 2.5; Fase 3):
   ≥ 30 señales cripto aprobadas y resueltas **registradas con VRP desde el 16-nov-2026** (lo
   anterior no cuenta); el filtro exacto del experimento E6 (percentil 33 causal); mismo signo en
   BTC y ETH; mejora ≥ +0,10R por operación; y revisión obligatoria si descarta > 60 % de los setups
   en mercado bajista. Después, puerta de validación con `--final` y **decisión tuya**.
6. **Costes reales:** tras ~20 fills (`/fill`), si el R ejecutado menos el R teórico es
   menor que −0,10R de media, se ajustan los costes del modelo a los reales.
   **Oro (base futuro/contado):** la entrada, el SL y el TP del oro salen del futuro GC=F,
   pero en XM se opera el contado, que cotiza ~0,3–0,5 % por debajo (medido del 29-oct al
   13-nov-2026 con PAXG/XAUT de Kraken: GC=F por encima en 11 de 12 días). Con el stop típico
   son ≈0,1R, el mismo tamaño que el umbral de esta regla: un short de oro ejecutado al
   instante saldrá "en contra" ~0,1R y un long "a favor" sin ser coste de ejecución. Antes de
   ajustar costes con fills de oro, hay que medir esa base aparte (p. ej. anotar en el /fill
   el precio de XM en el momento de la señal) o mirar el oro por separado de BTC/ETH.
   Si un /fill se escribió mal, otro /fill de la misma señal lo sustituye (cuenta el más
   reciente); un precio a más del 20 % de la entrada o más allá del SL/TP no se guarda sin
   confirmar.
7. **Si el SPRT dice "sin edge":** no se sube el riesgo y se revisa con calma qué dicen el
   registro sombra y los fills antes de decidir nada (pausar, seguir en papel o seguir igual
   es decisión tuya; no se cambian filtros a la vista de ese resultado).

## Riesgo por operación (Fase 3, 16-nov-2026): 0,25 % explícito

**Qué cambió.** Hasta el 16-nov-2026 el tamaño se calculaba con `risk_per_trade` = 1 % y un
**tope de nominal del 10 % del capital** que mandaba siempre, así que el riesgo real dependía de lo
lejos que estuviera el stop (de ~0,1 % a ~0,8 %) y la regla 3 no se cumplía en la mayoría de
señales de ETH ni en muchas de oro. Ahora (`config/settings.py`):

- `RISK["risk_per_trade"]` = **0,0025** (0,25 %, el tope de la regla 3): cada señal arriesga
  exactamente el 0,25 % del capital (`TRADING_CAPITAL`), sea cual sea el activo o el stop.
- El tope del 10 % de nominal se sustituye por `RISK["max_leverage"]` = **1×** (nominal ≤ capital).
  Con el 0,25 % solo actuaría con un stop a menos del 0,25 % del precio (el stop más corto de la
  historia aprobada es 0,81 %); si actuara, el riesgo bajaría y el mensaje lo diría.
- **No cambia qué señales salen** (check-live idéntico: BTC 64, ETH 63, oro 39, SPX 46) ni los
  resultados en R (SPRT, CUSUM, estadísticas). Solo cambia el tamaño en USD/unidades. Por eso
  **no se reinicia el SPRT** (`vigilancia_desde` no se toca). `pnl_usd` sigue siendo tamaño × %.

**Antes y después** con capital 1.000 USD y los 212 setups aprobados de la historia del arnés
(cripto desde 2022-05, oro y SPX desde 2006; script `fase3/A-eventos-riesgo/riesgo_antes_despues.py`):

| Activo | N | Stop (mediana) | Riesgo ANTES: mediana (mín–máx) | > 0,25 % antes | Nominal ANTES | Riesgo DESPUÉS | Nominal DESPUÉS: mediana (mín–máx) |
|---|---|---|---|---|---|---|---|
| BTC | 64 | 2,16 % | 0,22 % (0,11–0,47 %) | 17 de 64 | 10 % siempre | 0,25 % siempre | 11,6 % (5,3–22,2 %) |
| ETH | 63 | 2,97 % | 0,30 % (0,14–0,82 %) | 46 de 63 | 10 % siempre | 0,25 % siempre | 8,4 % (3,0–17,7 %) |
| Oro | 39 | 2,40 % | 0,24 % (0,13–0,68 %) | 18 de 39 | 10 % siempre | 0,25 % siempre | 10,4 % (3,7–19,9 %) |
| SPX (sombra) | 46 | 1,97 % | 0,20 % (0,08–0,34 %) | 13 de 46 | 10 % siempre | 0,25 % siempre | 12,7 % (7,3–30,7 %) |

El tope del 10 % actuaba en los 212 setups; el de apalancamiento 1× no actúa en ninguno. Con
1.000 USD el nominal típico es de 70–130 USD por operación (p. ej. 0,0016 BTC u 0,016 oz).

**Pendiente práctico (no cambia nada del sistema):** con 1.000 USD esos tamaños pueden quedar por
debajo del lote mínimo de XM. En cuanto se rellene `XM_CONTRACTS` el mensaje lo dirá ("no llega al
lote mínimo… con X lotes el riesgo sería Y %"); hasta entonces compruébalo en MT5 antes de operar.

## Control de cartera (Fase 3, 16-nov-2026): EN PRUEBAS, no bloquea

**Qué es.** Antes de cada señal el sistema mira su **cartera según el sistema** = las señales que
te envió, **te llegaron por Telegram** y siguen abiertas, **salvo las que marcaste con `/skip`** (si
luego envías `/fill`, vuelve a contar). Una señal cuyo envío falló (el resumen nocturno la lista en
"Señales que NO te llegaron") no cuenta ni como posición abierta ni en la pérdida del día: si el
reintento te la entrega, entra; si la ejecutaste igualmente, manda `/fill` y entra. Con eso comprueba cuatro reglas (`PORTFOLIO_RULES` en `config/settings.py`, código en
`risk/portfolio.py`):

| Regla | Qué dice |
|---|---|
| R1 | Como mucho **3 posiciones abiertas** a la vez. |
| R2 | **Nunca LONG y SHORT a la vez** en el mismo activo. |
| R3 | BTC y ETH se mueven juntos: el riesgo abierto **BTC + ETH** (incluida la señal nueva) no pasa del **0,50 %** del capital. Con el 0,25 % por operación: como mucho 2 cripto abiertas. |
| R4 | Si hoy (día UTC) ya se ha perdido un **3 %** del capital en operaciones cerradas, no abrir más. |

**Hoy (modo `PORTFOLIO_CONTROL_MODE = "shadow"`) no bloquea nada:** la señal te llega igual y el
mensaje añade "Cartera según el sistema: 2 abiertas (BTC long #12, …)" y, si incumple alguna regla,
"⚠️ Con esta señal: … (control de cartera en pruebas: no bloquea)". Queda guardado (columna
`portfolio_flag`) y el resumen nocturno cuenta, por regla, cuántas señales de las últimas 24 h y 30
días habría bloqueado. El modo `"enforce"` (la señal no se envía y queda en el registro sombra con
motivo `portfolio_R1`…`portfolio_R4`, sin reintentarla más tarde) está programado y probado, pero
**solo se activa si tú lo decides**.

**Regla de decisión** (fijada en la especificación de la Fase 3 y escrita antes de calcular nada).
Se **recomienda** pasar una regla a `"enforce"` solo si cumple las tres cosas:
1. bloquea **≤ 15 %** de las señales;
2. el **R total no cae más de un 10 %**: R total con la regla ≥ R total sin ella − 0,10 × |R total sin ella|;
3. **reduce la peor racha (drawdown máximo en R) más que quitar al azar** el mismo número de señales,
   con **p < 0,10**. p = proporción de las simulaciones al azar (5.000) que dejan una peor racha igual o
   menor que la de la regla (p pequeño = la regla lo hace mejor que el azar). Si la regla no quita
   ninguna señal, quitar 0 al azar da lo mismo que la base y p = 1.
Excepción: **R2** (LONG y SHORT a la vez) se recomienda si bloquea **al menos 1** señal y cumple el
punto 2, por coherencia operativa. Una regla que no bloquea ninguna señal no se recomienda (no hay
efecto medible). La recomendación es por regla ("todas juntas" se informa igual). Para decidir más
adelante con el registro en vivo se aplican estos mismos números.

**Medición histórica** (16-nov-2026, con la regla de arriba; la simulación usa las mismas funciones
de reglas que el vivo, `risk/portfolio.evaluate_portfolio_rules`). Señales que
habría emitido el vivo, simuladas con costes XM (swap 5 %), riesgo 0,25 % cada una, solo datos
anteriores a la reserva (15-nov-2025). Periodo principal: los 4 activos a la vez, 2022-05 → 2025-11
(125 señales: E[R] +0,42R, R total +52,8R, peor racha −7,95R):

| Regla | Señales que habría bloqueado | Efecto en R total y peor racha | Frente a quitar al azar (5.000 sim.; p de la peor racha) | Recomendación |
|---|---|---|---|---|
| R1 | 0 de 125 | ninguno | — (p = 1) | No activar: no habría actuado nunca |
| R2 | 0 de 125 | ninguno | — (p = 1) | No activar: no habría actuado nunca |
| R3 | 0 de 125 | ninguno | — (p = 1) | No activar: no habría actuado nunca |
| R4 | 0 de 125 | ninguno | — (p = 1) | No activar: no habría actuado nunca |

Lo mismo con solo BTC/ETH/oro (119 señales), con todo el histórico (180) y con los costes del
validador: **ninguna regla habría bloqueado ninguna señal**. Por qué: en la historia nunca hubo más
de 3 posiciones abiertas a la vez (R1 actúa con la 4.ª), nunca más de 2 cripto a la vez (R3 actúa con
la 3.ª), nunca una señal contraria con otra abierta del mismo activo, y la peor pérdida realizada en un
día fue 0,25 % (R4 actúa al 3 %). Se queda cerca del límite: 6 señales salieron con 2 posiciones ya
abiertas y 43 de 106 señales cripto con la otra cripto abierta (20 de ellas en la misma vela y
dirección).

**Recomendación por la regla fijada de antemano:** ninguna regla cumple "reduce la peor racha más que
quitar señales al azar" (no quitan ninguna) y R2 no bloquea ninguna, así que **no se recomienda pasar
ninguna a "enforce"**: el modo sigue en `"shadow"`. Esto no dice que las reglas sean malas: dice que
con el sistema actual no habrían cambiado nada; funcionan como cinturón de seguridad para situaciones
que la historia no ha mostrado (p. ej. muchas señales seguidas en un mercado nuevo). Activarlas o no
es decisión tuya; si se activa, se anota aquí la fecha. El registro en vivo dirá si alguna llega a
actuar (resumen nocturno, "Cartera según el sistema").

**Matiz de los primeros días:** las señales enviadas antes del 16-nov-2026 se dimensionaron con el
tope antiguo (riesgo 0,1–0,8 %); mientras sigan abiertas, R3 puede marcar una señal con una sola
cripto abierta. Desaparece solo en ≤ 4 días (horizonte de las cripto).

## Qué esperar (medido antes de empezar)

Simulación de 10.000 secuencias por caso (script de la Fase 1, `D-vigilancia/oc_sprt_cusum.py`):

| Si el R medio real fuera… | SPRT acaba en "edge aceptado" | "sin edge" | Sin decidir a las 150 | Señales hasta decidir (mediana) | CUSUM salta antes de 150 señales |
|---|---|---|---|---|---|
| −0,25R | 0 % | 100 % | 0 % | 26 | 99,8 % (mediana 47 señales) |
| 0 | 4,5 % | 95,5 % | 13 % | 57 | 74 % (mediana 101) |
| +0,10R | 27 % | 73 % | 29 % | 88 | 43 % |
| +0,25R | 82 % | 18 % | 24 % | 92 | 9 % (falsa alarma) |
| +0,40R | 97 % | 3 % | 3 % | 61 | 1 % |

- Con la forma real de los resultados (muchos −1 y +1,33/+1,5/+2, no una campana) los errores
  quedan parecidos: α real 3,6 %, potencia 83–84 %. Agrupando las señales de BTC y ETH que salen
  en la misma vela (23 de 143 velas; correlación de sus R 0,41), α real 4,5 %.
- A ~26 señales live al año, 57–92 señales son **2–4 años**. El registro sombra (más N) da una
  lectura antes, pero no sustituye a la de las señales live.
- El CUSUM con h = 20R es **lento a propósito** (pocas falsas alarmas): si la ventaja cae a 0,
  tarda una mediana de ~100 señales en saltar. Su función es parar un deterioro claro, no
  detectar uno pequeño.
- Aplicado hacia atrás a la historia (166 setups aprobados de BTC, ETH y oro), el SPRT habría
  aceptado el edge en la señal 37 (sin costes) o 47 (con costes) y el CUSUM nunca pasó de 8R.
  **Es dentro de muestra** (los filtros se eligieron con esos datos): no prueba nada sobre el futuro.

## Puerta de validación: criterios de APROBADO (fijados el 16-nov-2026, Fase 2)

**Código:** `backtest/protocol.py` (orquestación y veredicto, constantes `CRIT_*`) y
`backtest/robustness.py` (CPCV, PBO, DSR). **Comando (offline, nunca en el worker):**

    python main.py validate --protocol                                  # línea base de la config actual
    python main.py validate --protocol --candidate cand.json            # examen de un candidato (sin reserva)
    python main.py validate --protocol --candidate cand.json --final    # examen final: UNA sola vez por candidato

Cualquier cambio de la estrategia (umbral, filtro, activo, dirección…) tiene que pasar este
examen antes de llegar al vivo. Los criterios se escribieron **antes** de ver ningún resultado
de candidatos y no se cambian a la vista de los datos; si alguna vez se cambian, se anota aquí
con fecha y motivo.

### Qué mira el examen (en sencillo)

Todo se mide en R netos con los costes de XM y un swap supuesto del 5 % anual (escenario de
decisión), **solo con datos anteriores al 15-nov-2025**. Lo posterior es la **reserva** y solo
se mira con `--final`.

- **CPCV con re-selección:** se parte la historia en 6 trozos; 15 veces se coge "la mejor
  variante" de una rejilla de 15 (umbral actual ±0,05/±0,10 × ambas/solo largos/solo cortos)
  usando 4 trozos y se prueba en los otros 2, dejando un margen entre ambos (el horizonte de
  una operación y un 1 % más) para que no se "vean". Dice si elegir con datos funciona en datos
  no vistos.
- **PBO:** probabilidad de que la variante que parece mejor en una mitad de la historia quede
  en la mitad de abajo en la otra mitad (0,5 = como elegir al azar).
- **DSR:** probabilidad de que la ventaja por operación sea real **descontando todas las pruebas
  hechas** (registro `backtest/trials_registry.jsonl`: 400 históricas + cada variante evaluada).
  Cuantas más ideas se prueban, más exigente se vuelve.

### Criterios (un candidato se adopta SOLO si cumple los cinco)

a) **Vivo = validador** con la configuración del candidato (lo que se enviaría por Telegram es
   exactamente lo que se ha medido).
b) **Mejora del E[R] neto ≥ +0,10R** frente a la configuración actual en la **mediana de la
   CPCV** (diferencia candidato − actual en los trozos de prueba) **y** en **al menos 2 de los 3
   periodos** fijos (cripto: may-2022→2023, 2024, 2025→14-nov-2025; oro/SPX: 2006–2014,
   2015–2020, 2021→14-nov-2025).
c) **PBO** de la rejilla con el candidato **< 0,20**.
d) **DSR** del candidato **> 0,95**.
e) Con `--final`: en la **reserva** y, en cripto, en el **histórico de OKX 2019-12→2022-04**, la
   diferencia candidato − actual **no es negativa**. Si no hay datos de OKX para comprobarlo,
   cuenta como no cumplido. Solo cuentan las operaciones cuyo **plazo completo** (hasta la vela
   límite de cierre por tiempo) ya está en los datos, **hayan salido antes o no** por SL/TP: si se
   aceptaran las que tocaron SL/TP pronto y se dejaran fuera las aún abiertas, se elegirían
   operaciones según su resultado (el SL está más cerca que el TP: sesgo hacia pérdidas). Regla
   corregida el 16-nov-2026 tras la revisión; no cambió ninguna cifra del ejemplo de ETH de abajo
   (en la reserva de BTC deja fuera 1 operación que entraba por un TP temprano).

El informe empieza por una línea: **"APROBADO"**, **"NO APROBADO: falla b, d"** o, si a)–d) pasan
pero aún no se hizo `--final`, **"PASA a)–d); FALTA e)"** (todavía no se adopta). Aprobar no
cambia nada solo: adoptar el candidato es decisión tuya y se anota aquí con fecha (y reinicia el
SPRT, regla 1).

Reglas de uso:
- `--final` se usa **una sola vez por candidato**. Cada acceso queda en el registro, **también el
  de la configuración actual**, porque el informe enseña siempre sus cifras de la reserva (y de
  OKX) junto a las del candidato. Si el candidato o la configuración actual ya vieron la reserva,
  el informe lo avisa en grande y ese resultado ya no es independiente.
- Si el registro de pruebas no se puede leer o escribir, **no hay veredicto**.
- Parámetros fijos: 6 grupos y 2 de prueba en la CPCV, purga de `forward_bars` velas a cada
  lado + embargo del 1 % de las velas, 16 bloques en el PBO, mínimo de 10 operaciones para que
  una variante pueda ser "la elegida" (si ninguna llega, se queda la actual).

### Línea base de la configuración actual (16-nov-2026, datos hasta el 15-nov-2026)

Cifras **dentro de muestra** para la config actual (se eligió mirando estos datos), solo como
referencia para comparar candidatos:

| Activo | E[R] XM swap 5 % (N) | CPCV: "elegir la mejor" en datos no vistos (mediana) | Config actual fija en esas pruebas (mediana) | PBO | DSR (sin descontar pruebas) |
|---|---|---|---|---|---|
| BTC | +0,50R (51) | +0,60R (0 de 15 por debajo de 0) | +0,50R | 0,56 | 0,36 (1,00) |
| ETH | +0,30R (55) | +0,06R (7 de 15 por debajo de 0) | +0,31R | 0,86 | 0,61 (0,94) |
| Oro | +0,31R (33) | +0,16R (2 de 14) | +0,45R | 0,55 | 0,39 (0,93) |
| SPX (sombra) | +0,12R (41) | +0,09R (6 de 15) | +0,16R | 0,15 | 0,02 (0,73) |

En el informe, la CPCV da la mediana y el rango "en 8 de cada 10 pruebas entre X e Y" (percentiles
10 y 90), además del **peor** y el **mejor** caso (en el oro, por ejemplo, entre −0,08 y +1,26,
pero el peor es −1,02R). En el oro una de las 15 pruebas no tiene operaciones y no cuenta: por eso
pone "2 de 14".

Cómo leerlo: **ninguna** configuración actual pasaría hoy c) y d) a la vez. Con ~30–55
operaciones por activo y 460 pruebas hechas, el DSR no puede llegar a 0,95 salvo con una
ventaja muy grande; el PBO de BTC, ETH y oro está en la zona de "elegir al azar" entre
variantes parecidas. En SPX el PBO bajo no indica ventaja: solo que las variantes "solo largos"
ganan siempre a las que incluyen cortos (muy negativas). Es decir, el examen es **exigente a
propósito**: lo normal será que un cambio no lo pase y la configuración siga congelada.

### Ejemplo registrado: "ETH con umbral 0,45" (informe final §7, 3.3) — NO APROBADO, no se adopta

Único `--final` de la Fase 2 (16-nov-2026 00:57 UTC, queda en el registro):
a) OK (91 aprobados = 91 emitidos); b) falla: mediana CPCV −0,04R, periodos −0,00 / +0,16 / −0,25R
(1 de 3); c) falla: PBO 0,86; d) falla: DSR 0,57; e) falla: reserva −0,15R (actual +0,42R con
N=8, candidato +0,26R con N=14), OKX +0,20R (actual −0,18R N=19, candidato +0,02R N=25).
Este candidato ya vio la reserva: si se vuelve a evaluar, su resultado ya no es independiente.
**La ETH actual (umbral 0,55) también la vio** en ese mismo `--final` (sus cifras: reserva +0,42R
N=8, OKX −0,18R N=19), así que para ella ya ningún tramo de la reserva ni de OKX es independiente.
Ese acceso no se registró entonces (solo se apuntaba el del candidato) y se añadió al registro a
posteriori el 16-nov-2026. El −0,18R de OKX se debe sobre todo al spread fijo de 5 USD aplicado a un
ETH de 130–400 USD (0,29R por operación frente a 0,08R hoy); con los costes del validador da
+0,08R. La diferencia candidato − actual casi no cambia (+0,20R con XM, +0,21R con el validador).

## Candidatos evaluados (Fase 3, 16-nov-2026) — ninguno aprobado, NO se adopta nada

Cinco ideas del informe (§7, 3.3) pasadas por la puerta **sin `--final`** (la reserva no se ha
mirado). Sin `--final` lo máximo posible es "PASA a)–d); FALTA e)", y ni así se adoptaría: hace
falta el examen final y tu decisión. Datos hasta el 15-nov-2026, selección antes de la reserva,
costes XM con swap 5 %. "Solo a favor de la SMA200" = solo largos con el cierre diario por encima
de su media de 200 días y solo cortos por debajo (con el último diario ya cerrado: en cripto, el del
día anterior). Se evaluó como **filtro** (quitar los que van en contra); la versión "reducir el
tamaño a la mitad" del informe no se evaluó.

| Candidato | Operaciones y E[R] (actual → candidato) | b) mejora mediana CPCV; periodos | c) PBO | d) DSR | Veredicto |
|---|---|---|---|---|---|
| BTC solo a favor de la SMA200 | 51 / +0,50R → 30 / +0,31R | −0,17R; −0,28 / +0,13 / −0,05 | 0,50 | 0,12 | NO APROBADO (falla b, c, d) |
| ETH solo a favor de la SMA200 | 55 / +0,30R → 26 / +0,48R | +0,16R; +0,26 / +0,29 / −0,04 (pasa) | 0,65 | 0,75 | NO APROBADO (falla c, d) |
| Oro solo a favor de la SMA200 | 33 / +0,31R → 26 / +0,32R | +0,00R; +0,77 / +0,24 / −0,16 | 0,54 | 0,39 | NO APROBADO (falla b, c, d) |
| SPX sin umbral (umbral 0, solo largos) | 41 / +0,12R → 101 / +0,08R | −0,01R; −0,01 / +0,12 / −0,35 | 0,15 (pasa) | 0,00 | NO APROBADO (falla b, d) |
| ETH umbral 0,50 | 55 / +0,30R → 65 / +0,23R | −0,07R; −0,09 / +0,06 / −0,18 | 0,86 | 0,45 | NO APROBADO (falla b, c, d) |

En los cinco, a) vivo = validador con la configuración del candidato: OK. Pruebas registradas al
evaluarlos: 460 → 464 (4 variantes nuevas: los tres filtros y SPX umbral 0; ETH 0,50 ya estaba en
la rejilla y no cuenta otra vez). La N del DSR de cada informe va de 461 a 464 porque cada candidato
cuenta también su propia prueba.

Cómo leerlo:
- **SMA200:** confirma lo que ya decía la auditoría (F32). En ETH sube el E[R] por operación, pero
  quita más de la mitad de las operaciones y el **R total baja** (de ~16,6R a ~12,5R); el PBO de 0,65
  dice que elegirlo mirando la historia es casi como elegir al azar. En BTC empeora y en el oro no
  cambia nada en la mediana. Que solo "funcione" en uno de los tres es justo lo que la regla de los
  dos activos independientes pide no adoptar.
- **SPX sin umbral:** más operaciones (101) con peor E[R] (+0,08R); no mejora. Sigue en sombra.
- **ETH 0,50:** igual que el 0,45 del ejemplo anterior: más operaciones, peor E[R], PBO 0,86.
- Ninguno se ha llevado al `--final`: no se gasta un acceso a la reserva sin un candidato que pase a)–d).

### Variables registradas en sombra desde el 16-nov-2026 (columna `features` del registro sombra)

Desde este despliegue, cada setup que el sistema detecta (emitido o no) se guarda con: si va a favor
de la SMA200 diaria, el signo de la tendencia a 21/63/252 días, el percentil de volatilidad (ATR),
lo lejos que está ya el precio de la entrada (en ATR, positivo = a favor) y, en BTC y ETH, el
**DVOL** de Deribit (volatilidad implícita) y el **VRP** = DVOL / volatilidad realizada de 30 días.
No cambian ninguna señal ni el mensaje. Si Deribit no responde (timeout de 5 s), DVOL y VRP quedan
vacíos ese ciclo y todo sigue igual. Estas variables sirven para decidir más adelante con datos
propios, sin volver a mirar la misma historia.

### Regla de activación del VRP (fijada el 16-nov-2026; informe §7, 2.5; experimento E6)

El VRP no se activa como filtro hasta que se cumpla TODO esto, contado **solo con setups registrados
desde el 16-nov-2026** (el registro empieza hoy; lo anterior es la historia con la que se generó la
idea y no cuenta):
1. **≥ 30 señales cripto aprobadas** (BTC + ETH: emitidas o bloqueadas, del registro sombra) ya
   resueltas y con VRP registrado (no vacío).
2. Filtro evaluado = el del experimento E6, sin cambios: conservar el setup si su VRP es ≥ el
   percentil 33 de los VRP de los setups ANTERIORES del mismo activo (con al menos 30 anteriores;
   antes de eso no se filtra).
3. **Mismo signo en BTC y en ETH:** E[R] de los conservados − E[R] de todos > 0 en los dos.
4. **Mejora del total ≥ +0,10R** por operación (BTC + ETH juntos, conservados frente a todos).
5. **Revisión obligatoria del mercado bajista:** si entre los setups con el diario por debajo de la
   SMA200 el filtro descarta más del 60 %, NO se activa sin revisarlo antes (en el OOS de E6, con
   mercado bajista, el VRP iba en contra).
Si se cumple, el VRP pasa como candidato por la puerta (`validate --protocol`, con `--final`) y la
activación es **decisión tuya**. Ritmo esperado: ~2 señales cripto aprobadas al mes → unos 14 meses.
Nota técnica: en vivo se guarda el último valor horario del DVOL; E6 usaba velas de 12 h con un
retraso prudente. La diferencia es pequeña, pero por eso solo cuenta lo registrado en vivo.

### Pendiente (3.4): solo documentado, no implementado

- Instrumento **sin swap** para SPX y oro (CFD sobre futuros con vencimiento, cuenta sin swap o
  futuros micro MES/MGC en otro bróker). Necesita tus datos de XM; mientras, SPX sigue en sombra.
- **RSI(2) en SPX**, solo si SPX revive con un instrumento sin swap.
- **Datos de derivados** (interés abierto, funding, DVOL) como variables: revisar tras 9–12 meses de
  registro propio.
- Resolver la **vela de entrada con datos de 1 hora** (saber qué tocó antes, SL o TP, cuando una
  vela 4H/diaria toca los dos).
- **Umbral por percentil** cuando el SPRT haya decidido.

## Monitor de datos (Fase 2, 16-nov-2026): qué avisa y propuesta PENDIENTE

Desde la Fase 2 cada ciclo revisa las últimas 60 velas de cada activo (velas planas, duplicadas,
con fecha de fin de semana, huecos, precio congelado, saltos absurdos y cambio de fuente) y cada
noche compara BTC/ETH entre OKX y Gate.io y cuenta los setups de 30 días frente a lo normal.
Lo grave llega por Telegram ("datos sospechosos", como mucho un aviso cada 6 h por activo; si el
envío falla, se reintenta en los ciclos siguientes durante 24 h; desde la Fase 4 esos avisos
pendientes se guardan en la base de datos y sobreviven a un reinicio del worker); todo
va al resumen nocturno en "Calidad de datos". **Hoy NO bloquea ninguna señal.**

Qué hacer si llega un aviso de "datos sospechosos": antes de operar una señal de ese activo,
compara la última vela con TradingView o con XM. Si no cuadra, no entres.

**Propuesta pendiente de tu decisión (no implementada):** no emitir la señal de un activo si en
ese mismo ciclo hay una incidencia GRAVE en sus 2 últimas velas (vela plana, duplicada o en fin de
semana) o un salto absurdo en la última. Cuánto quitaría (medido sobre el histórico con el
monitor): en cripto, las 8 velas planas por serie que dejaba el fallo de Gate.io (ya corregido);
desde 2020, en oro 1 aviso (caída real del −10,8 % del 30-ene-2026) y en SPX 1 (−12 % del
16-mar-2020): días de mercado de verdad, en los que bloquear sería discutible. Antes de 2020
habría habido además 56 avisos de velas planas del oro (datos raros de Yahoo en 2006–2019) y 2
saltos del SPX (octubre de 2008). Si la quieres,
dilo y se implementa como cambio de configuración (cambia el vivo: necesita tu visto bueno).

Límites conocidos: oro y SPX no tienen una segunda fuente gratuita fiable para contrastar; en
oro y SPX la "deriva" solo puede avisar de demasiados setups (lo normal es 0 al mes); y si se
cambia la estrategia o un umbral hay que recalcular `DRIFT_BASELINE` (si no, avisará en falso).

## Filtro de eventos macro (Fase 2, 16-nov-2026): medición — QUITADO el 16-nov-2026 (Fase 3)

**Decisión tomada (tú, 16-nov-2026: "Sí, quita el filtro si no aporta nada").** Con los números
de abajo (0 señales bloqueadas o retrasadas por un FOMC/CPI/NFP desde 2015, y ninguna diferencia
concluyente de resultado con un evento cerca), **el bloqueo por eventos se quitó** del vivo
(`main._emit_best_setup` ya no llama a `is_event_blackout` y `EVENT_FILTER` desapareció de la
configuración). El motivo "evento macro cercano" del registro sombra ya no se produce (las filas
antiguas se conservan). **Se mantienen:** el calendario `data/econ_events.csv`, la línea de aviso
"⚠️ Evento de alto impacto dentro del plazo de la operación…" en cada mensaje y los eventos de las
próximas 48 h en el resumen nocturno. No cambia qué señales salen (check-live idéntico), así que
no reinicia el SPRT.

**Qué hacía (hasta el 16-nov-2026).** Antes de emitir una señal de oro o SPX, el sistema miraba si
había un evento de alto impacto de EE. UU. entre 1 h antes y 2 h después de ese momento; si lo había,
no la emitía en ese ciclo (lo volvía a intentar al ciclo siguiente). Cripto estaba exento. Desde la Fase 2 los eventos
salen de un calendario propio (`data/econ_events.csv`: FOMC, CPI y empleo/NFP desde 2015 hasta
2027, de federalreserve.gov y bls.gov) y el feed semanal de ForexFactory queda como complemento
(funciona hoy: `thisweek` responde; `nextweek` da 404 pero el sistema no lo usa).

**¿Actúa alguna vez? En la práctica, no.** El vivo evalúa el oro y el SPX después del cierre diario
(22:15–23:00 UTC en invierno, o desde las 00:00 de Nueva York = 04:00–05:00 UTC) y los eventos son
a las 12:30/13:30 UTC (CPI, empleo) y 18:00/19:00 UTC (Fed). Con todas las señales aprobadas desde
2015 (36 de oro, 28 de SPX) y probando los 60 minutos posibles del ciclo horario, **0 habrían sido
retrasadas o bloqueadas** por un FOMC/CPI/NFP (tampoco las de cripto si no estuviera exento: 0 de
127). Solo podría actuar con un evento del feed a una hora rara (p. ej. un discurso nocturno), y
entonces solo retrasa la señal 1–3 h. Tenía sentido antes de la Fase 0, cuando el sistema emitía
durante el día con la vela aún abierta.

**¿Son peores las operaciones con un evento cerca?** Medido sobre las aprobadas antes de la reserva
(cripto 2022-05 → 2025-11, oro y SPX 2015 → 2025-11; costes XM con swap 5 %; 159 operaciones):

| Ventana | Con evento | Sin evento | Diferencia (IC90) |
|---|---|---|---|
| Vela de entrada | 2 (+1,20R) | 157 (+0,34R) | no evaluable: solo 2 con evento |
| Primeras 24 h | 17 (+0,29R) | 142 (+0,36R) | −0,07R (−0,62 a +0,47) |
| Todo el plazo | 89 (+0,24R) | 70 (+0,49R) | −0,24R (−0,58 a +0,10); no evaluable: en oro y SPX el 100 % tiene evento |

Por grupos (primeras 24 h): cripto −0,42R (−1,07 a +0,26; ETH −1,01R con solo 6 operaciones),
oro y SPX +0,73R (−0,09 a +1,42). Todo el plazo, solo cripto: −0,26R (−0,70 a +0,19).
Muestras pequeñas: ninguna diferencia es concluyente y los signos de cripto y de oro/SPX se
contradicen.

**Regla fijada antes de medir:** bloquear solo si el IC90 del agregado queda entero por debajo de
−0,10R y la diferencia es negativa en cripto y en oro/SPX. **No se cumple en ninguna ventana →
recomendación: AVISAR, no bloquear.** Ya implementado (no cambia qué señales salen): el mensaje de
cada señal añade "⚠️ Evento de alto impacto dentro del plazo de la operación: CPI 10-dic 13:30 UTC"
(hasta 3 eventos entre la entrada y la hora límite de cierre por tiempo, seguidos de "(solo aviso: no
es motivo para no entrar)"; saldrá en casi todas las de oro y SPX y en ~1 de cada 3 de cripto) y el resumen nocturno lista los eventos de las próximas 48 h.

**Decisión (16-nov-2026):** (1) filtro quitado (ver arriba); (2) no se descartan setups por
eventos. Qué hacer con el aviso: nada especial (los datos no
dicen que esas operaciones sean peores; saltarlas por el evento va contra la regla de no saltarse
señales). Si aun así saltas una, apúntalo con `/skip ID evento` para que la medición siga siendo honesta. Mantenimiento: cuando la Fed o el BLS publiquen el calendario de 2028, añadirlo
al CSV (el resumen nocturno avisa 60 días antes de que se acabe). **Pendiente: añadir el calendario
de 2028 antes de octubre de 2027** (ver "Mensaje y registro (Fase 4)": no falta ningún FOMC de 2027).

## Cuándo llegan las señales (Fase 4, 16-nov-2026): ciclo a hora fija

**Qué cambió.** Antes el worker revisaba los mercados "cada 60 minutos desde que arrancó": tras el
último despliegue, a las :40 de cada hora (y unos segundos más tarde cada hora). Desde la Fase 4
revisa a horas **fijas en UTC** (`CYCLE_SCHEDULE` en `config/settings.py`):

- **Cada hora en HH:01 UTC.** La vela 4H de BTC y ETH cierra en punto (00, 04, 08, 12, 16 y 20 UTC);
  el minuto de margen deja que Gate.io la cierre. La señal llega ~1 minuto después del cierre.
- **Un ciclo extra a las 22:20 UTC** para la vela diaria del oro (y del SPX en sombra), que se da por
  cerrada a las 22:15 UTC. El sistema no acepta la vela del día desde las 18:00 de Nueva York hasta
  medianoche de Nueva York (en ese rato Yahoo puede estar mostrando ya la sesión siguiente). Por eso:
  - **Invierno** (horario estándar de EE. UU., de noviembre a marzo): la señal de oro llega a las
    **22:20 UTC** de ese mismo día.
  - **Verano** (de marzo a noviembre): las 18:00 de Nueva York son las 22:00 UTC, antes de las 22:15.
    La vela se acepta desde medianoche de Nueva York y la señal llega a las **04:01 UTC** del día
    siguiente. Antes llegaba en el primer ciclo después de las 04:00, por ejemplo a las 04:40.
- **Comandos /fill /skip /help:** se leen cada 2 minutos (antes, una vez por hora). La respuesta llega en
  1–2 minutos, o algo más si en ese momento el sistema está revisando los mercados.
- Sigue igual: una revisión nada más arrancar, el resumen nocturno de las 03:00 UTC (va justo antes del
  ciclo de las 03:01) y los reintentos de señales que no llegaron (a la hora y a las 2 horas).
- El startCommand de Render (`--interval 60`) no hay que tocarlo: 60 = cada hora en HH:01. Otro
  intervalo que divida el día (15, 30, 120, 240…) también se alinea a las 00:01 UTC. Uno que no lo divida
  (por ejemplo 50) vuelve al modo antiguo (cada N minutos desde el arranque) y el log lo avisa. En todos
  los casos se añade el ciclo de las 22:20.
- Latido (HEARTBEAT_URL): llega cada hora tras HH:01 (y a las 22:20). En el servicio externo pon un
  periodo de 1 hora y un margen de 30 minutos o más.

**Las señales son las mismas.** Cambia CUÁNDO llegan, no QUÉ llega. Comprobado con el worker real
(`main.cmd_schedule` con su planificador) y un reloj simulado sobre el snapshot del 15-nov-2026, sin red y
sin Telegram. Se comparó el código anterior (3518d41, despliegue a las :40) con el nuevo en cinco periodos:
8–16 jul 2025, 8–23 oct 2025, 27 sep–3 oct 2026 (verano, con oro), 8–15 nov 2024 y 29 nov–5 dic 2023
(invierno, con oro). Resultado: **13 de 13 señales iguales** (activo, vela, dirección, entrada, SL, TP y
modo) y los mismos motivos en el registro sombra (43 filas). Con un despliegue a las :07 (8–15 nov 2024)
también salen las mismas señales.

| Retraso desde el cierre de la vela | Antes (despliegue a las :40) | Ahora |
|---|---|---|
| Vela 4H (BTC/ETH), 277 cierres: mediana / máximo | 46 / 60 min | 1 / 1 min |
| Señales cripto emitidas (7): mediana / máximo | 45 / 51 min | 1 / 1 min |
| Oro, invierno (9 velas): mediana / máximo | 32 / 37 min | 5 / 5 min (22:20) |
| Oro, verano (22 velas): mediana / máximo | 6 h 31 min / 6 h 43 min | 5 h 46 min (04:01) |

Antes, el retraso dependía de la hora del último despliegue. Con un despliegue entre las :00 y las :15, el
oro de invierno no se evaluaba esa noche, sino a las ~05:00 UTC del día siguiente: en la simulación con
despliegue a las :07, la señal de oro del 11-nov-2024 llegó a las 05:14 UTC del 12-nov en vez de a las
22:20 UTC del 11-nov.

## Dashboard de solo lectura (Fase 4, 16-nov-2026)

**Las señales oficiales llegan solo por Telegram**, con su ID para responder /fill y /skip. El
dashboard ya no escribe en la base de datos ni envía nada:

- **Quitado "Registrar señal en DB" y "Enviar a Telegram".** Si registrabas un setup desde el
  dashboard antes que el worker, el worker lo descartaba como duplicado: no te llegaba por
  Telegram ni tenía ID para /fill. El envío desde el dashboard mandaba una señal sin ID.
- **Quitado el modelo ML** (pestaña, entrenamiento y descarga de datos macro). El ML está apagado
  en todos los activos y entrenar en el dashboard podía agotar la memoria.
- **"Ver lo que ve el worker"** (antes "Generar señal") es una vista. Usa las mismas velas que el
  worker (historia corta del vivo, solo velas cerradas), el mismo detector, el mismo umbral, la misma
  ventana y frescura, los mismos controles de duplicado y el mismo tamaño (TRADING_CAPITAL y riesgo del
  0,25 %). Dice qué haría el worker con cada setup de la ventana. Ver la vista no cambia lo que emite el
  worker. Comprobado con el worker real y un reloj simulado en los cinco periodos de arriba: la vista
  coincide con lo emitido en todos los instantes y no escribió nada en la base de datos.
- **Quitado "Resolver señales pendientes".** El worker ya cierra las señales en cada ciclo (cada hora),
  con velas cerradas y la misma regla que la validación.
- **Quitado el cierre a mano.** Sustituía el resultado del sistema por el tuyo en la vigilancia
  (SPRT/CUSUM), y la vigilancia mide el sistema con la entrada del mensaje. Lo que ejecutas de verdad se
  registra con /fill y /skip, y se compara aparte (regla 6). Las señales que ya cerraste a mano siguen
  contando como antes (`resolved_by='manual'`).
- **Nuevo en el dashboard**, todo solo leyendo la base de datos:
  - vigilancia (SPRT live y del registro sombra, CUSUM y "configuración congelada N/150");
  - cartera según el sistema y cuántas señales habría bloqueado cada regla;
  - tus /fill y /skip por señal, con el R teórico y el R ejecutado (misma salida, entrando a tu precio);
  - registro sombra: setups por motivo y activo, y resultados de los resueltos;
  - la línea de riesgo por operación.
- La prueba de Telegram se hace con `python trading_system/main.py telegram`.

## Mensaje y registro (Fase 4, 16-nov-2026; no cambia qué señales salen)

- **Distancias del SL y del TP en el mensaje** (todos los activos), justo debajo de los niveles:
  "Distancias desde la entrada: SL a 1.50 % (1,350.00 USD por BTC); TP a 2.25 % (…)". En el SPX van
  en puntos del índice. Sirven para colocar el SL y el TP sobre **tu** precio de entrada si el de tu
  bróker no coincide con el del mensaje.
- **Oro:** el mensaje añade que los niveles salen del futuro GC=F, que suele cotizar por encima del
  oro de XM (en nov-2026 ~0,3–0,5 %; tras el cambio de contrato puede ser ~0,7–1 %, estimación sin
  medir): "si tu precio en MT5 es distinto, coloca el SL y el TP a esas distancias de tu precio de
  entrada". Así el riesgo en R es el de la señal aunque el nivel absoluto sea otro (regla 6).
- **Precio al emitir:** cada señal nueva guarda en la columna `price_at_emit` el precio actual que
  usó el ciclo (cierre de la última vela cerrada; el mismo de la línea "Precio actual" y del filtro de
  frescura). Es solo registro (F14 del informe): permitirá medir cuánto se había movido el precio
  desde la entrada al llegar la señal. No cambia la resolución ni el R (siguen con la entrada del
  mensaje). Las señales anteriores quedan con el valor vacío.
- **Resumen nocturno, bloque "Últimos 12 meses":** nº de señales live resueltas en los últimos 365
  días y su R medio (total y por activo; sin costes del bróker), de los activos que hoy se notifican y
  sin las repeticiones que marcó la limpieza de la Fase 0. Incluye las anteriores a la Fase 0, así que
  **no** es lo que cuenta la vigilancia. Cómo leerlo: el sistema gana sobre todo en meses con
  tendencia (F21), así que un año flojo no prueba por sí solo que esté roto; las decisiones las toman
  el SPRT y el CUSUM.
- **Calendario de eventos (comprobado el 16-nov-2026 en federalreserve.gov, página "Meeting calendars",
  actualizada el 12-nov-2026):** en 2027 la Fed **no** tiene reunión en diciembre. Las de 2027 son
  26-27 ene, 16-17 mar, 27-28 abr, 8-9 jun, 27-28 jul, 14-15 sep, 26-27 oct (la página la marca como
  "Monetary Policy Strategy Meeting") y 16-17 nov; todas ya estaban en `data/econ_events.csv`. La
  siguiente es el 25-26 de enero de 2028: **no se añade todavía** para que el aviso del resumen ("el
  calendario se acaba en < 60 días") siga saltando a tiempo por el CPI/empleo de 2028, que el BLS aún no
  ha publicado. **Pendiente: añadir el calendario de 2028 (Fed y BLS) antes de octubre de 2027.**
