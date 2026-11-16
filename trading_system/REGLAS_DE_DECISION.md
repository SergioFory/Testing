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
  aunque no se enviaran (por frescura, evento macro, duplicado, etc.), resueltos con la misma
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
   **Cómo se aplica:** a mano, operando **la mitad de las unidades (o lotes)** que dice el
   mensaje en la línea "Para respetar el 0.25 %" (o en "Tamaño" si esa línea no aparece).
   Cambiar `risk_per_trade` en la configuración NO sirve: el tamaño casi siempre lo fija el
   tope del 10 % de nominal (ver abajo) y bajarlo a la mitad solo cambia 10 de 212 setups.
3. **El riesgo no sube del 0,25 %** por operación hasta que el SPRT con las señales live diga
   "edge aceptado". (En la práctica puede tardar años; esperar no cuesta rendimiento.)
   **Ojo: hoy el sistema NO lo cumple solo** (ver "Riesgo real hoy"). Cada mensaje compara el
   riesgo real con el 0,25 % y, si lo supera, añade la línea "Para respetar el 0.25 %: X
   unidades": es el tamaño que hay que operar para cumplir esta regla.
4. **SPX vuelve a notificarse solo si** el E[R] neto con los costes reales de XM es
   ≥ +0,10R en al menos 2 de 3 épocas **y** pasa la puerta de validación (Fase 2).
   Si no, sigue en sombra (se registra y se mide, sin coste).
5. **El filtro VRP (volatilidad implícita/realizada) solo se activa** tras ≥ 30 señales cripto
   aprobadas fuera de muestra (y con las demás condiciones de la Fase 2: mismo signo en BTC y
   ETH, mejora ≥ +0,10R).
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

## Riesgo real hoy (añadido en la revisión de la Fase 1, 15-nov-2026)

El tamaño lo calcula el sistema con `risk_per_trade` = 1 % y un **tope de nominal del 10 % del
capital**; casi siempre manda el tope, así que el riesgo real depende de lo lejos que esté el
stop. Medido con los setups aprobados de la historia (capital 10.000 USD, script
`G-correcciones/riesgo_real.py`):

| Activo | Últimos 12 meses: riesgo mediano | máximo | señales por encima del 0,25 % |
|---|---|---|---|
| BTC | 0,21 % | 0,34 % | 1 de 13 (8 %) |
| ETH | 0,27 % | 0,55 % | 5 de 8 (63 %) |
| Oro | 0,40 % | 0,68 % | 6 de 6 (100 %) |
| SPX (sombra) | 0,19 % | 0,22 % | 0 de 5 |

En toda la historia el rango va de ~0,1 % a ~0,8 %. Es decir: con el tamaño del mensaje, la
regla 3 no se cumple en el oro ni en la mayoría de señales de ETH. Para cumplirla hay dos
caminos: (a) operar el tamaño de la línea "Para respetar el 0.25 %" del mensaje (lo que se
recomienda mientras dure la congelación), o (b) cambiar la configuración a un tope de RIESGO
del 0,25 % en lugar del tope de nominal, o `risk_per_trade` = 0,0025 (que cambia el tamaño en
94 de 212 setups). La opción (b) es un cambio de configuración: no altera qué señales salen,
pero se anota aquí con fecha si se hace.

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
   cuenta como no cumplido.

El informe empieza por una línea: **"APROBADO"**, **"NO APROBADO: falla b, d"** o, si a)–d) pasan
pero aún no se hizo `--final`, **"PASA a)–d); FALTA e)"** (todavía no se adopta). Aprobar no
cambia nada solo: adoptar el candidato es decisión tuya y se anota aquí con fecha (y reinicia el
SPRT, regla 1).

Reglas de uso:
- `--final` se usa **una sola vez por candidato**. Cada acceso queda en el registro; si un
  candidato ya vio la reserva, el informe lo avisa en grande y su resultado ya no es independiente.
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

## Monitor de datos (Fase 2, 16-nov-2026): qué avisa y propuesta PENDIENTE

Desde la Fase 2 cada ciclo revisa las últimas 60 velas de cada activo (velas planas, duplicadas,
con fecha de fin de semana, huecos, precio congelado, saltos absurdos y cambio de fuente) y cada
noche compara BTC/ETH entre OKX y Gate.io y cuenta los setups de 30 días frente a lo normal.
Lo grave llega por Telegram ("datos sospechosos", como mucho un aviso cada 6 h por activo); todo
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

## Filtro de eventos macro (Fase 2, 16-nov-2026): medición y decisión PENDIENTE

**Qué hace hoy.** Antes de emitir una señal de oro o SPX, el sistema mira si hay un evento de alto
impacto de EE. UU. entre 1 h antes y 2 h después de ese momento; si lo hay, no la emite en ese
ciclo (lo vuelve a intentar al ciclo siguiente). Cripto está exento. Desde la Fase 2 los eventos
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
(hasta 3 eventos entre la entrada y la hora límite de cierre por tiempo; saldrá en casi todas las de
oro y SPX y en ~1 de cada 3 de cripto) y el resumen nocturno lista los eventos de las próximas 48 h.

**Decisión pendiente (tuya):** (1) dejar el filtro actual como está (no molesta, pero no protege de
nada) o quitarlo para simplificar; (2) no se recomienda descartar setups por eventos. Cualquiera de
las dos cambia el vivo y necesita tu visto bueno. Qué hacer con el aviso: nada especial (los datos no
dicen que esas operaciones sean peores; saltarlas por el evento va contra la regla de no saltarse
señales). Si aun así saltas una, apúntalo con `/skip ID evento` para que la medición siga siendo honesta. Mantenimiento: cuando la Fed o el BLS publiquen el calendario de 2028, añadirlo
al CSV (el resumen nocturno avisa 60 días antes de que se acabe).
