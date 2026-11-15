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
