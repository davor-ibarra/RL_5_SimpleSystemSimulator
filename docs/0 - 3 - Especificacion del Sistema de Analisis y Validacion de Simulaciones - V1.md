# Especificacion del Sistema de Analisis y Validacion de Simulaciones - V1

## 1. Proposito del documento

Este documento define los requerimientos funcionales, analiticos y de trazabilidad para construir un notebook y, eventualmente, codigo auxiliar de analisis que permita evaluar simulaciones del sistema de autosintonia RL de manera consistente, reutilizable y orientada a decisiones.

Este sistema debe concebirse como una herramienta externa de post-analisis. No forma parte del flujo de simulacion en linea ni del ciclo interno de ejecucion del entorno, los controladores o el entrenamiento. Su rol es consumir los artefactos ya exportados por una corrida terminada y transformarlos en evidencia util para la toma de decisiones.

El documento no propone codigo. Su funcion es dejar completamente especificado:

- que entradas deben analizarse,
- que bloques de analisis deben existir,
- que tablas y figuras deben generarse,
- como deben guardarse los artefactos,
- y que preguntas tecnicas deben poder responderse para apoyar la seleccion de hiperparametros y el refinamiento de la funcion de recompensa.

La expectativa central es que este sistema permita hacer una radiografia completa del experimento y, en particular, de cada hiperparametro relevante. Eso implica que el analisis no debe limitarse a listar valores configurados, sino que debe explicar:

- que hace cada hiperparametro,
- por que ese hiperparametro importa fisica o algoritimicamente,
- en que region operativa estuvo actuando realmente,
- que evidencia experimental muestra su efecto,
- y que justificacion existe para mantenerlo, modificarlo o desactivarlo.

## 2. Alcance

El sistema de analisis futuro debe cubrir, como minimo, cuatro dimensiones:

- evaluacion del aprendizaje a lo largo de la corrida,
- evaluacion fisica del comportamiento del sistema controlado,
- evaluacion estructural del comportamiento de la funcion de recompensa,
- y apoyo explicito a la toma de decisiones sobre hiperparametros y bloques activos del reward.

Debe operar despues de finalizada la simulacion, usando solo archivos ya persistidos. Por tanto, este documento no especifica cambios al loop de simulacion, sino los requerimientos de una capa externa de analisis comparable con lo que hoy representan `analysis/run_analysis.py`, `analysis/run_reward_behavior_notebook.py` y el notebook `0_design_functionReward.ipynb`.

No debe limitarse a "hacer graficos". Debe entregar evidencia util para responder si una corrida:

- mejora o empeora,
- por que mejora o empeora,
- en que fase de la maniobra aparece el cuello de botella,
- si la recompensa esta alineada o no con el comportamiento fisicamente deseado,
- y que ajustes concretos de parametros tienen mayor probabilidad de ser utiles en la siguiente simulacion.

## 3. Principios de diseno del sistema de analisis

El notebook o sistema futuro debe construirse con estos principios:

- Reproducibilidad: una misma corrida debe poder reanalizarse cuantas veces sea necesario obteniendo el mismo conjunto de artefactos.
- Persistencia: todas las figuras, tablas y resumentes deben guardarse dentro de la carpeta `analysis/<sim_id>/`.
- Modularidad: cada bloque de analisis debe poder ejecutarse de forma relativamente independiente.
- Trazabilidad: cada salida debe ser relacionable con la configuracion y los datos que la originaron.
- Orientacion a decisiones: cada bloque debe responder preguntas concretas de tuning, no solo producir visualizaciones esteticas.
- Comparabilidad: debe ser facil contrastar una corrida con otra.
- Externalidad: debe ejecutarse fuera del flujo de simulacion, como consumidor de resultados ya generados.
- No intrusividad: no debe depender de hooks en tiempo real ni alterar la ejecucion de la corrida que analiza.
- Auditabilidad de hiperparametros: cada hiperparametro activo debe poder rastrearse desde su valor configurado hasta su efecto observado.
- Explicabilidad causal: el sistema debe justificar, con apoyo en figuras y tablas, por que un hiperparametro parece ayudar, perjudicar o volverse irrelevante.

## 4. Modelo de radiografia completa de hiperparametros

La herramienta externa de analisis debe tratar los hiperparametros como objetos de estudio explicitos. No basta con mostrar un bloque de configuracion. Debe construirse una radiografia parametro por parametro.

Cada hiperparametro relevante deberia quedar documentado, al menos, con los siguientes campos analiticos:

- nombre canonico,
- bloque al que pertenece,
- valor activo en la corrida,
- rango o contexto esperado de operacion,
- rol funcional dentro del sistema,
- senales sobre las que actua,
- fase de la maniobra en la que deberia influir,
- evidencia empirica de activacion o no activacion,
- riesgos de valor demasiado bajo,
- riesgos de valor demasiado alto,
- conclusion diagnostica,
- recomendacion de ajuste.

Esta radiografia deberia cubrir, como minimo, cuatro familias:

- hiperparametros del reward principal,
- hiperparametros de extras y shaping,
- hiperparametros relevantes de entrenamiento RL,
- hiperparametros relevantes de condiciones de termino y bandas de evaluacion.

Si algun hiperparametro aparece en metadata pero no tiene impacto observable o no puede justificarse desde el analisis, eso mismo debe quedar explicitado. La ausencia de efecto observable tambien es un hallazgo.

## 5. Entradas minimas requeridas

El sistema de analisis debe poder trabajar, como minimo, con los siguientes archivos de una simulacion:

- `metadata.json`
- `summary.xlsx`
- `episodes_chunks_*.json`

Idealmente tambien debe reconocer y reutilizar:

- figuras ya generadas dentro de la carpeta de resultados,
- configuraciones exportadas,
- y cualquier artefacto previo de analisis que permita comparacion incremental.

### 4.1. Rol de cada entrada

`metadata.json` debe usarse para:

- reconstruir la configuracion activa,
- identificar parametros de recompensa, simulacion, entrenamiento y sistema,
- y dejar un snapshot claro del experimento.

`summary.xlsx` debe usarse para:

- analisis agregado por episodio,
- filtros por `last_n`, `top_reward`, `top_performance`,
- y contrastes rapidos entre subconjuntos.

`episodes_chunks_*.json` debe usarse para:

- reconstruir trayectorias temporales,
- analizar dinamica intra-episodio,
- inspeccionar reward por intervalo y por step cuando corresponda,
- y construir mapas de calor, perfiles temporales y analisis de fases.

## 5. Estructura de salida esperada

Cada corrida analizada debe producir una carpeta autocontenida:

- `analysis/<sim_id>/`

Dentro de ella, la estructura minima recomendada es:

- `analysis/<sim_id>/manifest/`
- `analysis/<sim_id>/hyperparameters/`
- `analysis/<sim_id>/summary/`
- `analysis/<sim_id>/learning/`
- `analysis/<sim_id>/episodes/`
- `analysis/<sim_id>/reward_behavior/`
- `analysis/<sim_id>/comparison/`
- `analysis/<sim_id>/decision_report/`

Ademas, deberian existir archivos consolidados en la raiz de `analysis/<sim_id>/`, por ejemplo:

- `analysis_results.xlsx`
- `analysis_notes.md`
- `analysis_decision_report.md`
- `hyperparameter_radiography.xlsx`

Esta estructura debe ser generada por el sistema externo de analisis. No debe ser una carpeta administrada por el runtime principal de simulacion ni por el loop de entrenamiento.

## 6. Parametros de ejecucion que el sistema de analisis deberia aceptar

El notebook o runner futuro deberia aceptar, como minimo, estos parametros:

- `sim_id`
- `compare_sim_ids`
- `last_n`
- `top_k_reward`
- `top_k_performance`
- `top_k_stability`
- `selected_episode_ids`
- `export_figures`
- `export_tables`
- `overwrite_existing`
- `time_grid`
- `angle_thresholds`
- `cart_thresholds`
- `reward_profile_grids`

La idea es que estos parametros permitan reutilizar el mismo flujo de analisis en distintas simulaciones sin tener que editar manualmente celdas o scripts cada vez.

Estos parametros pertenecen a la herramienta externa de analisis. No son hiperparametros de simulacion ni deben contaminar la configuracion operativa del experimento.

## 7. Bloques de analisis requeridos

Lo que sigue no es opcional. Es la especificacion de los bloques que el sistema futuro deberia cubrir para sustentar decisiones de tuning de forma seria.

### Bloque 0. Manifiesto de insumos y control de consistencia

Objetivo:

- validar que la corrida tiene todos los insumos requeridos,
- registrar rutas, fechas y huellas basicas,
- y dejar evidencia de que el analisis fue ejecutado sobre el conjunto correcto de archivos.

Salidas minimas:

- tabla con rutas de entrada encontradas,
- tabla con tamano y fecha de cada archivo,
- resumen de cantidad de episodios y chunks detectados.

Artefactos esperados:

- `manifest/input_manifest.csv`
- `manifest/input_manifest.md`

Preguntas que debe responder:

- Se analizaron los archivos correctos?
- Falta alguna fuente critica para interpretar la corrida?

### Bloque 1. Snapshot contextual de configuracion

Objetivo:

- reconstruir y presentar de forma legible la configuracion realmente usada en la corrida.

Contenido minimo:

- parametros de sistema fisico,
- parametros del entrenamiento RL,
- parametros de reward principal,
- parametros de extras,
- condiciones de termino,
- y seeds si existen.

Artefactos esperados:

- tabla de metadata resumida,
- seccion de reward activa,
- seccion de parametros comparativos contra corrida previa cuando corresponda.

Preguntas que debe responder:

- Que cambio exactamente entre una corrida y la siguiente?
- Cual era la hipotesis experimental detras de la configuracion activa?

### Bloque 1B. Radiografia y justificacion de hiperparametros

Objetivo:

- transformar la configuracion activa en un mapa analitico de hiperparametros, explicando el rol y la justificacion de cada uno.

Cobertura minima:

- hiperparametros del reward principal,
- hiperparametros de extras,
- hiperparametros RL relevantes,
- parametros de termination y bandas,
- y cualquier otro parametro que altere materialmente la interpretacion de resultados.

Para cada hiperparametro, el sistema deberia construir una ficha minima con:

- nombre,
- ruta completa dentro de metadata o configuracion,
- valor activo,
- bloque funcional al que pertenece,
- descripcion tecnica,
- hipotesis de efecto esperada,
- senales y metricas con las que se relaciona,
- evidencia observada en la corrida,
- conclusion diagnostica,
- recomendacion.

Artefactos minimos:

- tabla `hyperparameter_inventory`
- tabla `hyperparameter_radiography`
- reporte `hyperparameter_justification`

Preguntas que debe responder:

- Que hiperparametros fueron realmente determinantes en la corrida?
- Cuales estuvieron activos pero sin evidencia de efecto?
- Cuales parecen estar mal calibrados segun la region del espacio de estados efectivamente visitada?

### Bloque 2. Resumen global de resultados

Objetivo:

- entregar una lectura compacta del resultado general de la corrida antes de entrar en detalles.

Indicadores minimos:

- episodios totales,
- estabilizaciones,
- reward total medio y percentiles,
- performance medio y percentiles,
- duracion media de episodio,
- razones de fallo,
- mejores episodios por reward,
- mejores episodios por performance.

Figuras minimas:

- histograma de `total_reward`,
- histograma de `performance`,
- grafico de barras de causas de termino,
- tabla compacta de episodios destacados.

Preguntas que debe responder:

- La corrida, en terminos generales, fue mejor, peor o equivalente?
- El problema dominante esta en captura, recentrado, estabilizacion o termination temprana?

### Bloque 3. Evolucion del aprendizaje

Objetivo:

- analizar como cambian las metricas a lo largo de los episodios.

Figuras minimas:

- series temporales de `total_reward`,
- series temporales de `performance`,
- series temporales de duracion de episodio,
- medias moviles y bandas percentil,
- version recortada a `last_n`.

Tablas minimas:

- comparacion entre primeras, medias y ultimas ventanas,
- episodios top dentro de `last_n`,
- episodios top globales.

Preguntas que debe responder:

- Hay aprendizaje sostenido o hubo regresion tardia?
- La politica final representa lo mejor que la corrida descubrio?
- Los mejores episodios quedaron solo como hallazgos aislados?

### Bloque 4. Analisis de hiperparametros de control aprendidos

Objetivo:

- inspeccionar la evolucion de las ganancias PID propuestas por los agentes y su relacion con el desempeno.

Figuras minimas:

- evolucion de `kp`, `ki`, `kd` por controlador,
- distribuciones globales y en `last_n`,
- comparacion entre `top_reward`, `top_performance` y `last_n`.

Tablas minimas:

- percentiles de ganancias por lazo,
- resumen de combinaciones presentes en episodios destacados.

Preguntas que debe responder:

- Las ganancias convergen o siguen dispersas?
- La politica final esta cerca de los valores que generan episodios fisicamente utiles?
- Existe sesgo a ganancias extremas, saturadas o poco realistas?

### Bloque 5. Analisis de causas de fallo y restricciones fisicas

Objetivo:

- separar claramente si la corrida falla por angulo, por posicion del carro, por tiempo, por saturacion o por otra condicion activa.

Figuras minimas:

- barras apiladas de causas de termino,
- comparacion de causas en global, `last_n`, `top_reward` y `top_performance`,
- distribucion de maxima excursion del carro,
- distribucion de maximo angulo y maxima velocidad.

Preguntas que debe responder:

- Cual es la restriccion fisica que actualmente manda?
- Hay cambio de modo de fallo entre una etapa y otra del aprendizaje?

### Bloque 6. Seleccion estructurada de episodios representativos

Objetivo:

- construir subconjuntos de episodios que permitan mirar la corrida desde distintas perspectivas, sin depender solo del mejor episodio global.

Subconjuntos minimos:

- `top_k_reward`
- `top_k_performance`
- `top_k_stability`
- `last_n`
- episodios con mejor angulo minimo
- episodios con mejor retorno del carro
- episodios de fallo temprano representativo

Artefactos minimos:

- tabla de episodios seleccionados,
- archivo de referencia con sus `episode_id`,
- etiqueta explicita del criterio por el cual cada episodio fue seleccionado.

Preguntas que debe responder:

- Los episodios mejor rankeados por reward son tambien los mas deseables fisicamente?
- Donde aparecen las discrepancias entre reward, performance y politica final?

### Bloque 7. Dinamica temporal de estados y acciones

Objetivo:

- inspeccionar las trayectorias fisicas y de control dentro del episodio.

Senales minimas por episodio seleccionado:

- `pendulum_angle`
- `pendulum_velocity`
- `cart_position`
- `cart_velocity`
- `u_total`
- `u_eff_pendulum_angle`
- `u_eff_cart_position`
- `control_action_*`
- reward por intervalo cuando exista

Figuras minimas:

- overlays temporales por subconjunto de episodios,
- perfiles promedio con bandas percentil,
- paneles por episodio destacado.

Preguntas que debe responder:

- El carro se coloca debajo del pendulo a tiempo?
- El pendulo baja pero el carro se va?
- Hay cancelacion entre lazos?
- La maniobra fisica deseada aparece realmente o solo mejora el reward?

### Bloque 8. Analisis por fases de la maniobra

Objetivo:

- separar conceptualmente la dinamica en fases fisicamente relevantes.

Fases minimas que el sistema deberia poder distinguir, aunque sea mediante heuristicas configurables:

- fase de captura,
- fase de transicion,
- fase de recentrado,
- fase de preestabilizacion,
- fase de estabilizacion.

La salida no necesita imponer una etiqueta rigida perfecta, pero si debe ofrecer indicadores que permitan inferir en que fase falla la corrida.

Indicadores minimos:

- primer tiempo en que `|theta|` cruza umbrales de captura,
- primer tiempo en que el carro cruza umbrales de excursion,
- velocidad del carro cuando el angulo entra a banda,
- distancia del carro al origen cuando el pendulo entra a banda,
- duracion dentro de bandas.

Preguntas que debe responder:

- El problema actual es no capturar, capturar tarde, capturar pero no frenar, o capturar y no estabilizar?
- Los cambios de reward desplazan el cuello de botella de una fase a otra?

### Bloque 9. Mapas de calor y distribuciones tiempo-estado

Objetivo:

- ofrecer una lectura poblacional compacta del comportamiento temporal del sistema.

Mapas de calor minimos:

- tiempo vs `pendulum_angle`
- tiempo vs `cart_position`
- tiempo vs `cart_velocity`
- tiempo vs `u_total`

Mapas recomendados adicionales:

- tiempo vs `u_eff_pendulum_angle`
- tiempo vs `u_eff_cart_position`

Preguntas que debe responder:

- La poblacion esta realmente bajando el angulo?
- En que ventana temporal aparecen las mayores diferencias entre corridas?
- La captura existe a nivel poblacional o solo en unos pocos episodios?

### Bloque 10. Descomposicion completa de la recompensa observada

Objetivo:

- entender que componentes del reward estan dominando realmente el ranking de episodios.

Componentes minimos a reportar:

- `principal_reward`
- componentes por feature principal (`L_e`, `L_edot`, `L_I`, `L_u`)
- bonus de banda
- incentivos dinamicos
- cualquier bonus coordinativo futuro

Vista minima requerida:

- global,
- `last_n`,
- `top_reward`,
- `top_performance`,
- y episodios representativos.

Figuras minimas:

- barras apiladas de componentes,
- distribuciones de cada componente,
- comparacion entre episodios seleccionados.

Preguntas que debe responder:

- El reward total esta alineado con la conducta fisica deseada?
- Hay componentes que premian episodios cortos pero no utiles?
- Existen componentes activos que en la practica nunca se encienden?

### Bloque 11. Analisis teorico de la funcion de recompensa

Objetivo:

- visualizar el comportamiento teorico de la recompensa definida por configuracion, sin mezclarlo aun con datos empiricos de simulacion.

Contenido minimo:

- perfiles unidimensionales de cada bloque de reward relevante,
- cortes bidimensionales cuando haya dos variables activas en un mismo termino,
- tablas de soporte para niveles y saturaciones.

Preguntas que debe responder:

- Donde se activa realmente cada bonus o gate?
- La banda util es demasiado estrecha o demasiado amplia?
- El incentivo se satura demasiado pronto?

### Bloque 12. Analisis empirico de la recompensa sobre datos reales

Objetivo:

- cruzar los datos reales de la corrida con la formula teorica configurada.

Contenido minimo:

- histogramas de las senales que alimentan cada componente,
- porcentaje de muestras dentro de bandas,
- niveles de activacion efectiva por componente,
- comparacion entre teoria y datos observados.

Preguntas que debe responder:

- El reward fue disenado para una region del espacio de estados que casi nunca se visita?
- Los gates se quedan siempre cerrados o siempre abiertos?
- Hay shaping definido pero irrelevante en la practica?

### Bloque 13. Perfiles de reward sobre grillas de parametros

Objetivo:

- permitir disenar y afinar la recompensa visualizando como cambia el reward cuando se barre una o mas variables relevantes.

Perfiles requeridos:

- curvas de reward vs error,
- curvas de reward vs esfuerzo efectivo,
- mapas `error` vs `u_eff`,
- mapas `error` vs derivada del error cuando aplique,
- mapas `error` vs `control_action` si el termino lo usa.

La idea no es solo ver la formula. La idea es que el analista pueda comparar:

- region esperada de activacion,
- region observada en la simulacion,
- y region deseada desde el punto de vista fisico.

Preguntas que debe responder:

- El reward favorece justamente la transicion fisica que se desea?
- Existe una zona ciega donde deberia haber incentivo pero no lo hay?
- Se esta premiando una conducta demasiado agresiva o demasiado conservadora?

### Bloque 14. Diagnostico de coordinacion y cancelacion entre lazos

Objetivo:

- detectar si los controladores colaboran o se cancelan entre si durante las fases criticas.

Indicadores minimos:

- correlacion entre `u_eff_pendulum_angle` y `u_eff_cart_position`,
- comparacion entre magnitudes individuales y `u_total`,
- ventanas temporales donde la suma efectiva es pequena pese a esfuerzos grandes por lazo,
- relacion entre cancelacion y fracaso de captura o retorno.

Preguntas que debe responder:

- Los lazos se estan anulando en la fase inicial?
- El lazo de soporte empieza a pelear demasiado pronto contra el lazo principal?
- Hay evidencia de que la recompensa actual induzca conflicto entre controladores?

### Bloque 15. Comparacion entre corridas

Objetivo:

- convertir el analisis en una herramienta acumulativa y no aislada.

Comparaciones minimas:

- corrida actual vs corrida inmediatamente anterior,
- corrida actual vs mejor corrida historica segun criterio seleccionado,
- corrida actual vs una corrida de referencia manualmente indicada.

Dimensiones minimas de comparacion:

- resultados globales,
- captura,
- recentrado,
- estabilizacion,
- distribucion de reward,
- comportamiento de componentes del reward,
- dinamica de ganancias PID.

Preguntas que debe responder:

- Que cambio mejoro y que cambio empeoro?
- La nueva corrida movio el cuello de botella o solo lo reemplazo por otro?

### Bloque 16. Reporte final de decision de hiperparametros

Objetivo:

- sintetizar toda la evidencia anterior en una recomendacion de ajuste concreta y justificada.

Contenido minimo:

- diagnostico principal de la corrida,
- fase dominante del problema,
- componentes del reward que mas influyeron,
- evidencia a favor y en contra de mantener la configuracion actual,
- propuesta de ajustes priorizados,
- justificacion parametro por parametro para los hiperparametros criticos,
- riesgos de cada ajuste.

La salida de este bloque no debe ser una opinion breve. Debe ser una conclusion argumentada desde tablas y figuras previamente generadas.

Ademas, cuando la evidencia lo permita, el reporte final deberia clasificar cada hiperparametro relevante en una de estas categorias:

- mantener,
- subir,
- bajar,
- estrechar banda,
- ensanchar banda,
- revisar por saturacion,
- revisar por irrelevancia observada.

## 8. Reglas de decision que el sistema de analisis debe ayudar a aplicar

El sistema futuro debe facilitar, como minimo, las siguientes lecturas diagnosticas.

### Caso A. `top_performance` bueno pero `top_reward` malo

Interpretacion esperada:

- la politica encontro trayectorias fisicamente utiles,
- pero la recompensa no las esta valorizando correctamente.

Acciones tipicas que el analisis deberia sugerir:

- revisar descomposicion de reward,
- revisar episodios cortos con reward alto,
- revisar bonus o penalizaciones que dominen el ranking.

### Caso B. `top_reward` bueno pero `last_n` malo

Interpretacion esperada:

- hubo hallazgos utiles durante la exploracion,
- pero la politica final no los consolidó.

Acciones tipicas:

- revisar estabilidad del aprendizaje,
- revisar varianza de ganancias PID,
- revisar si la recompensa genera optimos locales pobres pero estables.

### Caso C. captura buena pero carro se fuga

Interpretacion esperada:

- la fase inicial ya esta desbloqueada,
- el cuello ahora esta en frenado y retorno.

Acciones tipicas:

- revisar `L_e_cart_position`,
- revisar `L_edot_cart_position`,
- revisar banda e incentivos de preestabilizacion,
- revisar dinamica de cancelacion tardia entre lazos.

### Caso D. reward favorece episodios conservadores que casi no mueven el carro

Interpretacion esperada:

- la recompensa esta castigando demasiado pronto la excursion del lazo de soporte,
- o el shaping de captura es demasiado debil.

Acciones tipicas:

- revisar bloques de captura,
- revisar gating o saturacion de componentes del carro,
- revisar mapas de reward vs error en region de captura.

### Caso E. todos los componentes estan activos pero ninguno discrimina

Interpretacion esperada:

- existe saturacion excesiva,
- o el espacio de activacion fue mal calibrado.

Acciones tipicas:

- revisar perfiles teoricos,
- revisar histogramas empiricos,
- mover umbrales, escalas o pesos a regiones mas informativas.

## 9. Requerimientos de almacenamiento de artefactos

Cada bloque debe guardar sus salidas de forma estructurada y consistente.

### 9.1. Figuras

Toda figura generada debe guardarse automaticamente. No deberia depender de que el usuario recuerde exportarla manualmente.

Convencion recomendada:

- `analysis/<sim_id>/<bloque>/<orden>_<nombre_figura>.png`

Ejemplos de nombres:

- `analysis/<sim_id>/learning/01_total_reward_evolution.png`
- `analysis/<sim_id>/episodes/03_top_reward_theta_profiles.png`
- `analysis/<sim_id>/reward_behavior/07_heatmap_error_vs_u_eff_capture.png`

### 9.2. Tablas

Toda tabla relevante deberia guardarse en formato legible y reutilizable.

Convencion recomendada:

- `csv` para reutilizacion tecnica,
- `xlsx` para consolidado,
- `md` para lectura rapida cuando aporte valor.

En particular, la radiografia de hiperparametros deberia dejar como minimo:

- `analysis/<sim_id>/hyperparameters/01_hyperparameter_inventory.csv`
- `analysis/<sim_id>/hyperparameters/02_hyperparameter_radiography.xlsx`
- `analysis/<sim_id>/hyperparameters/03_hyperparameter_justification.md`

### 9.3. Reportes de texto

Cada corrida deberia dejar al menos:

- un resumen tecnico global,
- y un reporte de decision de hiperparametros.

## 10. Integracion esperada con las herramientas externas de analisis ya existentes

La construccion futura del notebook o codigo de analisis deberia apoyarse directamente en los elementos externos que ya existen en el repositorio para post-procesar corridas terminadas.

### 10.1. `analysis/run_analysis.py`

Debe evolucionar hacia un runner que:

- ejecute bloques de analisis de resumen, aprendizaje, episodios y comparacion,
- exporte artefactos en la carpeta de analisis correspondiente,
- y consolide resultados en archivos tabulares estables.

Su naturaleza debe seguir siendo externa: recibe un `sim_id`, carga resultados ya exportados y genera nuevos artefactos de analisis sin intervenir en la corrida original.

### 10.2. `analysis/run_reward_behavior_notebook.py`

Debe encargarse de:

- materializar el bloque de analisis teorico y empirico de la recompensa,
- guardar tablas y figuras en `analysis/<sim_id>/reward_behavior/`,
- y operar como interfaz reproducible del notebook de diseno del reward.

Este runner tambien debe mantenerse desacoplado del runtime principal. Su entrada son resultados ya almacenados y su salida son artefactos de interpretacion.

### 10.3. `0_design_functionReward.ipynb`

Debe quedar reconvertido en un notebook estructurado por bloques claros:

- definicion contextual de la reward,
- perfiles teoricos,
- mapas de calor y superficies,
- comparacion teoria vs datos observados,
- y conclusiones de tuning.

No deberia permanecer como un notebook solo exploratorio o dependiente de ediciones manuales caso a caso.

### 10.4. Archivos consumidos desde la simulacion

La relacion con el flujo de simulacion debe ser unicamente de lectura posterior. El sistema externo de analisis debe consumir artefactos como:

- `metadata.json`
- `summary.xlsx`
- `episodes_chunks_*.json`
- figuras ya persistidas dentro de la carpeta de resultados, cuando aporte valor reutilizarlas

No deberia requerir hooks en tiempo de ejecucion, callbacks internos ni integracion al ciclo de entrenamiento.

### 10.5. Configuraciones de salida y resumen

Los archivos de configuracion del sistema deben seguir permitiendo que el analisis consuma:

- columnas por intervalo,
- columnas por episodio,
- metadata completa del experimento,
- y cualquier nuevo componente del reward que se agregue.

## 11. Requerimientos especificos para el futuro notebook

El notebook que se construya a partir de esta especificacion deberia cumplir, como minimo, con lo siguiente:

- una seccion inicial de parametros de ejecucion,
- una seccion de carga y validacion de insumos,
- una seccion por bloque de analisis,
- exportacion automatica de figuras y tablas,
- y una seccion final de conclusiones.

Ademas, cada seccion deberia poder ejecutarse de forma razonablemente independiente para no obligar a reejecutar todo el notebook cuando solo se necesite revisar un bloque concreto.

## 12. Limites y no-responsabilidades del sistema externo de analisis

Para evitar ambiguedades de arquitectura, este sistema no deberia:

- ejecutar simulaciones,
- modificar parametros de entrenamiento en tiempo real,
- participar del loop de control o del loop RL,
- depender de estados internos no exportados por la corrida,
- ni escribir de vuelta sobre la carpeta de resultados original, salvo lectura y generacion de artefactos dentro de `analysis/<sim_id>/`.

Su funcion es exclusivamente analitica y posterior a la ejecucion.

## 13. Criterios de aceptacion del sistema de analisis futuro

El notebook o sistema futuro deberia considerarse correctamente construido solo si cumple simultaneamente estos criterios:

- Reanaliza una corrida completa sin intervenciones manuales ad hoc.
- Guarda todas las figuras y tablas relevantes dentro de `analysis/<sim_id>/`.
- Construye una radiografia completa y auditable de los hiperparametros activos.
- Permite contrastar `top_reward`, `top_performance`, `last_n` y episodios seleccionados.
- Permite auditar la recompensa tanto desde la teoria como desde los datos reales.
- Permite comparar corridas distintas con criterio uniforme.
- Produce un reporte final que apoye explicitamente la seleccion de hiperparametros.

## 14. Orden recomendado de construccion

Para construir esta herramienta de forma robusta, el orden recomendado es:

1. bloque de manifiesto y snapshot de configuracion,
2. bloque de resumen global y evolucion del aprendizaje,
3. bloque de episodios representativos y dinamica temporal,
4. bloque de recompensa teorica y empirica,
5. bloque de comparacion entre corridas,
6. bloque final de reporte de decision.

Este orden permite que cada etapa entregue valor por si misma y, a la vez, construya una base estable para la siguiente.

## 15. Conclusion

El sistema de analisis que requiere este proyecto debe ser tratado como una pieza central del flujo experimental, no como un accesorio posterior.

Si se construye siguiendo esta especificacion, el notebook futuro no solo servira para mirar resultados, sino para explicar de forma consistente:

- que aprendio realmente la politica,
- como se comporta fisicamente el sistema,
- como esta actuando la funcion de recompensa,
- y que parametros conviene modificar en la siguiente iteracion experimental.
