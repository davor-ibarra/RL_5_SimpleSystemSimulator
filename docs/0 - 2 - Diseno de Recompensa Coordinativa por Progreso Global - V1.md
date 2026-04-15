# Diseno de Recompensa Coordinativa por Progreso Global - V1

## 1. Proposito del documento

Este documento define, en formato contextual, matematico e instruccional, una extension del diseno de recompensa actual del sistema para resolver una limitacion estructural observada durante el analisis del CartPole autosintonizado por RL.

La idea central es mantener la recompensa principal local por lazo, pero agregar un mecanismo adicional de coordinacion basado en progreso global de la tarea. Con esto se busca evitar que la arquitectura de recompensa quede amarrada a reglas manuales especificas del sistema actual, por ejemplo del tipo "el reward de `cart_position` depende explicitamente de `pendulum_angle`".

El objetivo no es proponer codigo en este documento. El objetivo es dejar claramente establecidos:

- el problema que se quiere resolver,
- la formulacion matematica recomendada,
- los parametros nuevos que tiene sentido incorporar,
- los criterios de diseno que deben preservarse,
- y los puntos exactos de la arquitectura donde deberia tocarse el codigo cuando se implemente.

## 2. Contexto del problema

El sistema actual ya dispone de una recompensa principal construida desde metricas por lazo, principalmente:

- `L_e`
- `L_edot`
- `L_I`
- `L_u`

Cada una de estas metricas se calcula por variable objetivo y luego alimenta la recompensa de cada controlador PID mediante el esquema vigente.

Este diseno tiene una gran ventaja: es limpio, interpretable y consistente con el principio de que cada agente deberia recibir credito principalmente por el comportamiento de su lazo.

Sin embargo, durante el analisis de resultados del CartPole aparecio una limitacion importante: existen etapas de la maniobra en las que un controlador de soporte debe empeorar temporalmente su objetivo local para ayudar al objetivo global del sistema.

En el caso del CartPole, esto ocurre cuando:

- el carro debe desplazarse lejos del origen para ponerse debajo del pendulo,
- aunque ese desplazamiento empeore temporalmente el error local de `cart_position`,
- porque dicho movimiento favorece la captura del pendulo y, por tanto, el progreso fisico real de la tarea.

Si la recompensa se mantiene estrictamente local por lazo, el sistema de aprendizaje queda sin una forma general de reconocer ese aporte cooperativo. Como consecuencia, aparecen dos salidas indeseadas:

- o bien se penaliza demasiado pronto el lazo de soporte y se inhibe la maniobra necesaria,
- o bien se introducen reglas cruzadas manuales entre lazos, lo que resuelve un caso puntual pero degrada la generalidad del diseno.

## 3. Problema de arquitectura que se quiere evitar

No es recomendable que el diseno de recompensa dependa de reglas especificas del sistema del tipo:

- el reward del lazo A depende directamente del error del lazo B,
- el lazo del carro se gatea manualmente usando el error del pendulo,
- o se agregan heuristicas particulares que solo tienen sentido para una planta concreta.

Ese tipo de soluciones puede ser util como parche, pero tiene tres problemas fuertes:

- hace que la recompensa deje de ser reusable en otros sistemas fisicos,
- obliga a redisenar la logica cada vez que se agregan mas controladores,
- y mezcla conocimiento de control especifico de una planta con la politica general de asignacion de credito.

La solucion recomendada no es "acoplar manualmente controladores". La solucion recomendada es introducir una capa de evaluacion de progreso global de tarea y luego repartir ese progreso entre los lazos segun su contribucion efectiva.

## 4. Objetivo funcional del nuevo bloque de recompensa

El nuevo bloque debe permitir que un lazo reciba credito adicional cuando:

- el sistema completo avanza hacia su objetivo global,
- y dicho lazo contribuyo de forma efectiva y correctiva a ese avance,
- incluso si su error local empeoro temporalmente durante una fase necesaria de la maniobra.

Al mismo tiempo, el nuevo bloque no debe reemplazar la recompensa principal actual. Debe complementarla.

Por tanto, la arquitectura objetivo es:

- recompensa principal local por lazo,
- extras locales ya existentes, cuando corresponda,
- y un nuevo bonus coordinativo basado en progreso global de la tarea.

## 5. Principios de diseno que deben preservarse

El bloque nuevo debe respetar simultaneamente estos principios:

- Debe ser genericamente reusable para sistemas con dos o mas controladores.
- Debe evitar dependencias manuales controlador-a-controlador.
- Debe usar senales ya coherentes con la arquitectura actual, especialmente metricas por lazo y esfuerzos efectivos.
- Debe entregar una senal continua, acotada e interpretable.
- Debe permitir trazabilidad completa en los archivos de analisis.
- Debe ser escalable a nuevas plantas sin obligar a redisenar la formula base.

## 6. Diseno matematico propuesto

### 6.1. Notacion

Se trabajara con dos niveles temporales:

- nivel step, indexado por `t`,
- nivel intervalo RL, indexado por `k`.

Se considerara un conjunto generico de variables objetivo:

\[
\mathcal{V} = \{v_1, v_2, \dots, v_m\}
\]

Para cada variable objetivo `v`, se asume que el sistema ya dispone, directa o indirectamente, de:

- error local \(e_v(t)\),
- esfuerzo efectivo del lazo \(u^{eff}_v(t)\),
- costo de error por intervalo \(L_{e,v}(k)\),
- costo de derivada del error por intervalo \(L_{\dot e,v}(k)\),
- y, si se requiere, otras metricas locales compatibles con la arquitectura actual.

### 6.2. Potencial global de tarea

Se define un potencial global de tarea por intervalo:

\[
\Phi(k) =
\sum_{v \in \mathcal{V}} \alpha_v L_{e,v}(k)
+
\sum_{v \in \mathcal{V}} \beta_v L_{\dot e,v}(k)
\]

donde:

- \(\alpha_v \ge 0\) pondera la relevancia del costo de error de cada lazo,
- \(\beta_v \ge 0\) pondera la relevancia del costo asociado a la dinamica del error.

Interpretacion:

- un valor alto de \(\Phi(k)\) indica peor estado global de la tarea,
- un valor bajo de \(\Phi(k)\) indica mejor estado global de la tarea.

La primera version recomendada debe ser deliberadamente simple. Por ello, no se recomienda partir usando `L_I` ni `L_u` dentro del potencial global, salvo que una necesidad fisica muy clara lo justifique despues.

### 6.3. Progreso global por intervalo

El progreso de tarea entre dos intervalos consecutivos se define como:

\[
\Delta \Phi(k) = \Phi(k-1) - \Phi(k)
\]

De esta forma:

- si \(\Delta \Phi(k) > 0\), el sistema mejoro globalmente,
- si \(\Delta \Phi(k) < 0\), el sistema empeoro globalmente.

Para la primera implementacion, se recomienda usar la parte positiva del progreso:

\[
\Delta \Phi^{+}(k) = \max(0, \Delta \Phi(k))
\]

La razon es practica: asi el bonus coordinativo premia solo avance real y evita introducir una capa adicional de castigo que podria interferir con la recompensa principal existente.

### 6.4. Contribucion efectiva local por lazo

Para cada variable objetivo `v`, se define una magnitud de contribucion correctiva instantanea:

\[
c_v(t) = \max(0, s_v \, e_v(t) \, u^{eff}_v(t))
\]

donde:

- \(s_v \in \{-1, +1\}\) representa el signo correctivo esperado del lazo `v`.

Interpretacion:

- si el lazo empuja en direccion no correctiva, la contribucion es nula,
- si el lazo empuja en direccion correctiva, la contribucion es positiva,
- si ademas el error y el esfuerzo efectivo son mayores, el credito bruto aumenta.

Luego se define la contribucion media por intervalo:

\[
\bar c_v(k) = \frac{1}{N_k}\sum_{t=1}^{N_k} c_v(t)
\]

donde \(N_k\) es la cantidad de steps ejecutados dentro del intervalo `k`.

### 6.5. Reparto de credito coordinativo

El progreso global del sistema no debe repartirse de forma uniforme. Debe asignarse segun contribucion relativa. Para ello se define:

\[
\rho_v(k) =
\frac{\bar c_v(k)}
{\varepsilon + \sum_{j \in \mathcal{V}} \bar c_j(k)}
\]

donde:

- \(\varepsilon > 0\) evita division por cero y estabiliza el cociente.

Interpretacion:

- \(\rho_v(k)\) es la fraccion del credito coordinativo atribuible al lazo `v`,
- si un lazo no contribuyo de forma correctiva, su fraccion se aproxima a cero,
- si un lazo fue dominante en la correccion, recibe una fraccion mayor del progreso.

### 6.6. Bonus coordinativo por lazo

La recompensa coordinativa recomendada para cada lazo es:

\[
R^{coord}_v(k) = \lambda \, \rho_v(k)\, \Delta \Phi^{+}(k)
\]

donde:

- \(\lambda \ge 0\) es la ganancia global del bonus coordinativo.

Esta expresion es la opcion base recomendada porque es:

- continua,
- interpretable,
- modular,
- y reusable.

### 6.7. Recompensa final por controlador

La recompensa final de cada lazo, a nivel conceptual, debe quedar compuesta por:

\[
R^{final}_v(k) =
R^{principal}_v(k)
+
R^{extras\_locales}_v(k)
+
R^{coord}_v(k)
\]

El nuevo termino no sustituye la logica actual. La complementa.

## 7. Parametros que tiene sentido incorporar

La primera version deberia introducir un bloque de configuracion nuevo, suficientemente explicito y consistente con el estilo del proyecto. Los parametros recomendados son los siguientes.

### 7.1. Parametros de activacion general

| Parametro | Rol esperado | Criterio de uso |
| --- | --- | --- |
| `enabled` | Activa o desactiva el bloque coordinativo | Debe permitir aislar facilmente el experimento |
| `reward_mode` | Nombre canonico del modo | Debe identificar sin ambiguedad que se trata de progreso global con asignacion de credito |
| `assign_mode` | Estrategia de asignacion del bonus | La primera version debe ser por lazo |
| `global_weight` | Ganancia \(\lambda\) del bonus coordinativo | Debe calibrar cuanto pesa este bloque respecto al reward principal |

### 7.2. Parametros del potencial global

| Parametro | Rol esperado | Criterio de uso |
| --- | --- | --- |
| `potential_features` | Lista de features usadas en \(\Phi(k)\) | La primera version deberia usar `L_e` y `L_edot` |
| `potential_weights` | Pesos \(\alpha_v\) y \(\beta_v\) por feature y por variable | Permiten expresar importancia relativa de cada lazo dentro del objetivo global |
| `progress_positive_only` | Indica si solo se usa \(\Delta \Phi^{+}(k)\) | Recomendado `true` en la primera etapa |
| `progress_clip_max` | Limite superior opcional del progreso premiable | Util si se desea evitar picos excesivos |

### 7.3. Parametros de asignacion de credito

| Parametro | Rol esperado | Criterio de uso |
| --- | --- | --- |
| `contribution_error_signal` | Senal de error usada en \(c_v(t)\) | Debe apuntar a `error_<var_obj>` |
| `contribution_effort_signal` | Senal de esfuerzo usada en \(c_v(t)\) | Debe apuntar a `u_eff_<var_obj>` |
| `correction_signs` | Signo correctivo \(s_v\) por lazo | Debe declararse por variable objetivo |
| `epsilon_credit` | \(\varepsilon\) del reparto de credito | Debe ser pequeno pero no cero |
| `credit_reduction_mode` | Forma de reducir de step a intervalo | La primera version deberia usar promedio temporal |

### 7.4. Parametros de trazabilidad y exportacion

| Parametro | Rol esperado | Criterio de uso |
| --- | --- | --- |
| `export_task_potential` | Persistencia de \(\Phi(k)\) | Recomendado `true` |
| `export_task_progress` | Persistencia de \(\Delta \Phi(k)\) y \(\Delta \Phi^{+}(k)\) | Recomendado `true` |
| `export_credit_raw` | Persistencia de \(\bar c_v(k)\) | Recomendado `true` |
| `export_credit_share` | Persistencia de \(\rho_v(k)\) | Recomendado `true` |
| `export_reward_component` | Persistencia de \(R^{coord}_v(k)\) | Recomendado `true` |

## 8. Restricciones conceptuales que deben preservarse

Durante la implementacion no deberia romperse ninguno de estos criterios:

- No usar reglas manuales del tipo "el reward de un lazo depende directamente del error de otro lazo".
- No usar `u_total` ni `u_total_raw` para medir contribucion local, porque eso mezcla esfuerzos y degrada el credito causal del lazo.
- No introducir tolerancia silenciosa en la logica principal que oculte errores de configuracion.
- No reemplazar la recompensa principal actual; el nuevo bloque debe ser aditivo y modular.
- No perder trazabilidad por intervalo, porque sin ella no se podra validar si el credito coordinativo esta funcionando como se espera.

## 9. Puntos de la arquitectura donde deberia tocarse el codigo

Esta seccion no propone codigo. Define, de forma instruccional, donde deberia integrarse la futura implementacion para que quede consistente con la arquitectura ya existente.

### 9.1. `config/config_CartPole.yaml`

Aqui deberia declararse el bloque de configuracion nuevo.

Responsabilidad esperada:

- definir el bloque coordinativo,
- habilitar o deshabilitar el feature,
- declarar pesos del potencial global,
- declarar senales de contribucion y signos correctivos por lazo,
- y dejar el experimento completamente trazable desde metadata.

La expectativa es que este bloque siga el estilo declarativo actual del proyecto, evitando configuraciones ambiguas o implícitas.

### 9.2. `rewards/reward_calculator_base.py`

Este es el punto natural para coordinar la nueva logica por dos razones:

- ya participa en la composicion final del reward,
- y es el lugar mas adecuado para mantener memoria de estado entre intervalos, en particular \(\Phi(k-1)\).

Responsabilidad esperada:

- recibir las metricas del intervalo actual,
- mantener el potencial global del intervalo anterior,
- calcular el progreso global,
- invocar el bloque coordinativo,
- sumar el bonus coordinativo al reward correspondiente de cada lazo,
- y resetear correctamente el estado al comienzo de cada episodio.

### 9.3. `rewards/extra_rewards_handler.py`

Este archivo podria reutilizarse o bien inspirar una nueva pieza dedicada, pero la responsabilidad del nuevo bloque deberia quedar claramente separada.

La recomendacion arquitectonica es:

- mantener los extras locales actuales en su espacio natural,
- y tratar el bonus coordinativo como un bloque independiente, porque depende de informacion global del intervalo y de memoria temporal.

Si se decide extender el handler actual, debe hacerse de forma muy explicita para no mezclar extras locales con credito global de tarea.

### 9.4. `metrics/metric_processing.py`

Este modulo ya produce gran parte de las metricas necesarias.

Responsabilidad esperada:

- asegurar disponibilidad consistente de `L_e_<var_obj>` y `L_edot_<var_obj>`,
- mantener la correspondencia actual de `L_u` con `u_eff_<var_obj>`,
- y, solo si hiciera falta, exponer de forma mas directa alguna metrica agregada requerida por el potencial global.

La primera implementacion deberia intentar no cargar nueva complejidad aqui si las metricas actuales ya son suficientes.

### 9.5. `metrics/metric_collector.py`

Responsabilidad esperada:

- persistir las nuevas columnas por intervalo,
- integrarlas al flujo actual de recoleccion sin romper la forma plana de las salidas.

Este punto es crucial para que luego el analisis pueda inspeccionar el nuevo bloque sin rehacer logs.

### 9.6. `config/sub_config_template_output_CartPole.yaml`

Aqui deberian declararse las nuevas claves planas que deben quedar almacenadas por intervalo.

Responsabilidad esperada:

- garantizar trazabilidad completa del nuevo bonus coordinativo,
- permitir exportacion consistente de columnas en los chunks y archivos de salida.

### 9.7. `config/sub_config_data_summary_CartPole.yaml`

Aqui deberian agregarse los nuevos indicadores relevantes que se quieran consolidar en `summary.xlsx`.

Responsabilidad esperada:

- permitir ver rapidamente si el bonus coordinativo estuvo activo,
- y evaluar si efectivamente se asocio con progreso global util o con comportamientos espurios.

### 9.8. `utils/result_handler.py`

Responsabilidad esperada:

- asegurar que metadata y artefactos incluyan el nuevo bloque configurado,
- y que las salidas finales mantengan la misma consistencia documental que el resto del sistema.

### 9.9. `analysis/run_analysis.py`

Responsabilidad esperada:

- incorporar tablas y graficos especificos del nuevo bonus coordinativo,
- comparar su comportamiento en `last_n`, `top_reward`, `top_performance` y episodios representativos.

### 9.10. `analysis/run_reward_behavior_notebook.py` y `0_design_functionReward.ipynb`

Responsabilidad esperada:

- incluir visualizaciones teoricas y empiricas del bloque coordinativo,
- revisar el perfil de la recompensa en funcion del progreso global y de la contribucion relativa por lazo,
- y guardar todos los artefactos resultantes dentro de la carpeta de analisis correspondiente a cada simulacion.

## 10. Requerimientos de estado y reinicio

Como el nuevo bloque usa progreso entre intervalos, requiere memoria.

Por tanto, la implementacion futura debera respetar estas reglas:

- al inicio de cada episodio no existe \(\Phi(k-1)\) valido,
- el primer intervalo del episodio no debe generar bonus coordinativo basado en diferencia previa,
- el estado debe resetearse siempre al terminar o reiniciar episodio,
- y no debe arrastrarse potencial global entre episodios distintos.

Esto es obligatorio. Si no se hace asi, el bonus coordinativo quedara contaminado por transiciones que no tienen significado fisico.

## 11. Requerimientos de trazabilidad

Para que la futura implementacion sea auditable, se recomienda registrar, como minimo, las siguientes claves planas por intervalo:

- `coordination_task_potential`
- `coordination_task_progress_delta`
- `coordination_task_progress_positive`
- `coordination_credit_raw_<var_obj>`
- `coordination_credit_share_<var_obj>`
- `extra_conditional_coordination_bonus_<var_obj>`

Si el sistema almacena tambien agregados de episodio o resumen, conviene incluir:

- promedio,
- maximo,
- percentiles relevantes,
- y conteos de activacion no nula.

## 12. Preguntas de validacion que la implementacion debe poder responder

Una vez implementado el bloque, el sistema de analisis deberia poder responder de forma directa preguntas como:

- Cuando mejora el objetivo global, que lazos reciben el credito?
- El credito se concentra en el lazo correcto durante cada fase de la maniobra?
- Existen episodios donde hay mejora global pero el credito cae sobre el lazo equivocado?
- El bonus coordinativo ayuda a preferir episodios fisicamente deseables frente a episodios solo "limpios" pero poco efectivos?
- El nuevo termino desplaza en exceso a la recompensa principal o actua solo como complemento?

## 13. Criterios de aceptacion de la futura implementacion

La implementacion futura deberia considerarse correcta solo si cumple simultaneamente lo siguiente:

- La recompensa principal actual sigue funcionando igual cuando el bloque coordinativo esta desactivado.
- El bloque nuevo no requiere reglas manuales cruzadas especificas entre controladores.
- El nuevo bonus se calcula usando progreso global y contribucion local efectiva.
- El estado se reinicia correctamente por episodio.
- Todas las magnitudes nuevas quedan registradas y exportadas.
- El analisis posterior puede reconstruir y explicar el valor del bonus coordinativo en cualquier intervalo relevante.

## 14. Recomendacion de alcance para la primera version

La primera version deberia ser deliberadamente minima.

Eso implica:

- usar solo `L_e` y `L_edot` dentro del potencial global,
- usar solo `error_<var_obj>` y `u_eff_<var_obj>` para la contribucion local,
- usar solo parte positiva del progreso,
- y repartir credito por promedio temporal de contribucion.

La recomendacion es no agregar variantes mas complejas hasta que esta version simple quede trazable y validada con resultados de simulacion.

## 15. Conclusion

La incorporacion de un bonus coordinativo por progreso global resuelve una necesidad real del sistema: reconocer cuando un lazo ayuda al objetivo total aunque temporalmente empeore su objetivo local.

La idea no es reemplazar la filosofia actual del proyecto, sino completarla de una forma mas general, reusable e interpretable.

Si esta extension se construye siguiendo los principios aqui definidos, el sistema podra evolucionar hacia plantas con mas controladores sin depender de reglas ad hoc disenadas especificamente para cada configuracion fisica.
