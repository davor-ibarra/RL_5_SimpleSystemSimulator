---
created: 20260202 10:02
update: 20260213-01:14
summary:
status:
link:
tags:
---
# Principios Instruccionales Generales a Considerar

El sistema ideal se concibe como una **arquitectura por capas claramente delimitadas**, donde cada componente tiene una responsabilidad única, contratos de datos explícitos y puntos de integración formales. La prioridad no es solo que funcione, sino que sea trazable, extensible, auditable, con código legible y eficiente en memoria y cómputo.

En el nivel superior se encuentra el `main` que inyecta dependencia y gestiona el inicio y término de la ejecución, y la **orquestación de la simulación** `simulation_manager`, cuyo único rol es coordinar el flujo `episode → interval → step`. Este componente no calcula dinámicas, no procesa métricas y no decide recompensas; simplemente invoca a cada subsistema en el orden correcto, mantiene el reloj lógico del episodio y gestiona los eventos de inicio, cierre de intervalo y cierre de episodio. La orquestación define los puntos formales donde se solicitan acciones al agente, donde se evalúa terminación y donde se dispara la persistencia final. Toda dependencia base debe haber sido inyectada, las clases específicas son manejadas internamente por su correspondiente orquestador (componente base) permitiendo intercambiar sistema dinámico, agentes o estrategias de recompensa sin alterar la estructura del bucle.

El **sistema dinámico** constituye una unidad autocontenida que encapsula parámetros físicos, estado interno, integración numérica y condiciones de terminación. Debe exponer un contrato claro: `reset()`, `step(u)`, `get_state(raw|normalized)`, `is_terminated()`. No debe conocer nada sobre recompensas ni agentes. La normalización del estado debe ser responsabilidad explícita del sistema dinámico, evitando ambigüedad en las escalas. El estado entregado debe ser canónico, estable y documentado, separando estado físico crudo de estado normalizado para aprendizaje. Además, el `dynamic_system_base` debe contar con un método para exponer parámetros internos de orquestación si corresponden y del sistema específico que corresponda.

La **capa de control** debe abstraer la noción de controlador como entidad autónoma. El controlador PID debe exponer su contribución individual (`u_local`), su estado interno (error, integral, derivada) y permitir exponer sus parámetros. La saturación global se gestiona en el controlador orquestador base, donde se determina si la suma total de acciones de control supera los límites configurados. El anti-windup condicional se activa únicamente cuando se detecta saturación efectiva respecto a los límites configurados, y no por comparación entre señales internas ambiguas. El controlador no debe tomar decisiones de aprendizaje ni registrar métricas; solo producir acción y reportar su estado interno.

Los **agentes de aprendizaje** deben operar bajo un contrato minimalista: construir su estado observable a partir de información explícita del sistema (no acceder a estructuras internas no declaradas), seleccionar acción, actualizar política y exponer sus parámetros. El agente no debe calcular recompensas; recibe una recompensa asignada según una política de distribución claramente definida (global, individual o por controlador). La discretización, selección de acciones, política (exploración/explotación) y aprendizaje (actualización de tablas) deben estar encapsuladas, evitando lógica dispersa en otros módulos.

El **pipeline de métricas** debe centralizar completamente la recolección en un único punto formal. La estructura de salida del episodio debe dividirse estrictamente en `step_data`, `interval_data` y `end_episode_data`. Cada uno debe crecer de forma incremental mediante `append`, sin reconstrucciones ni recalculado de historiales. La selección de variables a persistir debe ser gobernada exclusivamente por configuración declarativa; el colector nunca decide qué eliminar por “no uso”. Tampoco transforma hacia estructuras serializables o aplanadas ya que estas deben venir listas desde la fuente de datos, no durante la recolección.

El **procesamiento de métricas** debe ser declarativo: recibir datos crudos/normalizados del sistema durante el intervalo y producir métricas derivadas necesarias para recompensa o análisis, sin alterar el histórico base. Las funciones de normalización, agregación o cálculo intermedio deben ser puras y reproducibles. La lógica de recompensa no debe recomputar lo que ya fue procesado; debe consumir métricas específicas procesadas.

La **arquitectura de recompensa** debe estructurarse como una orquestación de componentes independientes: recompensa principal y recompensas adicionales. Cada componente produce un valor y un registro explicativo. El orquestador consolida resultados en una estructura `reward_info` canónica, manteniendo trazabilidad completa del cálculo. La asignación de recompensa a agentes debe ser responsabilidad de una política explícita y configurable, evitando condicionales dispersos, donde se asigna una recompensa global a cada agente, una recompensa específica por agente, o una recompensa calculada por lazo y asignada a los agentes correspondientes de las ganancias de ese controlador. El sistema debe permitir agregar nuevas funciones de recompensa sin modificar el núcleo del bucle de simulación.

La **persistencia y post-procesamiento** deben separarse de la simulación activa. La simulación produce una estructura por niveles de datos y filtrada; un manejador de resultados transforma esa estructura solo cuando sea necesario (por ejemplo, para JSON serializable o DataFrame). Esto permite eficiencia en memoria durante la ejecución y flexibilidad en análisis posterior.

En términos transversales, el sistema ideal debe cumplir:
- **Contratos de datos estables**: nombres canónicos, sin duplicidades semánticas.
- **Configuración como fuente de verdad**: selección de variables, modos de recompensa, límites de saturación, flags de logging.
- **Testabilidad estructural**: cada componente debe poder probarse en aislamiento.
- **Observabilidad mínima pero suficiente**: registros claros de terminación, saturación, decisiones del agente y descomposición de recompensa.
- **Extensibilidad real**: incorporar otro sistema dinámico, otro controlador o un agente profundo sin alterar el bucle maestro.

Finalmente, el sistema ideal no es el que acumula más funcionalidades, sino el que mantiene coherencia interna. Cada componente debe poder describirse en una sola frase sin ambigüedad sobre su responsabilidad. Si una función requiere conocer demasiado del resto del sistema, existe acoplamiento indebido. El objetivo no es solo optimizar el rendimiento del controlador o acelerar el aprendizaje, sino construir una arquitectura que permita experimentar, publicar y escalar sin introducir fragilidad estructural.