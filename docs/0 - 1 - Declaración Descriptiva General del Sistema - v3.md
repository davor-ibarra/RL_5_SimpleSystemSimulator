---
created: 20260202 10:02
update: 20260211-14:00
summary:
status:
link:
tags:
---
# Principios Instruccionales Generales a Considerar

ROL
Eres un agente senior de auditoría de arquitectura y calidad de integración (simulación + control + RL). Tu trabajo es producir un REPORTE de auditoría de lo que se te indique, mediante una comparación profesional del TO-BE vs el estado AS-IS del código actual. Debes evaluar coherencia global, consistencia de contratos, y consistencia/estandarización de nombres de variables y métodos.

RESTRICCIONES
- NO ejecutes nada.
- NO propongas planes de pruebas ni planes de ejecución.
- NO te limites a seguir instrucciones literales: debes inferir quiebres, inconsistencias y mejoras necesarias para alinear AS-IS con TO-BE.
- Tu salida debe incluir hallazgos + propuestas (mínimo 2 alternativas por hallazgo relevante) + una recomendación final.

OBJETIVO
Emitir un reporte de auditoría que:
A) Determine si el AS-IS implementa e integra correctamente el flujo TO-BE.
B) Detecte quiebres de integración, responsabilidades mal ubicadas, contratos ambiguos o incompletos.
C) Evalúe consistencia y coherencia de naming (métodos, variables, llaves de dicts, prefijos/sufijos).
D) Proponga mejoras concretas y alternativas viables, recomendando un camino final.

ENFOQUE DE TRABAJO (GUÍA)
1) Leer el flujo y convertirlo en “contratos explícitos”:
   - Qué componentes participan
   - Qué entra y qué sale en cada transición
   - Qué estructuras de datos existen (dicts/objetos) y sus llaves esperadas
   - Qué responsabilidades quedan asignadas a cada componente
2) Inspeccionar el AS-IS en el código:
   - Identificar el flujo real, los métodos usados y el orden real
3) Comparar TO-BE vs AS-IS:
   - Coincidencias (correcto)
   - Divergencias (quiebre o deuda)
   - Ambigüedades (no está definido o está duplicado)
4) Auditoría de naming:
   - Variables de estado
   - Acciones
   - Reward/metrics/info dicts
   - Métodos de ciclo de vida (reset, step, interval, episode, finalize)
   - Nombres de llaves en dicts (generalizados, snake_case, prefijos, consistencia semántica)
5) Propuestas:
   - Para cada problema: 2–3 alternativas (parche / alineación / refactor)
   - Indicar trade-offs y riesgos
   - Recomendar un camino final coherente con TO-BE

FORMATOS Y CRITERIOS
- Sé preciso, pero no excesivamente granular.
- Todo hallazgo debe estar anclado a: archivo + clase + método (y, si corresponde, nombre de variables, llaves).
- Clasifica severidad: CRÍTICO / MAYOR / MENOR.
- Evalúa coherencia semántica: no solo “mismo nombre”, sino “mismo significado”.
	- Se debe evitar asignación de nombres redundantes si ya ha sido nombrado de forma generalizada

NOTAS
- Si el flujo TO-BE tiene vacíos o ambigüedades, debes señalarlos explícitamente y proponer
  2 alternativas de definición.
- Mantén una postura “arquitectura limpia”: responsabilidades claras, contratos estables, nombres semánticos.



# Flujo Ideal General

## 0) Entrada única y preparación de corrida
1. **`main.main()` carga 3 YAMLs y no interpreta nada**:
    - `config_main` (ejecución), `config_data_save` (qué guardar), `config_visualization` (post-run).  
2. `main._build_run_id()` y `main._build_output_dir()` determinan nombre según fecha+horario de ejecución y el nombre de la carpeta de salida.
3. `main._build_components()` instancia explícitamente todos los componentes (sin DI).  
    Secuencia recomendada (ya alineada al código):
    - Sistema dinámico: `DynamicSystemBase(CartPoleDynamicSystem, config)`
    - Control: `ControllerBase` + `PIDController` por `var_obj`
    - Agente: `AgentBase(PIDQLearningAgent, config)`
    - Métricas: `MetricProcessing`
    - Recompensa: `LagrangeRewardCalculator` (ver nota de contrato abajo)
    - Persistencia: `ResultHandler(output_dir)`
    - Registro: `MetricCollector(result_handler, config_data_save)`
    - Orquestación: `SimulationManager(...)`
4. `main._save_metadata()` delega a `ResultHandler.save_metadata()`.

## 1) Ejecución de corrida
1. `main._run_simulation()` llama una sola vez a `SimulationManager.run_simulation()`.  
2. `SimulationManager.run_simulation()` itera episodios y al final ejecuta `metric_collector.finalize_run()`.  
    Justificación: cerrar buffers y flush final _una sola vez_.

## 2) Episodio (macro-ciclo)
1. `SimulationManager._run_episode(episode_id)` comienza con `_reset_episode()`.  
    Orden recomendado (ya está):
    - `dynamic_system_base.reset_episode(...)`
    - `controller_base.reset_episode()` que a su vez por cada controlador
    - `agent.reset_episode()` (que a su vez por cada agente y decay epsilon y learning_rate se deberían actualizar tambien)
    - `metric_processing.reset_episode()`
    - `reward_calculator.reset_episode()`
2. **Construcción del estado inicial de decisión**:
    - `prev_dynamic_state_dict = self.dynamic_system_base.current_state_dict`
    - `self.prev_dynamic_system_state_norm_dict = self.dynamic_system_base.current_state_norm_dict`
    - `current_controller_gains = self._get_controller_gains_dict()`
    - `prev_agent_state` = `self.agent_base.build_agent_state` el agente necesita “mundo + ganancias” como estado, y eso se arma _en un solo lugar_.
3. **Acción inicial canónica**:
    - `prev_actions_dict = _build_initial_actions_dict()` (todas “mantener” = 1).
4. **Inicio de buffers del episodio**:
- `metric_collector.on_episode_start(episode_id)`  
    Justificación: el colector es el único encargado de recibir _todo_ lo que se acumula antes de persistir.

## 3) Intervalo de decisión (ciclo medio)
1. **Selección de acción**:
- `actions_dict = agent.select_action(prev_agent_state)`
2. **Aplicación a controladores**:
- `_apply_actions_to_controllers(actions_dict)` agrupa por `var_obj` y llama `PIDController.update_gains(kp,ki,kd)` la actualización de ganancias ocurre _solo_ al inicio del intervalo; dentro del intervalo no se toca.
3. **Ejecución del intervalo**:
- `_run_interval(decision_id, current_time_sec, actions_dict, prev_agent_state)` retorna `(interval_data)`.
    Justificación: encapsular todo lo step-level e interval-level en un solo método mantiene legibilidad.


## 4) Step-loop interno del intervalo (micro-ciclo)
1. **Control total + record**:
- `u_total, controller_state_record = controller_base.compute_control(prev_dynamic_system_state_norm_dict)`  
    Dentro de `compute_control`:
- Cada `PIDController.compute(...)` produce acción normalizada ya que el sistema ha normalizado previamente sus variables físicas.
- Se suma `u_total_raw`, se satura globalmente y también se calcula `delta_u_total`
- Si hay saturación, se reparte anti-windup `apply_antiwindup_correction()`. Se aplica anti-windup conditional, es decir, se corrige integral solo si hay saturación.
    Justificación: control "global" (suma/saturación/anti-windup) y (PID interno) quedan claramente separados.
2. **Integración dinámica**:
- `current_state_dict, dynamic_system_params_record = dynamic_system_base.step(u_total, dt_sec)`  
    Dentro:
- `CartPoleDynamicSystem.apply_action(u_total)` convierte a fuerza
- `odeint(system_impl.dynamics, ...)` integra, luego `normalize_state` desde el propio sistema dinámico en específico 
    Justificación: el orquestador integra; la física vive en el sistema específico.
3. **Chequeo de terminación**:
- `terminated, reason = dynamic_system_base.check_termination()`  
    Justificación: condiciones declarativas por config, sin hardcode.
4. **Registro step-level en MetricCollector**:
- `metric_collector.on_step(current_state_dict, dynamic_system_params_record, controller_state_record, current_time)`
    Justificación: el collector aplana dicts crudos y hace append sobre llaves activas del template.
5. **Acumulación flat para MetricProcessing**:
- `flat_step_records.append(_flatten_controller_record(controller_state_record))`
    Justificación: se aplana el record nested `{controller_name: {signal: val}}` a llaves canónicas planas `{signal: val}`. MetricProcessing accede directamente por llave sin navegar sub-dicts.

## 5) Post-step del intervalo: métricas → recompensa → aprendizaje

1. **Procesamiento interval-level**:
- `processed_metrics_dict = metric_processing.process_interval_metrics(flat_step_records)`
    Justificación: MetricProcessing recibe lista de dicts planos con llaves canónicas (e.g. `error_<var_obj>`, `control_action_<var_obj>`). Agrega series temporales y aplica normalización declarativa desde config.
2. **Recompensa interval-level**:
- `reward_info = reward_calculator.calculate(processed_metrics_dict, termination_reason, end_time_sec)`  
    Justificación: recompensa solo depende de agregados del intervalo (no de cada step).
3. **Aprendizaje**:
- Construye `next_agent_state` (estado dinámico + ganancias actuales) y llama:  
    `learn_info = agent.learn(prev_agent_state, next_agent_state, actions_dict, reward_info, terminated)`

**Contrato único de salida del intervalo**:
```json
interval_result = {
	"interval_metadata": {
		"decision_id": decision_id,
		"t_start_sec": current_time_sec,
		"n_steps_executed": n_steps_executed
	},
	"interval_level_data": {
		"reward_info": reward_info,
		"learn_info": learn_info,
		"controller_info": {**actions_dict, **controller_state_record},
		"processed_metrics_dict": processed_metrics_dict,
		"simulation_state_dict": {
			"terminated": terminated,
			"termination_reason": termination_reason
		}
	}
}
```
Justificación: contrato limpio sin step_level_data; los datos step-level ya están en MetricCollector.

## 6) Registro y persistencia: "MetricCollector primero, ResultHandler después"

1. **Registro step-level** (dentro del step-loop, §4.4):
- `metric_collector.on_step(current_state_dict, dynamic_system_params_record, controller_state_record, current_time)`
    Justificación: el collector aplana y acumula incrementalmente; SM no guarda datos step-level.
2. **Registro interval-level**:
- `metric_collector.on_interval_end(flat_interval_data)`
    Donde `flat_interval_data = _build_interval_flat_data(interval_result, ...)` aplana: metadata, reward_component (`L_<feature>_<var_obj>`), reward per agent, principal_record, extra_record, learn_info, actions, y parámetros del agente.
    Justificación: `MetricCollector` solo acumula lo permitido según template, de forma eficiente.
3. **Fin del episodio**:
- `metric_collector.on_episode_end(episode_id, terminated, reason)` construye `episode_data` y delega a `result_handler.save_episode(episode_data)`.
- `ResultHandler.save_episode` auto-genera summary row via `calculate_episode_summary(episode_data)` y persiste por chunks.  
    Justificación: `ResultHandler` escribe a disco (chunking + summary), _nadie más_.
4. **Guardar estado del agente (periodicidad)**:
- `result_handler.save_agent_state_learn_dict(agent, episode_id)`  
    Justificación: desacopla "resultados" de "estado de aprendizaje".

## 7) Post-run: visualización y transformaciones

1. **Visualización post-run**:
- `VisualizationManager.run(...)` se ejecuta fuera del loop de simulación.
    Justificación: no contaminar el loop con I/O pesado.
	- `main._run_visualization()`





