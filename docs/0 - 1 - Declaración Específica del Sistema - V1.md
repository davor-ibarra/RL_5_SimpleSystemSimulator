---
created: 20260212 14:02
update: 20260213-14:22
summary:
status:
link:
tags:
---
# Declaración Ideal del Sistema (TO-BE)
Se declara el sistema ideal como **contrato operativo completo**, indicando **responsabilidades** y **métodos públicos e internos**, agrupados por funcionalidad. La regla base es: **cada método existe porque hay una responsabilidad explícita**; nada queda implícito.


## 1) Declaración de sistema (una frase por componente)
- **`main`**: punto de entrada que carga configuración, construye componentes, inyecta dependencias y delega ejecución/postproceso.
- **`SimulationManager`**: orquestador puro del flujo `episode -> interval -> step` sin lógica de dinámica, recompensa ni métricas derivadas.
- **`DynamicSystemBase` + sistema específico**: unidad autocontenida para reset, integración, normalización, estado canónico y terminación.
- **`ControllerBase` + `PIDController`**: producción de señal de control por lazo, suma global, saturación global y anti-windup condicional.
- **`AgentBase` + agente específico**: decisión y aprendizaje RL bajo estado observable explícito y contrato mínimo.
- **`MetricCollector`**: acumulación incremental y filtrada por template desde el inicio (sin recomputar ni podar por heurística).
- **`MetricProcessing`**: transformación declarativa de series step-level a métricas interval-level puras y reproducibles.
- **`RewardCalculatorBase` + componentes**: orquestación de recompensa principal y adicional con trazabilidad completa (`reward_info`).
- **`ResultHandler`**: persistencia y transformación post-simulación (chunks, summary, serialización final).
- **`VisualizationManager`**: post-procesamiento visual desacoplado del loop activo de simulación.


```text
main.main()
├── load_config() × 4
├── _build_run_id() + _build_output_dir()
├── _build_components()  → instanciación explícita
├── _save_metadata()
├── simulation_manager.run_simulation()
│   └── for episode_id in range(n_episodes):
│       ├── _reset_episode(episode_id)
│       ├── Construir estado inicial (prev_state, prev_actions)
│       ├── metric_collector.on_episode_start(episode_id)
│       └── while not terminated and time < max:
│           ├── agent.select_action(prev_agent_state)
│           ├── _apply_actions_to_controllers(actions_dict)
│           ├── _run_interval(...)
│           │   ├── for step in range(steps_per_interval):
│           │   │   ├── controller_base.compute_control(state_norm)
│           │   │   ├── dynamic_system_base.step(u_total, dt)
│           │   │   ├── dynamic_system_base.check_termination()
│           │   │   ├── metric_collector.on_step(...)
│	            │   │   └── flat_step_records.append(...)
│           │   ├── metric_processing.process_interval_metrics(flat)
│           │   ├── reward_calculator.calculate(processed, term, time)
│           │   ├── agent.learn(prev, next, actions, reward, term)
│           │   └── return interval_result
│           ├── _build_interval_flat_data(interval_result) → flat
│           ├── metric_collector.on_interval_end(flat)
│           └── Actualizar prev_states, time, decision_id
│       ├── metric_collector.on_episode_end(episode_id, end_data)
│       └── result_handler.save_agent_state (periódico)
│   └── metric_collector.finalize_run()
└── visualization_manager.run()
```

**Regla de arquitectura:** ninguna mejora puede violar este esqueleto. Todo cambio se evalúa por compatibilidad con esta secuencia.

---

## 2) Contratos públicos e internos por funcionalidad

## A. Bootstrap y wiring (`main.py`)
**Responsabilidades**
1. Cargar YAMLs sin reinterpretar reglas de negocio.
2. Construir identificadores de corrida.
3. Instanciar explícitamente todos los componentes.
4. Guardar metadata.
5. Disparar simulación y visualización.

**Públicos**
- `main()`
- `load_config(path)`

**Internos**
- `_build_run_id()`
- `_build_output_dir(base_dir, system_id, run_id)`
- `_build_metadata(run_id, timestamp, config_main, config_data_save, config_visualization)`
- `_build_components(config_main, config_data_save, config_template_output, output_dir)`
- `_save_metadata(output_dir, metadata_dict, result_handler)`
- `_run_simulation(simulation_manager)`
- `_run_visualization(visualization_manager, output_dir, config_visualization)`

**No permitido**
- Calcular reward/métricas/control.
- Operar paso a paso del episodio.

## B. Orquestación (`simulation_manager.py`)
**Responsabilidad única**
- Mantener el orden formal del esqueleto y el reloj lógico de ejecución.

**Públicos**
- `run_simulation()`

**Internos (flujo)**
- `_run_episode(episode_id)`
- `_run_interval(decision_id, current_time_sec, actions_dict, prev_agent_state, prev_dynamic_system_state_norm_dict)`
- `_reset_episode(episode_id)`

**Internos (adaptación/flat)**
- `_build_interval_flat_data(interval_result, decision_id, step_idx_global, actions_dict)`

**Internos (estado de decisión)**
- `_build_initial_actions_dict()`
- `_apply_actions_to_controllers(actions_dict)`
- `_get_controller_gains_dict()`
- `_build_agent_to_controller_map()`

**No permitido**
- Resolver física.
- Resolver reward directamente.
- Tratar estructura de dato que debiera venir lista
- Persistir directo a disco (excepto delegación explícita a handlers).

## C. Sistema dinámico (`dynamic_system/*`)
**Responsabilidad**
- Encapsular modelo físico y su estado canónico (raw + normalized), mientras `dynamic_system_base` se encarga de orquestar e interactuar con el simulador.

**Públicos mínimos de** `dynamic_system_base`
- `reset_episode()`
- `step(u_total, dt_sec)`
- `check_termination()`
- `get_records()`

**Público recomendado a incorporar (para cerrar contrato) de** `dynamic_system_base`
- `get_dynamic_system_state(mode="raw|normalized")` (evita depender de atributos internos).


**Públicos mínimos de** `cart_pole_dynamic_system`
- `reset_episode()`
- `apply_u_total(u_total)`
- `dynamics(state_vector, time_t, force)`
- `reference_system_correction(angle_values)`
- `normalize_state(state_arr)`
- `get_params_dict()`

**Internos típicos de** `cart_pole_dynamic_system`
- `_build_initial_state(initial_state_dict)`

**Reglas**
- La normalización vive en dinámica, no en agente, processing ni en orquestador.

## D. Control (`controllers/*`)
**Responsabilidad**
- Generar `u_local` por lazo, sumar `u_total`, aplicar saturación global y anti-windup condicional.

**Públicos de `ControllerBase`**
- `reset_episode()`
- `compute_control(dynamic_state_dict)`
- `update_gains(controller_name, kp, ki, kd)`
- `get_controller_state()` (exponer ganancias actuales)
- `get_records()`

**Internos de `ControllerBase`**
- `_apply_antiwindup_corrections(individual_actions, saturation_error)`
- `_build_controller_state_record(...)`

**Públicos de `PIDController`**
- `reset_episode()`
- `compute(dynamic_state_dict, dt_sec)`
- `apply_antiwindup_correction(correction, dt_sec)`
- `set_saturation_status(is_saturated, u_total_raw, u_total_saturated)`
- `update_current_gains(kp, ki, kd)`
- `get_current_controller_gains()`
- `get_params_dict()`

**Internos de `PIDController`**
- `_normalize_setpoint()`
- `_apply_conditional_antiwindup_correction(e_sat)`

**Regla crítica**
- El gatillo de anti-windup es saturación efectiva global respecto a límites configurados.

## E. Agentes (`agents/*`)
**Responsabilidad**
- Construir estado, seleccionar acción y aprender.

**Públicos de `AgentBase`**
- `reset_episode()`
- `build_agent_state(dynamic_state_dict, controller_gains_dict)`
- `select_action(agent_state)`
- `learn(prev_agent_state, next_agent_state, actions_dict, reward_info, terminated)`
- `get_agent_learn_state()`
- `get_records()`

**Públicos de `PIDQLearningAgent`**
- `reset_episode()`
- `build_agent_state(dynamic_state_dict, controller_gains_dict)`
- `select_action(agent_state)`
- `learn(prev_agent_state, next_agent_state, actions_dict, reward_info, terminated)`
- `get_agent_learn_state()`
- `get_params_dict()`

**Internos de `PIDQLearningAgent`**
- `_build_state_space()`
- `_initialize_q_tables()`
- `_discretize_value(value, var_name)`
- `_get_state_indices(agent_state, agent_name)`
- `_get_reward_for_agent(agent_name, reward_for_learning)`

**Reglas**
- El agente no calcula reward.
- El `PIDQLearningAgent` debe crear desde el inicio la matriz de aprendizaje
- El estado del agente se arma desde contratos explícitos (estado del sistema + ganancias), no inspección interna de otras clases.

## F. Pipeline de métricas (`metrics/*`)

### F.1 `MetricCollector`
**Responsabilidad**
- Recibir eventos del loop y acumular incrementalmente por nivel de dato.

**Públicos**
- `on_episode_start(episode_id)`
- `on_step(dynamic_system_state_norm_dict, dynamic_system_params_record, controller_state_record, current_time)`
- `on_interval_end(interval_flat_data)`
- `on_episode_end(episode_id, end_episode_data)`
- `finalize_run()`
- `get_step_records()`

**Internos**
- `_extract_list_keys(template_section, exclude=None)`

**Regla crítica (obligatoria)**
- El filtrado ocurre **desde el inicio** usando `config/sub_config_template_output_CartPole.yaml`.
- El collector **solo** hace append en llaves previamente habilitadas por template.
- El collector no decide aplanar o borrar variables por “no uso” ya que estos deberían venir listos desde la fuente de datos (lógica integrada en cada `get_records()`).

### F.2 `MetricProcessing`
**Responsabilidad**
- Calcular agregados y normalizaciones interval-level para recompensa/análisis.

**Públicos**
- `reset_episode()`
- `process_interval_metrics(flat_step_records)`
- `get_records()`

**Internos**
- `_build_var_obj_mapping()`
- `_extract_normalization_config()`
- `_process_var_obj_metrics(...)`
- `_process_global_vars(...)`
- `_aggregate_with_method(values, method)`
- `_normalize_value(value, value_range, output_limits, method)`

**Regla**
- Procesamiento declarativo y puro; no altera el histórico base.

## G. Recompensa (`rewards/*`)
**Responsabilidad**
- Orquestar reward principal + extras + asignación final por política.

**Públicos de `RewardCalculatorBase`**
- `reset_episode()`
- `calculate(processed_metrics_dict, termination_flag='unknown', current_time_sec=0.0)`
- `get_records()`

**Internos de `RewardCalculatorBase`**
- `_build_agent_to_var_obj_map()`
- `_instantiate_principal_reward()`
- `_instantiate_extra_rewards()`
- `_assign_rewards(global_reward, controller_rewards, extra_reward)`
- `_compute_individual_rewards(controller_rewards, individual_params)`

**Públicos de rewards concretas**
- `LagrangeRewardCalculator.reset_episode()`
- `LagrangeRewardCalculator.compute_reward(reward_component)`
- `LagrangeRewardCalculator.get_reward_params_dict()`
- `ExtraRewardsHandler.reset_episode()`
- `ExtraRewardsHandler.evaluate(extra_reward_component, termination_flag='unknown', current_time_sec=0.0)`
- `ExtraRewardsHandler.get_extra_reward_params_dict()`

**Internos de rewards concretas**
- `LagrangeRewardCalculator._extract_lineal_weights()`
- `LagrangeRewardCalculator._extract_exponential_weights()`
- `LagrangeRewardCalculator._compute_lineal()`
- `LagrangeRewardCalculator._compute_exponential()`
- `ExtraRewardsHandler._get_configured_var_objs()`
- `ExtraRewardsHandler._evaluate_penalties(self, extra_reward_component, var_objs)`
- `ExtraRewardsHandler._compute_instantaneous_penalty(self, config)`
- `ExtraRewardsHandler._compute_delta_var_penalty(self, config, extra_reward_component, var_objs)`
- `ExtraRewardsHandler._evaluate_bonuses(self, extra_reward_component, termination_flag, current_time_sec, var_objs)`
- `ExtraRewardsHandler._compute_goal_bonus(self, config, termination_flag, current_time_sec)`
- `ExtraRewardsHandler._compute_bandwidth_bonus(self, config, extra_reward_component, var_objs)`
- `ExtraRewardsHandler._evaluate_conditional(self, extra_reward_component, var_objs)`
- `ExtraRewardsHandler._compute_dynamic_penalty(self, config, extra_reward_component, var_objs)`
- `ExtraRewardsHandler._compute_dynamic_incentive(self, config, extra_reward_component, var_objs)`

**Regla de asignación**
- Política explícita configurable: global por agente / individual por agente / por lazo controlador.

## H. Persistencia y visualización (`utils/result_handler.py`, `visualization_manager.py`)
**Responsabilidad**
- Persistir estructura final por episodio y ejecutar postproceso visual.

**Públicos `ResultHandler`**
- `save_metadata(metadata_dict)`
- `save_episode(episode_data)`
- `save_episode_chunk(episodes_data, chunk_id)`
- `append_summary_row(summary_row)`
- `save_agent_state_learn_dict(state_dict, episode_id)`
- `finalize_run()`
- `load_all_chunks()`
- `load_summary()`
- `flatten_episode_to_steps(episode_data)`

**Internos `ResultHandler`**
- `_ensure_output_structure()`
- `_flush_episode_buffer()`
- `_flush_summary()`
- `get_chunk_filepath(chunk_id)`
- `get_summary_filepath()`
- `get_agent_state_filepath(episode_id)`

**Públicos `VisualizationManager`**
- `run()`

**Internos `VisualizationManager`**
- `_generate_heatmap_data_if_needed()`

**Regla**
- Metadata no va en cada `json` sino que va en un `json` propio de metadata en la misma carpeta y contiene información específica de la ejecución y todos los parámetros de configuración tal cual vienen en los `yaml`
- Todo registro y plot queda en la misma carpeta

---

## 3) Contrato de datos de salida (canónico y estricto)

## 3.1 Forma por episodio
```yaml
episode_data:
  episode_id: int
  step_data: dict[list]
  interval_data: dict[list]
  end_episode_data: dict
```

## 3.2 Reglas estrictas
1. `step_data` solo guarda señales step-level.
2. `interval_data` solo guarda agregados/decisiones interval-level.
3. `end_episode_data` solo guarda resumen final y metadata de cierre.
4. Las llaves deben ser canónicas y estables (snake_case, semántica única).
5. No duplicar semántica con llaves alternativas para la misma señal.

## 3.3 Template YAML como filtro temprano
El template de referencia es `config/sub_config_template_output_CartPole.yaml` y define qué se guarda desde el primer evento.

**Lista obligatoria de familias de llaves esperadas**
- Step-level: tiempo, estado raw/norm, parámetros del sistema, control global, contribuciones por controlador, señales PID canónicas por controlador.
- Interval-level: metadata del intervalo, terminación, métricas `L_*`, acciones por agente, reward global/individual, reward params opcionales, extras por variable objetivo, learn-info opcional.
- End-episode: terminación final, resumen de rewards, metadata de episodio y stats opcionales de agentes.

---

## 4) Contratos de integración entre capas
1. `SimulationManager` solo usa APIs públicas de cada subsistema (componente base u orquestador).
2. `MetricCollector` recibe datos planos/canónicos y no recalcula derivados.
3. `MetricProcessing` recibe listas de datos del intervalo; devuelve `processed_metrics_dict`.
4. `RewardCalculatorBase` consume únicamente procesados de intervalo y devuelve `reward_for_learning` trazable.
5. `Agent.learn(...)` consume `reward_for_learning` ya asignada según política para cada agent.
6. `ResultHandler` persiste al cierre de episodio/run, sin intervenir decisiones online.

---

## 5) Criterios de aceptación de arquitectura ideal
1. Cada clase explica su responsabilidad en una frase sin ambigüedad.
2. El esqueleto elemental se cumple sin saltos ni lógica lateral.
3. Contratos de datos apropiados y estables en nombres, niveles y tipos.
4. Configuración gobierna guardado, reward modes, límites y comportamiento declarativo.
5. Recompensa auditable extremo a extremo (`principal + extras + asignación`).
6. Extensibilidad real: cambiar sistema dinámico/controlador/agente/reward sin romper el loop maestro.
7. Testabilidad estructural: cada componente validable en aislamiento.

