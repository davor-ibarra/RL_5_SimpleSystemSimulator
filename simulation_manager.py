"""
simulation_manager.py

Responsabilidad:
Orquestador central de la simulación episódica.
Ejecuta el bucle episode → interval → step coordinando el pipeline completo:
Agent → Controllers → DynamicSystem → MetricProcessing → Reward → Learn → MetricCollector

Contrato de datos:
- Todos los registros que fluyen entre componentes usan llaves canónicas planas
  (e.g. error_<var_obj>, control_action_<var_obj>), sin estructuras anidadas.
- MetricProcessing y MetricCollector reciben dicts planos directamente.
- ResultHandler persiste en chunks JSON y genera summary.xlsx automáticamente
  via calculate_episode_summary al cierre de cada episodio.
"""


class SimulationManager:
    """
    Orquestador de la simulación. Ejecuta episodios e intervalos de decisión.
    """
    
    def __init__(self, dynamic_system_base, controller_base, agent_base, 
                 metric_processing, reward_calculator, metric_collector, result_handler,
                 config_main, config_data_summary, output_dir):
        """
        Inicializa el orquestador y deja todo preparado para ejecutar la simulación
        sin lógica condicional ni descubrimientos tardíos.
        
        Args:
            dynamic_system_base: Instancia del sistema dinámico base
            controller_base: Instancia del controlador base
            agent_base: Instancia del agente base
            metric_processing: Instancia de procesamiento de métricas
            reward_calculator: Instancia del calculador de recompensa
            metric_collector: Instancia del colector de métricas
            result_handler: Instancia del manejador de resultados
            config_main (dict): Configuración principal
            config_data_summary (dict): Configuración de resumen de datos
            output_dir (str): Directorio de salida
        """
        # Almacenar componentes
        self.dynamic_system_base = dynamic_system_base
        self.controller_base = controller_base
        self.agent_base = agent_base
        self.metric_processing = metric_processing
        self.reward_calculator = reward_calculator
        self.metric_collector = metric_collector
        self.result_handler = result_handler
        
        # Referencia directa a controladores desde controller_base
        self.controllers = controller_base.controllers
        
        # Guardar contexto de ejecución
        self.config_main = config_main
        self.config_data_summary = config_data_summary
        self.output_dir = output_dir
        
        # Extraer parámetros de simulación
        self.dt_sec = config_main['simulation']['dt_sec']
        self.decision_interval_sec = config_main['simulation']['decision_interval_sec']
        self.episode_duration_sec = config_main['simulation']['episode_duration_sec']
        self.n_episodes = config_main['simulation']['n_episodes']
        
        # Derivar constantes operativas
        self.steps_per_interval = int(self.decision_interval_sec / self.dt_sec)
        
        # Logging params
        self.current_time_sec = 0.0
        self.termination_reason = ""
        self.total_reward = 0.0
        self.total_controller_rewards = {}
        
        # Obtener nombres desde componentes instanciados
        self.controller_names = list(self.controllers.keys())
        self.agent_names = agent_base.agent_names
        
        # Precomputar mapping declarativo: agent_name → (controller_name, gain_type)
        self.agent_to_controller_map = self._build_agent_to_controller_map()
        
        # Preparar estado mínimo para deltas entre intervalos
        self.prev_actions_dict = None
        self.prev_dynamic_state_dict = None
        self.prev_agent_state = None
    
    def run_simulation(self):
        """
        Ejecuta la corrida completa con un flujo único: episodios → cierre de corrida.
        """
        print(f"[SIMULATION_MANAGER] Iniciando simulación: {self.n_episodes} episodios")
        
        for episode_id in range(self.n_episodes):
            print(f"\n[SIMULATION_MANAGER] === Episodio {episode_id + 1}/{self.n_episodes} ===")
            self._run_episode(episode_id)
            formatted_controller_rewards = self._format_controller_rewards()
            print(f"\n[SIMULATION_MANAGER] --- T_max = {self.current_time_sec}  |  Rewards = {formatted_controller_rewards}  |  Termination Reason = {self.termination_reason} ---")
            current_gains = self._get_gains_for_agent()
            formatted_gains = "  |  ".join([f"{k} = {round(v, 2)}" for k, v in current_gains.items()])
            print(f"\n[SIMULATION_MANAGER] --- {formatted_gains} ---")
        
        self.metric_collector.finalize_run()
        print(f"\n[SIMULATION_MANAGER] Simulación completada")
    
    def _run_episode(self, episode_id):
        """
        Ejecuta un episodio completo.
        
        Args:
            episode_id (int): Identificador del episodio
        """
        # 1. Resetear componentes
        self._reset_episode(episode_id)
        
        # 2. Inicializar reloj y control de término
        self.current_time_sec = 0.0
        decision_id = 0
        
        # 3. Inicializar estado previo necesario para decisiones
        self.prev_dynamic_system_state_dict = self.dynamic_system_base.get_dynamic_system_state('raw')
        self.prev_dynamic_system_state_norm_dict = self.dynamic_system_base.get_dynamic_system_state('normalized')
        self.prev_agent_state = self.agent_base.build_agent_state(self._collect_agent_context())
        
        # 4. Construir prev_actions_dict inicial (todas las acciones en "mantener" = 1)
        self.prev_actions_dict = self._build_initial_actions_dict()
        
        # 5. Inicializar buffers del episodio en el colector (captura t=0)
        self.metric_collector.on_episode_start(episode_id)
        
        # 5.1 Registrar paso explícito t=0 para que las ganancias iniciales queden impecablemente registradas
        # antes de ser mutadas por la primera acción del agente.
        step_flat_data_t0 = {'t_sec': 0.0}
        step_idx_global = 1
        step_flat_data_t0.update(self.dynamic_system_base.get_records())
        step_flat_data_t0.update(self.controller_base.get_records())
        self.metric_processing.normalize_step_record(step_flat_data_t0)
        self.metric_collector.on_step(step_flat_data_t0)
        
        # 6. Loop de intervalos hasta término por tiempo o condición del sistema
        terminated = False
        self.termination_reason = ""
        self.total_reward = 0.0
        self.total_controller_rewards = self._build_zero_controller_rewards()
        total_agent_decisions = 0
        max_episode_steps = int(round(self.episode_duration_sec / self.dt_sec))
        executed_episode_steps = 0
        
        while executed_episode_steps < max_episode_steps and not terminated:
            remaining_steps = max_episode_steps - executed_episode_steps
            # 6.1. Decide acciones del intervalo
            actions_dict = self.agent_base.select_action(self.prev_agent_state)
            
            # 6.2. Aplica acciones a controladores
            self._apply_actions_to_controllers(actions_dict)
            
            # 6.3. Ejecuta la simulación del intervalo y obtiene productos
            interval_result = self._run_interval(decision_id, 
                self.current_time_sec, 
                actions_dict, 
                self.prev_agent_state, 
                self.prev_dynamic_system_state_norm_dict,
                remaining_steps
            )
            
            # 6.4. Construir paquete plano para interval_data y registrar
            interval_flat_data = self._build_interval_flat_data(
                interval_result, decision_id, step_idx_global, actions_dict
            )
            self.metric_collector.on_interval_end(interval_flat_data)
            
            # 6.5. Actualiza acumuladores y estado para siguiente intervalo
            n_steps_executed = interval_result['interval_metadata']['n_steps_executed']
            terminated = interval_result['interval_level_data']['simulation_state_dict']['terminated']
            self.termination_reason = interval_result['interval_level_data']['simulation_state_dict']['termination_reason']
            self.total_reward += interval_result['interval_level_data']['global_interval_reward']
            interval_controller_rewards = self._aggregate_interval_controller_rewards(
                interval_result['interval_level_data']['all_interval_reward']
            )
            for controller_name, reward_value in interval_controller_rewards.items():
                self.total_controller_rewards[controller_name] += reward_value

            executed_episode_steps += n_steps_executed
            self.current_time_sec += n_steps_executed * self.dt_sec
            
            step_idx_global += n_steps_executed
            decision_id += 1
            total_agent_decisions += 1
            self.prev_dynamic_system_state_dict = self.dynamic_system_base.get_dynamic_system_state('raw')
            self.prev_dynamic_system_state_norm_dict = self.dynamic_system_base.get_dynamic_system_state('normalized')
            self.prev_agent_state = self.agent_base.build_agent_state(self._collect_agent_context())
            self.prev_actions_dict = actions_dict
        
        # 7. Si terminó por tiempo, asignar razón
        if not terminated:
            self.termination_reason = "time_limit"
        
        # 8. Cierre del episodio: construir end_episode_data y commit a disco
        episode_reward_summary = self.reward_calculator.get_episode_summary_rewards()
        
        end_episode_data = {
            'end_terminated': terminated,
            'end_termination_reason': self.termination_reason,
            'final_pendulum_angle_raw': self.prev_dynamic_system_state_dict['pendulum_angle'],
            'final_pendulum_velocity_raw': self.prev_dynamic_system_state_dict['pendulum_velocity'],
            'final_cart_position_raw': self.prev_dynamic_system_state_dict['cart_position'],
            'final_cart_velocity_raw': self.prev_dynamic_system_state_dict['cart_velocity'],
            'final_pendulum_angle_normalized': self.prev_dynamic_system_state_dict['pendulum_angle'],
            'final_pendulum_velocity_normalized': self.prev_dynamic_system_state_dict['pendulum_velocity'],
            'final_cart_position_normalized': self.prev_dynamic_system_state_dict['cart_position'],
            'final_cart_velocity_normalized': self.prev_dynamic_system_state_dict['cart_velocity'],
            'total_reward': self.total_reward,
            'total_agent_decisions': total_agent_decisions
        }
        end_episode_data.update(self._build_controller_reward_summary())
        end_episode_data.update(episode_reward_summary)
        end_episode_data.update(self.agent_base.get_agent_params_records())
        self.metric_collector.on_episode_end(episode_id, end_episode_data)
        
        # 9. Guardar estado del agente según periodicidad
        save_period = self.config_main['data_handling']['agent_state_save_frequency']
        if save_period and (episode_id + 1) % save_period == 0:
            self.result_handler.save_agent_state_learn_dict(self.agent_base.get_agent_state_learn_dict(), episode_id)
    
    def _run_interval(self, decision_id, current_time_sec, actions_dict, prev_agent_state, prev_dynamic_system_state_norm_dict, remaining_steps):
        """
        Orquesta un intervalo completo y retorna el contenedor canónico.
        
        Args:
            decision_id (int): Identificador de la decisión
            current_time_sec (float): Tiempo actual en segundos
            actions_dict (dict): Diccionario de acciones del agente
            prev_agent_state: Estado previo del agente
            prev_dynamic_system_state_dict: Estado previo del sistema dinámico
            remaining_steps (int): Pasos restantes en el episodio
            
        Returns:
            tuple: (interval_data)
        """
        # 1. Inicializar contenedor del intervalo (flat records)
        flat_step_records = []
        n_steps_executed = 0
        
        # 2. Ejecutar el step-loop del intervalo
        for step in range(min(self.steps_per_interval, remaining_steps)):
            n_steps_executed += 1
            
            # 2.1. Calcular acción total de control
            u_total = self.controller_base.compute_control(prev_dynamic_system_state_norm_dict)
            
            # 2.2. Integrar el sistema dinámico un paso
            current_state_norm_dict = self.dynamic_system_base.step(u_total, self.dt_sec)
            
            # 2.3. Evaluar condición de término
            terminated, self.termination_reason = self.dynamic_system_base.check_termination()
            
            # 2.4. Registrar step en el collector (dict plano vía get_records())
            step_flat_data = {'t_sec': self.dynamic_system_base.current_time}
            step_flat_data.update(self.dynamic_system_base.get_records())
            step_flat_data.update(self.controller_base.get_records())
            self.metric_processing.normalize_step_record(step_flat_data)
            self.metric_collector.on_step(step_flat_data)
            
            # 2.4b. Acumular step record para MetricProcessing
            flat_step_records.append(step_flat_data)

            # 2.5. Actualizar estado normalizado previo para siguiente step
            prev_dynamic_system_state_norm_dict = current_state_norm_dict
            
            # 2.6. Si terminó, cortar el loop inmediatamente
            if terminated:
                break
        
        # 3. Procesar métricas del intervalo (interval-level)
        processed_metrics_dict = self.metric_processing.process_interval_metrics(flat_step_records)
        
        # 4. Calcular recompensa del intervalo (interval-level)
        # Tiempo al final del intervalo para cálculo de decay en goal_bonus
        end_time_sec = current_time_sec + n_steps_executed * self.dt_sec
        reward_for_learning = self.reward_calculator.calculate(
            processed_metrics_dict,
            self.termination_reason,
            end_time_sec,
            actions_dict
        )
        
        # 5. Ejecutar aprendizaje del agente
        next_agent_state = self.agent_base.build_agent_state(self._collect_agent_context())
        
        # Distinction: Time limit truncation vs true boundary termination
        terminated_boundary = terminated and self.termination_reason != "time_limit"
        
        learn_info = self.agent_base.learn(
            prev_agent_state, next_agent_state, actions_dict, reward_for_learning, terminated_boundary
        )
        
        # 6. Armar el interval_result con metadata completa
        interval_result = {
            'interval_metadata': {
                'decision_id': decision_id,
                't_start_sec': current_time_sec,
                'n_steps_executed': n_steps_executed
            },
            'interval_level_data': {
                'global_interval_reward': self.reward_calculator._last_global_interval_reward,
                'all_interval_reward': reward_for_learning,
                'learn_info': learn_info,
                'simulation_state_dict': {'terminated': terminated, 'termination_reason': self.termination_reason}
            }
        }
        
        return interval_result
    
    def _build_interval_flat_data(self, interval_result, decision_id, step_idx_global, actions_dict):
        """
        Construye un dict plano con todas las llaves interval-level para el collector.
        Fusiona records planos de cada componente vía get_records().
        
        Args:
            interval_result (dict): Resultado completo del intervalo
            decision_id (int): ID de la decisión
            step_idx_global (int): Índice global del primer step del intervalo
            actions_dict (dict): Acciones del agente
            
        Returns:
            dict: Dict plano con todas las llaves interval-level
        """
        metadata = interval_result['interval_metadata']
        interval_level = interval_result['interval_level_data']
        n_steps = metadata['n_steps_executed']
        
        flat = {}
        
        # 1. Metadata del intervalo
        flat['interval_id'] = decision_id
        flat['interval_start_step_idx'] = step_idx_global
        flat['interval_end_step_idx'] = step_idx_global + n_steps - 1
        flat['interval_t_start_sec'] = metadata['t_start_sec']
        flat['interval_n_steps'] = n_steps
        
        # 2. Estado de simulación
        sim_state = interval_level['simulation_state_dict']
        flat['terminated'] = sim_state['terminated']
        flat['termination_reason'] = sim_state['termination_reason']
        
        # 3. Processed metrics (L_<feature>_<var_obj>) — via MetricProcessing.get_records()
        flat.update(self.metric_processing.get_records())
        
        # 4. Reward records (global_interval_reward, per-agent, principal, extra) — via get_records()
        flat.update(self.reward_calculator.get_records())
        
        # 5. Learn info (td_error_<agent>, q_value_<agent> — ya planas)
        learn_info = interval_level['learn_info']
        flat.update(learn_info)
        
        # 6. Actions: action_<agent_name> -> decisions and gain-delta diagnostics
        vars_decision = actions_dict['vars_decision']
        flat.update(vars_decision)
        vars_delta = actions_dict['vars_delta']
        flat.update(vars_delta)
        
        # 7. Agent parameters (epsilon, learning_rate, Q-stats) — via get_records()
        flat.update(self.agent_base.get_records())
        
        return flat
    
    def _reset_episode(self, episode_id):
        """
        Resetea todos los componentes al inicio de un episodio.
        
        Args:
            episode_id (int): Identificador del episodio
        """
        self.dynamic_system_base.reset_episode()
        self.controller_base.reset_episode()
        self.agent_base.reset_episode()
        self.metric_processing.reset_episode()
        self.reward_calculator.reset_episode()
    
    def _apply_actions_to_controllers(self, actions_dict):
        """
        Aplica las decisiones del agente a las ganancias de cada controlador.
        Clipea cada ganancia al rango [min, max] definido en agent_config.
        
        Args:
            actions_dict (dict): Diccionario de acciones del agente
        """
        agents_config = self.config_main['agent_base']['agent_config']['agents']
        
        # Obtener el mapping de acciones desde la configuración
        actions_space = self.config_main['agent_base']['agent_config']['actions']['actions_space']
        actions_dict['actions_applied'] = False
        actions_dict['vars_delta']['actions_applied'] = 0.0

        # Inicializar ganancias por controlador con sus valores actuales reales (evita errores de llaves faltantes)
        controller_gains = {}
        for controller_name in self.controller_names:
            controller_gains[controller_name] = self.controllers[controller_name].get_current_controller_gains()
            
        for agent_name, (controller_name, gain_type) in self.agent_to_controller_map.items():
            current_value = controller_gains[controller_name][gain_type]
            actions_dict['vars_values'][agent_name] = current_value
            action_decision = actions_dict['vars_decision'][f'action_{agent_name}']
            delta_gain = actions_dict['vars_delta'][f'delta_gain_{agent_name}']
            
            # Traducir el índice de acción a su significado explícito ('decrease', 'maintain', 'increase')
            action_str = actions_space[action_decision]
            
            if action_str == 'decrease':
                requested_delta_gain = -delta_gain
            elif action_str == 'increase':
                requested_delta_gain = delta_gain
            elif action_str == 'maintain':
                requested_delta_gain = 0.0
            else:
                raise ValueError(f"Unsupported action label for {agent_name}: {action_str}")

            unclipped_value = current_value + requested_delta_gain
            
            # Clipear al rango [min, max] definido en config del agente
            agent_cfg = agents_config[agent_name]
            new_value = max(agent_cfg['min'], min(agent_cfg['max'], unclipped_value))
            applied_delta_gain = new_value - current_value
            if requested_delta_gain == 0.0:
                blocked_fraction = 0.0
            else:
                blocked_fraction = 1.0 - abs(applied_delta_gain) / abs(requested_delta_gain)
                blocked_fraction = max(0.0, min(1.0, blocked_fraction))
            action_requested_move = float(action_str != 'maintain')
            action_blocked = float(action_requested_move > 0.0 and blocked_fraction > 0.0)

            actions_dict['vars_delta'][f'delta_gain_requested_{agent_name}'] = requested_delta_gain
            actions_dict['vars_delta'][f'delta_gain_applied_{agent_name}'] = applied_delta_gain
            actions_dict['vars_delta'][f'action_requested_move_{agent_name}'] = action_requested_move
            actions_dict['vars_delta'][f'action_blocked_{agent_name}'] = action_blocked
            actions_dict['vars_delta'][f'action_blocked_fraction_{agent_name}'] = blocked_fraction
            
            controller_gains[controller_name][gain_type] = new_value
        
        # Aplicar ganancias a cada controlador
        for controller_name, gains in controller_gains.items():
            self.controllers[controller_name].update_gains(gains['kp'], gains['ki'], gains['kd'])
        actions_dict['actions_applied'] = True
        actions_dict['vars_delta']['actions_applied'] = 1.0
    
    def _build_initial_actions_dict(self):
        """
        Construye el actions_dict inicial con todas las acciones en "mantener" (índice 1).
        
        Returns:
            dict: actions_dict inicial
        """
        initial_actions = {
            'actions_applied': False,
            'vars_values': {},
            'vars_decision': {},
            'vars_delta': {
                'actions_applied': 0.0
            }
        }
        
        # Usar mapping precomputado para obtener valores iniciales
        for agent_name, (controller_name, gain_type) in self.agent_to_controller_map.items():
            controller = self.controllers[controller_name]
            gains = controller.get_current_controller_gains()
            initial_actions['vars_values'][agent_name] = gains[gain_type]
            initial_actions['vars_decision'][f'action_{agent_name}'] = 1
            initial_actions['vars_delta'][f'delta_gain_{agent_name}'] = self.agent_base.agent_gain_steps[agent_name]
            initial_actions['vars_delta'][f'delta_gain_requested_{agent_name}'] = 0.0
            initial_actions['vars_delta'][f'delta_gain_applied_{agent_name}'] = 0.0
            initial_actions['vars_delta'][f'action_requested_move_{agent_name}'] = 0.0
            initial_actions['vars_delta'][f'action_blocked_{agent_name}'] = 0.0
            initial_actions['vars_delta'][f'action_blocked_fraction_{agent_name}'] = 0.0
        
        return initial_actions
    
    def _get_gains_for_agent(self):
        """
        Construye diccionario de ganancias actuales de controladores.
        Formato: {agent_name: value}
        
        Returns:
            dict: Ganancias por nombre de agente
        """
        gains_dict = {}
        
        for agent_name, (controller_name, gain_type) in self.agent_to_controller_map.items():
            controller = self.controllers[controller_name]
            gains = controller.get_current_controller_gains()
            gains_dict[agent_name] = gains[gain_type]
        
        return gains_dict

    def _build_zero_controller_rewards(self):
        """
        Inicializa el acumulador de reward por controlador.

        Returns:
            dict: {controller_name -> reward_total_episode}
        """
        return {controller_name: 0.0 for controller_name in self.controller_names}

    def _aggregate_interval_controller_rewards(self, assigned_rewards):
        """
        Agrega rewards asignados a agentes hacia rewards por controlador.
        Para controller_reward, los agentes del mismo controlador reciben el
        mismo valor; el promedio evita contar tres veces el mismo lazo.

        Args:
            assigned_rewards (dict): {agent_name -> reward_interval}

        Returns:
            dict: {controller_name -> reward_interval}
        """
        grouped_rewards = {
            controller_name: []
            for controller_name in self.controller_names
        }

        for agent_name, reward_value in assigned_rewards.items():
            if agent_name not in self.agent_to_controller_map:
                continue
            controller_name, _ = self.agent_to_controller_map[agent_name]
            grouped_rewards[controller_name].append(float(reward_value))

        controller_rewards = {}
        for controller_name, values in grouped_rewards.items():
            if values:
                controller_rewards[controller_name] = sum(values) / len(values)
            else:
                controller_rewards[controller_name] = 0.0

        return controller_rewards

    def _build_controller_reward_summary(self):
        """
        Construye columnas de summary para el reward total del episodio por controlador.
        """
        return {
            f'total_reward_{controller_name}': reward_value
            for controller_name, reward_value in self.total_controller_rewards.items()
        }

    def _format_controller_rewards(self):
        """
        Formatea rewards acumulados por controlador para salida de terminal.
        """
        if not self.total_controller_rewards:
            return "N/A"

        return "  |  ".join([
            f"{controller_name} = {round(reward_value, 4)}"
            for controller_name, reward_value in self.total_controller_rewards.items()
        ])

    def _collect_agent_context(self):
        """
        Reune snapshots planos disponibles y los entrega al agente sin interpretar
        su espacio de estados.
        """
        context = {}
        context.update(self.dynamic_system_base.get_dynamic_system_state('raw'))
        context.update(self.dynamic_system_base.get_records())
        context.update(self.controller_base.get_records())
        context.update(self.metric_processing.get_records())
        context.update(self.reward_calculator.get_records())
        context.update(self.reward_calculator.get_agent_state_records())
        context.update(self._get_gains_for_agent())
        return context
    
    def _build_agent_to_controller_map(self):
        """
        Construye mapping declarativo desde config: agent_name → (controller_name, gain_type).
        Se ejecuta una sola vez en __init__ para evitar parseo de strings en runtime.
        
        Returns:
            dict: {agent_name: (controller_name, gain_type)}
        """
        agent_to_controller = {}
        
        # Construir mapping inverso: var_obj → controller_name
        var_obj_to_controller = {}
        for controller_name in self.controller_names:
            # Extraer var_obj desde config del controlador (name_objective_var)
            controller_config = self.config_main['controller_base']['controllers'][controller_name]
            var_obj = controller_config['params']['name_objective_var']
            var_obj_to_controller[var_obj] = controller_name
        
        # Para cada agente habilitado, derivar controller_name y gain_type
        agents_config = self.config_main['agent_base']['agent_config']['agents']
        for agent_name, agent_cfg in agents_config.items():
            if not agent_cfg['enabled_agent']:
                continue
            
            # El agent_name sigue el patrón: {gain_type}_{var_obj}
            parts = agent_name.split('_', 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Enabled agent '{agent_name}' must follow the canonical name "
                    "{gain_type}_{var_obj}"
                )
            
            gain_type, var_obj = parts
            if var_obj not in var_obj_to_controller:
                raise ValueError(
                    f"Enabled agent '{agent_name}' references var_obj '{var_obj}', "
                    "but no controller declares that objective variable"
                )

            controller_name = var_obj_to_controller[var_obj]
            controller_config = self.config_main['controller_base']['controllers'][controller_name]
            if gain_type not in controller_config['initial_conditions']:
                raise ValueError(
                    f"Enabled agent '{agent_name}' references gain '{gain_type}', "
                    f"but controller '{controller_name}' does not declare it"
                )
            agent_to_controller[agent_name] = (controller_name, gain_type)
        
        return agent_to_controller
