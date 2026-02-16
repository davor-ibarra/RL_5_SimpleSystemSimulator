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
            print(f"\n[SIMULATION_MANAGER] --- T_max = {self.current_time_sec}  |  Total Reward = {self.total_reward}  |  Termination Reason = {self.termination_reason} ---")
        
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
        current_controller_gains = self._get_controller_gains_dict()
        self.prev_agent_state = self.agent_base.build_agent_state(self.prev_dynamic_system_state_dict, current_controller_gains)
        
        # 4. Construir prev_actions_dict inicial (todas las acciones en "mantener" = 1)
        self.prev_actions_dict = self._build_initial_actions_dict()
        
        # 5. Inicializar buffers del episodio en el colector (captura t=0)
        self.metric_collector.on_episode_start(episode_id)
        
        # 6. Loop de intervalos hasta término por tiempo o condición del sistema
        terminated = False
        self.termination_reason = ""
        self.total_reward = 0.0
        total_agent_decisions = 0
        step_idx_global = 0
        
        while self.current_time_sec < self.episode_duration_sec and not terminated:
            # 6.1. Decide acciones del intervalo
            actions_dict = self.agent_base.select_action(self.prev_agent_state)
            
            # 6.2. Aplica acciones a controladores
            self._apply_actions_to_controllers(actions_dict)
            
            # 6.3. Ejecuta la simulación del intervalo y obtiene productos
            interval_result = self._run_interval(decision_id, 
                self.current_time_sec, 
                actions_dict, 
                self.prev_agent_state, 
                self.prev_dynamic_system_state_norm_dict
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
            step_idx_global += n_steps_executed
            self.current_time_sec += n_steps_executed * self.dt_sec
            decision_id += 1
            total_agent_decisions += 1
            self.prev_dynamic_system_state_dict = self.dynamic_system_base.get_dynamic_system_state('raw')
            current_controller_gains = self._get_controller_gains_dict()
            self.prev_agent_state = self.agent_base.build_agent_state(self.prev_dynamic_system_state_dict, current_controller_gains)
            self.prev_actions_dict = actions_dict
        
        # 7. Si terminó por tiempo, asignar razón
        if not terminated:
            self.termination_reason = "time_limit"
        
        # 8. Cierre del episodio: construir end_episode_data y commit a disco
        episode_reward_summary = self.reward_calculator.get_episode_summary_rewards()
        
        end_episode_data = {
            'end_terminated': terminated,
            'end_termination_reason': self.termination_reason,
            'accumulated_band_bonus': episode_reward_summary['accumulated_band_bonus'],
            'goal_bonus': episode_reward_summary['goal_bonus'],
            'total_reward': self.total_reward,
            'total_agent_decisions': total_agent_decisions
        }
        self.metric_collector.on_episode_end(episode_id, end_episode_data)
        
        # 9. Guardar estado del agente según periodicidad
        save_period = self.config_main['data_handling']['agent_state_save_frequency']
        if save_period and (episode_id + 1) % save_period == 0:
            self.result_handler.save_agent_state_learn_dict(self.agent_base.get_agent_state_learn_dict(), episode_id)
    
    def _run_interval(self, decision_id, current_time_sec, actions_dict, prev_agent_state, prev_dynamic_system_state_norm_dict):
        """
        Orquesta un intervalo completo y retorna el contenedor canónico.
        
        Args:
            decision_id (int): Identificador de la decisión
            current_time_sec (float): Tiempo actual en segundos
            actions_dict (dict): Diccionario de acciones del agente
            prev_agent_state: Estado previo del agente
            prev_dynamic_system_state_dict: Estado previo del sistema dinámico
            
        Returns:
            tuple: (interval_data)
        """
        # 1. Inicializar contenedor del intervalo (flat records)
        flat_step_records = []
        n_steps_executed = 0
        
        # 2. Ejecutar el step-loop del intervalo
        for step in range(self.steps_per_interval):
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
        reward_for_learning = self.reward_calculator.calculate(processed_metrics_dict, self.termination_reason, end_time_sec)
        
        # 5. Ejecutar aprendizaje del agente
        current_controller_gains = self._get_controller_gains_dict()
        next_agent_state = self.agent_base.build_agent_state(current_state_norm_dict, current_controller_gains)
        learn_info = self.agent_base.learn(
            prev_agent_state, next_agent_state, actions_dict, reward_for_learning, terminated
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
        
        # 6. Actions: action_<agent_name> → decisions
        vars_decision = actions_dict['vars_decision']
        flat.update(vars_decision)
        
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
        
        Args:
            actions_dict (dict): Diccionario de acciones del agente
        """
        delta_gain = actions_dict['vars_delta']['delta_gain']
        
        # Agrupar nuevas ganancias por controlador usando mapping precomputado
        controller_gains = {}
        for agent_name, (controller_name, gain_type) in self.agent_to_controller_map.items():
            if controller_name not in controller_gains:
                controller_gains[controller_name] = {}
            
            current_value = actions_dict['vars_values'][agent_name]
            action_decision = actions_dict['vars_decision'][f'action_{agent_name}']
            new_value = current_value + (action_decision - 1) * delta_gain
            controller_gains[controller_name][gain_type] = new_value
        
        # Aplicar ganancias a cada controlador
        for controller_name, gains in controller_gains.items():
            self.controllers[controller_name].update_gains(gains['kp'], gains['ki'], gains['kd'])
    
    def _build_initial_actions_dict(self):
        """
        Construye el actions_dict inicial con todas las acciones en "mantener" (índice 1).
        
        Returns:
            dict: actions_dict inicial
        """
        initial_actions = {
            'vars_values': {},
            'vars_decision': {},
        }
        
        # Usar mapping precomputado para obtener valores iniciales
        for agent_name, (controller_name, gain_type) in self.agent_to_controller_map.items():
            controller = self.controllers[controller_name]
            initial_actions['vars_values'][agent_name] = getattr(controller, gain_type)
            initial_actions['vars_decision'][f'action_{agent_name}'] = 1
        
        return initial_actions
    
    def _get_controller_gains_dict(self):
        """
        Construye diccionario de ganancias actuales de controladores.
        Formato: {agent_name: value}
        
        Returns:
            dict: Ganancias por nombre de agente
        """
        gains_dict = {}
        
        for agent_name, (controller_name, gain_type) in self.agent_to_controller_map.items():
            controller = self.controllers[controller_name]
            gains_dict[agent_name] = getattr(controller, gain_type)
        
        return gains_dict
    
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
                continue
            
            gain_type, var_obj = parts
            if var_obj in var_obj_to_controller:
                controller_name = var_obj_to_controller[var_obj]
                agent_to_controller[agent_name] = (controller_name, gain_type)
        
        return agent_to_controller