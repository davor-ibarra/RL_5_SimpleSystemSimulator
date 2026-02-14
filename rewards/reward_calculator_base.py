"""
reward_calculator_base.py

Responsabilidad:
Orquestador del sistema de recompensas.
- Instancia dinámicamente principal_reward_impl y extra_rewards_handler
- Orquesta: principal + extras = global_interval_reward
- Asigna recompensas a agentes según reward_approach
"""

import importlib


class RewardCalculatorBase:
    """
    Orquestador del sistema de recompensas.
    
    Instancia dinámicamente:
    - principal_reward_impl (e.g., LagrangeRewardCalculator)
    - extra_rewards_handler (ExtraRewardsHandler)
    
    Orquesta el cálculo de recompensas y asignación a agentes.
    """
    
    def __init__(self, config_main):
        """
        Inicializa el orquestador con configuración principal.
        
        Args:
            config_main (dict): Configuración principal completa
        """
        self.config_main = config_main
        self.reward_base_config = config_main['reward_base']
        self.reward_config = self.reward_base_config['reward_config']
        self.reward_calculation_config = self.reward_base_config['reward_calculation']
        
        # Modo de asignación de recompensa
        self.reward_approach = self.reward_config['reward_approach']
        
        # Construir mapeo agent_name → var_obj desde config
        self.agent_to_var_obj_map = self._build_agent_to_var_obj_map()
        
        # Instanciar principal_reward_impl dinámicamente
        self.principal_reward_impl = self._instantiate_principal_reward()
        
        # Instanciar extra_rewards_handler dinámicamente
        self.extra_rewards_handler = self._instantiate_extra_rewards()
    
    def _build_agent_to_var_obj_map(self):
        """
        Construye mapeo agent_name → var_obj desde config.
        Solo incluye agentes habilitados (enabled_agent: true).
        
        Returns:
            dict: {agent_name: var_obj}
        """
        mapping = {}
        
        # Obtener config de controladores
        controllers_config = self.config_main['controller_base']['controllers']
        
        # Para cada controlador, extraer var_obj
        var_obj_to_controller = {}
        for controller_name, controller_cfg in controllers_config.items():
            var_obj = controller_cfg['params']['name_objective_var']
            var_obj_to_controller[var_obj] = controller_name
        
        # Obtener config de agentes
        agents_config = self.config_main['agent_base']['agent_config']['agents']
        
        # Mapear solo agentes habilitados a su var_obj
        for agent_name, agent_cfg in agents_config.items():
            # Filtrar agentes deshabilitados
            if not agent_cfg['enabled_agent']:
                continue
            
            # Buscar var_obj que coincida con el agent_name
            for var_obj in var_obj_to_controller.keys():
                if var_obj in agent_name:
                    mapping[agent_name] = var_obj
                    break
        
        return mapping
    
    def _instantiate_principal_reward(self):
        """
        Instancia dinámicamente el calculador de recompensa principal.
        
        Returns:
            object: Instancia del principal_reward_impl
        """
        principal_config = self.reward_calculation_config['principal_reward']
        
        if not principal_config['enabled']:
            return None
        
        module_path = principal_config['module_path']
        class_name = principal_config['class_name']
        
        module = importlib.import_module(module_path)
        reward_class = getattr(module, class_name)
        
        return reward_class(principal_config)
    
    def _instantiate_extra_rewards(self):
        """
        Instancia dinámicamente el manejador de extra rewards.
        
        Returns:
            object: Instancia del extra_rewards_handler
        """
        extra_config = self.reward_calculation_config['extra_rewards']
        
        if not extra_config['enabled']:
            return None
        
        module_path = extra_config['module_path']
        class_name = extra_config['class_name']
        
        module = importlib.import_module(module_path)
        handler_class = getattr(module, class_name)
        
        return handler_class(extra_config)
    
    def reset_episode(self):
        """
        Resetea el calculador al inicio de un episodio.
        """
        if self.principal_reward_impl:
            self.principal_reward_impl.reset_episode()
        if self.extra_rewards_handler:
            self.extra_rewards_handler.reset_episode()
    
    def calculate(self, processed_metrics_dict, termination_flag='unknown', current_time_sec=0.0):
        """
        Calcula la recompensa del intervalo y retorna solo las recompensas 
        asignadas a cada agente, listas para el aprendizaje.
        
        Args:
            processed_metrics_dict (dict): Métricas procesadas con estructura:
                - reward_component: {var_obj: {L_e, L_edot, ...}}
                - extra_reward_component: {error_<var_obj>: [serie], ...}
                - metrics_info: {}
            termination_flag (str): Flag de terminación del episodio
            current_time_sec (float): Tiempo actual en el episodio [s]
            
        Returns:
            dict: {agent_name: reward} — recompensas por agente para aprendizaje
        """
        reward_component = processed_metrics_dict['reward_component']
        extra_reward_component = processed_metrics_dict['extra_reward_component']
        
        # Inicializar defaults (en caso de que algún bloque esté deshabilitado)
        principal_reward = 0.0
        self._last_principal_record = {}
        controller_rewards = {}
        extra_reward = 0.0
        self._last_extra_record = {}
        
        # 1. Calcular recompensa principal        
        if self.principal_reward_impl:
            principal_reward, self._last_principal_record, controller_rewards = self.principal_reward_impl.compute_reward(reward_component)
        
        # 2. Calcular extra rewards (pasando termination_flag y current_time_sec)
        if self.extra_rewards_handler:
            extra_reward, self._last_extra_record = self.extra_rewards_handler.evaluate(extra_reward_component, termination_flag, current_time_sec)
        
        # 3. Calcular global_interval_reward
        self._last_global_interval_reward = principal_reward + extra_reward
        
        # 4. Asignar recompensas a nombres de agentes según reward_approach
        self._last_assign_internal_reward_dict = self._assign_rewards(
            self._last_global_interval_reward, controller_rewards, extra_reward, reward_component
        )
        
        return self._last_assign_internal_reward_dict
    
    def get_records(self):

        """
        Retorna dict plano con TODOS los parámetros de reward para el MetricCollector.
        Recoge y registra los parámetros relevantes de cada subcomponente
        (principal_reward_impl, extra_rewards_handler).
        
        Returns:
            dict: Registro plano interval-level del sistema de recompensas
        """
        records = {}
        
        # Recompensa global
        records['global_interval_reward'] = self._last_global_interval_reward
        
        # Recompensas per-agent
        for agent_name, reward_val in self._last_assign_internal_reward_dict.items():
            records[f'reward_{agent_name}'] = reward_val
        
        # Principal record (aplanar, separando controller_rewards)
        if self._last_principal_record:
            for key, value in self._last_principal_record.items():
                if key == 'controller_rewards':
                    # controller_rewards → L_<var_obj>
                    for var_obj, loss_val in value.items():
                        records[f'L_{var_obj}'] = loss_val
                elif isinstance(value, (int, float, bool, str)):
                    records[key] = value
        
        # Extra record (ya viene plano desde ExtraRewardsHandler)
        records.update(self._last_extra_record)
        
        return records
    
    def get_episode_summary_rewards(self):
        """
        Retorna resumen de rewards acumulados del episodio.
        Encapsula acceso a internos de extra_rewards_handler.
        
        Returns:
            dict: {accumulated_band_bonus, goal_bonus}
        """
        accumulated_band_bonus = 0.0
        goal_bonus = 0.0
        
        if self.extra_rewards_handler:
            accumulated_band_bonus = self.extra_rewards_handler.accumulated_band_bonus
            last_record = self.extra_rewards_handler.last_extra_reward_params_record
            goal_bonus = sum(v for k, v in last_record.items() if k.startswith('extra_bonus_goal_'))
        
        return {
            'accumulated_band_bonus': accumulated_band_bonus,
            'goal_bonus': goal_bonus
        }
    
    def _assign_rewards(self, global_reward, controller_rewards, extra_reward, reward_component):
        """
        Asigna recompensas a agentes según el reward_approach.
        
        Args:
            global_reward (float): Recompensa global (principal + extras)
            controller_rewards (dict): {var_obj: principal_reward_por_lazo}
            extra_reward (float): Extra reward global
            reward_component (dict): Componentes de recompensa para individual_reward
            
        Returns:
            dict: {agent_name: reward}
        """
        assign_dict = {}
        
        if self.reward_approach == 'unique_for_all_reward':
            # Todos los agentes reciben la misma recompensa global
            for agent_name in self.agent_to_var_obj_map.keys():
                assign_dict[agent_name] = global_reward
        
        elif self.reward_approach == 'controller_reward':
            # Cada agente recibe la recompensa de su lazo (sin extras)
            # Todos los agentes de un controlador reciben la misma recompensa
            for agent_name, var_obj in self.agent_to_var_obj_map.items():
                if var_obj in controller_rewards:
                    assign_dict[agent_name] = controller_rewards[var_obj]
                else:
                    assign_dict[agent_name] = 0.0
        
        elif self.reward_approach == 'individual_reward':
            # Cada agente recibe recompensa ponderada según individual_reward_params
            individual_params = self.reward_config['individual_reward_params']
            assign_dict = self._compute_individual_rewards(
                reward_component, controller_rewards, individual_params
            )
        
        else:
            # Fallback: todos reciben global
            for agent_name in self.agent_to_var_obj_map.keys():
                assign_dict[agent_name] = global_reward
        
        return assign_dict
    
    def _compute_individual_rewards(self, reward_component, controller_rewards, individual_params):
        """
        Computa recompensas individuales ponderadas por agente.
        Aplica pesos específicos por agente a cada componente L_* del var_obj.
        
        Fórmula: reward_agent = -Σ(w_feature * L_feature) para features del var_obj.
        
        Args:
            reward_component (dict): {var_obj: {L_e, L_edot, ...}} del proceso de métricas
            controller_rewards (dict): {var_obj: principal_reward_por_lazo} (fallback)
            individual_params (dict): Pesos por agente desde config
            
        Returns:
            dict: {agent_name: reward}
        """
        assign_dict = {}
        lineal_params = individual_params['lineal_combination_params']
        
        # Mapeo feature L_* → llave de peso w_*
        feature_to_weight_key = {
            'L_e': 'w_e',
            'L_edot': 'w_edot',
            'L_I': 'w_I',
            'L_u': 'w_u',
            'L_delta_u': 'w_s'
        }
        
        for agent_name, var_obj in self.agent_to_var_obj_map.items():
            # Obtener pesos individuales del agente desde config
            var_obj_weights = lineal_params[var_obj]
            agent_weights = var_obj_weights[agent_name]
            
            # Obtener features L_* del var_obj
            var_metrics = reward_component[var_obj] if var_obj in reward_component else {}
            
            if not var_metrics:
                # Fallback a recompensa del controlador
                assign_dict[agent_name] = controller_rewards[var_obj] if var_obj in controller_rewards else 0.0
                continue
            
            # Calcular lagrangiana ponderada individual
            agent_lagrangian = 0.0
            for feature_name, weight_key in feature_to_weight_key.items():
                if feature_name in var_metrics and weight_key in agent_weights:
                    agent_lagrangian += agent_weights[weight_key] * var_metrics[feature_name]
            
            # Recompensa = -lagrangiana
            assign_dict[agent_name] = -agent_lagrangian
        
        return assign_dict

