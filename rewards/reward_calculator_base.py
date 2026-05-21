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
        self.reward_composition_config = self.reward_config.get('reward_composition', {})
        self.reward_composition_enabled = self.reward_composition_config.get('enabled', False)
        self.reward_composition_strict = self.reward_composition_config.get('strict_weight_sum', True)
        
        # Construir mapeo agent_name → var_obj desde config
        self.agent_to_var_obj_map = self._build_agent_to_var_obj_map()
        
        # Instanciar principal_reward_impl dinámicamente
        self.principal_reward_impl = self._instantiate_principal_reward()
        
        # Instanciar extra_rewards_handler dinámicamente
        self.extra_rewards_handler = self._instantiate_extra_rewards()

        # Instanciar coordination_reward_handler dinámicamente
        self.coordination_reward_handler = self._instantiate_coordination_reward()
        self.reward_block_weights = self._build_reward_block_weights()
        
        # Precomputar pesos individuales si la estrategia es individual_reward
        self.agent_individual_weights = self._build_agent_individual_weights()

        self._last_principal_record = {}
        self._last_extra_record = {}
        self._last_coordination_record = {}
        self._last_reward_composition_record = {}
        self._last_global_interval_reward = 0.0
        self._last_assign_internal_reward_dict = {}
    
    def _build_agent_individual_weights(self):
        """
        Extrae y almacena los pesos individuales por agente y variable para acceso O(1) en ejecución.
        """
        weights = {}
        if self.reward_approach != 'individual_reward':
            return weights
            
        method = self.principal_reward_impl.method if self.principal_reward_impl else 'lineal_combination'
        individual_params = self.reward_config['individual_reward_params']
        
        if method == 'weighted_exponential':
            params = individual_params.get('weighted_exponential_params', {})
        else:
            params = individual_params.get('lineal_combination_params', {})
            
        for agent_name, var_obj in self.agent_to_var_obj_map.items():
            if var_obj in params and agent_name in params[var_obj]:
                weights[agent_name] = params[var_obj][agent_name]
            else:
                weights[agent_name] = None
        
        return weights
    
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
        
        return reward_class(self.config_main)
    
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

    def _instantiate_coordination_reward(self):
        """
        Instancia dinámicamente el manejador de reward coordinativo.

        Returns:
            object: Instancia del coordination_reward_handler
        """
        coordination_config = self.reward_calculation_config['coordination_reward']

        if not coordination_config['enabled']:
            return None

        module_path = coordination_config['module_path']
        class_name = coordination_config['class_name']

        module = importlib.import_module(module_path)
        handler_class = getattr(module, class_name)

        return handler_class(coordination_config)

    def _build_reward_block_weights(self):
        """
        Construye los pesos de composicion global por bloque de recompensa.
        """
        if not self.reward_composition_enabled:
            return {
                'principal_reward': 1.0,
                'extra_rewards': 1.0,
                'coordination_reward': 1.0
            }

        raw_weights = self.reward_composition_config.get('block_weights', {})
        weights = {
            'principal_reward': raw_weights.get('principal_reward', 0.0),
            'extra_rewards': raw_weights.get('extra_rewards', 0.0),
            'coordination_reward': raw_weights.get('coordination_reward', 0.0)
        }

        for block_name, weight in weights.items():
            if weight < 0.0:
                raise ValueError(f"Reward block weight must be non-negative: {block_name}={weight}")
            if weight > 0.0 and not self._is_reward_block_enabled(block_name):
                raise ValueError(f"Reward block {block_name} is disabled but has weight {weight}")

        weight_sum = sum(weights.values())
        if self.reward_composition_strict and abs(weight_sum - 1.0) > 1.0e-9:
            raise ValueError(f"Reward block weights must sum 1.0, got {weight_sum}")

        if weight_sum > 0.0 and not self.reward_composition_strict:
            weights = {
                block_name: weight / weight_sum
                for block_name, weight in weights.items()
            }

        return weights

    def _is_reward_block_enabled(self, block_name):
        """
        Consulta declarativa de activacion de cada bloque.
        """
        if block_name == 'principal_reward':
            return self.reward_calculation_config['principal_reward']['enabled']
        if block_name == 'extra_rewards':
            return self.reward_calculation_config['extra_rewards']['enabled']
        if block_name == 'coordination_reward':
            return self.reward_calculation_config['coordination_reward']['enabled']
        return False

    def _compose_interval_reward(self, principal_reward, extra_reward, coordination_reward):
        """
        Compone la recompensa global del intervalo con pesos declarativos.
        """
        raw_values = {
            'principal_reward': principal_reward,
            'extra_rewards': extra_reward,
            'coordination_reward': coordination_reward
        }

        weighted_values = {}
        for block_name, raw_value in raw_values.items():
            block_weight = self.reward_block_weights[block_name]
            weighted_values[block_name] = block_weight * raw_value

        self._last_reward_composition_record = {
            'reward_composition_enabled': float(self.reward_composition_enabled),
            'reward_block_weight_sum': sum(self.reward_block_weights.values()),
            'reward_block_weight_principal_reward': self.reward_block_weights['principal_reward'],
            'reward_block_weight_extra_rewards': self.reward_block_weights['extra_rewards'],
            'reward_block_weight_coordination_reward': self.reward_block_weights['coordination_reward'],
            'reward_raw_principal_reward': principal_reward,
            'reward_raw_extra_rewards': extra_reward,
            'reward_raw_coordination_reward': coordination_reward,
            'reward_weighted_principal_reward': weighted_values['principal_reward'],
            'reward_weighted_extra_rewards': weighted_values['extra_rewards'],
            'reward_weighted_coordination_reward': weighted_values['coordination_reward']
        }
        return sum(weighted_values.values())

    def _compose_loop_reward(self, principal_reward, extra_reward, coordination_reward):
        """
        Compone una senal de aprendizaje por lazo/agente.
        """
        return (
            self.reward_block_weights['principal_reward'] * principal_reward
            + self.reward_block_weights['extra_rewards'] * extra_reward
            + self.reward_block_weights['coordination_reward'] * coordination_reward
        )

    def reset_episode(self):
        """
        Resetea el calculador al inicio de un episodio.
        """
        self._last_principal_record = {}
        self._last_extra_record = {}
        self._last_coordination_record = {}
        self._last_reward_composition_record = {}
        self._last_global_interval_reward = 0.0
        self._last_assign_internal_reward_dict = {}

        if self.principal_reward_impl:
            self.principal_reward_impl.reset_episode()
        if self.extra_rewards_handler:
            self.extra_rewards_handler.reset_episode()
        if self.coordination_reward_handler:
            self.coordination_reward_handler.reset_episode()
    
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
        extra_rewards_by_var = {}
        self._last_extra_record = {}
        coordination_reward = 0.0
        coordination_rewards_by_var = {}
        self._last_coordination_record = {}
        
        # 1. Calcular recompensa principal        
        if self.principal_reward_impl:
            principal_reward, self._last_principal_record, controller_rewards = self.principal_reward_impl.compute_reward(reward_component)
        
        # 2. Calcular extra rewards (pasando termination_flag y current_time_sec)
        if self.extra_rewards_handler:
            extra_reward, self._last_extra_record = self.extra_rewards_handler.evaluate(extra_reward_component, reward_component, termination_flag, current_time_sec)
            for agent_name, var_obj in self.agent_to_var_obj_map.items():
                extra_total_key = f'extra_total_{var_obj}'
                if extra_total_key in self._last_extra_record:
                    extra_rewards_by_var[var_obj] = self._last_extra_record[extra_total_key]

        # 3. Calcular reward coordinativo
        if self.coordination_reward_handler:
            coordination_reward, self._last_coordination_record, coordination_rewards_by_var = (
                self.coordination_reward_handler.evaluate(reward_component, extra_reward_component)
            )
        
        # 4. Calcular global_interval_reward
        self._last_global_interval_reward = self._compose_interval_reward(
            principal_reward,
            extra_reward,
            coordination_reward
        )
        
        # 5. Asignar recompensas a nombres de agentes según reward_approach
        #    La recompensa base de cada agente sale de su lazo, y los extras
        #    actúan como complemento global sobre esa señal de aprendizaje.
        self._last_assign_internal_reward_dict = self._assign_rewards(
            self._last_global_interval_reward,
            controller_rewards,
            extra_rewards_by_var,
            coordination_rewards_by_var,
            reward_component
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

        # Coordination record
        records.update(self._last_coordination_record)

        # Reward composition record
        records.update(self._last_reward_composition_record)
        
        return records
    
    def get_episode_summary_rewards(self):
        """
        Retorna resumen de rewards acumulados del episodio.
        Encapsula acceso a internos de extra_rewards_handler.
        
        Returns:
            dict: {accumulated_band_bonus, accumulated_band_bonus_<var_obj>, goal_bonus}
        """
        accumulated_band_bonus = 0.0
        accumulated_band_bonus_by_var = {}
        goal_bonus = 0.0
        
        if self.extra_rewards_handler:
            for var_obj, bonus_val in self.extra_rewards_handler.accumulated_band_bonus.items():
                accumulated_band_bonus_by_var[f'accumulated_band_bonus_{var_obj}'] = bonus_val
                accumulated_band_bonus += bonus_val
            last_record = self.extra_rewards_handler.last_extra_reward_params_record
            for key, value in last_record.items():
                if key.startswith('extra_bonus_goal_'):
                    goal_bonus = value
                    break
        
        summary = {
            'accumulated_band_bonus': accumulated_band_bonus,
            'goal_bonus': goal_bonus
        }
        summary.update(accumulated_band_bonus_by_var)
        
        return summary
    
    def _assign_rewards(self, global_reward, controller_rewards, extra_rewards_by_var, coordination_rewards_by_var, flat_reward_component):
        """
        Asigna recompensas a agentes según el reward_approach.
        
        Args:
            global_reward (float): Recompensa global (principal + extras)
            controller_rewards (dict): {var_obj: principal_reward_por_lazo}
            extra_rewards_by_var (dict): {var_obj: extra_reward_por_lazo}
            coordination_rewards_by_var (dict): {var_obj: reward_coordinativo_por_lazo}
            flat_reward_component (dict): Componentes de recompensa aplanados
            
        Returns:
            dict: {agent_name: reward}
        """
        assign_dict = {}
        
        if self.reward_approach == 'unique_for_all_reward':
            # Todos los agentes reciben la misma recompensa global
            for agent_name in self.agent_to_var_obj_map.keys():
                assign_dict[agent_name] = global_reward
        
        elif self.reward_approach == 'controller_reward':
            # Cada agente recibe la recompensa de su lazo
            # más los extras globales del intervalo y el bonus coordinativo.
            for agent_name, var_obj in self.agent_to_var_obj_map.items():
                loop_extra_reward = 0.0
                if var_obj in extra_rewards_by_var:
                    loop_extra_reward = extra_rewards_by_var[var_obj]
                loop_coordination_reward = 0.0
                if var_obj in coordination_rewards_by_var:
                    loop_coordination_reward = coordination_rewards_by_var[var_obj]
                if var_obj in controller_rewards:
                    assign_dict[agent_name] = self._compose_loop_reward(
                        controller_rewards[var_obj],
                        loop_extra_reward,
                        loop_coordination_reward
                    )
                else:
                    assign_dict[agent_name] = self._compose_loop_reward(
                        0.0,
                        loop_extra_reward,
                        loop_coordination_reward
                    )
        
        elif self.reward_approach == 'individual_reward':
            # Delegar la matemática y extracción a la implementación específica,
            # manteniendo los extras y la coordinación como complemento global.
            for agent_name, var_obj in self.agent_to_var_obj_map.items():
                agent_weights = self.agent_individual_weights[agent_name]
                loop_extra_reward = 0.0
                if var_obj in extra_rewards_by_var:
                    loop_extra_reward = extra_rewards_by_var[var_obj]
                loop_coordination_reward = 0.0
                if var_obj in coordination_rewards_by_var:
                    loop_coordination_reward = coordination_rewards_by_var[var_obj]
                
                # Fallback a controller_reward si faltan pesos o no existe un principal_impl
                if not agent_weights or not self.principal_reward_impl:
                    assign_dict[agent_name] = self._compose_loop_reward(
                        controller_rewards[var_obj],
                        loop_extra_reward,
                        loop_coordination_reward
                    )
                else:
                    individual_principal_reward = self.principal_reward_impl.compute_agent_individual_reward(
                        flat_reward_component, var_obj, agent_weights
                    )
                    assign_dict[agent_name] = self._compose_loop_reward(
                        individual_principal_reward,
                        loop_extra_reward,
                        loop_coordination_reward
                    )
        
        else:
            # Fallback: todos reciben global
            for agent_name in self.agent_to_var_obj_map.keys():
                assign_dict[agent_name] = global_reward
        
        return assign_dict

