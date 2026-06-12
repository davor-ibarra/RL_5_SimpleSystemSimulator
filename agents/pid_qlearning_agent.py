"""
pid_qlearning_agent.py

Responsabilidad:
Implementación específica del agente Q-learning para ajuste de ganancias PID.
Contiene toda la lógica de:
- Espacio de estados y discretización
- Q-tables y su inicialización
- Política ε-greedy
- Actualización Q-learning
- Gestión de parámetros de aprendizaje (epsilon, learning_rate)
"""

import numpy as np


class PIDQLearningAgent:
    """
    Implementación específica del agente Q-learning para ajuste de ganancias PID.
    Cada agente corresponde a una ganancia (kp, ki, kd) de un controlador.
    El estado de cada agente se basa en su propia ganancia + state_vars opcionales.
    """
    
    def __init__(self, config_main):
        """
        Inicializa el agente PID Q-learning.
        
        Args:
            config_main (dict): Configuración principal
        """
        self.config_main = config_main
        self.config_agent = config_main['agent_base']
        
        # Parámetros de aprendizaje
        self.discount_factor = self.config_agent['params']['discount_factor']
        self.epsilon = self.config_agent['params']['epsilon']
        self.epsilon_decay_enabled = self.config_agent['params']['epsilon_decay']['enabled']
        self.epsilon_min = self.config_agent['params']['epsilon_decay']['epsilon_min']
        self.epsilon_decay_factor = self.config_agent['params']['epsilon_decay']['decay_factor']
        self.learning_rate = self.config_agent['params']['learning_rate']
        self.learning_rate_decay_enabled = self.config_agent['params']['learning_rate_decay']['enabled']
        self.learning_rate_min = self.config_agent['params']['learning_rate_decay']['learning_rate_min']
        self.learning_rate_decay_factor = self.config_agent['params']['learning_rate_decay']['decay_factor']
        self._is_first_episode = True
        
        # Espacio de acciones (debe estar antes de _build_state_space que usa num_actions)
        self.action_space = self.config_agent['agent_config']['actions']['actions_space']
        self.num_actions = len(self.action_space)
        
        # Per-agent delta_gain desde config (ANTES de _build_state_space que lo necesita)
        actions_values_config = self.config_agent['agent_config']['actions']['actions_values']
        self.actions_values_mode = actions_values_config['mode']
        self.agent_gain_steps = {}
        agents_config_temp = self.config_agent['agent_config']['agents']
        for var_name, var_cfg in agents_config_temp.items():
            if var_cfg['enabled_agent']:
                if self.actions_values_mode == 'universal':
                    self.agent_gain_steps[var_name] = actions_values_config['universal_params']['delta_gain']
                elif self.actions_values_mode == 'per_agent':
                    self.agent_gain_steps[var_name] = actions_values_config['per_agent_params']['delta_gain'][var_name]
        
        # Construir espacio de estados (usa agent_gain_steps para derivar bins)
        self._build_state_space()
        
        # Valor inicial de Q-table
        self.q_init_value = 0.0
        
        # Inicializar Q-tables
        self.q_tables = {}
        self.visit_counts = {}
        self._initialize_q_tables()
    
    def _build_state_space(self):
        """
        Construye diccionarios de discretización desde config.
        Pre-calcula todo lo necesario para acceso directo sin iteraciones en runtime.
        """
        agents_config = self.config_agent['agent_config']['agents']
        
        # Diccionarios de parámetros de discretización por variable
        self.var_names = []
        self.var_to_idx = {}
        self.var_mins = {}
        self.var_maxs = {}
        self.var_bins = {}
        self.var_steps = {}
        
        # Por agente habilitado
        self.agent_names = []
        self.agent_state_vars = {}
        self.required_state_vars = []
        self.agent_q_shapes = {}
        
        # Primera pasada: construir diccionarios de todas las variables
        idx = 0
        for var_name, var_cfg in agents_config.items():
            self.var_names.append(var_name)
            self.var_to_idx[var_name] = idx
            self.var_mins[var_name] = var_cfg['min']
            self.var_maxs[var_name] = var_cfg['max']
            
            # Derivar bins y step: desde delta_gain para agentes, desde config para el resto
            if var_name in self.agent_gain_steps:
                delta_gain = self.agent_gain_steps[var_name]
                n_bins = int(round((var_cfg['max'] - var_cfg['min']) / delta_gain)) + 1
                step = delta_gain
            else:
                n_bins = var_cfg['bins']
                step = (var_cfg['max'] - var_cfg['min']) / (n_bins - 1) if n_bins > 1 else 0.0
            
            self.var_bins[var_name] = n_bins
            self.var_steps[var_name] = step
            idx += 1
        
        # Segunda pasada: por cada agente habilitado
        for var_name, var_cfg in agents_config.items():
            if not var_cfg['enabled_agent']:
                continue
            
            agent_name = var_name
            self.agent_names.append(agent_name)
            
            # Variables de estado: primero la ganancia propia, luego state_vars
            state_vars = [agent_name]
            for sv in var_cfg['state_vars']:
                if sv not in self.var_to_idx:
                    raise ValueError(
                        f"State variable '{sv}' declared for agent '{agent_name}' "
                        "must have a discretization entry in agent_config.agents"
                    )
                if sv != agent_name:
                    state_vars.append(sv)
            
            self.agent_state_vars[agent_name] = state_vars
            for state_var in state_vars:
                if state_var not in self.required_state_vars:
                    self.required_state_vars.append(state_var)
            
            # Shape de Q-table: (bins_var1, bins_var2, ..., num_actions)
            shape = tuple([self.var_bins[sv] for sv in state_vars]) + (self.num_actions,)
            self.agent_q_shapes[agent_name] = shape
    
    def _initialize_q_tables(self):
        """
        Inicializa Q-tables con shapes pre-calculadas.
        """
        for agent_name, shape in self.agent_q_shapes.items():
            self.q_tables[agent_name] = np.full(shape, self.q_init_value, dtype=np.float32)
            self.visit_counts[agent_name] = np.zeros(shape, dtype=np.int32)
    
    def _discretize_value(self, value, var_name):
        """
        Discretiza un valor continuo a un índice de bin.
        
        Args:
            value: Valor a discretizar
            var_name: Nombre de la variable
            
        Returns:
            int or None: Índice del bin, o None si valor inválido
        """
        if not np.isfinite(value):
            return None
        
        var_min = self.var_mins[var_name]
        var_step = self.var_steps[var_name]
        n_bins = self.var_bins[var_name]
        
        clipped = np.clip(value, var_min, var_min + var_step * (n_bins - 1))
        idx = int(round((clipped - var_min) / var_step)) if var_step > 0 else 0
        return max(0, min(idx, n_bins - 1))
    
    def _get_state_indices(self, agent_state, agent_name):
        """
        Discretiza el estado para un agente específico.
        
        Args:
            agent_state (dict): Estado del agente
            agent_name (str): Nombre del agente
            
        Returns:
            tuple or None: Índices discretos, o None si valor inválido
        """
        indices = []
        
        for var_name in self.agent_state_vars[agent_name]:
            value = agent_state[var_name]
            
            if value is None:
                return None
            
            idx = self._discretize_value(value, var_name)
            if idx is None:
                return None
            
            indices.append(idx)
        
        return tuple(indices)
    
    def reset_episode(self):
        """
        Resetea el agente al inicio de un episodio.
        Aplica decay de epsilon y learning_rate (excepto primer episodio).
        """
        if self._is_first_episode:
            self._is_first_episode = False
            return
        
        if self.epsilon_decay_enabled:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_factor)
        
        if self.learning_rate_decay_enabled:
            self.learning_rate = max(self.learning_rate_min, self.learning_rate * self.learning_rate_decay_factor)
    
    def build_agent_state(self, agent_context):
        """
        Recibe un contexto plano y selecciona solo las variables declaradas
        en agent_state_vars. Los nombres faltantes fallan por acceso directo.

        Construye el estado del agente.
        El estado se basa en las ganancias de los controladores + variables adicionales.
        
        Args:
            dynamic_state_dict (dict): Estado del sistema dinámico
            controller_gains_dict (dict): Ganancias actuales {agent_name: value}
            
        Returns:
            dict: Estado del agente con todas las variables necesarias
        """
        return {
            var_name: agent_context[var_name]
            for var_name in self.required_state_vars
        }
    
    def select_action(self, agent_state):
        """
        Selecciona acciones usando política ε-greedy.
        
        Args:
            agent_state (dict): Estado del agente
            
        Returns:
            dict: actions_dict con formato canónico
        """
        actions_dict = {
            'actions_applied': False,
            'vars_values': {},
            'vars_decision': {},
            'vars_delta': {
                'actions_applied': 0.0
            }
        }
        
        for agent_name in self.agent_names:
            current_gain_value = agent_state[agent_name]
            actions_dict['vars_values'][agent_name] = current_gain_value
            
            state_indices = self._get_state_indices(agent_state, agent_name)
            
            if state_indices is None:
                # Estado inválido: explorar
                action_idx = np.random.randint(0, self.num_actions)
            elif np.random.rand() < self.epsilon:
                # Exploración
                action_idx = np.random.randint(0, self.num_actions)
            else:
                # Explotación: seleccionar mejor acción
                q_values = self.q_tables[agent_name][state_indices]
                max_q = np.max(q_values)
                best_actions = np.where(np.isclose(q_values, max_q))[0]
                action_idx = int(np.random.choice(best_actions))
            
            actions_dict['vars_decision'][f'action_{agent_name}'] = action_idx
            actions_dict['vars_delta'][f'delta_gain_{agent_name}'] = self.agent_gain_steps[agent_name]
            actions_dict['vars_delta'][f'delta_gain_requested_{agent_name}'] = 0.0
            actions_dict['vars_delta'][f'delta_gain_applied_{agent_name}'] = 0.0
            actions_dict['vars_delta'][f'action_requested_move_{agent_name}'] = 0.0
            actions_dict['vars_delta'][f'action_blocked_{agent_name}'] = 0.0
            actions_dict['vars_delta'][f'action_blocked_fraction_{agent_name}'] = 0.0
        
        return actions_dict
    
    def learn(self, prev_agent_state, next_agent_state, actions_dict, reward_for_learning, terminated):
        """
        Actualiza Q-tables usando Q-learning.
        
        Args:
            prev_agent_state (dict): Estado previo
            next_agent_state (dict): Estado siguiente
            actions_dict (dict): Acciones tomadas
            reward_for_learning (dict): {agent_name: reward} — recompensas por agente
            terminated (bool): Si el episodio terminó
            
        Returns:
            dict: learn_info con métricas de aprendizaje
        """
        learn_info = {}
        
        for agent_name in self.agent_names:
            s_indices = self._get_state_indices(prev_agent_state, agent_name)
            s_prime_indices = self._get_state_indices(next_agent_state, agent_name)
            
            if s_indices is None:
                continue
            
            action_idx = actions_dict['vars_decision'][f'action_{agent_name}']
            reward = self._get_reward_for_agent(agent_name, reward_for_learning)
            
            q_table = self.q_tables[agent_name]
            current_q = q_table[s_indices + (action_idx,)]
            
            # Calcular max Q(s', a')
            max_q_next = 0.0
            if not terminated and s_prime_indices is not None:
                max_q_next = np.max(q_table[s_prime_indices])
            
            # Actualización Q-learning inline
            target_q = reward if terminated else reward + self.discount_factor * max_q_next
            td_error = target_q - current_q
            new_q = current_q + self.learning_rate * td_error
            
            q_table[s_indices + (action_idx,)] = new_q
            self.visit_counts[agent_name][s_indices + (action_idx,)] += 1
            
            learn_info[f'td_error_{agent_name}'] = float(td_error)
            learn_info[f'q_value_{agent_name}'] = float(new_q)
        
        return learn_info
    
    def _get_reward_for_agent(self, agent_name, reward_for_learning):
        """
        Obtiene la recompensa para un agente desde assign_internal_reward_dict.
        RewardCalculatorBase ya resuelve la asignación según el approach configurado.
        
        Args:
            agent_name (str): Nombre del agente
            reward_for_learning (dict): Recompensa para el agente
            
        Returns:
            float: Recompensa para el agente
        """
        return reward_for_learning[agent_name]
    
    def get_records(self):
        """
        Expone parámetros de telemetría ligera del agente (estadísticas agregadas).
        Solo Q-table stats a nivel de intervalo.
        Para epsilon/learning_rate usar get_end_episode_records().
        
        Returns:
            dict: Parámetros agregados del agente (interval-level)
        """
        params = {}
        
        for agent_name in self.agent_names:
            q_table = self.q_tables[agent_name]
            visit_count = self.visit_counts[agent_name]
            
            params[f'q_mean_{agent_name}'] = float(np.mean(q_table))
            params[f'q_std_{agent_name}'] = float(np.std(q_table))
            params[f'q_max_{agent_name}'] = float(np.max(q_table))
            params[f'q_min_{agent_name}'] = float(np.min(q_table))
            params[f'visits_total_{agent_name}'] = int(np.sum(visit_count))
            params[f'visits_coverage_{agent_name}'] = float(np.mean(visit_count > 0))
        
        return params
    
    def get_end_episode_records(self):
        """
        Expone parámetros del agente que cambian solo por episodio.
        Diseñado para fusionarse con end_episode_data en SimulationManager.
        
        Returns:
            dict: Parámetros episode-level del agente
        """
        return {
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
        }
    
    def get_agent_state_learn_dict(self):
        """
        Expone Q-tables y visit_counts por agentes.
        
        Returns:
            dict: Estado serializable con:
                - q_tables: {agent_name: Q-table}
                - visit_counts: {agent_name: matriz de visitas}
                - metadata de discretizacion para persistencia tabular
        """
        # Q-tables y visit_counts en formato serializable (listas)
        q_tables_matrix = {}
        visit_counts_matrix = {}
        
        for agent_name in self.agent_names:
            q_tables_matrix[agent_name] = self.q_tables[agent_name]
            visit_counts_matrix[agent_name] = self.visit_counts[agent_name]
        
        return {
            'q_tables': q_tables_matrix,
            'visit_counts': visit_counts_matrix,
            'agent_state_vars': self.agent_state_vars,
            'required_state_vars': self.required_state_vars,
            'var_mins': self.var_mins,
            'var_steps': self.var_steps,
            'action_space': self.action_space,
            }
