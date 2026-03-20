"""
lagrange_reward_calculator.py

Responsabilidad:
Calcular recompensa lagrangiana desde reward_component.
r = -(α·L_e + β·L_edot + γ·L_I + η·L_delta_u)

Esta clase es un reward_impl puro, sin responsabilidad de asignación a agentes.
"""

import math

class LagrangeRewardCalculator:
    """
    Calculador de recompensa basado en formulación lagrangiana.
    
    Consume reward_component del MetricProcessing:
    {
        "pendulum_angle": {L_e, L_edot, L_I, L_u, L_delta_u},
        "cart_position": {...}
    }
    
    Retorna: (principal_reward, breakdown, controller_rewards)
    """
    
    def __init__(self, config_main):
        """
        Inicializa el calculador lagrangiano.
        
        Args:
            config_main (dict): Configuración global para extraer todas las dependencias directo.
        """
        self.config_main = config_main
        self.config = config_main['reward_base']['reward_calculation']['principal_reward']
        self.method = self.config['method']
        
        # Load params depending on method check
        if self.method == 'lineal_combination':
            self.config_lineal_combination = self.config['lineal_combination_params']
        elif self.method == 'weighted_exponential':
            self.config_weighted_exponential = self.config['weighted_exponential_params']
        
        # Extraer pesos según método
        if self.method == 'lineal_combination':
            self.weights = self._extract_lineal_weights()
        elif self.method == 'weighted_exponential':
            self.weights = self._extract_exponential_weights()
        else:
            self.weights = {}
            
        # Extraer var_objs desde controllers para evitar loop condicional posterior
        self.var_objs = self._extract_var_objs()
        
        # Mapping para agent individual weights (e.g. L_e -> w_e, L_delta_u -> w_s)
        self.feature_to_agent_weight_key = {
            'L_e': 'w_e',
            'L_edot': 'w_edot',
            'L_I': 'w_I',
            'L_u': 'w_u',
            'L_delta_u': 'w_s'
        }
        
        # Precompilar trabajos matemáticos
        self._compile_jobs()
        self._compile_agent_jobs()
        
        # Último reward_params_record (para auditoría)
        self.last_reward_params_record = {}
        
    def _compile_jobs(self):
        """
        Pre-construye las listas de ejecución matemática (tuplas estáticas) para evitar bucles de string e if-checks.
        """
        self.var_jobs = {}
        for var_obj in self.var_objs:
            self.var_jobs[var_obj] = []
            for feature_name, cfg in self.weights.items():
                flat_key = f"{feature_name}_{var_obj}"
                if isinstance(cfg, dict):
                    w, s, sp = cfg.get('weight', 0.0), cfg.get('scaled', 1.0), cfg.get('setpoint', 0.0)
                else:
                    w, s, sp = cfg, 1.0, 0.0
                self.var_jobs[var_obj].append((flat_key, feature_name, w, s, sp))
                
        self.global_jobs = []
        for feature_name, cfg in self.weights.items():
            if isinstance(cfg, dict):
                w, s, sp = cfg.get('weight', 0.0), cfg.get('scaled', 1.0), cfg.get('setpoint', 0.0)
            else:
                w, s, sp = cfg, 1.0, 0.0
            self.global_jobs.append((feature_name, w, s, sp))
            
    def _compile_agent_jobs(self):
        """
        Pre-construye las llaves referenciales del framework para `compute_agent_individual_reward`.
        """
        self.agent_feature_jobs = []
        for feature_name, weight_key in self.feature_to_agent_weight_key.items():
            cfg = self.weights.get(feature_name, {})
            if isinstance(cfg, dict):
                s, sp = cfg.get('scaled', 1.0), cfg.get('setpoint', 0.0)
            else:
                s, sp = 1.0, 0.0
            self.agent_feature_jobs.append((feature_name, weight_key, s, sp))
        
    def _extract_var_objs(self):
        """ Extrae las variables objetivo directamente de la configuración de controladores. """
        var_objs = []
        controllers_config = self.config_main['controller_base']['controllers']
        for ctrl_cfg in controllers_config.values():
            var_objs.append(ctrl_cfg['params']['name_objective_var'])
        return var_objs
    
    def _extract_lineal_weights(self):
        """
        Extrae pesos para combinación lineal desde config.
        Usa llave estándar 'weight' para cada feature.
        
        Returns:
            dict: {feature_name: weight}
        """
        features = self.config_lineal_combination['features']
        
        weights = {}
        for feature_name, feature_config in features.items():
            # Usar llave estándar 'weight'
            weights[feature_name] = feature_config['weight']
        
        return weights
    
    def _extract_exponential_weights(self):
        """
        Extrae pesos para método exponencial desde config_main.
        
        Returns:
            dict: {feature_name: {weight, scaled, setpoint}}
        """
        features = self.config_weighted_exponential['features']
        return features
    
    def reset_episode(self):
        """
        Resetea el calculador al inicio de un episodio.
        """
        self.last_reward_params_record = {}
    
    def compute_reward(self, flat_reward_component):
        """
        Calcula la recompensa lagrangiana desde el componente aplanado de métricas.
        
        Args:
            flat_reward_component (dict): Estructura {L_e_<var_obj>: val, ...}
            
        Returns:
            tuple: (principal_reward, reward_params_record, controller_rewards)
                - principal_reward: float (recompensa global)
                - reward_params_record: dict (desglose de componentes)
                - controller_rewards: dict (recompensas por controlador)
        """
        if self.method == 'lineal_combination':
            return self._compute_lineal(flat_reward_component)
        elif self.method == 'weighted_exponential':
            return self._compute_exponential(flat_reward_component)
        else:
            return 0.0, {}, {}
    
    def _compute_lineal(self, flat_reward_component):
        """
        Calcula recompensa usando combinación lineal iterando la lista compilada.
        """
        aggregated_L = {feature: 0.0 for feature in self.weights.keys()}
        controller_rewards = {}
        
        for var_obj, jobs in self.var_jobs.items():
            lazo_lagrangian = 0.0
            for flat_key, feature_name, w, _, _ in jobs:
                # Acceso dict ultra-veloz O(1) vía llave precompilada
                value = flat_reward_component[flat_key]
                aggregated_L[feature_name] += value
                lazo_lagrangian += w * value
            
            # Recompensa del lazo = -lagrangiana
            controller_rewards[var_obj] = -lazo_lagrangian
        
        global_lagrangian = 0.0
        for feature_name, w, _, _ in self.global_jobs:
            if feature_name in flat_reward_component:
                value = flat_reward_component[feature_name]
                aggregated_L[feature_name] += value
                global_lagrangian += w * value
        
        if global_lagrangian != 0.0:
            for var_obj in controller_rewards:
                controller_rewards[var_obj] -= global_lagrangian
        
        lagrangian_total = sum(
            self.weights[feature] * value 
            for feature, value in aggregated_L.items()
        )
        
        principal_reward = -lagrangian_total
        
        # Breakdown
        reward_params_record = {
            'principal_reward': principal_reward,
            'controller_rewards': controller_rewards,
            'lagrangian_total': lagrangian_total,
            **aggregated_L
        }
        
        self.last_reward_params_record = reward_params_record
        
        return principal_reward, reward_params_record, controller_rewards
    
    def _compute_exponential(self, flat_reward_component):
        """
        Calcula recompensa usando método exponencial iterando las tuplas compiladas.
        """
        aggregated_L = {feature: 0.0 for feature in self.weights.keys()}
        controller_rewards = {}
        
        for var_obj, jobs in self.var_jobs.items():
            lazo_reward = 0.0
            for flat_key, feature_name, w, s, sp in jobs:
                value = flat_reward_component[flat_key]
                aggregated_L[feature_name] += value
                
                exp_term = math.exp(-s * (value - sp) ** 2)
                lazo_reward -= w * (1 - exp_term)
            
            controller_rewards[var_obj] = lazo_reward
            
        global_lagrangian = 0.0
        for feature_name, w, s, sp in self.global_jobs:
            if feature_name in flat_reward_component:
                value = flat_reward_component[feature_name]
                aggregated_L[feature_name] += value
                
                exp_term = math.exp(-s * (value - sp) ** 2)
                global_lagrangian += w * (1 - exp_term)
        
        if global_lagrangian != 0.0:
            for var_obj in controller_rewards:
                controller_rewards[var_obj] -= global_lagrangian
        
        principal_reward = 0.0
        reward_params_record = {}
        
        for feature_name, w, s, sp in self.global_jobs:
            value = aggregated_L.get(feature_name, 0.0)
            
            exp_term = math.exp(-s * (value - sp) ** 2)
            contribution = -w * (1 - exp_term)
            
            principal_reward += contribution
            reward_params_record[feature_name] = {
                'value': value,
                'exp_term': exp_term,
                'contribution': contribution
            }
        
        reward_params_record['principal_reward'] = principal_reward
        reward_params_record['controller_rewards'] = controller_rewards
        self.last_reward_params_record = reward_params_record
        
        return principal_reward, reward_params_record, controller_rewards
    
    def compute_agent_individual_reward(self, flat_reward_component, var_obj, agent_weights):
        """
        Calcula la recompensa individual para un agente delegando la matemática aquí.
        """
        agent_reward = 0.0
        
        if self.method == 'weighted_exponential':
            for feature_name, weight_key, s, sp in self.agent_feature_jobs:
                flat_key = f"{feature_name}_{var_obj}"
                if weight_key in agent_weights:
                    value = flat_reward_component[flat_key]
                    w = agent_weights[weight_key]
                    exp_term = math.exp(-s * (value - sp) ** 2)
                    agent_reward -= w * (1 - exp_term)
                    
        else: # Default a lineal logic
            for feature_name, weight_key, _, _ in self.agent_feature_jobs:
                flat_key = f"{feature_name}_{var_obj}"
                if weight_key in agent_weights:
                    value = flat_reward_component[flat_key]
                    agent_reward -= agent_weights[weight_key] * value
                    
        return agent_reward
    
    def get_reward_params_record(self):
        """
        Retorna el desglose de la última recompensa calculada.
        
        Returns:
            dict: Desglose de componentes
        """
        return self.last_reward_params_record.copy()
