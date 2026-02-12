"""
lagrange_reward_calculator.py

Responsabilidad:
Calcular recompensa lagrangiana desde reward_component.
r = -(α·L_e + β·L_edot + γ·L_I + η·L_delta_u)

Esta clase es un reward_impl puro, sin responsabilidad de asignación a agentes.
"""


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
    
    def __init__(self, config):
        """
        Inicializa el calculador lagrangiano.
        
        Args:
            config (dict): Configuración de principal_reward con:
                - method: lineal_combination | weighted_exponential
                - lineal_combination_params: {features: {L_e: {alpha}, ...}}
                - weighted_exponential_params: {...}
        """
        self.config = config
        self.config_lineal_combination = config['lineal_combination_params']
        self.config_weighted_exponential = config['weighted_exponential_params']
        self.method = config['method']
        
        # Extraer pesos según método
        if self.method == 'lineal_combination':
            self.weights = self._extract_lineal_weights()
        elif self.method == 'weighted_exponential':
            self.weights = self._extract_exponential_weights()
        else:
            self.weights = {}
        
        # Último reward_params_record (para auditoría)
        self.last_reward_params_record = {}
    
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
    
    def compute_reward(self, reward_component):
        """
        Calcula la recompensa lagrangiana desde reward_component.
        
        Args:
            reward_component (dict): Estructura {var_obj: {L_e, L_edot, ...}}
            
        Returns:
            tuple: (principal_reward, reward_params_record, controller_rewards)
                - principal_reward: float (recompensa global)
                - reward_params_record: dict (desglose de componentes)
                - controller_rewards: dict (recompensas por controlador)
        """
        if self.method == 'lineal_combination':
            return self._compute_lineal(reward_component)
        elif self.method == 'weighted_exponential':
            return self._compute_exponential(reward_component)
        else:
            return 0.0, {}, {}
    
    def _compute_lineal(self, reward_component):
        """
        Calcula recompensa usando combinación lineal: r = -Σ(weight * L_feature)
        
        Args:
            reward_component (dict): {var_obj: {L_e, L_edot, ...}}
            
        Returns:
            tuple: (principal_reward, reward_params_record, controller_rewards)
        """
        # Agregar componentes L_* sobre todos los var_objs
        aggregated_L = {feature: 0.0 for feature in self.weights.keys()}
        controller_rewards = {}
        n_controllers = 0
        
        for var_obj, var_metrics in reward_component.items():
            if var_obj == 'global_vars':
                continue  # Procesar globales al final
            
            n_controllers += 1
            lazo_lagrangian = 0.0
            
            for feature_name, weight in self.weights.items():
                # Verificar que la feature existe en las métricas
                if feature_name not in var_metrics:
                    continue
                
                value = var_metrics[feature_name]
                aggregated_L[feature_name] += value
                lazo_lagrangian += weight * value
            
            # Recompensa del lazo = -lagrangiana
            controller_rewards[var_obj] = -lazo_lagrangian
        
        # Procesar global_vars si existe y tiene features con pesos
        if 'global_vars' in reward_component:
            global_metrics = reward_component['global_vars']
            for feature_name, weight in self.weights.items():
                if feature_name in global_metrics:
                    value = global_metrics[feature_name]
                    aggregated_L[feature_name] += value
        
        # Calcular lagrangiana total ponderada
        lagrangian_total = sum(
            self.weights[feature] * value 
            for feature, value in aggregated_L.items()
        )
        
        # Recompensa = -lagrangiana
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
    
    def _compute_exponential(self, reward_component):
        """
        Calcula recompensa usando método exponencial ponderado.
        r = -Σ weight * (1 - exp(-scaled * (value - setpoint)^2))
        
        Nota: Se aplica signo negativo para consistencia con método lineal.
        
        Args:
            reward_component (dict): {var_obj: {L_e, L_edot, ...}}
            
        Returns:
            tuple: (principal_reward, reward_params_record, controller_rewards)
        """
        import math
        
        # Agregar componentes sobre todos los var_objs
        aggregated_L = {feature: 0.0 for feature in self.weights.keys()}
        controller_rewards = {}
        
        for var_obj, var_metrics in reward_component.items():
            if var_obj == 'global_vars':
                continue
            
            lazo_reward = 0.0
            
            for feature_name, feature_config in self.weights.items():
                # Verificar que la feature existe
                if feature_name not in var_metrics:
                    continue
                    
                value = var_metrics[feature_name]
                aggregated_L[feature_name] += value
                
                weight = feature_config['weight']
                scaled = feature_config['scaled']
                setpoint = feature_config['setpoint']
                
                # exp(-scaled * (value - setpoint)^2) -> cercano a 1 cuando value ≈ setpoint
                exp_term = math.exp(-scaled * (value - setpoint) ** 2)
                # Penalidad: (1 - exp_term) es 0 cuando perfecto, ~1 cuando malo
                lazo_reward -= weight * (1 - exp_term)
            
            controller_rewards[var_obj] = lazo_reward
        
        # Procesar global_vars si existe
        if 'global_vars' in reward_component:
            global_metrics = reward_component['global_vars']
            for feature_name, feature_config in self.weights.items():
                if feature_name in global_metrics:
                    aggregated_L[feature_name] += global_metrics[feature_name]
        
        # Recompensa total (con signo negativo para consistencia)
        principal_reward = 0.0
        reward_params_record = {}
        
        for feature_name, feature_config in self.weights.items():
            value = aggregated_L[feature_name]
            weight = feature_config['weight']
            scaled = feature_config['scaled']
            setpoint = feature_config['setpoint']
            
            exp_term = math.exp(-scaled * (value - setpoint) ** 2)
            # Penalidad: -(weight * (1 - exp_term))
            contribution = -weight * (1 - exp_term)
            
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
    
    def get_reward_params_record(self):
        """
        Retorna el desglose de la última recompensa calculada.
        
        Returns:
            dict: Desglose de componentes
        """
        return self.last_reward_params_record.copy()
