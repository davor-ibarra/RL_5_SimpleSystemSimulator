"""
extra_rewards_handler.py

Responsabilidad:
Evaluar reglas declarativas de bonus/penalty desde extra_reward_component.
Produce output plano con llaves canónicas per-var_obj:
  extra_penalty_instantaneous_<var>, extra_penalty_<var>,
  extra_bonus_goal_<var>, extra_bonus_band_<var>,
  extra_conditional_dynamic_penalty_<var>, extra_conditional_dynamic_incentive_<var>,
  extra_total_<var>

Estructura de config:
- penalty_approach: penalty_instantaneous_reward, penalty_delta_var
- bonus_approach: goal_bonus_reward, bandwidth_bonus
- conditional_approach: dynamic_penalty, dynamic_incentive
"""

import math


class ExtraRewardsHandler:
    """
    Manejador declarativo de extra rewards (bonus/penalties).
    
    Consume extra_reward_component del MetricProcessing:
    {
        "error_pendulum_angle": [serie completa],
        "error_cart_position": [serie completa],
        "control_action_pendulum_angle": [serie completa],
        ...
    }
    """
    
    def __init__(self, config):
        """
        Inicializa el manejador con configuración de extra_rewards.
        
        Args:
            config (dict): Configuración de extra_rewards
        """
        self.config = config
        
        # Extraer configuraciones de cada approach
        self.penalty_config = config.get('penalty_approach', {})
        self.bonus_config = config.get('bonus_approach', {})
        self.conditional_config = config.get('conditional_approach', {})
        
        # Estado acumulativo para episodio
        self.accumulated_band_bonus = 0.0
        self.episode_time = 0.0
        
        # Último reward_params_record
        self.last_extra_reward_params_record = {}
    
    def reset_episode(self):
        """
        Resetea el manejador al inicio de un episodio.
        """
        self.accumulated_band_bonus = 0.0
        self.episode_time = 0.0
        self.last_extra_reward_params_record = {}
    
    def evaluate(self, extra_reward_component, termination_flag='unknown', current_time_sec=0.0):
        """
        Evalúa todas las reglas de extra rewards.
        
        Args:
            extra_reward_component (dict): Series crudas {var_name: [lista]}
            termination_flag (str): Flag de terminación del episodio
            current_time_sec (float): Tiempo actual en el episodio [s]
            
        Returns:
            tuple: (extra_reward_total, flat_record)
                flat_record: dict plano con llaves canónicas per-var_obj:
                    extra_penalty_instantaneous_<var>: float
                    extra_penalty_<var>: float (delta_var)
                    extra_bonus_goal_<var>: float
                    extra_bonus_band_<var>: float
                    extra_conditional_dynamic_penalty_<var>: float
                    extra_conditional_dynamic_incentive_<var>: float
                    extra_total_<var>: float
        """
        total_extra = 0.0
        flat = {}
        
        # Obtener var_objs desde las llaves configuradas
        var_objs = self._get_configured_var_objs()
        
        # Inicializar acumulador per-var_obj
        var_totals = {var: 0.0 for var in var_objs}
        
        # 1. Penalties
        penalty_total, penalty_flat = self._evaluate_penalties(extra_reward_component, var_objs)
        total_extra += penalty_total
        flat.update(penalty_flat)
        for var in var_objs:
            var_totals[var] += penalty_flat[f'extra_penalty_instantaneous_{var}']
            var_totals[var] += penalty_flat[f'extra_penalty_{var}']
        
        # 2. Bonuses
        bonus_total, bonus_flat = self._evaluate_bonuses(
            extra_reward_component, termination_flag, current_time_sec, var_objs
        )
        total_extra += bonus_total
        flat.update(bonus_flat)
        for var in var_objs:
            var_totals[var] += bonus_flat[f'extra_bonus_goal_{var}']
            var_totals[var] += bonus_flat[f'extra_bonus_band_{var}']
        
        # 3. Conditional
        cond_total, cond_flat = self._evaluate_conditional(extra_reward_component, var_objs)
        total_extra += cond_total
        flat.update(cond_flat)
        for var in var_objs:
            var_totals[var] += cond_flat[f'extra_conditional_dynamic_penalty_{var}']
            var_totals[var] += cond_flat[f'extra_conditional_dynamic_incentive_{var}']
        
        # Total per-var_obj
        for var in var_objs:
            flat[f'extra_total_{var}'] = var_totals[var]
        
        self.last_extra_reward_params_record = flat
        return total_extra, flat
    
    def _get_configured_var_objs(self):
        """
        Extrae las var_obj únicas desde las configuraciones de penalty, bonus y conditional.
        
        Returns:
            list: Lista ordenada de var_objs
        """
        var_objs = set()
        
        # Desde penalty_delta_var params
        delta_params = self.penalty_config['penalty_delta_var']['penalty_delta_var_params']
        var_objs.update(delta_params.keys())
        
        # Desde bandwidth_bonus ranges
        band_ranges = self.bonus_config['bandwidth_bonus']['ranges']
        for var_name in band_ranges.keys():
            # Extraer var_obj desde el nombre de la variable (e.g., error_pendulum_angle → pendulum_angle)
            parts = var_name.split('_', 1)
            if len(parts) == 2:
                var_objs.add(parts[1])
        
        # Desde dynamic_penalty params
        dyn_pen_params = self.conditional_config['dynamic_penalty']['dynamic_penalty_params']
        var_objs.update(dyn_pen_params.keys())
        
        # Desde dynamic_incentive params
        for key in ['dynamic_incentive_adapt_params', 'dynamic_incentive_lineal_params']:
            dyn_inc_params = self.conditional_config['dynamic_incentive'][key]
            var_objs.update(dyn_inc_params.keys())
        
        return sorted(var_objs)
    
    def _evaluate_penalties(self, extra_reward_component, var_objs):
        """
        Evalúa penalty_approach: penalty_instantaneous_reward, penalty_delta_var.
        Retorna flat record con llaves per-var_obj.
        
        Returns:
            tuple: (penalty_total, flat_record)
        """
        total_penalty = 0.0
        flat = {}
        
        # penalty_instantaneous_reward (global → distribuir equitativamente entre var_objs)
        inst_config = self.penalty_config['penalty_instantaneous_reward']
        if inst_config['enabled']:
            penalty = self._compute_instantaneous_penalty(inst_config)
            total_penalty += penalty
            per_var = penalty / len(var_objs) if var_objs else 0.0
            for var in var_objs:
                flat[f'extra_penalty_instantaneous_{var}'] = per_var
        else:
            for var in var_objs:
                flat[f'extra_penalty_instantaneous_{var}'] = 0.0
        
        # penalty_delta_var (per-var_obj)
        delta_config = self.penalty_config['penalty_delta_var']
        if delta_config['enabled']:
            penalty, delta_flat = self._compute_delta_var_penalty(delta_config, extra_reward_component, var_objs)
            total_penalty += penalty
            flat.update(delta_flat)
        else:
            for var in var_objs:
                flat[f'extra_penalty_{var}'] = 0.0
        
        return total_penalty, flat
    
    def _compute_instantaneous_penalty(self, config):
        """
        Penalización por tiempo (instantánea).
        Método: lineal o quadratic.
        """
        method = config['method']
        
        if method == 'lineal':
            params = config['penalty_lineal_params']
            time_coeff = params['time']
            return -time_coeff  # Penalización fija por step
        
        elif method == 'quadratic':
            params = config['penalty_quadratic_params']
            time_coeff = params['time']
            self.episode_time += 1
            return -time_coeff * (self.episode_time ** 2)
        
        return 0.0
    
    def _compute_delta_var_penalty(self, config, extra_reward_component, var_objs):
        """
        Penalización por variación de variables.
        Retorna (total, flat_record) con llaves extra_penalty_<var>.
        """
        method = config['method']
        params = config['penalty_delta_var_params']
        
        total_penalty = 0.0
        flat = {}
        
        for var_name, var_config in params.items():
            weight = var_config['weight']
            
            # Buscar serie de la variable
            series = extra_reward_component[var_name]
            if len(series) < 2:
                flat[f'extra_penalty_{var_name}'] = 0.0
                continue
            
            # Calcular variación promedio
            deltas = [abs(series[i] - series[i-1]) for i in range(1, len(series))]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
            
            if method == 'quadratic':
                penalty = -weight * (avg_delta ** 2)
            else:
                penalty = -weight * avg_delta
            
            total_penalty += penalty
            flat[f'extra_penalty_{var_name}'] = penalty
        
        # Asegurar llaves para var_objs sin config
        for var in var_objs:
            if f'extra_penalty_{var}' not in flat:
                flat[f'extra_penalty_{var}'] = 0.0
        
        return total_penalty, flat
    
    def _evaluate_bonuses(self, extra_reward_component, termination_flag, current_time_sec, var_objs):
        """
        Evalúa bonus_approach: goal_bonus_reward, bandwidth_bonus.
        Retorna flat record con llaves per-var_obj.
        
        Returns:
            tuple: (bonus_total, flat_record)
        """
        total_bonus = 0.0
        flat = {}
        
        # goal_bonus_reward
        goal_config = self.bonus_config['goal_bonus_reward']
        if goal_config['enabled']:
            bonus = self._compute_goal_bonus(goal_config, termination_flag, current_time_sec)
            total_bonus += bonus
            for var in var_objs:
                flat[f'extra_bonus_goal_{var}'] = bonus
        else:
            for var in var_objs:
                flat[f'extra_bonus_goal_{var}'] = 0.0
        
        # bandwidth_bonus (per-var_obj)
        band_config = self.bonus_config['bandwidth_bonus']
        if band_config['enabled']:
            bonus, band_flat = self._compute_bandwidth_bonus(band_config, extra_reward_component, var_objs)
            total_bonus += bonus
            flat.update(band_flat)
        else:
            for var in var_objs:
                flat[f'extra_bonus_band_{var}'] = 0.0
        
        return total_bonus, flat
    
    def _compute_goal_bonus(self, config, termination_flag, current_time_sec):
        """
        Bono por alcanzar objetivo.
        Soporta método static (valor fijo) y decay (exponencial/lineal con tiempo).
        
        Args:
            config (dict): Configuración del goal_bonus
            termination_flag (str): Flag de terminación actual
            current_time_sec (float): Tiempo actual en el episodio [s]
            
        Returns:
            float: Bono (positivo si objetivo alcanzado, 0 en otro caso)
        """
        # Obtener flags que activan el bonus
        success_flags = config['success_flags']
        
        # Verificar si el termination_flag indica éxito
        if termination_flag not in success_flags:
            return 0.0
        
        # Aplicar método correspondiente
        method = config['method']
        
        if method == 'static':
            static_params = config['static_params']
            return static_params['bonus_value']
        
        elif method == 'decay':
            decay_params = config['decay_params']
            decay_type = decay_params['decay_type']
            base_value = decay_params['base_value']
            min_value = decay_params['min_value']
            tau = decay_params['time_constant_sec']
            
            if decay_type == 'exponential':
                bonus = min_value + (base_value - min_value) * math.exp(-current_time_sec / tau)
            elif decay_type == 'linear':
                bonus = max(min_value, base_value - (base_value - min_value) * (current_time_sec / tau))
            else:
                bonus = base_value
            
            return bonus
        
        return 0.0
    
    def _compute_bandwidth_bonus(self, config, extra_reward_component, var_objs):
        """
        Bono por permanecer dentro de banda.
        Cuenta steps por var_obj donde la variable está en rango.
        Retorna (total, flat_record) con llaves extra_bonus_band_<var>.
        """
        per_step_bonus = config['per_step_band_bonus']
        max_bonus = config['max_total_band_bonus']
        ranges = config['ranges']
        
        flat = {}
        total_bonus = 0.0
        
        if not ranges:
            for var in var_objs:
                flat[f'extra_bonus_band_{var}'] = 0.0
            return 0.0, flat
        
        # Mapear rangos a var_objs
        var_ranges = {}
        for var_name, (min_val, max_val) in ranges.items():
            parts = var_name.split('_', 1)
            var_obj = parts[1] if len(parts) == 2 else var_name
            if var_obj not in var_ranges:
                var_ranges[var_obj] = []
            var_ranges[var_obj].append((var_name, min_val, max_val))
        
        for var in var_objs:
            if var not in var_ranges:
                flat[f'extra_bonus_band_{var}'] = 0.0
                continue
            
            # Contar steps donde TODAS las features de este var_obj están en rango
            range_entries = var_ranges[var]
            series_list = [(extra_reward_component[vn], mn, mx) for vn, mn, mx in range_entries]
            
            if not series_list or any(len(s) == 0 for s, _, _ in series_list):
                flat[f'extra_bonus_band_{var}'] = 0.0
                continue
            
            min_len = min(len(s) for s, _, _ in series_list)
            steps_in_band = 0
            for t in range(min_len):
                all_in = all(mn <= s[t] <= mx for s, mn, mx in series_list)
                if all_in:
                    steps_in_band += 1
            
            bonus = steps_in_band * per_step_bonus
            
            # Aplicar tope acumulativo global
            potential_total = self.accumulated_band_bonus + bonus
            if potential_total > max_bonus:
                bonus = max(0.0, max_bonus - self.accumulated_band_bonus)
            
            self.accumulated_band_bonus += bonus
            total_bonus += bonus
            flat[f'extra_bonus_band_{var}'] = bonus
        
        # Asegurar llaves para var_objs sin rangos
        for var in var_objs:
            if f'extra_bonus_band_{var}' not in flat:
                flat[f'extra_bonus_band_{var}'] = 0.0
        
        return total_bonus, flat
    
    def _evaluate_conditional(self, extra_reward_component, var_objs):
        """
        Evalúa conditional_approach: dynamic_penalty, dynamic_incentive.
        Retorna flat record con llaves per-var_obj.
        
        Returns:
            tuple: (conditional_total, flat_record)
        """
        total_conditional = 0.0
        flat = {}
        
        # dynamic_penalty (per-var_obj)
        dyn_penalty_config = self.conditional_config['dynamic_penalty']
        if dyn_penalty_config['enabled']:
            penalty, pen_flat = self._compute_dynamic_penalty(dyn_penalty_config, extra_reward_component, var_objs)
            total_conditional += penalty
            flat.update(pen_flat)
        else:
            for var in var_objs:
                flat[f'extra_conditional_dynamic_penalty_{var}'] = 0.0
        
        # dynamic_incentive (per-var_obj)
        dyn_incentive_config = self.conditional_config['dynamic_incentive']
        if dyn_incentive_config['enabled']:
            incentive, inc_flat = self._compute_dynamic_incentive(dyn_incentive_config, extra_reward_component, var_objs)
            total_conditional += incentive
            flat.update(inc_flat)
        else:
            for var in var_objs:
                flat[f'extra_conditional_dynamic_incentive_{var}'] = 0.0
        
        return total_conditional, flat
    
    def _compute_dynamic_penalty(self, config, extra_reward_component, var_objs):
        """
        Penalización dinámica condicionada a otra variable.
        Retorna (total, flat_record) con llaves extra_conditional_dynamic_penalty_<var>.
        """
        method = config['method']
        params = config['dynamic_penalty_params']
        
        total_penalty = 0.0
        flat = {}
        
        for var_name, var_config in params.items():
            weight = var_config['weight']
            condition = var_config['condition']
            
            # Obtener series
            series = extra_reward_component[var_name]
            if not series:
                flat[f'extra_conditional_dynamic_penalty_{var_name}'] = 0.0
                continue
            
            # Obtener condición
            condition_feature = condition['feature']
            condition_series = extra_reward_component[condition_feature]
            
            if not condition_series:
                flat[f'extra_conditional_dynamic_penalty_{var_name}'] = 0.0
                continue
            
            # Promedios
            avg_var = sum(abs(v) for v in series) / len(series)
            avg_cond = sum(abs(v) for v in condition_series) / len(condition_series)
            
            # Calcular condición exponencial
            cond_type = condition['type']
            scaled = condition['scaled']
            setpoint = condition['setpoint']
            
            if cond_type == 'exp':
                cond_factor = math.exp(-scaled * (avg_cond - setpoint) ** 2)
            else:
                cond_factor = 1.0
            
            # Penalización
            if method == 'quadratic':
                penalty = -weight * (avg_var ** 2) * cond_factor
            else:
                penalty = -weight * avg_var * cond_factor
            
            total_penalty += penalty
            flat[f'extra_conditional_dynamic_penalty_{var_name}'] = penalty
        
        # Asegurar llaves para var_objs sin config
        for var in var_objs:
            if f'extra_conditional_dynamic_penalty_{var}' not in flat:
                flat[f'extra_conditional_dynamic_penalty_{var}'] = 0.0
        
        return total_penalty, flat
    
    def _compute_dynamic_incentive(self, config, extra_reward_component, var_objs):
        """
        Incentivo dinámico adaptativo.
        Retorna (total, flat_record) con llaves extra_conditional_dynamic_incentive_<var>.
        """
        method = config['method']
        
        if method == 'adaptative':
            params = config['dynamic_incentive_adapt_params']
        else:
            params = config['dynamic_incentive_lineal_params']
        
        total_incentive = 0.0
        flat = {}
        
        for var_name, var_config in params.items():
            # Obtener series
            series = extra_reward_component[var_name]
            if not series:
                flat[f'extra_conditional_dynamic_incentive_{var_name}'] = 0.0
                continue
            
            # Último valor de la serie
            last_value = abs(series[-1])
            
            # Configuración
            y_max = var_config['y_max']
            
            if method == 'adaptative':
                strength = var_config['strength']
                x_feature = var_config['x']
                x_sp = var_config['x_sp']
                f_reward_config = var_config['f_reward']
                
                # Obtener x_series
                x_series = extra_reward_component[x_feature]
                x_val = abs(x_series[-1]) if x_series else 0.0
                
                # Tanh adaptativo
                tanh_factor = math.tanh(strength * abs(x_val - x_sp))
                
                # f_reward exponencial
                f_weight = f_reward_config['weight']
                f_scaled = f_reward_config['scaled']
                f_setpoint = f_reward_config['setpoint']
                
                exp_term = math.exp(-f_scaled * (last_value - f_setpoint) ** 2)
                
                incentive = y_max * tanh_factor * f_weight * exp_term
            else:
                x_max = var_config['x_max']
                incentive = y_max * min(1.0, last_value / x_max) if x_max > 0 else 0.0
            
            total_incentive += incentive
            flat[f'extra_conditional_dynamic_incentive_{var_name}'] = incentive
        
        # Asegurar llaves para var_objs sin config
        for var in var_objs:
            if f'extra_conditional_dynamic_incentive_{var}' not in flat:
                flat[f'extra_conditional_dynamic_incentive_{var}'] = 0.0
        
        return total_incentive, flat
    
    def get_extra_reward_params_record(self):
        """
        Retorna desglose plano de extra rewards aplicados.
        
        Returns:
            dict: Flat record con llaves canónicas per-var_obj
        """
        return self.last_extra_reward_params_record.copy()

