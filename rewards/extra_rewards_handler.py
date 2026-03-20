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
        self.penalty_config = config['penalty_approach']
        self.bonus_config = config['bonus_approach']
        self.conditional_config = config['conditional_approach']
        
        # Estado acumulativo para episodio
        self.accumulated_band_bonus = 0.0
        self.episode_time = 0.0
        
        # Pre-computar var_objs una sola vez en el init
        self.var_objs = self._get_configured_var_objs()
        
        # Precompilar listas de ejecución
        self._compile_rules()
        
        # Último reward_params_record
        self.last_extra_reward_params_record = {}
    
    def reset_episode(self):
        """
        Resetea el manejador al inicio de un episodio.
        """
        self.accumulated_band_bonus = 0.0
        self.episode_time = 0.0
        self.last_extra_reward_params_record = {}
        
    def _compile_rules(self):
        """
        Extrae y precompila los parámetros activos desde los diccionarios de configuración.
        Esto elimina la sobrecarga de evaluación condicional (if enabled, type==, etc) durante los loops de step.
        """
        self.inst_penalty_rule = None
        self.delta_rules = []
        self.goal_bonus_rule = None
        self.band_ranges = {}
        self.band_config_cache = None
        self.dyn_pen_rules = []
        self.dyn_inc_rules = []
        
        # 1. Penalties
        inst_cfg = self.penalty_config['penalty_instantaneous_reward']
        if inst_cfg.get('enabled', False):
            method = inst_cfg['method']
            if method == 'lineal':
                self.inst_penalty_rule = ('lineal', inst_cfg['penalty_lineal_params']['time'])
            elif method == 'quadratic':
                self.inst_penalty_rule = ('quadratic', inst_cfg['penalty_quadratic_params']['time'])
                
        delta_cfg = self.penalty_config['penalty_delta_var']
        if delta_cfg.get('enabled', False):
            method = delta_cfg['method']
            for v_name, cfg in delta_cfg['penalty_delta_var_params'].items():
                self.delta_rules.append((v_name, self._extract_var_obj(v_name), method, cfg['weight']))
                
        # 2. Bonuses
        goal_cfg = self.bonus_config['goal_bonus_reward']
        if goal_cfg.get('enabled', False):
            method = goal_cfg['method']
            self.goal_bonus_rule = {
                'flags': goal_cfg['success_flags'],
                'method': method,
                'static_val': goal_cfg['static_params']['bonus_value'] if method == 'static' else 0.0,
                'decay': goal_cfg.get('decay_params', {})
            }
            
        band_cfg = self.bonus_config['bandwidth_bonus']
        if band_cfg.get('enabled', False):
            self.band_config_cache = (band_cfg['per_step_band_bonus'], band_cfg['max_total_band_bonus'])
            for v_name, (mn, mx) in band_cfg['ranges'].items():
                v_obj = self._extract_var_obj(v_name)
                if v_obj not in self.band_ranges:
                    self.band_ranges[v_obj] = []
                self.band_ranges[v_obj].append((v_name, mn, mx))
                
        # 3. Conditionals
        dp_cfg = self.conditional_config['dynamic_penalty']
        if dp_cfg.get('enabled', False):
            method = dp_cfg['method']
            for v_name, cfg in dp_cfg['dynamic_penalty_params'].items():
                cond = cfg['condition']
                v_obj = self._extract_var_obj(v_name)
                self.dyn_pen_rules.append((
                    v_name, v_obj, method, cfg['weight'], cond['feature'], 
                    cond['type'], cond.get('scaled', 1.0), cond.get('setpoint', 0.0)
                ))
                
        di_cfg = self.conditional_config['dynamic_incentive']
        if di_cfg.get('enabled', False):
            method = di_cfg['method']
            params = di_cfg['dynamic_incentive_adapt_params'] if method == 'adaptative' else di_cfg['dynamic_incentive_lineal_params']
            for v_name, cfg in params.items():
                v_obj = self._extract_var_obj(v_name)
                self.dyn_inc_rules.append((v_name, v_obj, method, cfg))
    
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
        
        # Obtener var_objs precomputados
        var_objs = self.var_objs
        
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
        for var_name in delta_params.keys():
            var_objs.add(self._extract_var_obj(var_name))
        
        # Desde bandwidth_bonus ranges
        band_ranges = self.bonus_config['bandwidth_bonus']['ranges']
        for var_name in band_ranges.keys():
            var_objs.add(self._extract_var_obj(var_name))
        
        # Desde dynamic_penalty params
        dyn_pen_params = self.conditional_config['dynamic_penalty']['dynamic_penalty_params']
        for var_name in dyn_pen_params.keys():
            var_objs.add(self._extract_var_obj(var_name))
        
        # Desde dynamic_incentive params
        for key in ['dynamic_incentive_adapt_params', 'dynamic_incentive_lineal_params']:
            dyn_inc_params = self.conditional_config['dynamic_incentive'][key]
            for var_name in dyn_inc_params.keys():
                var_objs.add(self._extract_var_obj(var_name))
        
        return sorted(var_objs)

    def _extract_var_obj(self, var_name):
        """
        Extrae el nombre del var_obj subyacente removiendo prefijos conocidos.
        """
        prefixes = [
            'delta_control_action_',
            'control_action_',
            'delta_u_eff_',
            'u_eff_',
            'error_'
        ]
        for p in prefixes:
            if var_name.startswith(p):
                return var_name[len(p):]
        parts = var_name.split('_', 1)
        return parts[1] if len(parts) == 2 else var_name

    def _evaluate_penalties(self, extra_reward_component, var_objs):
        """
        Evalúa penalty_approach iternando las listas de reglas pre-compiladas.
        """
        total_penalty = 0.0
        flat = {f'extra_penalty_instantaneous_{v}': 0.0 for v in var_objs}
        flat.update({f'extra_penalty_{v}': 0.0 for v in var_objs})
        
        # 1. Instantaneous Penalty
        if self.inst_penalty_rule:
            method, coeff = self.inst_penalty_rule
            if method == 'lineal':
                penalty = -coeff
            else: # quadratic
                self.episode_time += 1
                penalty = -coeff * (self.episode_time ** 2)
                
            total_penalty += penalty
            per_var = penalty / len(var_objs) if var_objs else 0.0
            for var in var_objs:
                flat[f'extra_penalty_instantaneous_{var}'] = per_var
                
        # 2. Delta Penalty
        for v_name, var_obj, method, weight in self.delta_rules:
            series = extra_reward_component.get(v_name, [])
            if len(series) < 2:
                continue
                
            # Calcular variación promedio
            deltas = [abs(series[i] - series[i-1]) for i in range(1, len(series))]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
            
            penalty = -weight * (avg_delta ** 2) if method == 'quadratic' else -weight * avg_delta
            
            total_penalty += penalty
            flat[f'extra_penalty_{var_obj}'] += penalty
            
        return total_penalty, flat
    
    def _evaluate_bonuses(self, extra_reward_component, termination_flag, current_time_sec, var_objs):
        """
        Evalúa bonus_approach iternando las listas de reglas pre-compiladas.
        """
        total_bonus = 0.0
        flat = {f'extra_bonus_goal_{v}': 0.0 for v in var_objs}
        flat.update({f'extra_bonus_band_{v}': 0.0 for v in var_objs})
        
        # 1. Goal Bonus
        if self.goal_bonus_rule and termination_flag in self.goal_bonus_rule['flags']:
            method = self.goal_bonus_rule['method']
            if method == 'static':
                bonus = self.goal_bonus_rule['static_val']
            else: # decay
                decay = self.goal_bonus_rule['decay']
                dtype, base, min_v, tau = decay['decay_type'], decay['base_value'], decay['min_value'], decay['time_constant_sec']
                if dtype == 'exponential':
                    bonus = min_v + (base - min_v) * math.exp(-current_time_sec / tau)
                elif dtype == 'linear':
                    bonus = max(min_v, base - (base - min_v) * (current_time_sec / tau))
                else:
                    bonus = base
                    
            total_bonus += bonus
            for var in var_objs:
                flat[f'extra_bonus_goal_{var}'] = bonus
                
        # 2. Bandwidth Bonus
        if self.band_config_cache:
            per_step_bonus, max_bonus = self.band_config_cache
            
            for var, range_entries in self.band_ranges.items():
                series_list = [(extra_reward_component.get(vn, []), mn, mx) for vn, mn, mx in range_entries]
                
                if not series_list or any(len(s) == 0 for s, _, _ in series_list):
                    continue
                    
                min_len = min(len(s) for s, _, _ in series_list)
                steps_in_band = 0
                for t in range(min_len):
                    if all(mn <= s[t] <= mx for s, mn, mx in series_list):
                        steps_in_band += 1
                        
                bonus = steps_in_band * per_step_bonus
                potential_total = self.accumulated_band_bonus + bonus
                if potential_total > max_bonus:
                    bonus = max(0.0, max_bonus - self.accumulated_band_bonus)
                    
                self.accumulated_band_bonus += bonus
                total_bonus += bonus
                flat[f'extra_bonus_band_{var}'] = bonus
                
        return total_bonus, flat
    
    def _evaluate_conditional(self, extra_reward_component, var_objs):
        """
        Evalúa conditional_approach iternando las listas de reglas pre-compiladas.
        """
        total_conditional = 0.0
        flat = {f'extra_conditional_dynamic_penalty_{v}': 0.0 for v in var_objs}
        flat.update({f'extra_conditional_dynamic_incentive_{v}': 0.0 for v in var_objs})
        
        # 1. Dynamic Penalty
        for v_name, var_obj, method, weight, cond_feature, cond_type, cond_s, cond_sp in self.dyn_pen_rules:
            series = extra_reward_component.get(v_name, [])
            condition_series = extra_reward_component.get(cond_feature, [])
            
            if not series or not condition_series:
                continue
                
            avg_var = sum(abs(v) for v in series) / len(series)
            avg_cond = sum(abs(v) for v in condition_series) / len(condition_series)
            
            cond_factor = math.exp(-cond_s * (avg_cond - cond_sp) ** 2) if cond_type == 'exp' else 1.0
            penalty = -weight * (avg_var ** 2) * cond_factor if method == 'quadratic' else -weight * avg_var * cond_factor
            
            total_conditional += penalty
            flat[f'extra_conditional_dynamic_penalty_{var_obj}'] += penalty
            
        # 2. Dynamic Incentive
        for v_name, var_obj, method, cfg in self.dyn_inc_rules:
            series = extra_reward_component.get(v_name, [])
            if not series:
                continue
                
            last_value = abs(series[-1])
            y_max = cfg['y_max']
            
            if method == 'adaptative':
                x_series = extra_reward_component.get(cfg['x'], [])
                x_val = abs(x_series[-1]) if x_series else 0.0
                
                tanh_factor = math.tanh(cfg['strength'] * abs(x_val - cfg['x_sp']))
                f_cfg = cfg['f_reward']
                exp_term = math.exp(-f_cfg['scaled'] * (last_value - f_cfg['setpoint']) ** 2)
                incentive = y_max * tanh_factor * f_cfg['weight'] * exp_term
            else: # lineal
                x_max = cfg['x_max']
                incentive = y_max * min(1.0, last_value / x_max) if x_max > 0 else 0.0
                
            total_conditional += incentive
            flat[f'extra_conditional_dynamic_incentive_{var_obj}'] += incentive
            
        return total_conditional, flat
    
    def get_extra_reward_params_record(self):
        """
        Retorna desglose plano de extra rewards aplicados.
        
        Returns:
            dict: Flat record con llaves canónicas per-var_obj
        """
        return self.last_extra_reward_params_record.copy()

