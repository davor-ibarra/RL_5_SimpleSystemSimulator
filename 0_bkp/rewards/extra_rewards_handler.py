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
        self.normalized_reward_mode = config['normalized_reward_mode']
        
        # Extraer configuraciones de cada approach
        self.penalty_config = config['penalty_approach']
        self.bonus_config = config['bonus_approach']
        self.conditional_config = config['conditional_approach']
        
        # Estado acumulativo para episodio
        self.accumulated_band_bonus = {}
        self.episode_time = 0.0
        
        # Pre-computar var_objs una sola vez en el init
        self.var_objs = self._get_configured_var_objs()
        
        # Precompilar listas de ejecución
        self._compile_rules()
        self.reset_episode()
        
        # Último reward_params_record
        self.last_extra_reward_params_record = {}
    
    def reset_episode(self):
        """
        Resetea el manejador al inicio de un episodio.
        """
        self.accumulated_band_bonus = {var_obj: 0.0 for var_obj in self.band_ranges.keys()}
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
        self.band_per_step_bonus = 0.0
        self.band_max_bonus = {}
        self.band_gate_rules = {}
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
            self.band_per_step_bonus = band_cfg['per_step_band_bonus']
            for v_name, (mn, mx) in band_cfg['ranges'].items():
                v_obj = self._extract_var_obj(v_name)
                if v_obj not in self.band_ranges:
                    self.band_ranges[v_obj] = []
                self.band_ranges[v_obj].append((v_name, mn, mx))
                self.band_max_bonus[v_obj] = band_cfg['max_total_band_bonus'][v_obj]
            
            if 'gates' in band_cfg:
                for var_obj, gate_cfg in band_cfg['gates'].items():
                    self.band_gate_rules[var_obj] = []
                    for rule_cfg in gate_cfg:
                        self.band_gate_rules[var_obj].append((
                            rule_cfg['source'],
                            rule_cfg['type'],
                            rule_cfg['scaled'],
                            rule_cfg['setpoint'],
                            rule_cfg['min_factor'],
                            rule_cfg['max_factor'],
                            rule_cfg['invert']
                        ))
                
        # 3. Conditionals
        dp_cfg = self.conditional_config['dynamic_penalty']
        if dp_cfg.get('enabled', False):
            method = dp_cfg['method']
            for v_name, cfg in dp_cfg['dynamic_penalty_params'].items():
                cond = cfg['condition']
                v_obj = self._resolve_var_obj(v_name, cfg)
                signal_key = cfg.get('signal', v_name)
                cond_type = cond['type']
                cond_gain = self._get_condition_gain(cond)
                cond_setpoint = self._get_condition_setpoint(cond)
                self.dyn_pen_rules.append((
                    signal_key, v_obj, method, cfg['weight'], cond['feature'],
                    cond_type, cond_gain, cond_setpoint
                ))
                
        di_cfg = self.conditional_config['dynamic_incentive']
        if di_cfg.get('enabled', False):
            method = di_cfg['method']
            params = di_cfg['dynamic_incentive_adapt_params'] if method == 'adaptative' else di_cfg['dynamic_incentive_lineal_params']
            for v_name, cfg in params.items():
                v_obj = self._resolve_var_obj(v_name, cfg)
                if method == 'adaptative' and cfg['reward_mode'] == 'capture_directional_effort':
                    signal_key = cfg['u']
                else:
                    signal_key = cfg.get('y', cfg.get('signal', v_name))
                self.dyn_inc_rules.append((signal_key, v_obj, method, cfg))
    
    def evaluate(self, extra_reward_component, reward_component, termination_flag='unknown', current_time_sec=0.0):
        """
        Evalúa todas las reglas de extra rewards.
        
        Args:
            extra_reward_component (dict): Series crudas {var_name: [lista]}
            reward_component (dict): Métricas interval-level ya procesadas.
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
        flat = {}
        
        # Obtener var_objs precomputados
        var_objs = self.var_objs
        normalization_bounds = self._compute_extra_normalization_bounds(extra_reward_component, var_objs)
        
        # Inicializar acumulador per-var_obj
        var_totals = {var: 0.0 for var in var_objs}
        family_totals = {
            var: {'penalty': 0.0, 'bonus': 0.0, 'incentive': 0.0}
            for var in var_objs
        }
        
        # 1. Penalties
        penalty_total, penalty_flat = self._evaluate_penalties(extra_reward_component, var_objs)
        flat.update(penalty_flat)
        for var in var_objs:
            penalty_var = (
                penalty_flat[f'extra_penalty_instantaneous_{var}']
                + penalty_flat[f'extra_penalty_{var}']
            )
            var_totals[var] += penalty_var
            family_totals[var]['penalty'] += penalty_var
        
        # 2. Bonuses
        bonus_total, bonus_flat = self._evaluate_bonuses(
            extra_reward_component, reward_component, termination_flag, current_time_sec, var_objs
        )
        flat.update(bonus_flat)
        for var in var_objs:
            bonus_var = (
                bonus_flat[f'extra_bonus_goal_{var}']
                + bonus_flat[f'extra_bonus_band_{var}']
            )
            var_totals[var] += bonus_var
            family_totals[var]['bonus'] += bonus_var
        
        # 3. Conditional
        cond_total, cond_flat = self._evaluate_conditional(extra_reward_component, var_objs)
        flat.update(cond_flat)
        for var in var_objs:
            conditional_penalty_var = cond_flat[f'extra_conditional_dynamic_penalty_{var}']
            conditional_incentive_var = cond_flat[f'extra_conditional_dynamic_incentive_{var}']
            var_totals[var] += conditional_penalty_var + conditional_incentive_var
            family_totals[var]['penalty'] += conditional_penalty_var
            family_totals[var]['incentive'] += conditional_incentive_var
        
        normalized_totals, normalization_flat = self._compute_extra_normalized_totals(
            family_totals,
            normalization_bounds,
            var_objs
        )
        flat.update(normalization_flat)

        total_extra = 0.0
        composition_score_total = 0.0
        for var in var_objs:
            flat[f'extra_raw_total_{var}'] = var_totals[var]
            if self.normalized_reward_mode:
                flat[f'extra_total_{var}'] = normalized_totals[var]
            else:
                flat[f'extra_total_{var}'] = var_totals[var]
            total_extra += flat[f'extra_total_{var}']
            composition_score_total += flat[f'extra_composition_score_01_{var}']
        if self.normalized_reward_mode and var_objs:
            total_extra /= len(var_objs)
        flat['extra_composition_score_01'] = (
            composition_score_total / len(var_objs)
            if var_objs else 0.5
        )
        flat['extra_composition_cost_01'] = 1.0 - flat['extra_composition_score_01']
        
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
        for var_name, cfg in dyn_pen_params.items():
            var_objs.add(self._resolve_var_obj(var_name, cfg))
        
        # Desde dynamic_incentive params
        for key in ['dynamic_incentive_adapt_params', 'dynamic_incentive_lineal_params']:
            dyn_inc_params = self.conditional_config['dynamic_incentive'][key]
            for var_name, cfg in dyn_inc_params.items():
                var_objs.add(self._resolve_var_obj(var_name, cfg))
        
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

        if var_name.endswith('_raw'):
            return var_name[:-4]

        return var_name

    def _resolve_var_obj(self, rule_name, cfg=None):
        """
        Resuelve el lazo al que pertenece una regla.
        """
        if isinstance(cfg, dict) and cfg.get('assign_to'):
            return cfg['assign_to']
        return self._extract_var_obj(rule_name)

    def _get_condition_gain(self, cond, default=1.0):
        """
        Obtiene la ganancia principal de una condiciÃ³n soportando aliases
        histÃ³ricos (`scaled`) y el nombre canÃ³nico (`strength`).
        """
        return cond.get('strength', cond.get('scaled', default))

    def _get_condition_setpoint(self, cond, default=0.0):
        """
        Obtiene el setpoint de una condiciÃ³n soportando ambos esquemas:
        `setpoint` y `x_sp`.
        """
        return cond.get('x_sp', cond.get('setpoint', default))

    def _compute_condition_factor(self, factor_type, value, gain, setpoint):
        """
        EvalÃºa una compuerta condicional en funciÃ³n de la distancia al setpoint.

        - exp: campana gaussiana centrada en el setpoint.
        - tanh: funciÃ³n S monÃ³tona sobre la distancia al setpoint.
        """
        if gain < 0:
            raise ValueError(f"Conditional gain must be non-negative, got {gain}")

        distance = abs(value - setpoint)
        if factor_type == 'exp':
            if gain == 0:
                return 0.0
            return math.exp(-((distance / gain) ** 2))
        if factor_type == 'tanh':
            return math.tanh(gain * distance)

        raise ValueError(f"Unsupported conditional factor type: {factor_type}")

    def _compute_exp_reward(self, value, scaled, setpoint, weight):
        """
        Recompensa gaussiana con semantica del notebook:
        weight * exp(-((value - setpoint) / scaled)^2)
        """
        if scaled is None or scaled <= 0:
            raise ValueError(f"Exp reward scaled must be positive, got {scaled}")

        normalized_distance = (value - setpoint) / scaled
        return weight * math.exp(-(normalized_distance ** 2))

    def _safe_unit_ratio(self, value, bound):
        if bound <= 0.0:
            return 0.0
        return min(1.0, max(0.0, value / bound))

    def _compute_extra_normalization_bounds(self, extra_reward_component, var_objs):
        """
        Construye cotas declarativas por familia sin usar min/max observado.
        """
        bounds = {
            var: {'penalty': 0.0, 'bonus': 0.0, 'incentive': 0.0}
            for var in var_objs
        }
        if not var_objs:
            return bounds

        if self.inst_penalty_rule:
            method, coeff = self.inst_penalty_rule
            if method == 'quadratic':
                next_episode_time = self.episode_time + 1.0
                penalty_bound = abs(coeff * (next_episode_time ** 2))
            else:
                penalty_bound = abs(coeff)
            per_var_bound = penalty_bound / len(var_objs)
            for var in var_objs:
                bounds[var]['penalty'] += per_var_bound

        for _, var_obj, method, weight in self.delta_rules:
            max_delta = 2.0
            if method == 'quadratic':
                bounds[var_obj]['penalty'] += weight * (max_delta ** 2)
            else:
                bounds[var_obj]['penalty'] += weight * max_delta

        if self.goal_bonus_rule:
            if self.goal_bonus_rule['method'] == 'static':
                goal_bound = abs(self.goal_bonus_rule['static_val'])
            else:
                goal_bound = abs(self.goal_bonus_rule['decay'].get('base_value', 0.0))
            for var in var_objs:
                bounds[var]['bonus'] += goal_bound

        for var, range_entries in self.band_ranges.items():
            if var not in bounds:
                continue
            n_steps = self._get_band_window_steps(extra_reward_component, range_entries)
            gate_bound = self._compute_band_gate_bound(var)
            interval_bound = n_steps * self.band_per_step_bonus * gate_bound
            remaining_bound = max(
                0.0,
                self.band_max_bonus.get(var, 0.0) - self.accumulated_band_bonus.get(var, 0.0)
            )
            bounds[var]['bonus'] += min(interval_bound, remaining_bound)

        for _, var_obj, method, weight, _, _, _, _ in self.dyn_pen_rules:
            signal_bound = 1.0
            if method == 'quadratic':
                bounds[var_obj]['penalty'] += weight * (signal_bound ** 2)
            else:
                bounds[var_obj]['penalty'] += weight * signal_bound

        for _, var_obj, method, cfg in self.dyn_inc_rules:
            bounds[var_obj]['incentive'] += self._compute_incentive_bound(method, cfg)

        return bounds

    def _get_band_window_steps(self, extra_reward_component, range_entries):
        lengths = []
        for v_name, _, _ in range_entries:
            series = extra_reward_component.get(v_name, [])
            if series:
                lengths.append(len(series))
        return min(lengths) if lengths else 0

    def _compute_band_gate_bound(self, var_obj):
        if var_obj not in self.band_gate_rules:
            return 1.0

        gate_bound = 1.0
        for _, _, _, _, _, max_factor, _ in self.band_gate_rules[var_obj]:
            gate_bound *= max(0.0, max_factor)
        return gate_bound

    def _compute_incentive_bound(self, method, cfg):
        if method != 'adaptative':
            return abs(cfg.get('y_max', 0.0))

        reward_mode = cfg.get('reward_mode', 'legacy')
        uses_tracking_shape = (
            reward_mode == 'tanh_tracking'
            or 'track_weight' in cfg
            or 'track_scaled' in cfg
            or 'base_weight' in cfg
            or 'base_scaled' in cfg
        )

        if reward_mode in {
            'capture_directional_effort',
            'directional_dense',
            'directional_effort'
        }:
            return abs(cfg.get('weight', 0.0))

        if uses_tracking_shape:
            return (
                abs(cfg.get('track_weight', 0.0))
                + abs(cfg.get('base_weight', 0.0))
                + abs(cfg.get('band_bonus_per_step', 0.0))
            )

        f_cfg = cfg.get('f_reward', {})
        return abs(cfg.get('y_max', 0.0) * f_cfg.get('weight', 0.0))

    def _compute_extra_normalized_totals(self, family_totals, bounds, var_objs):
        normalized_totals = {}
        flat = {
            'extra_normalized_reward_mode': float(self.normalized_reward_mode)
        }

        for var in var_objs:
            penalty_bound = bounds[var]['penalty']
            bonus_bound = bounds[var]['bonus']
            incentive_bound = bounds[var]['incentive']

            penalty_cost_01 = self._safe_unit_ratio(
                abs(min(0.0, family_totals[var]['penalty'])),
                penalty_bound
            )
            bonus_reward_01 = self._safe_unit_ratio(
                max(0.0, family_totals[var]['bonus']),
                bonus_bound
            )
            incentive_reward_01 = self._safe_unit_ratio(
                max(0.0, family_totals[var]['incentive']),
                incentive_bound
            )

            positive_family_count = 0
            if bonus_bound > 0.0:
                positive_family_count += 1
            if incentive_bound > 0.0:
                positive_family_count += 1

            if positive_family_count > 0:
                positive_reward_01 = (
                    bonus_reward_01 + incentive_reward_01
                ) / positive_family_count
            else:
                positive_reward_01 = 0.0

            normalized_total = positive_reward_01 - penalty_cost_01
            normalized_totals[var] = normalized_total
            composition_score_01 = self._signed_unit_to_score(normalized_total)

            flat[f'extra_penalty_bound_{var}'] = penalty_bound
            flat[f'extra_bonus_bound_{var}'] = bonus_bound
            flat[f'extra_incentive_bound_{var}'] = incentive_bound
            flat[f'extra_penalty_total_01_{var}'] = penalty_cost_01
            flat[f'extra_bonus_total_01_{var}'] = bonus_reward_01
            flat[f'extra_incentive_total_01_{var}'] = incentive_reward_01
            flat[f'extra_positive_total_01_{var}'] = positive_reward_01
            flat[f'extra_total_01_{var}'] = normalized_total
            flat[f'extra_composition_score_01_{var}'] = composition_score_01
            flat[f'extra_composition_cost_01_{var}'] = 1.0 - composition_score_01

        return normalized_totals, flat

    def _signed_unit_to_score(self, value):
        signed_value = min(1.0, max(-1.0, value))
        return 0.5 * (signed_value + 1.0)

    def _resolve_incentive_bucket(self, reward_mode, cfg):
        """
        Clasifica un incentivo por familia para facilitar el analisis.
        """
        if reward_mode == 'capture_directional_effort':
            return 'capture'
        if reward_mode == 'directional_dense':
            return 'direction'
        if reward_mode == 'directional_effort':
            return 'effort'
        if reward_mode == 'tanh_tracking' or 'track_weight' in cfg or 'base_weight' in cfg:
            return 'tracking'
        return 'tracking'

    def _compute_directional_alignment(self, error_value, response_value, direction_sign, error_setpoint):
        """
        Magnitud correctiva positiva basada en error firmado y respuesta.
        """
        centered_error = error_value - error_setpoint
        return max(0.0, direction_sign * centered_error * response_value)

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
    
    def _evaluate_bonuses(self, extra_reward_component, reward_component, termination_flag, current_time_sec, var_objs):
        """
        Evalúa bonus_approach iternando las listas de reglas pre-compiladas.
        """
        total_bonus = 0.0
        flat = {f'extra_bonus_goal_{v}': 0.0 for v in var_objs}
        flat.update({f'extra_bonus_band_{v}': 0.0 for v in var_objs})
        flat.update({f'extra_bonus_band_gate_{v}': 1.0 for v in var_objs})
        
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
        if self.band_ranges:
            for var, range_entries in self.band_ranges.items():
                series_list = [(extra_reward_component[vn], mn, mx) for vn, mn, mx in range_entries]
                
                if not series_list or any(len(s) == 0 for s, _, _ in series_list):
                    continue
                    
                min_len = min(len(s) for s, _, _ in series_list)
                steps_in_band = 0
                for t in range(min_len):
                    if all(mn <= s[t] <= mx for s, mn, mx in series_list):
                        steps_in_band += 1
                
                gate_factor = self._compute_band_gate_factor(reward_component, var)
                bonus = steps_in_band * self.band_per_step_bonus * gate_factor

                potential_total = self.accumulated_band_bonus[var] + bonus
                if potential_total > self.band_max_bonus[var]:
                    bonus = max(0.0, self.band_max_bonus[var] - self.accumulated_band_bonus[var])
                    
                self.accumulated_band_bonus[var] += bonus
                total_bonus += bonus
                flat[f'extra_bonus_band_{var}'] = bonus
                flat[f'extra_bonus_band_gate_{var}'] = gate_factor
                
        return total_bonus, flat
    
    def _compute_band_gate_factor(self, reward_component, var_obj):
        """
        Calcula la compuerta del bandwidth_bonus desde métricas interval-level.
        """
        if var_obj not in self.band_gate_rules:
            return 1.0

        gate_factor = 1.0
        for source_key, factor_type, gain, setpoint, min_factor, max_factor, invert in self.band_gate_rules[var_obj]:
            source_value = reward_component[source_key]
            distance = abs(source_value - setpoint)

            if factor_type == 'exp':
                factor = math.exp(-((distance / gain) ** 2)) if gain > 0.0 else 0.0
            else:
                factor = math.tanh(gain * distance)

            if invert:
                factor = 1.0 - factor

            gate_factor *= min(max_factor, max(min_factor, factor))

        return gate_factor

    def _evaluate_conditional(self, extra_reward_component, var_objs):
        """
        Evalúa conditional_approach iternando las listas de reglas pre-compiladas.
        """
        total_conditional = 0.0
        flat = {f'extra_conditional_dynamic_penalty_{v}': 0.0 for v in var_objs}
        flat.update({f'extra_conditional_dynamic_incentive_{v}': 0.0 for v in var_objs})
        flat.update({f'extra_conditional_dynamic_incentive_tracking_{v}': 0.0 for v in var_objs})
        flat.update({f'extra_conditional_dynamic_incentive_capture_{v}': 0.0 for v in var_objs})
        flat.update({f'extra_conditional_dynamic_incentive_direction_{v}': 0.0 for v in var_objs})
        flat.update({f'extra_conditional_dynamic_incentive_effort_{v}': 0.0 for v in var_objs})
        
        # 1. Dynamic Penalty
        for signal_key, var_obj, method, weight, cond_feature, cond_type, cond_gain, cond_sp in self.dyn_pen_rules:
            series = extra_reward_component.get(signal_key, [])
            condition_series = extra_reward_component.get(cond_feature, [])

            if not series or not condition_series:
                continue

            min_len = min(len(series), len(condition_series))
            step_penalties = []
            for idx in range(min_len):
                signal_value = abs(series[idx])
                cond_factor = self._compute_condition_factor(
                    cond_type,
                    condition_series[idx],
                    cond_gain,
                    cond_sp
                )
                if method == 'quadratic':
                    step_penalties.append(-weight * (signal_value ** 2) * cond_factor)
                else:
                    step_penalties.append(-weight * signal_value * cond_factor)

            penalty = sum(step_penalties) / len(step_penalties) if step_penalties else 0.0
            total_conditional += penalty
            flat[f'extra_conditional_dynamic_penalty_{var_obj}'] += penalty

        # 2. Dynamic Incentive
        for signal_key, var_obj, method, cfg in self.dyn_inc_rules:
            series = extra_reward_component.get(signal_key, [])
            if not series:
                continue

            if method == 'adaptative':
                reward_mode = cfg.get('reward_mode', 'legacy')
                bucket = self._resolve_incentive_bucket(reward_mode, cfg)
                uses_tracking_shape = (
                    reward_mode == 'tanh_tracking'
                    or 'track_weight' in cfg
                    or 'track_scaled' in cfg
                    or 'base_weight' in cfg
                    or 'base_scaled' in cfg
                )

                if reward_mode == 'capture_directional_effort':
                    x_series = extra_reward_component[cfg['x']]
                    u_series = extra_reward_component[cfg['u']]

                    min_len = min(len(x_series), len(u_series))
                    step_rewards = []

                    x_sp = cfg['x_sp']
                    capture_threshold = cfg['capture_threshold']
                    direction_sign = cfg['direction_sign']
                    capture_strength = cfg['capture_strength']
                    alignment_strength = cfg['alignment_strength']
                    weight = cfg['weight']

                    for idx in range(min_len):
                        e_val = x_series[idx] - x_sp
                        u_eff_val = u_series[idx]

                        capture_gate = math.tanh(
                            capture_strength * max(0.0, abs(e_val) - capture_threshold)
                        )
                        corrective_drive = max(0.0, direction_sign * e_val * u_eff_val)
                        alignment_gate = math.tanh(alignment_strength * corrective_drive)

                        reward = weight * capture_gate * alignment_gate
                        step_rewards.append(reward)

                    incentive = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
                elif reward_mode == 'directional_dense':
                    x_series = extra_reward_component.get(cfg['x'], [])
                    y_series = extra_reward_component.get(cfg.get('y', signal_key), [])
                    if not x_series or not y_series:
                        continue

                    min_len = min(len(x_series), len(y_series))
                    step_rewards = []
                    error_strength = cfg.get('error_strength', self._get_condition_gain(cfg))
                    error_setpoint = self._get_condition_setpoint(cfg)
                    direction_sign = cfg.get('direction_sign', cfg.get('target_sign', 1.0))
                    alignment_strength = cfg.get('alignment_strength', 1.0)
                    weight = cfg.get('weight', 0.0)

                    for idx in range(min_len):
                        x_val = x_series[idx]
                        y_val = y_series[idx]
                        alignment = self._compute_directional_alignment(
                            x_val, y_val, direction_sign, error_setpoint
                        )
                        reward = (
                            weight
                            * math.tanh(error_strength * abs(x_val - error_setpoint))
                            * math.tanh(alignment_strength * alignment)
                        )
                        step_rewards.append(reward)

                    incentive = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
                elif reward_mode == 'directional_effort':
                    x_series = extra_reward_component.get(cfg['x'], [])
                    direction_series = extra_reward_component.get(cfg['direction_y'], [])
                    effort_signal = cfg.get('u', cfg.get('effort_signal', signal_key))
                    effort_series = extra_reward_component.get(effort_signal, [])
                    if not x_series or not direction_series or not effort_series:
                        continue

                    min_len = min(len(x_series), len(direction_series), len(effort_series))
                    step_rewards = []
                    error_strength = cfg.get('error_strength', self._get_condition_gain(cfg))
                    error_setpoint = self._get_condition_setpoint(cfg)
                    direction_sign = cfg.get('direction_sign', cfg.get('target_sign', 1.0))
                    alignment_strength = cfg.get('alignment_strength', 1.0)
                    effort_strength = cfg.get('effort_strength', 1.0)
                    weight = cfg.get('weight', 0.0)

                    for idx in range(min_len):
                        x_val = x_series[idx]
                        direction_val = direction_series[idx]
                        effort_val = effort_series[idx]
                        alignment = self._compute_directional_alignment(
                            x_val, direction_val, direction_sign, error_setpoint
                        )
                        reward = (
                            weight
                            * math.tanh(error_strength * abs(x_val - error_setpoint))
                            * math.tanh(alignment_strength * alignment)
                            * math.tanh(effort_strength * abs(effort_val))
                        )
                        step_rewards.append(reward)

                    incentive = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
                elif uses_tracking_shape:
                    x_series = extra_reward_component.get(cfg['x'], [])
                    y_series = extra_reward_component.get(cfg.get('y', signal_key), [])
                    if not x_series or not y_series:
                        continue

                    min_len = min(len(x_series), len(y_series))
                    step_rewards = []
                    y_max = cfg['y_max']
                    strength = self._get_condition_gain(cfg)
                    x_sp = self._get_condition_setpoint(cfg)
                    target_sign = cfg.get('target_sign', 1.0)
                    track_weight = cfg.get('track_weight', 0.0)
                    track_scaled = cfg.get('track_scaled')
                    base_weight = cfg.get('base_weight', 0.0)
                    base_scaled = cfg.get('base_scaled')
                    base_setpoint = cfg.get('base_setpoint', 0.0)
                    band_low = cfg.get('band_low')
                    band_high = cfg.get('band_high')
                    band_bonus_per_step = cfg.get('band_bonus_per_step', 0.0)

                    for idx in range(min_len):
                        x_val = x_series[idx]
                        y_val = y_series[idx]
                        target = target_sign * y_max * math.tanh(strength * (x_val - x_sp))

                        reward = 0.0
                        if track_scaled is not None and track_weight > 0.0:
                            reward += self._compute_exp_reward(y_val, track_scaled, target, track_weight)
                        if base_scaled is not None and base_weight > 0.0:
                            reward += self._compute_exp_reward(y_val, base_scaled, base_setpoint, base_weight)
                        if (
                            band_bonus_per_step > 0.0
                            and band_low is not None
                            and band_high is not None
                            and band_low <= y_val <= band_high
                        ):
                            reward += band_bonus_per_step

                        step_rewards.append(reward)

                    incentive = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
                else:
                    last_value = abs(series[-1])
                    y_max = cfg['y_max']
                    x_series = extra_reward_component.get(cfg['x'], [])
                    x_val = abs(x_series[-1]) if x_series else 0.0

                    tanh_factor = self._compute_condition_factor(
                        cfg.get('type', 'tanh'),
                        x_val,
                        self._get_condition_gain(cfg),
                        self._get_condition_setpoint(cfg)
                    )
                    f_cfg = cfg['f_reward']
                    exp_term = self._compute_condition_factor(
                        f_cfg.get('type', 'exp'),
                        last_value,
                        self._get_condition_gain(f_cfg),
                        self._get_condition_setpoint(f_cfg)
                    )
                    incentive = y_max * tanh_factor * f_cfg['weight'] * exp_term
            else: # lineal
                bucket = 'tracking'
                last_value = abs(series[-1])
                y_max = cfg['y_max']
                x_max = cfg['x_max']
                incentive = y_max * min(1.0, last_value / x_max) if x_max > 0 else 0.0

            total_conditional += incentive
            flat[f'extra_conditional_dynamic_incentive_{var_obj}'] += incentive
            flat[f'extra_conditional_dynamic_incentive_{bucket}_{var_obj}'] += incentive
            
        return total_conditional, flat
    
    def get_extra_reward_params_record(self):
        """
        Retorna desglose plano de extra rewards aplicados.
        
        Returns:
            dict: Flat record con llaves canónicas per-var_obj
        """
        return self.last_extra_reward_params_record.copy()

