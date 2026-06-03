"""
internal_risk_penalty_reward.py

Responsabilidad:
Calcular el riesgo estrategico interno del lazo a partir de primitivas ya
procesadas por la recompensa. Este bloque no premia bajo riesgo: solo expone
una penalizacion pura para la dimension estrategica de A/C/S.
"""


class InternalRiskPenaltyReward:
    """
    Penalizacion estrategica por riesgo operativo interno.

    El costo compuesto representa combinaciones peligrosas de expansion,
    memoria integral, perdida de autoridad e interferencia. La recompensa
    estrategica es R_S = -C_risk.
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.config = config_main['reward_base']['reward_calculation']['internal_risk_penalty']
        self.normalized_reward_mode = self.config['normalized_reward_mode']
        if not self.normalized_reward_mode:
            raise ValueError("InternalRiskPenaltyReward requires normalized_reward_mode=true")

        self.signals = self.config['signals']
        self.reduction = self.config['reduction']
        self.normalization_bounds = self.config['normalization_bounds']
        self.risk_weights = self.config['risk_weights']
        self.dynamic_expansive_weights = self.config['dynamic_expansive_weights']
        self.memory_integral_weights = self.config['memory_integral_weights']
        self.authority_weights = self.config['authority_weights']
        self.conflict_weights = self.config['conflict_weights']
        self.var_objs = self._extract_var_objs()
        self._compile_jobs()
        self.reset_episode()

    def _extract_var_objs(self):
        var_objs = []
        controllers_config = self.config_main['controller_base']['controllers']
        for controller_cfg in controllers_config.values():
            var_objs.append(controller_cfg['params']['name_objective_var'])
        return var_objs

    def _compile_jobs(self):
        self._validate_weight_group('risk_weights', self.risk_weights)
        self._validate_weight_group('dynamic_expansive_weights', self.dynamic_expansive_weights)
        self._validate_weight_group('memory_integral_weights', self.memory_integral_weights)
        self._validate_weight_group('authority_weights', self.authority_weights)
        self._validate_weight_group('conflict_weights', self.conflict_weights)

    def _validate_weight_group(self, group_name, weights):
        weight_sum = 0.0
        for key, value in weights.items():
            if value < 0.0:
                raise ValueError(f"{group_name}.{key} must be non-negative, got {value}")
            weight_sum += value

        if self.config['strict_weight_sum'] and abs(weight_sum - 1.0) > self.config['weight_tolerance']:
            raise ValueError(f"{group_name} weights must sum 1.0, got {weight_sum}")
        if weight_sum <= 0.0:
            raise ValueError(f"{group_name} weights must be positive")

    def reset_episode(self):
        self.last_record = {}
        self.last_components_by_var = {}

    def evaluate(self, local_records, extra_reward_component):
        records = {}
        components_by_var = {}

        for var_obj in self.var_objs:
            error_energy_cost = local_records[f'local_error_energy_cost_01_{var_obj}']
            expansive_cost = local_records[f'local_error_power_expansive_cost_01_{var_obj}']
            residual_cost = local_records[f'local_residual_velocity_cost_01_{var_obj}']
            integral_cost = local_records[f'local_integral_cost_01_{var_obj}']
            effort_cost = local_records[f'local_allocated_effort_cost_01_{var_obj}']

            conflict_series = extra_reward_component[
                self.signals['conflict_effort'].format(var_obj=var_obj)
            ]
            saturation_series = extra_reward_component[
                self.signals['saturation'].format(var_obj=var_obj)
            ]
            self._require_same_length(var_obj, conflict_series, saturation_series)

            conflict_effort_cost = self._normalized_series_cost(
                conflict_series,
                self.normalization_bounds['conflict_effort'][var_obj],
                self.reduction['conflict_effort'],
                f'conflict_effort_{var_obj}'
            )
            saturation_cost = self._reduce_unit_series(
                saturation_series,
                self.reduction['saturation'],
                f'saturation_{var_obj}'
            )

            dynamic_expansive_risk = self._dynamic_expansive_risk(
                expansive_cost,
                error_energy_cost,
                residual_cost
            )
            memory_integral_risk = self._memory_integral_risk(
                integral_cost,
                effort_cost,
                saturation_cost
            )
            authority_risk = self._authority_risk(
                saturation_cost,
                effort_cost,
                expansive_cost
            )
            conflict_risk = self._conflict_risk(
                conflict_effort_cost,
                expansive_cost
            )
            internal_risk_cost = self._clip_01(
                self.risk_weights['dynamic_expansive'] * dynamic_expansive_risk
                + self.risk_weights['memory_integral'] * memory_integral_risk
                + self.risk_weights['authority'] * authority_risk
                + self.risk_weights['conflict'] * conflict_risk
            )
            internal_risk_penalty_signed = -internal_risk_cost

            records[f'internal_risk_dynamic_expansive_cost_01_{var_obj}'] = dynamic_expansive_risk
            records[f'internal_risk_memory_integral_cost_01_{var_obj}'] = memory_integral_risk
            records[f'internal_risk_authority_cost_01_{var_obj}'] = authority_risk
            records[f'internal_risk_conflict_cost_01_{var_obj}'] = conflict_risk
            records[f'internal_risk_conflict_effort_cost_01_{var_obj}'] = conflict_effort_cost
            records[f'internal_risk_saturation_cost_01_{var_obj}'] = saturation_cost
            records[f'internal_risk_effort_cost_01_{var_obj}'] = effort_cost
            records[f'internal_risk_cost_01_{var_obj}'] = internal_risk_cost
            records[f'internal_risk_penalty_signed_{var_obj}'] = internal_risk_penalty_signed

            components_by_var[var_obj] = {
                'strategy_reward_signed': internal_risk_penalty_signed
            }

        self.last_record = records
        self.last_components_by_var = components_by_var
        return records, components_by_var

    def _dynamic_expansive_risk(self, expansive_cost, error_energy_cost, residual_cost):
        return self._clip_01(
            self.dynamic_expansive_weights['expansive'] * expansive_cost
            + self.dynamic_expansive_weights['expansive_error'] * expansive_cost * error_energy_cost
            + self.dynamic_expansive_weights['expansive_residual'] * expansive_cost * residual_cost
        )

    def _memory_integral_risk(self, integral_cost, effort_cost, saturation_cost):
        gate = (
            self.memory_integral_weights['base']
            + self.memory_integral_weights['effort'] * effort_cost
            + self.memory_integral_weights['saturation'] * saturation_cost
        )
        return self._clip_01(integral_cost * gate)

    def _authority_risk(self, saturation_cost, effort_cost, expansive_cost):
        return self._clip_01(
            self.authority_weights['saturation'] * saturation_cost
            + self.authority_weights['effort_expansive'] * effort_cost * expansive_cost
        )

    def _conflict_risk(self, conflict_effort_cost, expansive_cost):
        gate = (
            self.conflict_weights['base']
            + self.conflict_weights['expansive'] * expansive_cost
        )
        return self._clip_01(conflict_effort_cost * gate)

    def _normalized_series_cost(self, values, bound, method, key):
        if not values:
            raise ValueError(f"Cannot reduce internal risk series: {key}")
        if bound <= 0.0:
            raise ValueError(f"Internal risk normalization bound must be positive for {key}")

        if method == 'mean':
            return self._clip_01(abs(sum(values) / len(values)) / bound)
        if method == 'mean_abs' or method == 'abs':
            return self._clip_01(sum(abs(value) for value in values) / len(values) / bound)
        if method == 'rms':
            return self._clip_01((sum(value ** 2 for value in values) / len(values)) ** 0.5 / bound)
        if method == 'keep_last':
            return self._clip_01(abs(values[-1]) / bound)
        if method == 'accum':
            return self._clip_01(abs(sum(values)) / bound)

        raise ValueError(f"Unsupported internal risk reduction method: {method}")

    def _reduce_unit_series(self, values, method, key):
        if not values:
            raise ValueError(f"Cannot reduce internal risk series: {key}")

        if method == 'mean':
            return self._clip_01(sum(values) / len(values))
        if method == 'mean_abs' or method == 'abs':
            return self._clip_01(sum(abs(value) for value in values) / len(values))
        if method == 'rms':
            return self._clip_01((sum(value ** 2 for value in values) / len(values)) ** 0.5)
        if method == 'keep_last':
            return self._clip_01(abs(values[-1]))
        if method == 'proportion':
            return self._clip_01(sum(1.0 for value in values if value > 0.0) / len(values))

        raise ValueError(f"Unsupported internal risk reduction method: {method}")

    def _require_same_length(self, var_obj, first_series, second_series):
        if not first_series:
            raise ValueError(f"Cannot compute internal risk with empty series for {var_obj}")
        if len(first_series) != len(second_series):
            raise ValueError(
                f"Internal risk series must have same length for {var_obj}: "
                f"{len(first_series)} != {len(second_series)}"
            )

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))
