"""
cooperative_control_quality_reward.py

Responsabilidad:
Calcular costo cooperativo absoluto, mejora marginal global y asignacion de
credito entre controladores.
"""


class CooperativeControlQualityReward:
    """
    Recompensa cooperativa agnostica al sistema.

    Usa costos locales ya normalizados para construir un potencial global y
    reparte la senal marginal global segun credito helpful/harmful declarado.
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.config = config_main['reward_base']['reward_calculation']['cooperative_control_quality']
        self.normalized_reward_mode = self.config['normalized_reward_mode']
        if not self.normalized_reward_mode:
            raise ValueError("CooperativeControlQualityReward requires normalized_reward_mode=true")

        self.global_potential = self.config['global_potential']
        self.marginal_baseline = self.config['marginal_baseline']
        self.credit_assignment = self.config['credit_assignment']
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
        self.feature_names = self.global_potential['features']
        self.feature_weights = self.global_potential['feature_weights']
        self.variable_weights = self.global_potential['variable_weights']

        variable_weight_sum = 0.0
        for var_obj in self.var_objs:
            variable_weight_sum += self.variable_weights[var_obj]
            feature_weight_sum = 0.0
            for feature_name in self.feature_names:
                weight = self.feature_weights[feature_name][var_obj]
                if weight < 0.0:
                    raise ValueError(
                        f"Cooperative feature weight must be non-negative: {feature_name}_{var_obj}={weight}"
                    )
                feature_weight_sum += weight

            if (
                self.global_potential['strict_feature_weight_sum']
                and abs(feature_weight_sum - 1.0) > self.global_potential['weight_tolerance']
            ):
                raise ValueError(
                    f"Cooperative feature weights for {var_obj} must sum 1.0, got {feature_weight_sum}"
                )
            if feature_weight_sum <= 0.0:
                raise ValueError(f"Cooperative feature weights for {var_obj} must be positive")

        if (
            self.global_potential['strict_variable_weight_sum']
            and abs(variable_weight_sum - 1.0) > self.global_potential['weight_tolerance']
        ):
            raise ValueError(f"Cooperative variable weights must sum 1.0, got {variable_weight_sum}")
        if variable_weight_sum <= 0.0:
            raise ValueError("Cooperative variable weights must be positive")

        self.variable_weight_sum = variable_weight_sum

    def reset_episode(self):
        self.global_baseline_cost = self.marginal_baseline['initial_cost_01']
        self.last_record = {}
        self.last_components_by_var = {}

    def evaluate(self, reward_component, extra_reward_component, local_records):
        cooperative_cost_by_var = self._compute_cooperative_cost_by_var(local_records)
        global_cost = self._compute_global_cost(cooperative_cost_by_var)
        global_quality = 1.0 - global_cost
        global_quality_signed = 1.0 - 2.0 * global_cost

        baseline_cost = self.global_baseline_cost
        global_cost_delta = baseline_cost - global_cost
        global_marginal_signed = self._clip_signed(
            global_cost_delta / self.marginal_baseline['delta_cost_ref']
        )
        self.global_baseline_cost = self._clip_01(
            (1.0 - self.marginal_baseline['tau']) * baseline_cost
            + self.marginal_baseline['tau'] * global_cost
        )

        helpful_raw, harmful_raw = self._compute_credit_raw(extra_reward_component)
        helpful_share = self._compute_credit_share(helpful_raw)
        harmful_share = self._compute_credit_share(harmful_raw)

        records = {
            'cooperative_control_quality_normalized_mode': float(self.normalized_reward_mode),
            'cooperative_global_cost_01': global_cost,
            'cooperative_global_quality_01': global_quality,
            'cooperative_global_quality_signed': global_quality_signed,
            'cooperative_global_baseline_cost_01': baseline_cost,
            'cooperative_global_cost_delta': global_cost_delta,
            'cooperative_global_marginal_signed': global_marginal_signed
        }
        components_by_var = {}

        for var_obj in self.var_objs:
            expected_share = self.credit_assignment['expected_share'][var_obj]
            if expected_share <= 0.0:
                raise ValueError(f"credit_assignment.expected_share must be positive for {var_obj}")

            helpful_multiplier = min(
                self.credit_assignment['max_multiplier'],
                helpful_share[var_obj] / expected_share
            )
            harmful_multiplier = min(
                self.credit_assignment['max_multiplier'],
                harmful_share[var_obj] / expected_share
            )
            if global_marginal_signed >= 0.0:
                cooperative_marginal_signed = self._clip_signed(
                    global_marginal_signed * helpful_multiplier
                )
            else:
                cooperative_marginal_signed = self._clip_signed(
                    global_marginal_signed * harmful_multiplier
                )

            cooperative_cost = cooperative_cost_by_var[var_obj]
            cooperative_quality = 1.0 - cooperative_cost
            cooperative_quality_signed = 1.0 - 2.0 * cooperative_cost

            records[f'cooperative_cost_01_{var_obj}'] = cooperative_cost
            records[f'cooperative_quality_01_{var_obj}'] = cooperative_quality
            records[f'cooperative_quality_signed_{var_obj}'] = cooperative_quality_signed
            records[f'credit_helpful_raw_{var_obj}'] = helpful_raw[var_obj]
            records[f'credit_harmful_raw_{var_obj}'] = harmful_raw[var_obj]
            records[f'credit_helpful_share_{var_obj}'] = helpful_share[var_obj]
            records[f'credit_harmful_share_{var_obj}'] = harmful_share[var_obj]
            records[f'credit_expected_share_{var_obj}'] = expected_share
            records[f'credit_helpful_multiplier_{var_obj}'] = helpful_multiplier
            records[f'credit_harmful_multiplier_{var_obj}'] = harmful_multiplier
            records[f'cooperative_marginal_quality_signed_{var_obj}'] = cooperative_marginal_signed

            components_by_var[var_obj] = {
                'cooperative_cost_01': cooperative_cost,
                'cooperative_quality_signed': cooperative_quality_signed,
                'cooperative_marginal_quality_signed': cooperative_marginal_signed
            }

        self.last_record = records
        self.last_components_by_var = components_by_var
        return records, components_by_var

    def _compute_cooperative_cost_by_var(self, local_records):
        cooperative_cost_by_var = {}
        for var_obj in self.var_objs:
            cost = 0.0
            for feature_name in self.feature_names:
                record_key = f'{feature_name}_{var_obj}'
                cost += self.feature_weights[feature_name][var_obj] * local_records[record_key]
            cooperative_cost_by_var[var_obj] = self._clip_01(cost)
        return cooperative_cost_by_var

    def _compute_global_cost(self, cooperative_cost_by_var):
        weighted_cost = 0.0
        for var_obj in self.var_objs:
            weighted_cost += self.variable_weights[var_obj] * cooperative_cost_by_var[var_obj]
        if self.global_potential['strict_variable_weight_sum']:
            return self._clip_01(weighted_cost)
        return self._clip_01(weighted_cost / self.variable_weight_sum)

    def _compute_credit_raw(self, extra_reward_component):
        helpful_raw = {}
        harmful_raw = {}
        reduction_mode = self.credit_assignment['reduction_mode']

        for var_obj in self.var_objs:
            error_series = extra_reward_component[
                self.credit_assignment['error_signal'].format(var_obj=var_obj)
            ]
            effort_series = extra_reward_component[
                self.credit_assignment['allocated_effort_signal'].format(var_obj=var_obj)
            ]
            conflict_series = extra_reward_component[
                self.credit_assignment['conflict_effort_signal'].format(var_obj=var_obj)
            ]
            self._require_same_length(var_obj, error_series, effort_series)
            self._require_same_length(var_obj, error_series, conflict_series)

            helpful_sum = 0.0
            harmful_sum = 0.0
            correction_sign = self.credit_assignment['correction_signs'][var_obj]
            conflict_weight = self.credit_assignment['harmful_conflict_weight'][var_obj]
            for idx in range(len(error_series)):
                alignment = correction_sign * error_series[idx] * effort_series[idx]
                helpful_sum += max(0.0, alignment)
                harmful_sum += max(0.0, -alignment) + conflict_weight * abs(conflict_series[idx])

            if reduction_mode == 'average':
                helpful_raw[var_obj] = helpful_sum / len(error_series)
                harmful_raw[var_obj] = harmful_sum / len(error_series)
            elif reduction_mode == 'sum':
                helpful_raw[var_obj] = helpful_sum
                harmful_raw[var_obj] = harmful_sum
            else:
                raise ValueError(f"Unsupported cooperative credit reduction_mode: {reduction_mode}")

        return helpful_raw, harmful_raw

    def _compute_credit_share(self, raw_credit):
        raw_total = sum(raw_credit[var_obj] for var_obj in self.var_objs)
        denominator = self.credit_assignment['epsilon'] + raw_total
        if denominator <= 0.0:
            raise ValueError("Cooperative credit denominator must be positive")
        return {
            var_obj: raw_credit[var_obj] / denominator
            for var_obj in self.var_objs
        }

    def _require_same_length(self, var_obj, first_series, second_series):
        if not first_series:
            raise ValueError(f"Cannot compute cooperative credit with empty series for {var_obj}")
        if len(first_series) != len(second_series):
            raise ValueError(
                f"Cooperative credit series must have same length for {var_obj}: "
                f"{len(first_series)} != {len(second_series)}"
            )

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))

    def _clip_signed(self, value):
        return min(1.0, max(-1.0, value))

    def get_state_record(self):
        return {
            'cooperative_global_baseline_cost_01': self.global_baseline_cost
        }
