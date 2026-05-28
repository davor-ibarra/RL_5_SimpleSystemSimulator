"""
local_control_quality_reward.py

Responsabilidad:
Calcular costo local absoluto y calidad marginal local por controlador.

El modulo no conoce el sistema dinamico. Consume senales declaradas por
plantilla y escalas analiticas fijadas en YAML.
"""

import math


class LocalControlQualityReward:
    """
    Recompensa local por calidad del lazo.

    Expone costos en [0, 1], calidad firmada en [-1, 1] y una senal marginal
    respecto de una linea base suavizada declarada por controlador.
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.config = config_main['reward_base']['reward_calculation']['local_control_quality']
        self.normalized_reward_mode = self.config['normalized_reward_mode']
        if not self.normalized_reward_mode:
            raise ValueError("LocalControlQualityReward requires normalized_reward_mode=true")

        self.signals = self.config['signals']
        self.aggregation = self.config['aggregation']
        self.normalization_bounds = self.config['normalization_bounds']
        self.cost_weights = self.config['cost_weights']
        self.residual_velocity_gate = self.config['residual_velocity_gate']
        self.marginal_baseline = self.config['marginal_baseline']

        self.var_objs = self._extract_var_objs()
        self.cost_feature_names = [
            'error_energy',
            'error_power_expansive',
            'residual_velocity',
            'integral_memory',
            'allocated_effort',
            'conflict_effort'
        ]
        self._compile_jobs()
        self.reset_episode()

    def _extract_var_objs(self):
        var_objs = []
        controllers_config = self.config_main['controller_base']['controllers']
        for controller_cfg in controllers_config.values():
            var_objs.append(controller_cfg['params']['name_objective_var'])
        return var_objs

    def _compile_jobs(self):
        self.weight_sum_by_var = {}
        for var_obj in self.var_objs:
            weight_sum = 0.0
            for feature_name in self.cost_feature_names:
                weight = self.cost_weights[feature_name][var_obj]
                if weight < 0.0:
                    raise ValueError(
                        f"Local cost weight must be non-negative: {feature_name}_{var_obj}={weight}"
                    )
                weight_sum += weight

            if self.config['strict_weight_sum'] and abs(weight_sum - 1.0) > self.config['weight_tolerance']:
                raise ValueError(f"Local cost weights for {var_obj} must sum 1.0, got {weight_sum}")
            if weight_sum <= 0.0:
                raise ValueError(f"Local cost weights for {var_obj} must be positive")
            self.weight_sum_by_var[var_obj] = weight_sum

    def reset_episode(self):
        self.local_baseline_cost = {
            var_obj: self.marginal_baseline['initial_cost_01'][var_obj]
            for var_obj in self.var_objs
        }
        self.last_record = {}
        self.last_components_by_var = {}

    def evaluate(self, reward_component, extra_reward_component):
        records = {
            'local_control_quality_normalized_mode': float(self.normalized_reward_mode)
        }
        components_by_var = {}

        for var_obj in self.var_objs:
            error_series = extra_reward_component[self.signals['error'].format(var_obj=var_obj)]
            derivative_error_series = extra_reward_component[
                self.signals['derivative_error'].format(var_obj=var_obj)
            ]
            integral_error_series = extra_reward_component[
                self.signals['integral_error'].format(var_obj=var_obj)
            ]
            allocated_effort_series = extra_reward_component[
                self.signals['allocated_effort'].format(var_obj=var_obj)
            ]
            conflict_effort_series = extra_reward_component[
                self.signals['conflict_effort'].format(var_obj=var_obj)
            ]

            self._require_same_length(var_obj, error_series, derivative_error_series)
            error_power_expansive_series = []
            error_power_dissipative_series = []
            for idx in range(len(error_series)):
                error_power = error_series[idx] * derivative_error_series[idx]
                error_power_expansive_series.append(max(0.0, error_power))
                error_power_dissipative_series.append(max(0.0, -error_power))

            error_energy_cost = self._normalized_series_cost(
                error_series,
                self.normalization_bounds['error'][var_obj],
                self.aggregation['error_energy'],
                f'error_{var_obj}'
            )
            error_power_bound = self.normalization_bounds['error_power'][var_obj]
            error_power_expansive_cost = self._normalized_series_cost(
                error_power_expansive_series,
                error_power_bound,
                self.aggregation['error_power_expansive'],
                f'error_power_expansive_{var_obj}'
            )
            error_power_dissipative = self._normalized_series_cost(
                error_power_dissipative_series,
                error_power_bound,
                self.aggregation['error_power_dissipative'],
                f'error_power_dissipative_{var_obj}'
            )
            derivative_cost = self._normalized_series_cost(
                derivative_error_series,
                self.normalization_bounds['derivative_error'][var_obj],
                self.aggregation['residual_velocity'],
                f'derivative_error_{var_obj}'
            )
            residual_velocity_gate = math.exp(
                -((error_energy_cost / self.residual_velocity_gate['sigma_error'][var_obj]) ** 2)
            )
            residual_velocity_cost = self._clip_01(residual_velocity_gate * derivative_cost)
            integral_cost = self._normalized_series_cost(
                integral_error_series,
                self.normalization_bounds['integral_error'][var_obj],
                self.aggregation['integral_memory'],
                f'integral_error_{var_obj}'
            )
            allocated_effort_cost = self._normalized_series_cost(
                allocated_effort_series,
                self.normalization_bounds['allocated_effort'][var_obj],
                self.aggregation['allocated_effort'],
                f'allocated_effort_{var_obj}'
            )
            conflict_effort_cost = self._normalized_series_cost(
                conflict_effort_series,
                self.normalization_bounds['conflict_effort'][var_obj],
                self.aggregation['conflict_effort'],
                f'conflict_effort_{var_obj}'
            )

            local_cost = self._weighted_local_cost(
                var_obj,
                error_energy_cost,
                error_power_expansive_cost,
                residual_velocity_cost,
                integral_cost,
                allocated_effort_cost,
                conflict_effort_cost
            )
            local_quality = 1.0 - local_cost
            local_quality_signed = 1.0 - 2.0 * local_cost

            baseline_cost = self.local_baseline_cost[var_obj]
            local_cost_delta = baseline_cost - local_cost
            local_marginal_quality_signed = self._clip_signed(
                local_cost_delta / self.marginal_baseline['delta_cost_ref'][var_obj]
            )
            next_baseline = (
                (1.0 - self.marginal_baseline['tau'][var_obj]) * baseline_cost
                + self.marginal_baseline['tau'][var_obj] * local_cost
            )
            self.local_baseline_cost[var_obj] = self._clip_01(next_baseline)

            records[f'local_error_energy_cost_01_{var_obj}'] = error_energy_cost
            records[f'local_error_power_expansive_cost_01_{var_obj}'] = error_power_expansive_cost
            records[f'local_error_power_dissipative_01_{var_obj}'] = error_power_dissipative
            records[f'local_residual_velocity_gate_{var_obj}'] = residual_velocity_gate
            records[f'local_residual_velocity_cost_01_{var_obj}'] = residual_velocity_cost
            records[f'local_integral_cost_01_{var_obj}'] = integral_cost
            records[f'local_allocated_effort_cost_01_{var_obj}'] = allocated_effort_cost
            records[f'local_conflict_effort_cost_01_{var_obj}'] = conflict_effort_cost
            records[f'local_cost_01_{var_obj}'] = local_cost
            records[f'local_quality_01_{var_obj}'] = local_quality
            records[f'local_quality_signed_{var_obj}'] = local_quality_signed
            records[f'local_baseline_cost_01_{var_obj}'] = baseline_cost
            records[f'local_cost_delta_{var_obj}'] = local_cost_delta
            records[f'local_marginal_quality_signed_{var_obj}'] = local_marginal_quality_signed

            components_by_var[var_obj] = {
                'local_cost_01': local_cost,
                'local_quality_signed': local_quality_signed,
                'local_marginal_quality_signed': local_marginal_quality_signed
            }

        self.last_record = records
        self.last_components_by_var = components_by_var
        return records, components_by_var

    def _weighted_local_cost(
        self,
        var_obj,
        error_energy_cost,
        error_power_expansive_cost,
        residual_velocity_cost,
        integral_cost,
        allocated_effort_cost,
        conflict_effort_cost
    ):
        weighted_cost = (
            self.cost_weights['error_energy'][var_obj] * error_energy_cost
            + self.cost_weights['error_power_expansive'][var_obj] * error_power_expansive_cost
            + self.cost_weights['residual_velocity'][var_obj] * residual_velocity_cost
            + self.cost_weights['integral_memory'][var_obj] * integral_cost
            + self.cost_weights['allocated_effort'][var_obj] * allocated_effort_cost
            + self.cost_weights['conflict_effort'][var_obj] * conflict_effort_cost
        )
        if self.config['strict_weight_sum']:
            return self._clip_01(weighted_cost)
        return self._clip_01(weighted_cost / self.weight_sum_by_var[var_obj])

    def _normalized_series_cost(self, values, bound, method, key):
        if not values:
            raise ValueError(f"Cannot reduce empty local reward series: {key}")
        if bound <= 0.0:
            raise ValueError(f"Local normalization bound must be positive for {key}")

        if method == 'rms':
            value = math.sqrt(sum(sample ** 2 for sample in values) / len(values))
            return self._clip_01(value / bound)
        if method == 'mean_squared':
            value = sum(sample ** 2 for sample in values) / len(values)
            return self._clip_01(value / (bound ** 2))
        if method == 'mean_abs' or method == 'abs':
            value = sum(abs(sample) for sample in values) / len(values)
            return self._clip_01(value / bound)
        if method == 'mean':
            value = sum(values) / len(values)
            return self._clip_01(abs(value) / bound)
        if method == 'keep_last':
            return self._clip_01(abs(values[-1]) / bound)
        if method == 'accum':
            return self._clip_01(abs(sum(values)) / bound)

        raise ValueError(f"Unsupported local reward aggregation method: {method}")

    def _require_same_length(self, var_obj, first_series, second_series):
        if len(first_series) != len(second_series):
            raise ValueError(
                f"Local reward error and derivative series must have same length for {var_obj}: "
                f"{len(first_series)} != {len(second_series)}"
            )

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))

    def _clip_signed(self, value):
        return min(1.0, max(-1.0, value))

    def get_state_record(self):
        state_record = {}
        for var_obj in self.var_objs:
            state_record[f'local_baseline_cost_01_{var_obj}'] = self.local_baseline_cost[var_obj]
        return state_record
