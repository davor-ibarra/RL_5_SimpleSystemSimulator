"""
coupling_penalty_reward.py

Responsabilidad:
Calcular la compatibilidad estrategica de la accion local dentro del sistema
acoplado.
"""


class CouplingPenaltyReward:
    """
    Penaliza conflicto entre contribuciones y saturacion atribuible.
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.config = config_main['reward_base']['reward_calculation']['coupling_penalty']
        self.normalized_reward_mode = self.config['normalized_reward_mode']
        if not self.normalized_reward_mode:
            raise ValueError("CouplingPenaltyReward requires normalized_reward_mode=true")

        self.signals = self.config['signals']
        self.reduction = self.config['reduction']
        self.normalization_bounds = self.config['normalization_bounds']
        self.cost_weights = self.config['cost_weights']
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
        self.weight_sum_by_var = {}
        for var_obj in self.var_objs:
            weight_sum = (
                self.cost_weights['conflict'][var_obj]
                + self.cost_weights['saturation'][var_obj]
            )
            if self.cost_weights['conflict'][var_obj] < 0.0:
                raise ValueError(f"Strategic conflict weight must be non-negative for {var_obj}")
            if self.cost_weights['saturation'][var_obj] < 0.0:
                raise ValueError(f"Strategic saturation weight must be non-negative for {var_obj}")
            if self.config['strict_weight_sum'] and abs(weight_sum - 1.0) > self.config['weight_tolerance']:
                raise ValueError(f"Strategic weights for {var_obj} must sum 1.0, got {weight_sum}")
            if weight_sum <= 0.0:
                raise ValueError(f"Strategic weights for {var_obj} must be positive")
            self.weight_sum_by_var[var_obj] = weight_sum

    def reset_episode(self):
        self.last_record = {}
        self.last_components_by_var = {}

    def evaluate(self, extra_reward_component):
        records = {}
        components_by_var = {}

        for var_obj in self.var_objs:
            conflict_series = extra_reward_component[
                self.signals['conflict_effort'].format(var_obj=var_obj)
            ]
            saturation_series = extra_reward_component[
                self.signals['saturation'].format(var_obj=var_obj)
            ]

            self._require_same_length(var_obj, conflict_series, saturation_series)

            conflict_cost = self._normalized_series_cost(
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
            strategic_cost = (
                self.cost_weights['conflict'][var_obj] * conflict_cost
                + self.cost_weights['saturation'][var_obj] * saturation_cost
            )
            if not self.config['strict_weight_sum']:
                strategic_cost = strategic_cost / self.weight_sum_by_var[var_obj]
            strategic_cost = self._clip_01(strategic_cost)
            strategy_reward_signed = -strategic_cost

            records[f'strategic_conflict_cost_01_{var_obj}'] = conflict_cost
            records[f'strategic_saturation_cost_01_{var_obj}'] = saturation_cost
            records[f'strategic_cost_01_{var_obj}'] = strategic_cost
            records[f'reward_strategy_signed_{var_obj}'] = strategy_reward_signed

            components_by_var[var_obj] = {
                'strategy_reward_signed': strategy_reward_signed
            }

        self.last_record = records
        self.last_components_by_var = components_by_var
        return records, components_by_var

    def _normalized_series_cost(self, values, bound, method, key):
        if not values:
            raise ValueError(f"Cannot reduce empty strategic series: {key}")
        if bound <= 0.0:
            raise ValueError(f"Strategic normalization bound must be positive for {key}")

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

        raise ValueError(f"Unsupported strategic reduction method: {method}")

    def _reduce_unit_series(self, values, method, key):
        if not values:
            raise ValueError(f"Cannot reduce empty strategic series: {key}")

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

        raise ValueError(f"Unsupported strategic reduction method: {method}")

    def _require_same_length(self, var_obj, first_series, second_series):
        if not first_series:
            raise ValueError(f"Cannot compute strategic reward with empty series for {var_obj}")
        if len(first_series) != len(second_series):
            raise ValueError(
                f"Strategic reward series must have same length for {var_obj}: "
                f"{len(first_series)} != {len(second_series)}"
            )

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))
