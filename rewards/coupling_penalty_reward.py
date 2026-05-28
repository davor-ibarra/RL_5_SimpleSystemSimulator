"""
coupling_penalty_reward.py

Responsabilidad:
Calcular penalizacion firmada por acoplamiento improductivo.
"""


class CouplingPenaltyReward:
    """
    Penaliza conflicto local y saturacion atribuible por controlador.
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.config = config_main['reward_base']['reward_calculation']['coupling_penalty']
        self.normalized_reward_mode = self.config['normalized_reward_mode']
        if not self.normalized_reward_mode:
            raise ValueError("CouplingPenaltyReward requires normalized_reward_mode=true")

        self.signals = self.config['signals']
        self.reduction = self.config['reduction']
        self.conflict_weight = self.config['conflict_weight']
        self.saturation_weight = self.config['saturation_weight']
        self.var_objs = self._extract_var_objs()
        self.last_record = {}
        self.last_components_by_var = {}

    def _extract_var_objs(self):
        var_objs = []
        controllers_config = self.config_main['controller_base']['controllers']
        for controller_cfg in controllers_config.values():
            var_objs.append(controller_cfg['params']['name_objective_var'])
        return var_objs

    def reset_episode(self):
        self.last_record = {}
        self.last_components_by_var = {}

    def evaluate(self, extra_reward_component, local_records):
        records = {
            'coupling_penalty_normalized_mode': float(self.normalized_reward_mode)
        }
        components_by_var = {}

        for var_obj in self.var_objs:
            conflict_cost = local_records[f'local_conflict_effort_cost_01_{var_obj}']
            saturation_series = extra_reward_component[
                self.signals['saturation'].format(var_obj=var_obj)
            ]
            saturation_cost = self._reduce_saturation_series(
                saturation_series,
                self.reduction['saturation'],
                f'saturation_{var_obj}'
            )
            penalty_cost = self._clip_01(
                self.conflict_weight[var_obj] * conflict_cost
                + self.saturation_weight[var_obj] * saturation_cost
            )
            penalty_signed = -penalty_cost

            records[f'coupling_conflict_cost_01_{var_obj}'] = conflict_cost
            records[f'coupling_saturation_cost_01_{var_obj}'] = saturation_cost
            records[f'coupling_penalty_signed_{var_obj}'] = penalty_signed

            components_by_var[var_obj] = {
                'coupling_penalty_signed': penalty_signed
            }

        self.last_record = records
        self.last_components_by_var = components_by_var
        return records, components_by_var

    def _reduce_saturation_series(self, values, method, key):
        if not values:
            raise ValueError(f"Cannot reduce empty coupling series: {key}")

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

        raise ValueError(f"Unsupported coupling reduction method: {method}")

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))
