"""
cooperative_transition_evaluator.py

Responsabilidad:
Calcular la calidad cooperativa global y el progreso global de la transicion.

Este bloque no asigna credito fino. Solo expone R_GQ y Delta_G para que una
capa posterior decida si el progreso se entrega por controlador, por ganancia
o como recompensa comun.
"""


class CooperativeTransitionEvaluator:
    """
    Evaluador cooperativo de transicion.

    Produce costo/calidad global, potencial cooperativo por lazo y mejora
    global respecto de una linea base suavizada. La mezcla rho_G y la
    asignacion de Delta_G no pertenecen a este bloque.
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.config = config_main['reward_base']['reward_calculation'][
            'cooperative_transition_evaluator'
        ]
        self.normalized_reward_mode = self.config['normalized_reward_mode']
        if not self.normalized_reward_mode:
            raise ValueError("CooperativeTransitionEvaluator requires normalized_reward_mode=true")

        self.global_potential = self.config['global_potential']
        self.marginal_baseline = self.config['marginal_baseline']
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
                        f"Cooperative feature weight must be non-negative: "
                        f"{feature_name}_{var_obj}={weight}"
                    )
                feature_weight_sum += weight

            if (
                self.global_potential['strict_feature_weight_sum']
                and abs(feature_weight_sum - 1.0) > self.global_potential['weight_tolerance']
            ):
                raise ValueError(
                    f"Cooperative feature weights for {var_obj} must sum 1.0, "
                    f"got {feature_weight_sum}"
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
        self.last_components = {}

    def evaluate(self, reward_component, extra_reward_component, local_records):
        cooperative_potential_cost_by_var = self._compute_cooperative_potential_by_var(local_records)
        global_cost = self._compute_global_cost(cooperative_potential_cost_by_var)
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

        records = {
            'cooperative_global_cost_01': global_cost,
            'cooperative_global_quality_01': global_quality,
            'cooperative_global_quality_signed': global_quality_signed,
            'cooperative_global_baseline_cost_01': baseline_cost,
            'cooperative_global_cost_delta': global_cost_delta,
            'cooperative_global_marginal_signed': global_marginal_signed
        }

        for var_obj in self.var_objs:
            potential_cost = cooperative_potential_cost_by_var[var_obj]
            potential_quality = 1.0 - potential_cost
            potential_quality_signed = 1.0 - 2.0 * potential_cost

            records[f'cooperative_potential_cost_01_{var_obj}'] = potential_cost
            records[f'cooperative_potential_quality_01_{var_obj}'] = potential_quality
            records[f'cooperative_potential_quality_signed_{var_obj}'] = potential_quality_signed

        components = {
            'global_quality_signed': global_quality_signed,
            'global_marginal_signed': global_marginal_signed
        }

        self.last_record = records
        self.last_components = components
        return records, components

    def _compute_cooperative_potential_by_var(self, local_records):
        cooperative_potential_by_var = {}
        for var_obj in self.var_objs:
            cost = 0.0
            feature_weight_sum = 0.0
            for feature_name in self.feature_names:
                record_key = f'{feature_name}_{var_obj}'
                cost += self.feature_weights[feature_name][var_obj] * local_records[record_key]
                feature_weight_sum += self.feature_weights[feature_name][var_obj]

            if self.global_potential['strict_feature_weight_sum']:
                cooperative_potential_by_var[var_obj] = self._clip_01(cost)
            else:
                cooperative_potential_by_var[var_obj] = self._clip_01(cost / feature_weight_sum)
        return cooperative_potential_by_var

    def _compute_global_cost(self, cooperative_potential_by_var):
        weighted_cost = 0.0
        for var_obj in self.var_objs:
            weighted_cost += self.variable_weights[var_obj] * cooperative_potential_by_var[var_obj]
        if self.global_potential['strict_variable_weight_sum']:
            return self._clip_01(weighted_cost)
        return self._clip_01(weighted_cost / self.variable_weight_sum)

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))

    def _clip_signed(self, value):
        return min(1.0, max(-1.0, value))

    def get_state_record(self):
        return {
            'cooperative_global_baseline_cost_01': self.global_baseline_cost
        }
