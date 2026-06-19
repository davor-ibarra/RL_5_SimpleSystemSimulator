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
        self._compile_marginal_baseline()
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
        self.aggregation_mode = self.global_potential['aggregation_mode']
        if self.aggregation_mode == 'additive_sync_blend':
            self.sync_quality_config = self.global_potential['sync_quality']
            if not self.sync_quality_config['enabled']:
                raise ValueError(
                    "aggregation_mode=additive_sync_blend requires sync_quality.enabled=true"
            )
            self.sync_quality_weight = self.sync_quality_config['weight']
            self.sync_quality_epsilon = self.sync_quality_config['epsilon']
            self.sync_quality_variable_weights_mode = self.sync_quality_config[
                'variable_weights_mode'
            ]
            if self.sync_quality_weight < 0.0 or self.sync_quality_weight > 1.0:
                raise ValueError(
                    f"sync_quality.weight must be in [0, 1], got {self.sync_quality_weight}"
                )
            if self.sync_quality_epsilon <= 0.0:
                raise ValueError(
                    f"sync_quality.epsilon must be positive, got {self.sync_quality_epsilon}"
                )
            if self.sync_quality_variable_weights_mode not in ('uniform', 'global_potential'):
                raise ValueError(
                    "Unsupported sync_quality variable_weights_mode: "
                    f"{self.sync_quality_variable_weights_mode}"
                )
        elif self.aggregation_mode == 'weighted_cost':
            self.sync_quality_config = None
            self.sync_quality_weight = 0.0
            self.sync_quality_epsilon = 0.0
            self.sync_quality_variable_weights_mode = None
        else:
            raise ValueError(
                f"Unsupported cooperative global_potential aggregation_mode: {self.aggregation_mode}"
            )

        variable_weight_sum = 0.0
        for var_obj in self.var_objs:
            variable_weight = self.variable_weights[var_obj]
            if variable_weight < 0.0:
                raise ValueError(
                    f"Cooperative variable weight must be non-negative: {var_obj}={variable_weight}"
                )
            variable_weight_sum += variable_weight
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

    def _compile_marginal_baseline(self):
        self.marginal_baseline_mode = self.marginal_baseline['mode']

        if self.marginal_baseline_mode == 'smoothed_baseline':
            params = self.marginal_baseline['smoothed_baseline_params']
            self._validate_unit_interval(
                'marginal_baseline.smoothed_baseline_params.initial_cost_01',
                params['initial_cost_01']
            )
            self._validate_unit_interval(
                'marginal_baseline.smoothed_baseline_params.tau',
                params['tau']
            )
            self._validate_positive(
                'marginal_baseline.smoothed_baseline_params.delta_cost_ref',
                params['delta_cost_ref']
            )
        elif self.marginal_baseline_mode == 'asymmetric_smoothed_baseline':
            params = self.marginal_baseline['asymmetric_smoothed_baseline_params']
            self._validate_unit_interval(
                'marginal_baseline.asymmetric_smoothed_baseline_params.initial_cost_01',
                params['initial_cost_01']
            )
            self._validate_unit_interval(
                'marginal_baseline.asymmetric_smoothed_baseline_params.tau_improve',
                params['tau_improve']
            )
            self._validate_unit_interval(
                'marginal_baseline.asymmetric_smoothed_baseline_params.tau_worsen',
                params['tau_worsen']
            )
            self._validate_positive(
                'marginal_baseline.asymmetric_smoothed_baseline_params.delta_cost_ref',
                params['delta_cost_ref']
            )
            if params['tau_improve'] <= params['tau_worsen']:
                raise ValueError(
                    "asymmetric_smoothed_baseline requires tau_improve > tau_worsen"
                )
        else:
            raise ValueError(
                f"Unsupported cooperative marginal baseline mode: {self.marginal_baseline_mode}"
            )

        self.marginal_baseline_params = params

    def reset_episode(self):
        self.global_baseline_cost = self.marginal_baseline_params['initial_cost_01']
        self.last_record = {}
        self.last_components = {}

    def evaluate(self, reward_component, extra_reward_component, local_records):
        cooperative_potential_cost_by_var = self._compute_cooperative_potential_by_var(local_records)
        global_potential_values = self._compute_global_potential_values(
            cooperative_potential_cost_by_var
        )
        global_cost = global_potential_values['global_cost']
        global_quality = global_potential_values['global_quality']
        global_quality_signed = 1.0 - 2.0 * global_cost

        baseline_cost = self.global_baseline_cost
        global_cost_delta = baseline_cost - global_cost
        global_marginal_signed = self._clip_signed(
            global_cost_delta / self.marginal_baseline_params['delta_cost_ref']
        )
        next_baseline_cost, baseline_update_tau = self._update_global_baseline(
            baseline_cost,
            global_cost
        )
        self.global_baseline_cost = next_baseline_cost

        records = {
            'cooperative_global_cost_01': global_cost,
            'cooperative_global_quality_01': global_quality,
            'cooperative_global_quality_signed': global_quality_signed,
            'cooperative_global_linear_cost_01': global_potential_values['linear_cost'],
            'cooperative_global_linear_quality_01': global_potential_values['linear_quality'],
            'cooperative_global_linear_quality_signed': (
                2.0 * global_potential_values['linear_quality'] - 1.0
            ),
            'cooperative_global_sync_cost_01': global_potential_values['sync_cost'],
            'cooperative_global_sync_quality_01': global_potential_values['sync_quality'],
            'cooperative_global_sync_quality_signed': (
                2.0 * global_potential_values['sync_quality'] - 1.0
            ),
            'cooperative_global_sync_weight': self.sync_quality_weight,
            'cooperative_global_baseline_cost_01': baseline_cost,
            'cooperative_global_next_baseline_cost_01': next_baseline_cost,
            'cooperative_global_baseline_update_tau': baseline_update_tau,
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

    def _compute_global_potential_values(self, cooperative_potential_by_var):
        linear_cost = self._compute_linear_global_cost(cooperative_potential_by_var)
        linear_quality = 1.0 - linear_cost

        if self.aggregation_mode == 'weighted_cost':
            sync_quality = linear_quality
            global_quality = linear_quality
        elif self.aggregation_mode == 'additive_sync_blend':
            sync_quality = self._compute_sync_quality(cooperative_potential_by_var)
            global_quality = self._clip_01(
                (1.0 - self.sync_quality_weight) * linear_quality
                + self.sync_quality_weight * sync_quality
            )
        else:
            raise ValueError(
                f"Unsupported cooperative global_potential aggregation_mode: {self.aggregation_mode}"
            )

        return {
            'linear_cost': linear_cost,
            'linear_quality': linear_quality,
            'sync_cost': 1.0 - sync_quality,
            'sync_quality': sync_quality,
            'global_cost': 1.0 - global_quality,
            'global_quality': global_quality
        }

    def _compute_linear_global_cost(self, cooperative_potential_by_var):
        weighted_cost = 0.0
        for var_obj in self.var_objs:
            weighted_cost += self.variable_weights[var_obj] * cooperative_potential_by_var[var_obj]
        if self.global_potential['strict_variable_weight_sum']:
            return self._clip_01(weighted_cost)
        return self._clip_01(weighted_cost / self.variable_weight_sum)

    def _compute_sync_quality(self, cooperative_potential_by_var):
        sync_quality = 1.0

        if self.sync_quality_variable_weights_mode == 'uniform':
            variable_weight = 1.0 / len(self.var_objs)
            for var_obj in self.var_objs:
                local_quality = 1.0 - cooperative_potential_by_var[var_obj]
                sync_quality *= (local_quality + self.sync_quality_epsilon) ** variable_weight
            return self._clip_01(sync_quality)

        if self.sync_quality_variable_weights_mode == 'global_potential':
            for var_obj in self.var_objs:
                variable_weight = self.variable_weights[var_obj]
                if not self.global_potential['strict_variable_weight_sum']:
                    variable_weight = variable_weight / self.variable_weight_sum
                local_quality = 1.0 - cooperative_potential_by_var[var_obj]
                sync_quality *= (local_quality + self.sync_quality_epsilon) ** variable_weight
            return self._clip_01(sync_quality)

        raise ValueError(
            "Unsupported sync_quality variable_weights_mode: "
            f"{self.sync_quality_variable_weights_mode}"
        )

    def _update_global_baseline(self, baseline_cost, global_cost):
        if self.marginal_baseline_mode == 'smoothed_baseline':
            tau = self.marginal_baseline_params['tau']
        elif self.marginal_baseline_mode == 'asymmetric_smoothed_baseline':
            if global_cost < baseline_cost:
                tau = self.marginal_baseline_params['tau_improve']
            else:
                tau = self.marginal_baseline_params['tau_worsen']
        else:
            raise ValueError(
                f"Unsupported cooperative marginal baseline mode: {self.marginal_baseline_mode}"
            )

        next_baseline_cost = self._clip_01(
            (1.0 - tau) * baseline_cost + tau * global_cost
        )
        return next_baseline_cost, tau

    def _validate_unit_interval(self, key, value):
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{key} must be in [0, 1], got {value}")

    def _validate_positive(self, key, value):
        if value <= 0.0:
            raise ValueError(f"{key} must be positive, got {value}")

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))

    def _clip_signed(self, value):
        return min(1.0, max(-1.0, value))

    def get_state_record(self):
        return {
            'cooperative_global_baseline_cost_01': self.global_baseline_cost
        }
