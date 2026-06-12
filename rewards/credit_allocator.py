"""
credit_allocator.py

Responsabilidad:
Asignar el progreso cooperativo global al nivel de aprendizaje activo.

La calidad global y Delta_G vienen del CooperativeTransitionEvaluator. Este
bloque decide una sola granularidad por corrida:
- controller_reward: credito por lazo/controlador.
- agent_reward: credito fino por agente de ganancia.
- unique_for_all_reward: credito comun, sin asignacion individual.
"""


class CreditAllocator:
    """
    Asignador cooperativo en dominio signed_unit.

    No compone A/C/S y no calcula el potencial global. Solo transforma Delta_G
    en el marginal cooperativo que luego usa RewardComposer.
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.config = config_main['reward_base']['reward_calculation']['credit_allocator']
        self.reward_approach = config_main['reward_base']['reward_config']['reward_approach']
        self.controller_assignment = self.config['controller_assignment']
        self.agent_assignment = self.config['agent_assignment']
        self.common_assignment = self.config['common_assignment']
        self.var_objs = self._extract_var_objs()
        self.agent_to_var_obj_map, self.agent_to_gain_type_map = self._build_agent_maps()
        self.var_obj_to_agent_names = self._build_var_obj_to_agent_names()
        self.last_record = {}

    def _extract_var_objs(self):
        var_objs = []
        controllers_config = self.config_main['controller_base']['controllers']
        for controller_cfg in controllers_config.values():
            var_objs.append(controller_cfg['params']['name_objective_var'])
        return var_objs

    def _build_agent_maps(self):
        mapping = {}
        gain_mapping = {}
        controllers_config = self.config_main['controller_base']['controllers']
        var_obj_to_controller = {}
        for controller_name, controller_cfg in controllers_config.items():
            var_obj = controller_cfg['params']['name_objective_var']
            var_obj_to_controller[var_obj] = controller_name

        agents_config = self.config_main['agent_base']['agent_config']['agents']
        for agent_name, agent_cfg in agents_config.items():
            if not agent_cfg['enabled_agent']:
                continue

            parts = agent_name.split('_', 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Enabled agent '{agent_name}' must follow the canonical name "
                    "{gain_type}_{var_obj}"
                )

            gain_type, var_obj = parts
            if var_obj not in var_obj_to_controller:
                raise ValueError(
                    f"Enabled agent '{agent_name}' references var_obj '{var_obj}', "
                    "but no controller declares that objective variable"
                )

            mapping[agent_name] = var_obj
            gain_mapping[agent_name] = gain_type

        return mapping, gain_mapping

    def _build_var_obj_to_agent_names(self):
        mapping = {
            var_obj: []
            for var_obj in self.var_objs
        }
        for agent_name, var_obj in self.agent_to_var_obj_map.items():
            mapping[var_obj].append(agent_name)
        return mapping

    def reset_episode(self):
        self.last_record = {}

    def allocate(self, transition_components, extra_reward_component, actions_dict):
        if self.reward_approach == 'controller_reward':
            records, controller_components = self._allocate_controller_credit(
                transition_components,
                extra_reward_component
            )
            self.last_record = records
            return records, controller_components, {}, {}

        if self.reward_approach == 'agent_reward':
            records, agent_components = self._allocate_agent_credit(
                transition_components,
                extra_reward_component,
                actions_dict
            )
            self.last_record = records
            return records, {}, agent_components, {}

        if self.reward_approach == 'unique_for_all_reward':
            records, common_component = self._allocate_common_credit(transition_components)
            self.last_record = records
            return records, {}, {}, common_component

        raise ValueError(f"Unsupported reward_approach for credit allocation: {self.reward_approach}")

    def _allocate_controller_credit(self, transition_components, extra_reward_component):
        mode = self.controller_assignment['mode']
        if mode != 'loop_effort_alignment':
            raise ValueError(f"Unsupported controller credit assignment mode: {mode}")

        records = {}
        components_by_var = {}
        config = self.controller_assignment
        signals = config['signals']

        for var_obj in self.var_objs:
            error_series = extra_reward_component[signals['error'].format(var_obj=var_obj)]
            effort_series = extra_reward_component[
                signals['allocated_effort'].format(var_obj=var_obj)
            ]
            conflict_series = extra_reward_component[
                signals['conflict_effort'].format(var_obj=var_obj)
            ]
            self._require_same_length(var_obj, error_series, effort_series, 'controller credit')
            self._require_same_length(var_obj, error_series, conflict_series, 'controller credit')

            helpful_sum = 0.0
            harmful_sum = 0.0
            correction_sign = config['correction_sign'][var_obj]
            credit_gain = config['credit_gain'][var_obj]
            conflict_weight = config['harmful_conflict_weight'][var_obj]
            for idx in range(len(error_series)):
                alignment = credit_gain * correction_sign * error_series[idx] * effort_series[idx]
                helpful_sum += max(0.0, alignment)
                harmful_sum += max(0.0, -alignment) + conflict_weight * abs(conflict_series[idx])

            helpful_raw, harmful_raw = self._reduce_credit_sums(
                helpful_sum,
                harmful_sum,
                len(error_series),
                config['reduction_mode'],
                'controller credit'
            )
            (
                helpful_fraction,
                harmful_fraction,
                cooperative_marginal_signed
            ) = self._compute_fractional_marginal(
                transition_components['global_marginal_signed'],
                helpful_raw,
                harmful_raw,
                config['epsilon'],
                config['inactive_credit_threshold']
            )

            records[f'credit_helpful_raw_{var_obj}'] = helpful_raw
            records[f'credit_harmful_raw_{var_obj}'] = harmful_raw
            records[f'credit_helpful_fraction_{var_obj}'] = helpful_fraction
            records[f'credit_harmful_fraction_{var_obj}'] = harmful_fraction
            records[f'credit_active_{var_obj}'] = float(
                helpful_raw + harmful_raw > config['inactive_credit_threshold']
            )
            records[f'cooperative_marginal_quality_signed_{var_obj}'] = cooperative_marginal_signed

            components_by_var[var_obj] = {
                'global_quality_signed': transition_components['global_quality_signed'],
                'cooperative_marginal_quality_signed': cooperative_marginal_signed
            }

        return records, components_by_var

    def _allocate_agent_credit(self, transition_components, extra_reward_component, actions_dict):
        self._require_applied_actions(actions_dict, 'agent_reward credit allocation')

        mode = self.agent_assignment['mode']
        if mode != 'pid_incremental_control_credit':
            raise ValueError(f"Unsupported agent credit assignment mode: {mode}")

        records = {}
        components_by_agent = {}
        raw_delta_u_by_agent = {}
        var_series_cache = {}
        config = self.agent_assignment
        signals = config['signals']
        effective_scale_config = config['effective_scale']

        for var_obj in self.var_objs:
            error_series = extra_reward_component[signals['error'].format(var_obj=var_obj)]
            conflict_effort_series = extra_reward_component[
                signals['conflict_effort'].format(var_obj=var_obj)
            ]
            scale_control_action_series = extra_reward_component[
                effective_scale_config['control_action'].format(var_obj=var_obj)
            ]
            scale_allocated_effort_series = extra_reward_component[
                effective_scale_config['allocated_effort'].format(var_obj=var_obj)
            ]
            self._require_same_length(var_obj, error_series, conflict_effort_series, 'agent credit')
            self._require_same_length(var_obj, error_series, scale_control_action_series, 'agent credit')
            self._require_same_length(var_obj, error_series, scale_allocated_effort_series, 'agent credit')

            var_series_cache[var_obj] = {
                'error': error_series,
                'scale_control_action': scale_control_action_series,
                'scale_allocated_effort': scale_allocated_effort_series,
                'conflict_effort': conflict_effort_series
            }

            for agent_name in self.var_obj_to_agent_names[var_obj]:
                gain_type = self.agent_to_gain_type_map[agent_name]
                basis_name = config['gain_basis'][gain_type]
                basis_series = extra_reward_component[
                    signals[basis_name].format(var_obj=var_obj)
                ]
                self._require_same_length(var_obj, error_series, basis_series, f'agent credit {agent_name}')

                delta_gain_applied = actions_dict['vars_delta'][
                    f'delta_gain_applied_{agent_name}'
                ]
                raw_delta_u_by_agent[agent_name] = [
                    delta_gain_applied * basis_value
                    for basis_value in basis_series
                ]

        for var_obj in self.var_objs:
            series_bundle = var_series_cache[var_obj]
            interval_length = len(series_bundle['error'])
            conflict_weight = config['harmful_conflict_weight'][var_obj]
            correction_sign = config['correction_sign'][var_obj]
            credit_gain = config['credit_gain'][var_obj]

            for agent_name in self.var_obj_to_agent_names[var_obj]:
                helpful_sum = 0.0
                harmful_sum = 0.0
                raw_delta_u_series = raw_delta_u_by_agent[agent_name]

                for idx in range(interval_length):
                    abs_delta_u_sum = self._sum_abs_agent_delta_u_at_step(
                        var_obj,
                        raw_delta_u_by_agent,
                        idx
                    )
                    conflict_share = abs(raw_delta_u_series[idx]) / (
                        abs_delta_u_sum + config['epsilon']
                    )
                    effective_scale = self._compute_agent_effective_scale(
                        series_bundle['scale_control_action'][idx],
                        series_bundle['scale_allocated_effort'][idx]
                    )
                    effective_delta_u = raw_delta_u_series[idx] * effective_scale
                    alignment = (
                        credit_gain
                        * correction_sign
                        * series_bundle['error'][idx]
                        * effective_delta_u
                    )
                    helpful_sum += max(0.0, alignment)
                    harmful_sum += (
                        max(0.0, -alignment)
                        + conflict_weight * abs(series_bundle['conflict_effort'][idx]) * conflict_share
                    )

                helpful_raw, harmful_raw = self._reduce_credit_sums(
                    helpful_sum,
                    harmful_sum,
                    interval_length,
                    config['reduction_mode'],
                    'agent credit'
                )
                (
                    helpful_fraction,
                    harmful_fraction,
                    cooperative_marginal_signed
                ) = self._compute_fractional_marginal(
                    transition_components['global_marginal_signed'],
                    helpful_raw,
                    harmful_raw,
                    config['epsilon'],
                    config['inactive_credit_threshold']
                )

                records[f'agent_credit_helpful_raw_{agent_name}'] = helpful_raw
                records[f'agent_credit_harmful_raw_{agent_name}'] = harmful_raw
                records[f'agent_credit_helpful_fraction_{agent_name}'] = helpful_fraction
                records[f'agent_credit_harmful_fraction_{agent_name}'] = harmful_fraction
                records[f'agent_credit_active_{agent_name}'] = float(
                    helpful_raw + harmful_raw > config['inactive_credit_threshold']
                )
                records[f'agent_credit_marginal_signed_{agent_name}'] = cooperative_marginal_signed

                components_by_agent[agent_name] = {
                    'global_quality_signed': transition_components['global_quality_signed'],
                    'cooperative_marginal_quality_signed': cooperative_marginal_signed
                }

        return records, components_by_agent

    def _allocate_common_credit(self, transition_components):
        mode = self.common_assignment['mode']
        if mode != 'global_transition':
            raise ValueError(f"Unsupported common credit assignment mode: {mode}")

        records = {
            'cooperative_common_marginal_quality_signed': (
                transition_components['global_marginal_signed']
            )
        }
        component = {
            'global_quality_signed': transition_components['global_quality_signed'],
            'cooperative_marginal_quality_signed': transition_components['global_marginal_signed']
        }
        return records, component

    def _compute_fractional_marginal(
        self,
        global_marginal_signed,
        helpful_raw,
        harmful_raw,
        epsilon,
        inactive_credit_threshold
    ):
        contribution_raw = helpful_raw + harmful_raw
        if contribution_raw <= inactive_credit_threshold:
            return 0.0, 0.0, 0.0

        denominator = contribution_raw + epsilon
        if denominator <= 0.0:
            raise ValueError("Credit denominator must be positive")

        helpful_fraction = helpful_raw / denominator
        harmful_fraction = harmful_raw / denominator
        cooperative_marginal_signed = self._compute_marginal_credit(
            global_marginal_signed,
            helpful_fraction,
            harmful_fraction
        )
        return helpful_fraction, harmful_fraction, cooperative_marginal_signed

    def _compute_marginal_credit(
        self,
        global_marginal_signed,
        helpful_fraction,
        harmful_fraction
    ):
        if global_marginal_signed >= 0.0:
            return self._clip_signed(global_marginal_signed * (helpful_fraction - harmful_fraction))

        adverse_share = harmful_fraction + 0.50 * helpful_fraction
        return self._clip_signed(global_marginal_signed * adverse_share)

    def _sum_abs_agent_delta_u_at_step(self, var_obj, raw_delta_u_by_agent, idx):
        abs_delta_u_sum = 0.0
        for agent_name in self.var_obj_to_agent_names[var_obj]:
            abs_delta_u_sum += abs(raw_delta_u_by_agent[agent_name][idx])
        return abs_delta_u_sum

    def _compute_agent_effective_scale(self, control_action, allocated_effort):
        scale_config = self.agent_assignment['effective_scale']
        mode = scale_config['mode']
        if mode == 'none':
            return 1.0
        if mode == 'allocated_over_control_action':
            if abs(control_action) <= scale_config['min_abs_denominator']:
                return 0.0
            return allocated_effort / control_action
        raise ValueError(f"Unsupported agent credit effective_scale mode: {mode}")

    def _reduce_credit_sums(self, helpful_sum, harmful_sum, interval_length, reduction_mode, context):
        if reduction_mode == 'average':
            return helpful_sum / interval_length, harmful_sum / interval_length
        if reduction_mode == 'sum':
            return helpful_sum, harmful_sum
        raise ValueError(f"Unsupported {context} reduction_mode: {reduction_mode}")

    def _require_applied_actions(self, actions_dict, context):
        if actions_dict is None:
            raise ValueError(f"{context} requires actions_dict")
        if not actions_dict['actions_applied']:
            raise ValueError(f"{context} requires post-application actions_dict")
        if actions_dict['vars_delta']['actions_applied'] != 1.0:
            raise ValueError(f"{context} requires vars_delta.actions_applied=1.0")

    def _require_same_length(self, var_obj, first_series, second_series, context):
        if not first_series:
            raise ValueError(f"Cannot compute {context} with empty series for {var_obj}")
        if len(first_series) != len(second_series):
            raise ValueError(
                f"{context} series must have same length for {var_obj}: "
                f"{len(first_series)} != {len(second_series)}"
            )

    def _clip_signed(self, value):
        return min(1.0, max(-1.0, value))
