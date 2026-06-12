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
        self.recoverability_risk = self.config['recoverability_risk']
        self.action_feasibility_penalty = self.config['action_feasibility_penalty']
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
        self.recoverability_jobs = self._compile_recoverability_jobs()

    def _compile_recoverability_jobs(self):
        jobs = {}
        if not self.recoverability_risk['enabled']:
            return jobs

        signals = self.recoverability_risk['signals']
        barriers = self.recoverability_risk['barriers']
        for recoverability_var in signals:
            jobs[recoverability_var] = []
            for feature_name, signal_key in signals[recoverability_var].items():
                jobs[recoverability_var].append((
                    feature_name,
                    signal_key,
                    barriers[recoverability_var][feature_name]
                ))
        return jobs

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
        self.last_components_by_agent = {}

    def evaluate(
        self,
        local_records,
        extra_reward_component,
        actions_dict=None,
        agent_to_var_obj_map=None
    ):
        records = {}
        components_by_var = {}
        recoverability_risk_cost, recoverability_records = self._recoverability_risk(
            extra_reward_component
        )
        records.update(recoverability_records)

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
            local_risk_cost = self._clip_01(
                self.risk_weights['dynamic_expansive'] * dynamic_expansive_risk
                + self.risk_weights['memory_integral'] * memory_integral_risk
                + self.risk_weights['authority'] * authority_risk
                + self.risk_weights['conflict'] * conflict_risk
            )
            strategy_risk_cost = max(local_risk_cost, recoverability_risk_cost)
            internal_risk_penalty_signed = -strategy_risk_cost

            records[f'internal_risk_dynamic_expansive_cost_01_{var_obj}'] = dynamic_expansive_risk
            records[f'internal_risk_memory_integral_cost_01_{var_obj}'] = memory_integral_risk
            records[f'internal_risk_authority_cost_01_{var_obj}'] = authority_risk
            records[f'internal_risk_conflict_cost_01_{var_obj}'] = conflict_risk
            records[f'internal_risk_conflict_effort_cost_01_{var_obj}'] = conflict_effort_cost
            records[f'internal_risk_saturation_cost_01_{var_obj}'] = saturation_cost
            records[f'internal_risk_effort_cost_01_{var_obj}'] = effort_cost
            records[f'internal_risk_local_cost_01_{var_obj}'] = local_risk_cost
            records[f'internal_risk_cost_01_{var_obj}'] = strategy_risk_cost
            records[f'internal_risk_penalty_signed_{var_obj}'] = internal_risk_penalty_signed

            components_by_var[var_obj] = {
                'strategy_reward_signed': internal_risk_penalty_signed,
                'strategy_risk_cost_01': strategy_risk_cost,
                'strategy_decision_penalty_signed': 0.0
            }

        action_records, components_by_agent = self._action_feasibility_components(
            actions_dict,
            agent_to_var_obj_map,
            components_by_var
        )
        records.update(action_records)

        self.last_record = records
        self.last_components_by_var = components_by_var
        self.last_components_by_agent = components_by_agent
        return records, components_by_var, components_by_agent

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

    def _recoverability_risk(self, extra_reward_component):
        records = {}
        if not self.recoverability_risk['enabled']:
            for recoverability_var in self.recoverability_risk['signals']:
                for feature_name in self.recoverability_risk['signals'][recoverability_var]:
                    records[
                        f'internal_risk_recoverability_{feature_name}_cost_01_{recoverability_var}'
                    ] = 0.0
                records[f'internal_risk_recoverability_cost_01_{recoverability_var}'] = 0.0
            records['internal_risk_recoverability_cost_01'] = 0.0
            return 0.0, records

        recoverability_costs = []
        for recoverability_var in self.recoverability_jobs:
            feature_costs = []
            for feature_name, signal_key, barrier_config in self.recoverability_jobs[recoverability_var]:
                signal_values = extra_reward_component[signal_key]
                feature_cost = self._barrier_series_cost(
                    signal_values,
                    barrier_config,
                    self.recoverability_risk['reduction'],
                    f'recoverability_{feature_name}_{recoverability_var}'
                )
                records[
                    f'internal_risk_recoverability_{feature_name}_cost_01_{recoverability_var}'
                ] = feature_cost
                feature_costs.append(feature_cost)

            if not feature_costs:
                raise ValueError(f"Recoverability risk requires declared signals for {recoverability_var}")

            recoverability_var_cost = self._compose_recoverability_cost(feature_costs)
            records[f'internal_risk_recoverability_cost_01_{recoverability_var}'] = (
                recoverability_var_cost
            )
            recoverability_costs.append(recoverability_var_cost)

        if not recoverability_costs:
            raise ValueError("Recoverability risk requires at least one declared signal")

        recoverability_risk_cost = self._compose_recoverability_cost(recoverability_costs)
        records['internal_risk_recoverability_cost_01'] = recoverability_risk_cost
        return recoverability_risk_cost, records

    def _action_feasibility_components(
        self,
        actions_dict,
        agent_to_var_obj_map,
        components_by_var
    ):
        records = {}
        components_by_agent = {}
        if agent_to_var_obj_map is None:
            return records, components_by_agent
        self._require_applied_actions(actions_dict, 'Action feasibility risk')

        config = self.action_feasibility_penalty
        actions_space = self.config_main['agent_base']['agent_config']['actions']['actions_space']

        for agent_name, var_obj in agent_to_var_obj_map.items():
            action_requested_move = actions_dict['vars_delta'][
                f'action_requested_move_{agent_name}'
            ]
            action_blocked = actions_dict['vars_delta'][f'action_blocked_{agent_name}']
            blocked_fraction = actions_dict['vars_delta'][
                f'action_blocked_fraction_{agent_name}'
            ]
            action_idx = actions_dict['vars_decision'][f'action_{agent_name}']
            action_label = actions_space[action_idx]

            if not config['enabled']:
                action_feasibility_cost = 0.0
            elif action_label == 'maintain' and not config['penalize_maintain']:
                action_feasibility_cost = 0.0
            elif action_requested_move <= 0.0:
                action_feasibility_cost = 0.0
            elif action_blocked <= 0.0:
                action_feasibility_cost = 0.0
            elif blocked_fraction >= 1.0:
                action_feasibility_cost = config['blocked_decision_cost']
            else:
                action_feasibility_cost = config['partial_clip_decision_cost'] * blocked_fraction

            action_feasibility_cost = self._clip_01(action_feasibility_cost)
            strategy_decision_penalty_signed = -action_feasibility_cost
            agent_strategy_risk_cost = components_by_var[var_obj]['strategy_risk_cost_01']
            agent_strategy_penalty_signed = components_by_var[var_obj]['strategy_reward_signed']

            records[f'internal_risk_action_requested_move_{agent_name}'] = action_requested_move
            records[f'internal_risk_action_feasibility_cost_01_{agent_name}'] = (
                action_feasibility_cost
            )
            records[f'internal_risk_agent_barrier_cost_01_{agent_name}'] = agent_strategy_risk_cost
            records[f'internal_risk_agent_decision_penalty_signed_{agent_name}'] = (
                strategy_decision_penalty_signed
            )
            records[f'internal_risk_agent_penalty_signed_{agent_name}'] = (
                agent_strategy_penalty_signed
            )
            components_by_agent[agent_name] = {
                'strategy_reward_signed': agent_strategy_penalty_signed,
                'strategy_risk_cost_01': agent_strategy_risk_cost,
                'strategy_decision_penalty_signed': strategy_decision_penalty_signed
            }

        return records, components_by_agent

    def _require_applied_actions(self, actions_dict, context):
        if actions_dict is None:
            raise ValueError(f"{context} requires actions_dict")
        if not actions_dict['actions_applied']:
            raise ValueError(f"{context} requires post-application actions_dict")
        if actions_dict['vars_delta']['actions_applied'] != 1.0:
            raise ValueError(f"{context} requires vars_delta.actions_applied=1.0")

    def _barrier_series_cost(self, values, barrier_config, method, key):
        if not values:
            raise ValueError(f"Cannot reduce recoverability risk series: {key}")

        warn = barrier_config['warn']
        limit = barrier_config['limit']
        exponent = barrier_config['exponent']
        denominator = limit - warn
        if denominator <= 0.0:
            raise ValueError(f"Recoverability barrier limit must be greater than warn for {key}")

        barrier_values = []
        for value in values:
            normalized_distance = (abs(value) - warn) / denominator
            barrier_values.append(self._clip_01(normalized_distance) ** exponent)

        return self._reduce_unit_series(barrier_values, method, key)

    def _compose_recoverability_cost(self, component_costs):
        composition = self.recoverability_risk['composition']
        if composition == 'max':
            return self._clip_01(max(component_costs))
        if composition == 'mean':
            return self._clip_01(sum(component_costs) / len(component_costs))

        raise ValueError(f"Unsupported recoverability risk composition: {composition}")

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
        if method == 'max':
            return self._clip_01(max(abs(value) for value in values))
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
