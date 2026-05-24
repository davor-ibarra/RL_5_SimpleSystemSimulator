"""
lagrange_reward_calculator.py

Responsibility:
Compute the principal reward from flattened reward components.
"""

import math


class LagrangeRewardCalculator:
    """
    Principal reward calculator using a lagrangian-style formulation.

    Expected flattened input:
    {
        "L_e_pendulum_angle": ...,
        "L_edot_pendulum_angle": ...,
        ...
    }
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.config = config_main['reward_base']['reward_calculation']['principal_reward']
        self.method = self.config['method']
        self.normalized_reward_mode = self.config['normalized_reward_mode']
        self.budget_conservation_config = self.config['feature_budget_conservation']
        self.budget_conservation_enabled = (
            self.normalized_reward_mode
            and self.budget_conservation_config.get('enabled', False)
        )
        self.budget_conservation_strict = self.budget_conservation_config['strict_weight_sum']

        if self.method == 'lineal_combination':
            self.config_lineal_combination = self.config['lineal_combination_params']
            self.weights = self._extract_lineal_weights()
            self.feature_gates = self.config_lineal_combination.get('feature_gates', {})
        elif self.method == 'weighted_exponential':
            self.config_weighted_exponential = self.config['weighted_exponential_params']
            self.weights = self._extract_exponential_weights()
            self.feature_gates = self.config_weighted_exponential.get('feature_gates', {})
        elif self.method == 'nonlinear_local_cost':
            self.config_nonlinear_local_cost = self.config['nonlinear_local_cost_params']
            self.config_nonlinear_component_weights = self.config_nonlinear_local_cost['component_weights']
            self.weights = {'L_e': 1.0, 'L_edot': 1.0, 'L_u': 1.0}
            self.feature_gates = {}
        else:
            self.weights = {}
            self.feature_gates = {}

        self.var_objs = self._extract_var_objs()
        self.feature_weight_sum = self._compute_feature_weight_sum()
        self.budget_redistribution = self._compile_budget_redistribution()

        self.feature_to_agent_weight_key = {
            'L_e': 'w_e',
            'L_edot': 'w_edot',
            'L_I': 'w_I',
            'L_u': 'w_u',
            'L_delta_u': 'w_s',
        }

        self._compile_jobs()
        self._compile_agent_jobs()
        self.last_reward_params_record = {}
        self.last_controller_rewards = {}
        self.last_controller_composition_scores = {}
        self.previous_nonlinear_cost = {var_obj: None for var_obj in self.var_objs}

    def _compile_jobs(self):
        self.var_jobs = {}
        for var_obj in self.var_objs:
            self.var_jobs[var_obj] = []
            for feature_name, cfg in self.weights.items():
                flat_key = f"{feature_name}_{var_obj}"
                if isinstance(cfg, dict):
                    raw_weight = cfg.get('weight', 0.0)
                    scaled = cfg.get('scaled', 1.0)
                    setpoint = cfg.get('setpoint', 0.0)
                else:
                    raw_weight = cfg
                    scaled = 1.0
                    setpoint = 0.0
                weight = self._effective_feature_weight(raw_weight)
                gate_cfg = self._resolve_gate_config(feature_name, var_obj)
                self.var_jobs[var_obj].append(
                    (flat_key, feature_name, weight, raw_weight, scaled, setpoint, gate_cfg)
                )

        self.global_jobs = []
        for feature_name, cfg in self.weights.items():
            if isinstance(cfg, dict):
                raw_weight = cfg.get('weight', 0.0)
                scaled = cfg.get('scaled', 1.0)
                setpoint = cfg.get('setpoint', 0.0)
            else:
                raw_weight = cfg
                scaled = 1.0
                setpoint = 0.0
            weight = self._effective_feature_weight(raw_weight)
            self.global_jobs.append((feature_name, weight, raw_weight, scaled, setpoint))

    def _compile_agent_jobs(self):
        self.agent_feature_jobs = []
        for feature_name, weight_key in self.feature_to_agent_weight_key.items():
            if feature_name not in self.weights:
                continue
            cfg = self.weights.get(feature_name, {})
            if isinstance(cfg, dict):
                scaled = cfg.get('scaled', 1.0)
                setpoint = cfg.get('setpoint', 0.0)
            else:
                scaled = 1.0
                setpoint = 0.0
            self.agent_feature_jobs.append((feature_name, weight_key, scaled, setpoint))

    def _extract_var_objs(self):
        var_objs = []
        controllers_config = self.config_main['controller_base']['controllers']
        for ctrl_cfg in controllers_config.values():
            var_objs.append(ctrl_cfg['params']['name_objective_var'])
        return var_objs

    def _extract_lineal_weights(self):
        features = self.config_lineal_combination['features']
        return {feature_name: feature_config['weight'] for feature_name, feature_config in features.items()}

    def _extract_exponential_weights(self):
        return self.config_weighted_exponential['features']

    def _compute_feature_weight_sum(self):
        total = 0.0
        for cfg in self.weights.values():
            if isinstance(cfg, dict):
                weight = cfg.get('weight', 0.0)
            else:
                weight = cfg
            total += max(0.0, weight)
        return total if total > 0.0 else 1.0

    def _effective_feature_weight(self, raw_weight):
        if not self.normalized_reward_mode:
            return raw_weight
        return max(0.0, raw_weight) / self.feature_weight_sum

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))

    def _normalized_input_cost(self, value):
        return self._clip_01(abs(value))

    def _exponential_cost_01(self, value, scaled, setpoint):
        if scaled < 0:
            raise ValueError(f"Exponential reward scale must be non-negative, got {scaled}")
        if scaled == 0:
            return 0.0

        value_01 = self._clip_01(abs(value))
        raw_cost = 1.0 - math.exp(-scaled * ((value_01 - setpoint) ** 2))
        bound_0 = 1.0 - math.exp(-scaled * ((0.0 - setpoint) ** 2))
        bound_1 = 1.0 - math.exp(-scaled * ((1.0 - setpoint) ** 2))
        raw_bound = max(bound_0, bound_1, 1.0e-12)
        return self._clip_01(raw_cost / raw_bound)

    def _principal_reward_from_controller_rewards(self, controller_rewards):
        if not controller_rewards:
            return 0.0
        if self.normalized_reward_mode:
            return sum(controller_rewards.values()) / len(controller_rewards)
        return sum(controller_rewards.values())

    def _compute_agent_weight_sum(self, agent_weights):
        total = 0.0
        for _, weight_key, _, _ in self.agent_feature_jobs:
            total += max(0.0, agent_weights.get(weight_key, 0.0))
        return total if total > 0.0 else 1.0

    def _effective_agent_weight(self, agent_weights, weight_key, agent_weight_sum):
        raw_weight = agent_weights[weight_key]
        if not self.normalized_reward_mode:
            return raw_weight
        return max(0.0, raw_weight) / agent_weight_sum

    def _compile_budget_redistribution(self):
        """
        Precompila el mapa declarativo source_feature -> [(target_feature, ratio)].
        """
        redistribution_config = self.budget_conservation_config.get('redistribution', {})
        redistribution = {}

        for source_feature, target_map in redistribution_config.items():
            if source_feature not in self.weights:
                continue
            if not isinstance(target_map, dict) or not target_map:
                raise ValueError(
                    f"Budget redistribution for {source_feature} must define target weights"
                )

            target_total = 0.0
            target_jobs = []
            for target_feature, ratio in target_map.items():
                if target_feature not in self.weights:
                    raise ValueError(
                        f"Budget redistribution target {target_feature} is not an active feature"
                    )
                if ratio < 0.0:
                    raise ValueError(
                        f"Budget redistribution ratio must be non-negative, got {ratio}"
                    )
                target_total += ratio
                target_jobs.append((target_feature, ratio))

            if self.budget_conservation_strict and abs(target_total - 1.0) > 1.0e-9:
                raise ValueError(
                    f"Budget redistribution for {source_feature} must sum 1.0, got {target_total}"
                )

            if target_total > 0.0 and not self.budget_conservation_strict:
                target_jobs = [
                    (target_feature, ratio / target_total)
                    for target_feature, ratio in target_jobs
                ]
            redistribution[source_feature] = target_jobs

        if self.budget_conservation_enabled and self.budget_conservation_strict:
            for feature_name in self.weights.keys():
                gate_cfg = self.feature_gates.get(feature_name, {})
                if gate_cfg.get('enabled', False) and feature_name not in redistribution:
                    raise ValueError(
                        f"Enabled gate {feature_name} requires budget redistribution config"
                    )

        return redistribution

    def _compute_budget_allocation(self, base_weights, gate_factors):
        """
        Calcula pesos efectivos conservando presupuesto si el modo esta activo.
        """
        effective_weights = {}
        residual_weights = {}
        for feature_name, base_weight in base_weights.items():
            gate_factor = gate_factors.get(feature_name, 1.0)
            gated_weight = base_weight * gate_factor
            residual_weight = base_weight - gated_weight
            effective_weights[feature_name] = gated_weight
            residual_weights[feature_name] = residual_weight

        if not self.budget_conservation_enabled:
            return effective_weights, residual_weights

        for source_feature, residual_weight in residual_weights.items():
            if residual_weight <= 0.0:
                continue

            target_jobs = self.budget_redistribution.get(source_feature)
            if not target_jobs:
                if self.budget_conservation_strict:
                    raise ValueError(
                        f"Missing budget redistribution for gated feature {source_feature}"
                    )
                effective_weights[source_feature] += residual_weight
                continue

            for target_feature, ratio in target_jobs:
                effective_weights[target_feature] += residual_weight * ratio

        return effective_weights, residual_weights

    def _resolve_gate_config(self, feature_name, var_obj):
        gate_cfg = self.feature_gates.get(feature_name)
        if not gate_cfg or not gate_cfg.get('enabled', False):
            return None

        resolved = dict(gate_cfg)
        per_var = resolved.pop('per_var', {})
        if var_obj in per_var:
            override = per_var[var_obj] or {}
            merged = dict(resolved)
            merged.update(override)
            resolved = merged

        source = resolved.get('source', 'error_{var_obj}_mean_squared')
        if isinstance(source, str):
            resolved['source'] = source.format(var_obj=var_obj)
        else:
            resolved['source'] = source

        resolved.setdefault('type', 'exp')
        resolved.setdefault('scaled', resolved.get('strength', 1.0))
        resolved.setdefault('setpoint', resolved.get('x_sp', 0.0))
        resolved.setdefault('use_sqrt', False)
        resolved.setdefault('use_abs', True)
        resolved.setdefault('invert', False)
        resolved.setdefault('min_factor', 0.0)
        resolved.setdefault('max_factor', 1.0)
        return resolved

    def _compute_gate_factor(self, flat_reward_component, gate_cfg):
        if not gate_cfg:
            return 1.0

        source_key = gate_cfg['source']
        source_value = flat_reward_component[source_key]

        value = abs(source_value) if gate_cfg.get('use_abs', True) else source_value
        if gate_cfg.get('use_sqrt', False):
            value = math.sqrt(max(value, 0.0))

        gain = gate_cfg.get('strength', gate_cfg.get('scaled', 1.0))
        setpoint = gate_cfg.get('x_sp', gate_cfg.get('setpoint', 0.0))
        factor_type = gate_cfg.get('type', 'exp')

        if gain < 0:
            raise ValueError(f"Gate gain must be non-negative, got {gain}")

        if factor_type == 'exp':
            factor = math.exp(-(((value - setpoint) / gain) ** 2)) if gain > 0 else 0.0
        elif factor_type == 'tanh':
            factor = math.tanh(gain * abs(value - setpoint))
        else:
            raise ValueError(f"Unsupported principal gate type: {factor_type}")

        if gate_cfg.get('invert', False):
            factor = 1.0 - factor

        min_factor = gate_cfg.get('min_factor', 0.0)
        max_factor = gate_cfg.get('max_factor', 1.0)
        return min(max_factor, max(min_factor, factor))

    def reset_episode(self):
        self.last_reward_params_record = {}
        self.last_controller_rewards = {}
        self.last_controller_composition_scores = {}
        self.previous_nonlinear_cost = {var_obj: None for var_obj in self.var_objs}

    def compute_reward(self, flat_reward_component):
        if self.method == 'lineal_combination':
            return self._compute_lineal(flat_reward_component)
        if self.method == 'weighted_exponential':
            return self._compute_exponential(flat_reward_component)
        if self.method == 'nonlinear_local_cost':
            return self._compute_nonlinear_local_cost(flat_reward_component)
        return 0.0, {}, {}

    def _compute_lineal(self, flat_reward_component):
        aggregated_values = {feature: 0.0 for feature in self.weights.keys()}
        feature_contributions = {feature: 0.0 for feature in self.weights.keys()}
        controller_rewards = {}
        controller_composition_scores = {}
        loop_cost_records = {}
        composition_score_records = {}
        budget_records = {}
        gate_records = {}
        feature_cost_records = {}
        contribution_records = {}

        for var_obj, jobs in self.var_jobs.items():
            loop_reward = 0.0
            loop_cost_01 = 0.0
            feature_values = {}
            feature_costs = {}
            base_weights = {}
            raw_weights = {}
            gate_factors = {}

            for flat_key, feature_name, weight, raw_weight, _, _, gate_cfg in jobs:
                value = flat_reward_component[flat_key]
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                value_cost_01 = self._normalized_input_cost(value)

                aggregated_values[feature_name] += value
                feature_values[feature_name] = value
                feature_costs[feature_name] = value_cost_01
                base_weights[feature_name] = weight
                raw_weights[feature_name] = raw_weight
                gate_factors[feature_name] = gate_factor

                feature_cost_records[f'principal_feature_cost_01_{feature_name}_{var_obj}'] = value_cost_01
                gate_records[f'principal_gate_{feature_name}_{var_obj}'] = gate_factor

            effective_weights, residual_weights = self._compute_budget_allocation(
                base_weights,
                gate_factors
            )

            for feature_name, value in feature_values.items():
                if self.normalized_reward_mode:
                    contribution = -effective_weights[feature_name] * feature_costs[feature_name]
                else:
                    contribution = (
                        -raw_weights[feature_name]
                        * gate_factors[feature_name]
                        * value
                    )

                feature_contributions[feature_name] += contribution
                loop_reward += contribution
                loop_cost_01 += effective_weights[feature_name] * feature_costs[feature_name]

                budget_records[f'principal_base_weight_{feature_name}_{var_obj}'] = base_weights[feature_name]
                budget_records[f'principal_effective_weight_{feature_name}_{var_obj}'] = effective_weights[feature_name]
                budget_records[f'principal_weight_residual_{feature_name}_{var_obj}'] = residual_weights[feature_name]
                contribution_records[f'principal_contribution_{feature_name}_{var_obj}'] = contribution

            loop_cost_01 = self._clip_01(loop_cost_01)
            loop_score_01 = 1.0 - loop_cost_01
            controller_rewards[var_obj] = loop_reward
            controller_composition_scores[var_obj] = loop_score_01
            loop_cost_records[f'principal_loop_cost_01_{var_obj}'] = loop_cost_01
            composition_score_records[f'principal_composition_score_01_{var_obj}'] = loop_score_01
            budget_records[f'principal_effective_weight_sum_{var_obj}'] = sum(effective_weights.values())
            budget_records[f'principal_residual_weight_sum_{var_obj}'] = sum(residual_weights.values())

        global_reward = 0.0
        for feature_name, weight, raw_weight, _, _ in self.global_jobs:
            if feature_name in flat_reward_component:
                value = flat_reward_component[feature_name]
                aggregated_values[feature_name] += value
                if self.normalized_reward_mode:
                    contribution = -weight * self._normalized_input_cost(value)
                else:
                    contribution = -raw_weight * value
                feature_contributions[feature_name] += contribution
                global_reward += contribution

        if global_reward != 0.0:
            for var_obj in controller_rewards:
                controller_rewards[var_obj] += global_reward

        principal_reward = self._principal_reward_from_controller_rewards(controller_rewards)
        principal_cost_01 = (
            sum(loop_cost_records.values()) / len(loop_cost_records)
            if loop_cost_records else 0.0
        )
        principal_cost_01 = self._clip_01(principal_cost_01)
        principal_composition_score_01 = self._clip_01(
            sum(controller_composition_scores.values()) / len(controller_composition_scores)
            if controller_composition_scores else 0.0
        )
        lagrangian_total = -principal_reward

        reward_params_record = {
            'principal_reward': principal_reward,
            'principal_reward_normalized_mode': float(self.normalized_reward_mode),
            'principal_budget_conservation_enabled': float(self.budget_conservation_enabled),
            'principal_cost_01': principal_cost_01,
            'principal_score_01': 1.0 - principal_cost_01,
            'principal_composition_score_01': principal_composition_score_01,
            'principal_reward_bound_abs': 1.0,
            'controller_rewards': controller_rewards,
            'lagrangian_total': lagrangian_total,
        }
        for feature_name, value in aggregated_values.items():
            reward_params_record[f'principal_feature_value_{feature_name}'] = value
            reward_params_record[f'principal_feature_reward_{feature_name}'] = feature_contributions[feature_name]
        reward_params_record.update(loop_cost_records)
        reward_params_record.update(composition_score_records)
        reward_params_record.update(budget_records)
        reward_params_record.update(feature_cost_records)
        reward_params_record.update(gate_records)
        reward_params_record.update(contribution_records)

        self.last_reward_params_record = reward_params_record
        self.last_controller_composition_scores = controller_composition_scores.copy()
        return principal_reward, reward_params_record, controller_rewards

    def _compute_exponential(self, flat_reward_component):
        aggregated_values = {feature: 0.0 for feature in self.weights.keys()}
        feature_contributions = {feature: 0.0 for feature in self.weights.keys()}
        controller_rewards = {}
        controller_composition_scores = {}
        loop_cost_records = {}
        composition_score_records = {}
        budget_records = {}
        gate_records = {}
        feature_cost_records = {}
        contribution_records = {}

        for var_obj, jobs in self.var_jobs.items():
            loop_reward = 0.0
            loop_cost_01 = 0.0
            feature_values = {}
            feature_costs = {}
            base_weights = {}
            raw_weights = {}
            gate_factors = {}
            exp_terms = {}

            for flat_key, feature_name, weight, raw_weight, scaled, setpoint, gate_cfg in jobs:
                value = flat_reward_component[flat_key]
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                exp_term = math.exp(-scaled * (value - setpoint) ** 2)
                value_cost_01 = self._exponential_cost_01(value, scaled, setpoint)

                aggregated_values[feature_name] += value
                feature_values[feature_name] = value
                feature_costs[feature_name] = value_cost_01
                base_weights[feature_name] = weight
                raw_weights[feature_name] = raw_weight
                gate_factors[feature_name] = gate_factor
                exp_terms[feature_name] = exp_term

                feature_cost_records[f'principal_feature_cost_01_{feature_name}_{var_obj}'] = value_cost_01
                gate_records[f'principal_gate_{feature_name}_{var_obj}'] = gate_factor

            effective_weights, residual_weights = self._compute_budget_allocation(
                base_weights,
                gate_factors
            )

            for feature_name, value in feature_values.items():
                if self.normalized_reward_mode:
                    contribution = -effective_weights[feature_name] * feature_costs[feature_name]
                else:
                    contribution = (
                        -gate_factors[feature_name]
                        * raw_weights[feature_name]
                        * (1 - exp_terms[feature_name])
                    )

                feature_contributions[feature_name] += contribution
                loop_reward += contribution
                loop_cost_01 += effective_weights[feature_name] * feature_costs[feature_name]

                budget_records[f'principal_base_weight_{feature_name}_{var_obj}'] = base_weights[feature_name]
                budget_records[f'principal_effective_weight_{feature_name}_{var_obj}'] = effective_weights[feature_name]
                budget_records[f'principal_weight_residual_{feature_name}_{var_obj}'] = residual_weights[feature_name]
                contribution_records[f'principal_contribution_{feature_name}_{var_obj}'] = contribution

            loop_cost_01 = self._clip_01(loop_cost_01)
            loop_score_01 = 1.0 - loop_cost_01
            controller_rewards[var_obj] = loop_reward
            controller_composition_scores[var_obj] = loop_score_01
            loop_cost_records[f'principal_loop_cost_01_{var_obj}'] = loop_cost_01
            composition_score_records[f'principal_composition_score_01_{var_obj}'] = loop_score_01
            budget_records[f'principal_effective_weight_sum_{var_obj}'] = sum(effective_weights.values())
            budget_records[f'principal_residual_weight_sum_{var_obj}'] = sum(residual_weights.values())

        global_reward = 0.0
        for feature_name, weight, raw_weight, scaled, setpoint in self.global_jobs:
            if feature_name in flat_reward_component:
                value = flat_reward_component[feature_name]
                exp_term = math.exp(-scaled * (value - setpoint) ** 2)
                if self.normalized_reward_mode:
                    contribution = -weight * self._exponential_cost_01(value, scaled, setpoint)
                else:
                    contribution = -raw_weight * (1 - exp_term)

                aggregated_values[feature_name] += value
                feature_contributions[feature_name] += contribution
                global_reward += contribution

        if global_reward != 0.0:
            for var_obj in controller_rewards:
                controller_rewards[var_obj] += global_reward

        if self.normalized_reward_mode:
            principal_reward = self._principal_reward_from_controller_rewards(controller_rewards)
        else:
            principal_reward = sum(feature_contributions.values())
        principal_cost_01 = (
            sum(loop_cost_records.values()) / len(loop_cost_records)
            if loop_cost_records else 0.0
        )
        principal_cost_01 = self._clip_01(principal_cost_01)
        principal_composition_score_01 = self._clip_01(
            sum(controller_composition_scores.values()) / len(controller_composition_scores)
            if controller_composition_scores else 0.0
        )
        reward_params_record = {
            'principal_reward': principal_reward,
            'principal_reward_normalized_mode': float(self.normalized_reward_mode),
            'principal_budget_conservation_enabled': float(self.budget_conservation_enabled),
            'principal_cost_01': principal_cost_01,
            'principal_score_01': 1.0 - principal_cost_01,
            'principal_composition_score_01': principal_composition_score_01,
            'principal_reward_bound_abs': 1.0,
            'controller_rewards': controller_rewards,
        }
        for feature_name, value in aggregated_values.items():
            reward_params_record[f'principal_feature_value_{feature_name}'] = value
            reward_params_record[f'principal_feature_reward_{feature_name}'] = feature_contributions[feature_name]
        reward_params_record.update(loop_cost_records)
        reward_params_record.update(composition_score_records)
        reward_params_record.update(budget_records)
        reward_params_record.update(feature_cost_records)
        reward_params_record.update(gate_records)
        reward_params_record.update(contribution_records)

        self.last_reward_params_record = reward_params_record
        self.last_controller_composition_scores = controller_composition_scores.copy()
        return principal_reward, reward_params_record, controller_rewards

    def _compute_nonlinear_local_cost(self, flat_reward_component):
        aggregated_values = {feature: 0.0 for feature in self.weights.keys()}
        feature_contributions = {feature: 0.0 for feature in self.weights.keys()}
        controller_rewards = {}
        controller_composition_scores = {}
        loop_cost_records = {}
        local_cost_records = {}
        composition_score_records = {}
        budget_records = {}
        feature_cost_records = {}
        contribution_records = {}
        mode = self.config_nonlinear_local_cost['mode']
        reward_form = self.config_nonlinear_local_cost['reward_form']

        for var_obj in self.var_objs:
            terms = self._compute_nonlinear_terms(flat_reward_component, var_obj)
            base_component_weight, progress_component_weight = self._compute_nonlinear_component_weights(
                var_obj,
                mode
            )
            previous_cost = self.previous_nonlinear_cost[var_obj]
            cost_delta = 0.0 if previous_cost is None else previous_cost - terms['J']
            progress_signal = 0.0
            deterioration_signal = 0.0
            if mode == 'nonlinear_cost_with_progress':
                progress_signal = max(0.0, cost_delta)
                deterioration_signal = max(0.0, -cost_delta)
                if self.normalized_reward_mode:
                    progress_signal = self._clip_01(progress_signal)
                    deterioration_signal = self._clip_01(deterioration_signal)

            score_reward = 1.0 - terms['J']
            cost_reward = -terms['J']
            if reward_form == 'score':
                base_reward = score_reward
            else:
                base_reward = cost_reward

            base_component_reward = base_component_weight * base_reward
            if reward_form == 'score':
                progress_component_reward = progress_component_weight * progress_signal
                progress_component_score = progress_signal
            else:
                progress_component_reward = -progress_component_weight * deterioration_signal
                progress_component_score = 1.0 - deterioration_signal
            loop_reward = base_component_reward + progress_component_reward
            loop_composition_score_01 = self._clip_01(
                base_component_weight * score_reward
                + progress_component_weight * progress_component_score
            )
            controller_rewards[var_obj] = loop_reward
            controller_composition_scores[var_obj] = loop_composition_score_01

            aggregated_values['L_e'] += terms['L_e']
            aggregated_values['L_edot'] += terms['L_edot']
            aggregated_values['L_u'] += terms['L_u']

            contribution_e = -terms['effective_weight_L_e'] * terms['L_e']
            contribution_edot = -terms['effective_weight_L_edot'] * terms['L_edot']
            contribution_u = -terms['effective_weight_L_u'] * terms['L_u']
            feature_contributions['L_e'] += contribution_e
            feature_contributions['L_edot'] += contribution_edot
            feature_contributions['L_u'] += contribution_u

            loop_cost_records[f'principal_loop_cost_01_{var_obj}'] = self._clip_01(terms['J'])
            composition_score_records[f'principal_composition_score_01_{var_obj}'] = loop_composition_score_01
            local_cost_records[f'principal_local_error_potential_{var_obj}'] = terms['E']
            local_cost_records[f'principal_effort_gate_{var_obj}'] = terms['g_u']
            local_cost_records[f'principal_local_cost_J_{var_obj}'] = terms['J']
            local_cost_records[f'principal_local_cost_delta_{var_obj}'] = cost_delta
            local_cost_records[f'principal_local_score_reward_{var_obj}'] = score_reward
            local_cost_records[f'principal_local_cost_reward_{var_obj}'] = cost_reward
            local_cost_records[f'principal_local_base_reward_{var_obj}'] = base_reward
            local_cost_records[f'principal_local_base_component_reward_{var_obj}'] = base_component_reward
            local_cost_records[f'principal_local_progress_signal_01_{var_obj}'] = self._clip_01(progress_signal)
            local_cost_records[f'principal_local_deterioration_signal_01_{var_obj}'] = self._clip_01(deterioration_signal)
            local_cost_records[f'principal_local_progress_reward_{var_obj}'] = progress_component_reward
            budget_records[f'principal_component_weight_base_{var_obj}'] = base_component_weight
            budget_records[f'principal_component_weight_progress_{var_obj}'] = progress_component_weight
            feature_cost_records[f'principal_feature_cost_01_L_e_{var_obj}'] = self._clip_01(terms['L_e'])
            feature_cost_records[f'principal_feature_cost_01_L_edot_{var_obj}'] = self._clip_01(terms['L_edot'])
            feature_cost_records[f'principal_feature_cost_01_L_u_{var_obj}'] = self._clip_01(terms['L_u'])
            budget_records[f'principal_effective_weight_L_e_{var_obj}'] = terms['effective_weight_L_e']
            budget_records[f'principal_effective_weight_L_edot_{var_obj}'] = terms['effective_weight_L_edot']
            budget_records[f'principal_effective_weight_L_u_{var_obj}'] = terms['effective_weight_L_u']
            budget_records[f'principal_effective_weight_sum_{var_obj}'] = (
                terms['effective_weight_L_e']
                + terms['effective_weight_L_edot']
                + terms['effective_weight_L_u']
            )
            budget_records[f'principal_weight_residual_L_e_{var_obj}'] = 0.0
            budget_records[f'principal_weight_residual_L_edot_{var_obj}'] = 0.0
            budget_records[f'principal_weight_residual_L_u_{var_obj}'] = terms['effort_residual_weight']
            budget_records[f'principal_residual_weight_sum_{var_obj}'] = terms['effort_residual_weight']
            contribution_records[f'principal_contribution_L_e_{var_obj}'] = contribution_e
            contribution_records[f'principal_contribution_L_edot_{var_obj}'] = contribution_edot
            contribution_records[f'principal_contribution_L_u_{var_obj}'] = contribution_u

        principal_reward = self._principal_reward_from_controller_rewards(controller_rewards)
        principal_cost_01 = (
            sum(loop_cost_records.values()) / len(loop_cost_records)
            if loop_cost_records else 0.0
        )
        principal_cost_01 = self._clip_01(principal_cost_01)
        principal_composition_score_01 = self._clip_01(
            sum(controller_composition_scores.values()) / len(controller_composition_scores)
            if controller_composition_scores else 0.0
        )
        reward_params_record = {
            'principal_reward': principal_reward,
            'principal_reward_normalized_mode': float(self.normalized_reward_mode),
            'principal_budget_conservation_enabled': float(self.budget_conservation_enabled),
            'principal_cost_01': principal_cost_01,
            'principal_score_01': 1.0 - principal_cost_01,
            'principal_composition_score_01': principal_composition_score_01,
            'principal_reward_bound_abs': 1.0,
            'controller_rewards': controller_rewards,
            'principal_nonlinear_mode': mode,
            'principal_nonlinear_reward_form': reward_form,
            'lagrangian_total': -principal_reward,
        }
        for feature_name, value in aggregated_values.items():
            reward_params_record[f'principal_feature_value_{feature_name}'] = value
            reward_params_record[f'principal_feature_reward_{feature_name}'] = feature_contributions[feature_name]
        reward_params_record.update(loop_cost_records)
        reward_params_record.update(local_cost_records)
        reward_params_record.update(composition_score_records)
        reward_params_record.update(budget_records)
        reward_params_record.update(feature_cost_records)
        reward_params_record.update(contribution_records)

        self.previous_nonlinear_cost = {
            var_obj: reward_params_record[f'principal_local_cost_J_{var_obj}']
            for var_obj in self.var_objs
        }
        self.last_reward_params_record = reward_params_record
        self.last_controller_rewards = controller_rewards.copy()
        self.last_controller_composition_scores = controller_composition_scores.copy()
        return principal_reward, reward_params_record, controller_rewards

    def _compute_nonlinear_terms(self, flat_reward_component, var_obj):
        alpha_e = self._resolve_var_value(
            self.config_nonlinear_local_cost['alpha_e'],
            var_obj
        )
        effort_weight = self._resolve_var_value(
            self.config_nonlinear_local_cost['effort_weight'],
            var_obj
        )
        effort_gate_kappa = self._resolve_var_value(
            self.config_nonlinear_local_cost['effort_gate_kappa'],
            var_obj
        )

        L_e = self._local_cost_input(flat_reward_component[f'L_e_{var_obj}'])
        L_edot = self._local_cost_input(flat_reward_component[f'L_edot_{var_obj}'])
        L_u = self._local_cost_input(flat_reward_component[f'L_u_{var_obj}'])
        E = alpha_e * L_e + (1.0 - alpha_e) * L_edot
        g_u = math.exp(-effort_gate_kappa * E)
        denominator = 1.0 + effort_weight
        J = (E + effort_weight * g_u * L_u) / denominator

        return {
            'L_e': L_e,
            'L_edot': L_edot,
            'L_u': L_u,
            'E': E,
            'g_u': g_u,
            'J': J,
            'effective_weight_L_e': alpha_e / denominator,
            'effective_weight_L_edot': (1.0 - alpha_e) / denominator,
            'effective_weight_L_u': effort_weight * g_u / denominator,
            'effort_residual_weight': effort_weight * (1.0 - g_u) / denominator
        }

    def _local_cost_input(self, value):
        if self.normalized_reward_mode:
            return self._normalized_input_cost(value)
        return abs(value)

    def _compute_nonlinear_component_weights(self, var_obj, mode):
        base_weight = self._resolve_var_value(
            self.config_nonlinear_component_weights['base'],
            var_obj
        )
        progress_weight = self._resolve_var_value(
            self.config_nonlinear_component_weights['progress'],
            var_obj
        )
        if mode != 'nonlinear_cost_with_progress':
            progress_weight = 0.0

        if self.normalized_reward_mode:
            weight_sum = base_weight + progress_weight
            if weight_sum <= 0.0:
                raise ValueError(f"Nonlinear component weights must be positive for {var_obj}")
            base_weight = base_weight / weight_sum
            progress_weight = progress_weight / weight_sum

        return base_weight, progress_weight

    def _resolve_var_value(self, value_config, var_obj):
        if isinstance(value_config, dict):
            return value_config[var_obj]
        return value_config

    def compute_agent_individual_reward(self, flat_reward_component, var_obj, agent_weights):
        if self.method == 'nonlinear_local_cost':
            return self.last_controller_rewards[var_obj]

        agent_reward = 0.0
        agent_weight_sum = self._compute_agent_weight_sum(agent_weights)
        feature_values = {}
        feature_costs = {}
        base_weights = {}
        gate_factors = {}
        exp_terms = {}

        if self.method == 'weighted_exponential':
            for feature_name, weight_key, scaled, setpoint in self.agent_feature_jobs:
                flat_key = f"{feature_name}_{var_obj}"
                if weight_key not in agent_weights:
                    continue
                value = flat_reward_component[flat_key]
                weight = self._effective_agent_weight(agent_weights, weight_key, agent_weight_sum)
                gate_cfg = self._resolve_gate_config(feature_name, var_obj)
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                feature_values[feature_name] = value
                feature_costs[feature_name] = self._exponential_cost_01(value, scaled, setpoint)
                base_weights[feature_name] = weight
                gate_factors[feature_name] = gate_factor
                exp_terms[feature_name] = math.exp(-scaled * (value - setpoint) ** 2)

            effective_weights, _ = self._compute_budget_allocation(base_weights, gate_factors)
            for feature_name, value in feature_values.items():
                if self.normalized_reward_mode:
                    agent_reward -= effective_weights[feature_name] * feature_costs[feature_name]
                else:
                    agent_reward -= (
                        gate_factors[feature_name]
                        * base_weights[feature_name]
                        * (1 - exp_terms[feature_name])
                    )
        else:
            for feature_name, weight_key, _, _ in self.agent_feature_jobs:
                flat_key = f"{feature_name}_{var_obj}"
                if weight_key not in agent_weights:
                    continue
                value = flat_reward_component[flat_key]
                weight = self._effective_agent_weight(agent_weights, weight_key, agent_weight_sum)
                gate_cfg = self._resolve_gate_config(feature_name, var_obj)
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                feature_values[feature_name] = value
                feature_costs[feature_name] = self._normalized_input_cost(value)
                base_weights[feature_name] = weight
                gate_factors[feature_name] = gate_factor

            effective_weights, _ = self._compute_budget_allocation(base_weights, gate_factors)
            for feature_name, value in feature_values.items():
                if self.normalized_reward_mode:
                    agent_reward -= effective_weights[feature_name] * feature_costs[feature_name]
                else:
                    agent_reward -= (
                        gate_factors[feature_name]
                        * base_weights[feature_name]
                        * value
                    )

        return agent_reward

    def compute_agent_individual_composition_score(self, flat_reward_component, var_obj, agent_weights):
        if self.method == 'nonlinear_local_cost':
            return self.last_controller_composition_scores[var_obj]

        agent_weight_sum = self._compute_agent_weight_sum(agent_weights)
        feature_values = {}
        feature_costs = {}
        base_weights = {}
        gate_factors = {}

        if self.method == 'weighted_exponential':
            for feature_name, weight_key, scaled, setpoint in self.agent_feature_jobs:
                flat_key = f"{feature_name}_{var_obj}"
                if weight_key not in agent_weights:
                    continue
                value = flat_reward_component[flat_key]
                weight = self._effective_agent_weight(agent_weights, weight_key, agent_weight_sum)
                gate_cfg = self._resolve_gate_config(feature_name, var_obj)
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                feature_values[feature_name] = value
                feature_costs[feature_name] = self._exponential_cost_01(value, scaled, setpoint)
                base_weights[feature_name] = weight
                gate_factors[feature_name] = gate_factor
        else:
            for feature_name, weight_key, _, _ in self.agent_feature_jobs:
                flat_key = f"{feature_name}_{var_obj}"
                if weight_key not in agent_weights:
                    continue
                value = flat_reward_component[flat_key]
                weight = self._effective_agent_weight(agent_weights, weight_key, agent_weight_sum)
                gate_cfg = self._resolve_gate_config(feature_name, var_obj)
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                feature_values[feature_name] = value
                feature_costs[feature_name] = self._normalized_input_cost(value)
                base_weights[feature_name] = weight
                gate_factors[feature_name] = gate_factor

        effective_weights, _ = self._compute_budget_allocation(base_weights, gate_factors)
        cost_01 = 0.0
        for feature_name in feature_values.keys():
            cost_01 += effective_weights[feature_name] * feature_costs[feature_name]

        return 1.0 - self._clip_01(cost_01)

    def get_reward_params_record(self):
        return self.last_reward_params_record.copy()
