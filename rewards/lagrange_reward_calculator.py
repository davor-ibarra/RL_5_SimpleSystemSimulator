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
        self.normalized_reward_mode = self.config.get('normalized_reward_mode', False)

        if self.method == 'lineal_combination':
            self.config_lineal_combination = self.config['lineal_combination_params']
            self.weights = self._extract_lineal_weights()
            self.feature_gates = self.config_lineal_combination.get('feature_gates', {})
        elif self.method == 'weighted_exponential':
            self.config_weighted_exponential = self.config['weighted_exponential_params']
            self.weights = self._extract_exponential_weights()
            self.feature_gates = self.config_weighted_exponential.get('feature_gates', {})
        else:
            self.weights = {}
            self.feature_gates = {}

        self.var_objs = self._extract_var_objs()
        self.feature_weight_sum = self._compute_feature_weight_sum()

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
        source_value = flat_reward_component.get(source_key)
        if source_value is None:
            return 1.0

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

    def compute_reward(self, flat_reward_component):
        if self.method == 'lineal_combination':
            return self._compute_lineal(flat_reward_component)
        if self.method == 'weighted_exponential':
            return self._compute_exponential(flat_reward_component)
        return 0.0, {}, {}

    def _compute_lineal(self, flat_reward_component):
        aggregated_values = {feature: 0.0 for feature in self.weights.keys()}
        feature_contributions = {feature: 0.0 for feature in self.weights.keys()}
        controller_rewards = {}
        loop_cost_records = {}
        gate_records = {}
        feature_cost_records = {}
        contribution_records = {}

        for var_obj, jobs in self.var_jobs.items():
            loop_reward = 0.0
            loop_cost_01 = 0.0
            for flat_key, feature_name, weight, raw_weight, _, _, gate_cfg in jobs:
                value = flat_reward_component[flat_key]
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                value_cost_01 = self._normalized_input_cost(value)
                gated_cost_01 = gate_factor * value_cost_01

                aggregated_values[feature_name] += value
                if self.normalized_reward_mode:
                    contribution = -weight * gated_cost_01
                else:
                    contribution = -raw_weight * gate_factor * value

                feature_contributions[feature_name] += contribution
                loop_reward += contribution
                loop_cost_01 += weight * gated_cost_01

                feature_cost_records[f'principal_feature_cost_01_{feature_name}_{var_obj}'] = value_cost_01
                gate_records[f'principal_gate_{feature_name}_{var_obj}'] = gate_factor
                contribution_records[f'principal_contribution_{feature_name}_{var_obj}'] = contribution

            controller_rewards[var_obj] = loop_reward
            loop_cost_records[f'principal_loop_cost_01_{var_obj}'] = loop_cost_01

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
        lagrangian_total = -principal_reward

        reward_params_record = {
            'principal_reward': principal_reward,
            'principal_reward_normalized_mode': float(self.normalized_reward_mode),
            'principal_cost_01': principal_cost_01,
            'principal_score_01': 1.0 - principal_cost_01,
            'principal_reward_bound_abs': 1.0,
            'controller_rewards': controller_rewards,
            'lagrangian_total': lagrangian_total,
        }
        for feature_name, value in aggregated_values.items():
            reward_params_record[f'principal_feature_value_{feature_name}'] = value
            reward_params_record[f'principal_feature_reward_{feature_name}'] = feature_contributions[feature_name]
        reward_params_record.update(loop_cost_records)
        reward_params_record.update(feature_cost_records)
        reward_params_record.update(gate_records)
        reward_params_record.update(contribution_records)

        self.last_reward_params_record = reward_params_record
        return principal_reward, reward_params_record, controller_rewards

    def _compute_exponential(self, flat_reward_component):
        aggregated_values = {feature: 0.0 for feature in self.weights.keys()}
        feature_contributions = {feature: 0.0 for feature in self.weights.keys()}
        controller_rewards = {}
        loop_cost_records = {}
        gate_records = {}
        feature_cost_records = {}
        contribution_records = {}

        for var_obj, jobs in self.var_jobs.items():
            loop_reward = 0.0
            loop_cost_01 = 0.0
            for flat_key, feature_name, weight, raw_weight, scaled, setpoint, gate_cfg in jobs:
                value = flat_reward_component[flat_key]
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                exp_term = math.exp(-scaled * (value - setpoint) ** 2)
                value_cost_01 = self._exponential_cost_01(value, scaled, setpoint)
                if self.normalized_reward_mode:
                    contribution = -gate_factor * weight * value_cost_01
                else:
                    contribution = -gate_factor * raw_weight * (1 - exp_term)

                aggregated_values[feature_name] += value
                feature_contributions[feature_name] += contribution
                loop_reward += contribution
                loop_cost_01 += gate_factor * weight * value_cost_01

                feature_cost_records[f'principal_feature_cost_01_{feature_name}_{var_obj}'] = value_cost_01
                gate_records[f'principal_gate_{feature_name}_{var_obj}'] = gate_factor
                contribution_records[f'principal_contribution_{feature_name}_{var_obj}'] = contribution

            controller_rewards[var_obj] = loop_reward
            loop_cost_records[f'principal_loop_cost_01_{var_obj}'] = loop_cost_01

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
        reward_params_record = {
            'principal_reward': principal_reward,
            'principal_reward_normalized_mode': float(self.normalized_reward_mode),
            'principal_cost_01': principal_cost_01,
            'principal_score_01': 1.0 - principal_cost_01,
            'principal_reward_bound_abs': 1.0,
            'controller_rewards': controller_rewards,
        }
        for feature_name, value in aggregated_values.items():
            reward_params_record[f'principal_feature_value_{feature_name}'] = value
            reward_params_record[f'principal_feature_reward_{feature_name}'] = feature_contributions[feature_name]
        reward_params_record.update(loop_cost_records)
        reward_params_record.update(feature_cost_records)
        reward_params_record.update(gate_records)
        reward_params_record.update(contribution_records)

        self.last_reward_params_record = reward_params_record
        return principal_reward, reward_params_record, controller_rewards

    def compute_agent_individual_reward(self, flat_reward_component, var_obj, agent_weights):
        agent_reward = 0.0
        agent_weight_sum = self._compute_agent_weight_sum(agent_weights)

        if self.method == 'weighted_exponential':
            for feature_name, weight_key, scaled, setpoint in self.agent_feature_jobs:
                flat_key = f"{feature_name}_{var_obj}"
                if weight_key not in agent_weights:
                    continue
                value = flat_reward_component[flat_key]
                weight = self._effective_agent_weight(agent_weights, weight_key, agent_weight_sum)
                gate_cfg = self._resolve_gate_config(feature_name, var_obj)
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                if self.normalized_reward_mode:
                    agent_reward -= gate_factor * weight * self._exponential_cost_01(value, scaled, setpoint)
                else:
                    exp_term = math.exp(-scaled * (value - setpoint) ** 2)
                    agent_reward -= gate_factor * weight * (1 - exp_term)
        else:
            for feature_name, weight_key, _, _ in self.agent_feature_jobs:
                flat_key = f"{feature_name}_{var_obj}"
                if weight_key not in agent_weights:
                    continue
                value = flat_reward_component[flat_key]
                weight = self._effective_agent_weight(agent_weights, weight_key, agent_weight_sum)
                gate_cfg = self._resolve_gate_config(feature_name, var_obj)
                gate_factor = self._compute_gate_factor(flat_reward_component, gate_cfg)
                if self.normalized_reward_mode:
                    agent_reward -= gate_factor * weight * self._normalized_input_cost(value)
                else:
                    agent_reward -= gate_factor * weight * value

        return agent_reward

    def get_reward_params_record(self):
        return self.last_reward_params_record.copy()
