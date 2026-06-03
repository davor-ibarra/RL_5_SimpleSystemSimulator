"""
coordination_reward_handler.py

Responsabilidad:
Calcular el modulo de recompensa coordinativa por progreso global de tarea.

Cada contribucion se representa en tres niveles:
- raw: magnitud nativa del concepto.
- 01: magnitud normalizada por cota analitica/declarativa.
- weighted: contribucion luego de aplicar pesos conceptuales internos.
"""

import math


class CoordinationRewardHandler:
    """
    Manejador del modulo coordinativo por progreso global.

    Consume:
    - reward_component: metricas agregadas del intervalo.
    - extra_reward_component: series crudas por step.
    """

    def __init__(self, config):
        self.config = config
        self.normalized_reward_mode = config['normalized_reward_mode']
        self.global_weight = config['global_weight']
        self.potential_features = config['potential_features']
        self.potential_weights = config['potential_weights']
        self.progress_positive_only = config['progress_positive_only']
        self.progress_clip_max = config['progress_clip_max']
        self.potential_weight_normalization = config['potential_weight_normalization']
        self.potential_weight_normalization_enabled = self.potential_weight_normalization['enabled']
        self.potential_weight_strict_sum = self.potential_weight_normalization['strict_weight_sum']
        self.potential_weight_tolerance = self.potential_weight_normalization['tolerance']
        self.potential_transform = config['potential_transform']
        self.potential_value_power = self.potential_transform['value_power']
        if self.potential_transform['square_terms']:
            self.potential_value_power = 2.0
        self.coordination_bonus_clip = config['coordination_bonus_clip']
        self.coordination_bonus_clip_enabled = self.coordination_bonus_clip['enabled']
        self.coordination_bonus_progress_max = self.coordination_bonus_clip['progress_max']
        self.contribution_error_signal = config['contribution_error_signal']
        self.contribution_effort_signal = config['contribution_effort_signal']
        self.correction_signs = config['correction_signs']
        self.epsilon_credit = config['epsilon_credit']
        self.credit_reduction_mode = config['credit_reduction_mode']
        self.component_weights_config = config['component_weights']
        self.local_error_decay = config['local_error_decay']
        self.local_error_decay_enabled = self.local_error_decay['enabled']
        self.local_error_decay_method = self.local_error_decay['method']
        self.local_error_decay_signal = self.local_error_decay['error_signal']
        self.local_error_decay_reduction = self.local_error_decay['source_reduction']
        self.local_error_thresholds = self.local_error_decay['thresholds']
        self.local_error_lambdas = self.local_error_decay['lambdas']
        self.local_error_hard_zero = self.local_error_decay['hard_zero_above_threshold']
        self.local_potential_weights = config['local_potential_weights']
        self.local_progress = config['local_progress']
        self.local_progress_enabled = self.local_progress['enabled']
        self.local_progress_weight = self.local_progress['weight']
        self.local_progress_positive_only = self.local_progress['positive_only']
        self.local_progress_condition_mode = self.local_progress['condition_mode']
        self.local_progress_clip_enabled = self.local_progress['clip_enabled']
        self.local_progress_clip_max = self.local_progress['clip_max']
        self.local_debt = config['local_debt']
        self.local_debt_enabled = self.local_debt['enabled']
        self.local_debt_persistence = self.local_debt['persistence']
        self.local_debt_penalty_weight = self.local_debt['penalty_weight']
        self.local_debt_clip_enabled = self.local_debt['clip_enabled']
        self.local_debt_clip_max = self.local_debt['clip_max']

        self.var_objs = list(self.correction_signs.keys())
        self._compile_jobs()
        self.reset_episode()

    def _compile_jobs(self):
        if self.potential_value_power <= 0.0:
            raise ValueError("Coordination potential value_power must be positive")
        if self.epsilon_credit <= 0.0:
            raise ValueError("Coordination epsilon_credit must be positive")
        if self.normalized_reward_mode and not self.progress_positive_only:
            raise ValueError("Normalized coordination progress requires progress_positive_only=true")
        if (
            self.normalized_reward_mode
            and self.local_progress_enabled
            and not self.local_progress_positive_only
        ):
            raise ValueError("Normalized local progress requires positive_only=true")

        potential_jobs_raw_by_var = {var_obj: [] for var_obj in self.var_objs}
        self.potential_weight_sum_raw_by_var = {var_obj: 0.0 for var_obj in self.var_objs}
        self.potential_weight_sum_raw = 0.0

        for feature_name in self.potential_features:
            feature_weights = self.potential_weights[feature_name]
            for var_obj in self.var_objs:
                flat_key = f'{feature_name}_{var_obj}'
                weight = feature_weights[var_obj]
                if weight < 0.0:
                    raise ValueError(f"Coordination potential weight must be non-negative: {flat_key}={weight}")
                potential_jobs_raw_by_var[var_obj].append((flat_key, weight))
                self.potential_weight_sum_raw_by_var[var_obj] += weight
                self.potential_weight_sum_raw += weight

        self._validate_potential_weight_sum()

        self.potential_jobs = []
        self.potential_weight_sum_effective_by_var = {}
        for var_obj in self.var_objs:
            normalizer = (
                self.potential_weight_sum_raw_by_var[var_obj]
                if self.potential_weight_normalization_enabled
                else 1.0
            )
            effective_sum = 0.0
            for flat_key, weight in potential_jobs_raw_by_var[var_obj]:
                effective_weight = weight / normalizer
                self.potential_jobs.append((var_obj, flat_key, effective_weight))
                effective_sum += effective_weight
            self.potential_weight_sum_effective_by_var[var_obj] = effective_sum

        self.potential_weight_sum_effective = sum(weight for _, _, weight in self.potential_jobs)
        if self.potential_weight_sum_effective <= 0.0:
            raise ValueError("Coordination potential effective weight sum must be positive")
        self.task_potential_bound = self.potential_weight_sum_effective
        self.task_progress_bound = self.task_potential_bound
        self.task_progress_normalization_bound = self._compute_task_progress_normalization_bound()

        self.contribution_jobs = []
        for var_obj in self.var_objs:
            error_key = self.contribution_error_signal.format(var_obj=var_obj)
            effort_key = self.contribution_effort_signal.format(var_obj=var_obj)
            correction_sign = self.correction_signs[var_obj]
            self.contribution_jobs.append((var_obj, error_key, effort_key, correction_sign))

        self.local_potential_jobs = self._compile_local_potential_jobs(self.local_potential_weights)
        self.local_potential_bound_by_var = {
            var_obj: sum(weight for _, weight in jobs)
            for var_obj, jobs in self.local_potential_jobs.items()
        }
        if self.local_progress_enabled:
            for var_obj, bound in self.local_potential_bound_by_var.items():
                if bound <= 0.0:
                    raise ValueError(f"local_progress requires a positive local potential bound for {var_obj}")
        self.local_progress_bound_by_var = self.local_potential_bound_by_var.copy()
        self.local_debt_bound_by_var = {
            var_obj: self._compute_local_debt_bound(var_obj)
            for var_obj in self.var_objs
        }
        if self.local_debt_enabled:
            for var_obj, bound in self.local_debt_bound_by_var.items():
                if bound <= 0.0:
                    raise ValueError(f"local_debt requires a positive debt bound for {var_obj}")

        self.component_weights = self._compile_component_weights()
        self.local_progress_share_by_var = self._compile_var_weight_share(
            self.local_progress_weight,
            self.local_progress_enabled,
            'local_progress'
        )
        self.local_debt_share_by_var = self._compile_var_weight_share(
            self.local_debt_penalty_weight,
            self.local_debt_enabled,
            'local_debt'
        )

    def _validate_potential_weight_sum(self):
        for var_obj in self.var_objs:
            weight_sum = self.potential_weight_sum_raw_by_var[var_obj]
            if self.potential_weight_normalization_enabled and weight_sum <= 0.0:
                raise ValueError(
                    f"Coordination potential weight sum must be positive for {var_obj}"
                )

            if self.potential_weight_strict_sum and abs(weight_sum - 1.0) > self.potential_weight_tolerance:
                raise ValueError(
                    f"Coordination potential weights for {var_obj} must sum 1.0, got {weight_sum}"
                )

    def _compile_local_potential_jobs(self, weight_config):
        jobs = {}
        for var_obj in self.var_objs:
            jobs[var_obj] = []
            for feature_name, weight_by_var in weight_config.items():
                weight = self._resolve_var_value(weight_by_var, var_obj)
                if weight < 0.0:
                    raise ValueError(f"Local potential weight must be non-negative: {feature_name}_{var_obj}={weight}")
                flat_key = f'{feature_name}_{var_obj}'
                jobs[var_obj].append((flat_key, weight))
        return jobs

    def _compute_local_debt_bound(self, var_obj):
        if not self.local_debt_enabled:
            return 1.0

        if self.local_debt_clip_enabled:
            if self.local_debt_clip_max <= 0.0:
                raise ValueError("local_debt.clip_max must be positive when local_debt is enabled")
            return self.local_debt_clip_max

        if self.local_debt_persistence < 0.0:
            raise ValueError("local_debt.persistence must be non-negative")
        if self.local_debt_persistence < 1.0:
            return self.local_potential_bound_by_var[var_obj] / (1.0 - self.local_debt_persistence)

        raise ValueError(
            "local_debt requires clip_enabled=true or persistence < 1.0 to define an analytic bound"
        )

    def _compile_component_weights(self):
        weights = {
            'global_progress': self.component_weights_config['global_progress'],
            'local_progress': self.component_weights_config['local_progress'],
            'local_debt': self.component_weights_config['local_debt']
        }

        for component_name, weight in weights.items():
            if weight < 0.0:
                raise ValueError(f"Coordination component weight must be non-negative: {component_name}={weight}")

        if not self.local_progress_enabled and weights['local_progress'] > 0.0:
            raise ValueError("local_progress is disabled but has positive coordination component weight")
        if not self.local_debt_enabled and weights['local_debt'] > 0.0:
            raise ValueError("local_debt is disabled but has positive coordination component weight")

        if self.normalized_reward_mode:
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > self.potential_weight_tolerance:
                raise ValueError(f"Coordination component weights must sum 1.0, got {weight_sum}")

        return weights

    def _compile_var_weight_share(self, weight_config, enabled, component_name):
        if not enabled:
            return {var_obj: 0.0 for var_obj in self.var_objs}

        raw_weights = {}
        for var_obj in self.var_objs:
            raw_weight = self._resolve_var_value(weight_config, var_obj)
            if raw_weight < 0.0:
                raise ValueError(f"{component_name} variable weight must be non-negative: {var_obj}={raw_weight}")
            raw_weights[var_obj] = raw_weight

        if not self.normalized_reward_mode:
            return raw_weights

        weight_sum = sum(raw_weights.values())
        if weight_sum <= 0.0:
            raise ValueError(f"{component_name} variable weights must be positive")
        return {
            var_obj: raw_weights[var_obj] / weight_sum
            for var_obj in self.var_objs
        }

    def reset_episode(self):
        self.previous_task_potential = None
        self.previous_local_potential = {var_obj: None for var_obj in self.var_objs}
        self.local_debt_accumulated = {var_obj: 0.0 for var_obj in self.var_objs}
        self.last_coordination_reward_params_record = {}
        self.last_coordination_state_record = self._build_state_record()

    def evaluate(self, reward_component, extra_reward_component):
        task_potential = self._compute_task_potential(reward_component)
        task_progress_delta = self._compute_task_progress_delta(task_potential)
        task_progress_rewardable = self._compute_rewardable_progress(task_progress_delta)
        task_progress_01 = self._normalize_by_bound(
            task_progress_rewardable,
            self.task_progress_normalization_bound
        )
        coordination_bonus_total = self._compute_coordination_bonus_total(
            task_progress_rewardable,
            task_progress_01
        )

        credit_raw = self._compute_credit_raw(extra_reward_component)
        credit_share = self._compute_credit_share(credit_raw)
        error_abs_by_var = self._compute_local_error_abs(extra_reward_component)
        error_decay_by_var = self._compute_error_decay(error_abs_by_var)
        global_reward_raw_by_var, global_component_01_by_var = self._compute_global_components(
            credit_share,
            coordination_bonus_total,
            task_progress_01,
            error_decay_by_var
        )

        local_potential_by_var = self._compute_local_potential(reward_component)
        local_progress_delta_by_var = self._compute_local_progress_delta(local_potential_by_var)
        (
            local_progress_reward_raw_by_var,
            local_progress_01_by_var,
            local_progress_component_01_by_var,
            local_progress_condition_by_var
        ) = self._compute_local_progress_components(
            local_progress_delta_by_var,
            task_progress_delta
        )
        (
            debt_penalty_raw_by_var,
            debt_01_by_var,
            debt_component_01_by_var
        ) = self._compute_debt_components(local_potential_by_var)

        weighted_global_by_var = {}
        weighted_progress_by_var = {}
        weighted_debt_by_var = {}
        coordination_module_rewards_by_var = {}
        for var_obj in self.var_objs:
            if self.normalized_reward_mode:
                weighted_global_by_var[var_obj] = (
                    self.component_weights['global_progress']
                    * global_component_01_by_var[var_obj]
                )
                weighted_progress_by_var[var_obj] = (
                    self.component_weights['local_progress']
                    * local_progress_component_01_by_var[var_obj]
                )
                weighted_debt_by_var[var_obj] = (
                    self.component_weights['local_debt']
                    * debt_component_01_by_var[var_obj]
                )
            else:
                weighted_global_by_var[var_obj] = global_reward_raw_by_var[var_obj]
                weighted_progress_by_var[var_obj] = local_progress_reward_raw_by_var[var_obj]
                weighted_debt_by_var[var_obj] = debt_penalty_raw_by_var[var_obj]

            coordination_module_rewards_by_var[var_obj] = (
                weighted_global_by_var[var_obj]
                + weighted_progress_by_var[var_obj]
                - weighted_debt_by_var[var_obj]
            )

        coordination_reward_total = sum(coordination_module_rewards_by_var.values())
        flat_record = self._build_flat_record(
            task_potential,
            task_progress_delta,
            task_progress_rewardable,
            task_progress_01,
            coordination_bonus_total,
            credit_raw,
            credit_share,
            error_abs_by_var,
            error_decay_by_var,
            global_reward_raw_by_var,
            global_component_01_by_var,
            weighted_global_by_var,
            local_potential_by_var,
            local_progress_delta_by_var,
            local_progress_reward_raw_by_var,
            local_progress_01_by_var,
            local_progress_component_01_by_var,
            local_progress_condition_by_var,
            weighted_progress_by_var,
            debt_penalty_raw_by_var,
            debt_01_by_var,
            debt_component_01_by_var,
            weighted_debt_by_var,
            coordination_module_rewards_by_var,
            coordination_reward_total
        )

        self.previous_task_potential = task_potential
        self.previous_local_potential = local_potential_by_var.copy()
        self.last_coordination_reward_params_record = flat_record
        self.last_coordination_state_record = self._build_state_record()

        return coordination_reward_total, flat_record, coordination_module_rewards_by_var

    def _compute_task_potential(self, reward_component):
        task_potential = 0.0
        for _, flat_key, weight in self.potential_jobs:
            value = self._reward_cost_input_01(reward_component[flat_key], flat_key)
            task_potential += weight * self._transform_potential_value(value)
        return task_potential

    def _transform_potential_value(self, value):
        if self.potential_value_power == 1.0:
            return value
        return abs(value) ** self.potential_value_power

    def _compute_task_progress_delta(self, current_task_potential):
        if self.previous_task_potential is None:
            return 0.0
        return self.previous_task_potential - current_task_potential

    def _compute_rewardable_progress(self, task_progress_delta):
        progress_value = max(0.0, task_progress_delta) if self.progress_positive_only else task_progress_delta

        if self.normalized_reward_mode:
            return min(self.task_progress_normalization_bound, progress_value)

        if self.coordination_bonus_clip_enabled:
            progress_value = min(progress_value, self.coordination_bonus_progress_max)
        elif self.progress_clip_max > 0.0:
            progress_value = min(progress_value, self.progress_clip_max)

        if self.progress_positive_only:
            progress_value = max(0.0, progress_value)
        return progress_value

    def _compute_coordination_bonus_total(self, task_progress_rewardable, task_progress_01):
        if self.normalized_reward_mode:
            return task_progress_01
        return self.global_weight * task_progress_rewardable

    def _compute_task_progress_normalization_bound(self):
        if self.coordination_bonus_clip_enabled:
            if self.coordination_bonus_progress_max <= 0.0:
                raise ValueError("coordination_bonus_clip.progress_max must be positive when enabled")
            return min(self.task_progress_bound, self.coordination_bonus_progress_max)
        if self.progress_clip_max > 0.0:
            return min(self.task_progress_bound, self.progress_clip_max)
        return self.task_progress_bound

    def _compute_credit_raw(self, extra_reward_component):
        credit_raw = {}
        for var_obj, error_key, effort_key, correction_sign in self.contribution_jobs:
            error_series = extra_reward_component[error_key]
            effort_series = extra_reward_component[effort_key]
            n_steps = min(len(error_series), len(effort_series))
            if n_steps <= 0:
                raise ValueError(f"Empty coordination credit series for {var_obj}")

            raw_credit_sum = 0.0
            for step_idx in range(n_steps):
                corrective_contribution = correction_sign * error_series[step_idx] * effort_series[step_idx]
                raw_credit_sum += max(0.0, corrective_contribution)

            if self.credit_reduction_mode == 'average':
                credit_raw[var_obj] = raw_credit_sum / n_steps
            elif self.credit_reduction_mode == 'sum':
                credit_raw[var_obj] = raw_credit_sum
            else:
                raise ValueError(f"Unsupported coordination credit_reduction_mode: {self.credit_reduction_mode}")

        return credit_raw

    def _compute_credit_share(self, credit_raw):
        credit_raw_total = sum(credit_raw[var_obj] for var_obj in self.var_objs)
        denominator = self.epsilon_credit + credit_raw_total
        return {
            var_obj: credit_raw[var_obj] / denominator
            for var_obj in self.var_objs
        }

    def _compute_local_error_abs(self, extra_reward_component):
        error_abs_by_var = {}
        for var_obj in self.var_objs:
            error_key = self.local_error_decay_signal.format(var_obj=var_obj)
            error_series = extra_reward_component[error_key]
            error_value = self._reduce_series(error_series, self.local_error_decay_reduction)
            error_abs_by_var[var_obj] = abs(error_value)
        return error_abs_by_var

    def _compute_error_decay(self, error_abs_by_var):
        decay_by_var = {}
        for var_obj in self.var_objs:
            if not self.local_error_decay_enabled:
                decay_by_var[var_obj] = 1.0
                continue

            threshold = self._resolve_var_value(self.local_error_thresholds, var_obj)
            if threshold <= 0.0:
                raise ValueError(f"local_error_decay threshold must be positive for {var_obj}")

            ratio = error_abs_by_var[var_obj] / threshold
            if self.local_error_decay_method == 'linear':
                factor = max(0.0, 1.0 - ratio)
            elif self.local_error_decay_method == 'exponential':
                lambda_v = self._resolve_var_value(self.local_error_lambdas, var_obj)
                factor = math.exp(-lambda_v * (ratio ** 2))
            elif self.local_error_decay_method == 'hard':
                factor = 1.0 if ratio <= 1.0 else 0.0
            else:
                raise ValueError(f"Unsupported coordination error decay method: {self.local_error_decay_method}")

            if self.local_error_hard_zero and ratio > 1.0:
                factor = 0.0

            decay_by_var[var_obj] = min(1.0, max(0.0, factor))

        return decay_by_var

    def _compute_global_components(self, credit_share, coordination_bonus_total, task_progress_01, error_decay_by_var):
        raw_by_var = {}
        component_01_by_var = {}
        for var_obj in self.var_objs:
            raw_by_var[var_obj] = (
                credit_share[var_obj]
                * coordination_bonus_total
                * error_decay_by_var[var_obj]
            )
            component_01_by_var[var_obj] = (
                credit_share[var_obj]
                * task_progress_01
                * error_decay_by_var[var_obj]
            )
        return raw_by_var, component_01_by_var

    def _compute_local_potential(self, reward_component):
        local_potential_by_var = {}
        for var_obj in self.var_objs:
            potential = 0.0
            for flat_key, weight in self.local_potential_jobs[var_obj]:
                value = self._reward_cost_input_01(reward_component[flat_key], flat_key)
                potential += weight * value
            local_potential_by_var[var_obj] = potential
        return local_potential_by_var

    def _compute_local_progress_delta(self, local_potential_by_var):
        progress_delta_by_var = {}
        for var_obj in self.var_objs:
            previous = self.previous_local_potential[var_obj]
            if previous is None:
                progress_delta_by_var[var_obj] = 0.0
            else:
                progress_delta_by_var[var_obj] = previous - local_potential_by_var[var_obj]
        return progress_delta_by_var

    def _compute_local_progress_components(self, local_progress_delta_by_var, task_progress_delta):
        condition_factor_by_var = self._compute_local_progress_condition(
            local_progress_delta_by_var,
            task_progress_delta
        )
        progress_reward_raw_by_var = {}
        progress_01_by_var = {}
        progress_component_01_by_var = {}

        for var_obj in self.var_objs:
            if not self.local_progress_enabled:
                progress_reward_raw_by_var[var_obj] = 0.0
                progress_01_by_var[var_obj] = 0.0
                progress_component_01_by_var[var_obj] = 0.0
                continue

            progress_value = local_progress_delta_by_var[var_obj]
            if self.local_progress_positive_only:
                progress_value = max(0.0, progress_value)

            if self.normalized_reward_mode:
                progress_value = min(progress_value, self.local_progress_bound_by_var[var_obj])
                progress_01 = self._normalize_by_bound(
                    progress_value,
                    self.local_progress_bound_by_var[var_obj]
                )
                progress_reward_raw_by_var[var_obj] = progress_value * condition_factor_by_var[var_obj]
                progress_01_by_var[var_obj] = progress_01
                progress_component_01_by_var[var_obj] = (
                    self.local_progress_share_by_var[var_obj]
                    * progress_01
                    * condition_factor_by_var[var_obj]
                )
            else:
                if self.local_progress_clip_enabled:
                    progress_value = min(progress_value, self.local_progress_clip_max)
                weight = self.local_progress_share_by_var[var_obj]
                progress_reward_raw_by_var[var_obj] = (
                    weight
                    * progress_value
                    * condition_factor_by_var[var_obj]
                )
                progress_01_by_var[var_obj] = self._normalize_by_bound(
                    max(0.0, progress_value),
                    self.local_progress_bound_by_var[var_obj]
                )
                progress_component_01_by_var[var_obj] = (
                    self.local_progress_share_by_var[var_obj]
                    * progress_01_by_var[var_obj]
                    * condition_factor_by_var[var_obj]
                )

        return (
            progress_reward_raw_by_var,
            progress_01_by_var,
            progress_component_01_by_var,
            condition_factor_by_var
        )

    def _compute_local_progress_condition(self, local_progress_delta_by_var, task_progress_delta):
        if self.local_progress_condition_mode == 'always':
            return {var_obj: 1.0 for var_obj in self.var_objs}

        if self.local_progress_condition_mode == 'global_progress_positive':
            factor = 1.0 if task_progress_delta > 0.0 else 0.0
            return {var_obj: factor for var_obj in self.var_objs}

        if self.local_progress_condition_mode == 'all_local_progress_positive':
            all_positive = all(
                local_progress_delta_by_var[var_obj] > 0.0
                for var_obj in self.var_objs
            )
            factor = 1.0 if all_positive else 0.0
            return {var_obj: factor for var_obj in self.var_objs}

        if self.local_progress_condition_mode == 'global_and_all_local_positive':
            all_positive = all(
                local_progress_delta_by_var[var_obj] > 0.0
                for var_obj in self.var_objs
            )
            factor = 1.0 if task_progress_delta > 0.0 and all_positive else 0.0
            return {var_obj: factor for var_obj in self.var_objs}

        raise ValueError(f"Unsupported local progress condition mode: {self.local_progress_condition_mode}")

    def _compute_debt_components(self, local_potential_by_var):
        debt_penalty_raw_by_var = {}
        debt_01_by_var = {}
        debt_component_01_by_var = {}

        for var_obj in self.var_objs:
            if not self.local_debt_enabled:
                self.local_debt_accumulated[var_obj] = 0.0
                debt_penalty_raw_by_var[var_obj] = 0.0
                debt_01_by_var[var_obj] = 0.0
                debt_component_01_by_var[var_obj] = 0.0
                continue

            debt_value = (
                self.local_debt_persistence * self.local_debt_accumulated[var_obj]
                + local_potential_by_var[var_obj]
            )
            if self.local_debt_clip_enabled:
                debt_value = min(debt_value, self.local_debt_clip_max)

            self.local_debt_accumulated[var_obj] = max(0.0, debt_value)
            debt_01 = self._normalize_by_bound(
                self.local_debt_accumulated[var_obj],
                self.local_debt_bound_by_var[var_obj]
            )

            if self.normalized_reward_mode:
                debt_penalty_raw_by_var[var_obj] = self.local_debt_accumulated[var_obj]
                debt_component_01_by_var[var_obj] = self.local_debt_share_by_var[var_obj] * debt_01
            else:
                weight = self.local_debt_share_by_var[var_obj]
                debt_penalty_raw_by_var[var_obj] = weight * self.local_debt_accumulated[var_obj]
                debt_component_01_by_var[var_obj] = self.local_debt_share_by_var[var_obj] * debt_01

            debt_01_by_var[var_obj] = debt_01

        return debt_penalty_raw_by_var, debt_01_by_var, debt_component_01_by_var

    def _build_flat_record(
        self,
        task_potential,
        task_progress_delta,
        task_progress_rewardable,
        task_progress_01,
        coordination_bonus_total,
        credit_raw,
        credit_share,
        error_abs_by_var,
        error_decay_by_var,
        global_reward_raw_by_var,
        global_component_01_by_var,
        weighted_global_by_var,
        local_potential_by_var,
        local_progress_delta_by_var,
        local_progress_reward_raw_by_var,
        local_progress_01_by_var,
        local_progress_component_01_by_var,
        local_progress_condition_by_var,
        weighted_progress_by_var,
        debt_penalty_raw_by_var,
        debt_01_by_var,
        debt_component_01_by_var,
        weighted_debt_by_var,
        coordination_module_rewards_by_var,
        coordination_reward_total
    ):
        global_component_01_total = sum(global_component_01_by_var.values())
        progress_component_01_total = sum(local_progress_component_01_by_var.values())
        debt_component_01_total = sum(debt_component_01_by_var.values())
        weighted_global_total = sum(weighted_global_by_var.values())
        weighted_progress_total = sum(weighted_progress_by_var.values())
        weighted_debt_total = sum(weighted_debt_by_var.values())
        coordination_composition_score_by_var = self._compute_composition_scores_by_var(
            global_component_01_by_var,
            local_progress_01_by_var,
            debt_01_by_var
        )
        coordination_composition_score_01 = self._clip_01(
            sum(coordination_composition_score_by_var.values())
        )

        flat_record = {
            'coordination_reward_normalized_mode': float(self.normalized_reward_mode),
            'coordination_reward_bound': self._compute_coordination_reward_bound(),
            'coordination_potential_weight_sum_raw': self.potential_weight_sum_raw,
            'coordination_potential_weight_sum_effective': self.potential_weight_sum_effective,
            'coordination_potential_value_power': self.potential_value_power,
            'coordination_global_potential_bound': self.task_potential_bound,
            'coordination_global_progress_bound': self.task_progress_normalization_bound,
            'coordination_global_progress_potential_bound': self.task_progress_bound,
            'coordination_component_weight_global_progress': self.component_weights['global_progress'],
            'coordination_component_weight_local_progress': self.component_weights['local_progress'],
            'coordination_component_weight_local_debt': self.component_weights['local_debt'],
            'coordination_component_weight_sum': sum(self.component_weights.values()),
            'coordination_bonus_clip_enabled': float(self.coordination_bonus_clip_enabled),
            'coordination_bonus_total': coordination_bonus_total,
            'coordination_global_reward_total': sum(global_reward_raw_by_var.values()),
            'coordination_global_component_01_total': global_component_01_total,
            'coordination_progress_component_01_total': progress_component_01_total,
            'coordination_debt_component_01_total': debt_component_01_total,
            'coordination_component_01_global_progress_total': global_component_01_total,
            'coordination_component_01_local_progress_total': progress_component_01_total,
            'coordination_component_01_local_debt_total': debt_component_01_total,
            'coordination_component_weighted_global_progress_total': weighted_global_total,
            'coordination_component_weighted_local_progress_total': weighted_progress_total,
            'coordination_component_weighted_local_debt_total': weighted_debt_total,
            'coordination_task_potential': task_potential,
            'coordination_task_progress_delta': task_progress_delta,
            'coordination_task_progress_positive': task_progress_rewardable,
            'coordination_task_progress_01': task_progress_01,
            'coordination_reward_total_01': coordination_composition_score_01,
            'coordination_composition_score_01': coordination_composition_score_01,
            'coordination_composition_cost_01': 1.0 - coordination_composition_score_01,
            'coordination_reward_total': coordination_reward_total,
            'coordination_module_reward_total': coordination_reward_total,
            'coordination_progress_reward_total': weighted_progress_total,
            'coordination_debt_penalty_total': weighted_debt_total
        }

        for var_obj in self.var_objs:
            flat_record[f'coordination_credit_raw_{var_obj}'] = credit_raw[var_obj]
            flat_record[f'coordination_credit_share_{var_obj}'] = credit_share[var_obj]
            flat_record[f'coordination_global_credit_raw_{var_obj}'] = credit_raw[var_obj]
            flat_record[f'coordination_global_credit_share_01_{var_obj}'] = credit_share[var_obj]
            flat_record[f'coordination_potential_weight_sum_raw_{var_obj}'] = self.potential_weight_sum_raw_by_var[var_obj]
            flat_record[f'coordination_potential_weight_sum_effective_{var_obj}'] = self.potential_weight_sum_effective_by_var[var_obj]
            flat_record[f'coordination_error_abs_{var_obj}'] = error_abs_by_var[var_obj]
            flat_record[f'coordination_error_decay_{var_obj}'] = error_decay_by_var[var_obj]
            flat_record[f'coordination_global_error_decay_01_{var_obj}'] = error_decay_by_var[var_obj]
            flat_record[f'coordination_reward_01_{var_obj}'] = global_component_01_by_var[var_obj]
            flat_record[f'coordination_global_reward_{var_obj}'] = weighted_global_by_var[var_obj]
            flat_record[f'coordination_global_raw_{var_obj}'] = global_reward_raw_by_var[var_obj]
            flat_record[f'coordination_global_component_01_{var_obj}'] = global_component_01_by_var[var_obj]
            flat_record[f'coordination_global_component_weighted_{var_obj}'] = weighted_global_by_var[var_obj]
            flat_record[f'coordination_component_01_global_progress_{var_obj}'] = global_component_01_by_var[var_obj]
            flat_record[f'coordination_component_weighted_global_progress_{var_obj}'] = weighted_global_by_var[var_obj]
            flat_record[f'coordination_composition_score_01_{var_obj}'] = coordination_composition_score_by_var[var_obj]
            flat_record[f'coordination_composition_cost_01_{var_obj}'] = 1.0 - coordination_composition_score_by_var[var_obj]

            flat_record[f'coordination_local_potential_{var_obj}'] = local_potential_by_var[var_obj]
            flat_record[f'coordination_local_progress_delta_{var_obj}'] = local_progress_delta_by_var[var_obj]
            flat_record[f'coordination_progress_bound_{var_obj}'] = self.local_progress_bound_by_var[var_obj]
            flat_record[f'coordination_progress_raw_{var_obj}'] = local_progress_reward_raw_by_var[var_obj]
            flat_record[f'coordination_progress_01_{var_obj}'] = local_progress_01_by_var[var_obj]
            flat_record[f'coordination_progress_share_{var_obj}'] = self.local_progress_share_by_var[var_obj]
            flat_record[f'coordination_progress_condition_01_{var_obj}'] = local_progress_condition_by_var[var_obj]
            flat_record[f'coordination_progress_component_01_{var_obj}'] = local_progress_component_01_by_var[var_obj]
            flat_record[f'coordination_progress_component_weighted_{var_obj}'] = weighted_progress_by_var[var_obj]
            flat_record[f'coordination_component_01_local_progress_{var_obj}'] = local_progress_component_01_by_var[var_obj]
            flat_record[f'coordination_component_weighted_local_progress_{var_obj}'] = weighted_progress_by_var[var_obj]
            flat_record[f'coordination_progress_reward_{var_obj}'] = weighted_progress_by_var[var_obj]

            flat_record[f'coordination_debt_bound_{var_obj}'] = self.local_debt_bound_by_var[var_obj]
            flat_record[f'coordination_debt_raw_{var_obj}'] = self.local_debt_accumulated[var_obj]
            flat_record[f'coordination_debt_{var_obj}'] = debt_01_by_var[var_obj]
            flat_record[f'coordination_debt_01_{var_obj}'] = debt_01_by_var[var_obj]
            flat_record[f'coordination_debt_share_{var_obj}'] = self.local_debt_share_by_var[var_obj]
            flat_record[f'coordination_debt_component_01_{var_obj}'] = debt_component_01_by_var[var_obj]
            flat_record[f'coordination_debt_component_weighted_{var_obj}'] = weighted_debt_by_var[var_obj]
            flat_record[f'coordination_component_01_local_debt_{var_obj}'] = debt_component_01_by_var[var_obj]
            flat_record[f'coordination_component_weighted_local_debt_{var_obj}'] = weighted_debt_by_var[var_obj]
            flat_record[f'coordination_debt_penalty_{var_obj}'] = weighted_debt_by_var[var_obj]

            flat_record[f'coordination_module_reward_{var_obj}'] = coordination_module_rewards_by_var[var_obj]
            flat_record[f'extra_conditional_coordination_bonus_{var_obj}'] = weighted_global_by_var[var_obj]

        return flat_record

    def _reward_cost_input_01(self, value, key):
        if not self.normalized_reward_mode:
            return abs(value)

        tolerance = self.potential_weight_tolerance
        if value < -tolerance or value > 1.0 + tolerance:
            raise ValueError(f"Normalized reward cost {key} must be in [0, 1], got {value}")
        return min(1.0, max(0.0, value))

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))

    def _normalize_by_bound(self, value, bound):
        if bound <= 0.0:
            raise ValueError("Normalization bound must be positive")
        return min(1.0, max(0.0, value / bound))

    def _normalized_component_weights_for_score(self):
        weights = {
            name: max(0.0, value)
            for name, value in self.component_weights.items()
        }
        weight_sum = sum(weights.values())
        if weight_sum <= 0.0:
            return {name: 0.0 for name in weights}
        return {
            name: value / weight_sum
            for name, value in weights.items()
        }

    def _normalized_var_shares_for_score(self, raw_shares, enabled):
        if not enabled:
            return {var_obj: 0.0 for var_obj in self.var_objs}
        share_sum = sum(max(0.0, raw_shares[var_obj]) for var_obj in self.var_objs)
        if share_sum <= 0.0:
            return {var_obj: 0.0 for var_obj in self.var_objs}
        return {
            var_obj: max(0.0, raw_shares[var_obj]) / share_sum
            for var_obj in self.var_objs
        }

    def _compute_composition_scores_by_var(
        self,
        global_component_01_by_var,
        local_progress_01_by_var,
        debt_01_by_var
    ):
        component_weights = self._normalized_component_weights_for_score()
        progress_shares = self._normalized_var_shares_for_score(
            self.local_progress_share_by_var,
            self.local_progress_enabled
        )
        debt_shares = self._normalized_var_shares_for_score(
            self.local_debt_share_by_var,
            self.local_debt_enabled
        )

        scores = {}
        for var_obj in self.var_objs:
            global_score = (
                component_weights['global_progress']
                * self._clip_01(global_component_01_by_var[var_obj])
            )
            progress_score = (
                component_weights['local_progress']
                * progress_shares[var_obj]
                * self._clip_01(local_progress_01_by_var[var_obj])
            )
            debt_score = (
                component_weights['local_debt']
                * debt_shares[var_obj]
                * (1.0 - self._clip_01(debt_01_by_var[var_obj]))
            )
            scores[var_obj] = self._clip_01(global_score + progress_score + debt_score)
        return scores

    def _compute_coordination_reward_bound(self):
        if self.normalized_reward_mode:
            return 1.0
        if self.coordination_bonus_clip_enabled:
            return max(0.0, self.global_weight * self.coordination_bonus_progress_max)
        if self.progress_clip_max > 0.0:
            return max(0.0, self.global_weight * self.progress_clip_max)
        return 0.0

    def _resolve_var_value(self, value_config, var_obj):
        if isinstance(value_config, dict):
            return value_config[var_obj]
        return value_config

    def _reduce_series(self, values, reduction_mode):
        if not values:
            raise ValueError(f"Cannot reduce empty coordination series with mode {reduction_mode}")

        if reduction_mode == 'keep_last':
            return values[-1]
        if reduction_mode == 'mean':
            return sum(values) / len(values)
        if reduction_mode == 'mean_abs':
            return sum(abs(value) for value in values) / len(values)
        if reduction_mode == 'rms':
            return math.sqrt(sum(value ** 2 for value in values) / len(values))

        raise ValueError(f"Unsupported coordination series reduction: {reduction_mode}")

    def _build_state_record(self):
        state_record = {}
        for var_obj in self.var_objs:
            state_record[f'coordination_debt_raw_{var_obj}'] = self.local_debt_accumulated[var_obj]
            state_record[f'coordination_debt_{var_obj}'] = self._normalize_by_bound(
                self.local_debt_accumulated[var_obj],
                self.local_debt_bound_by_var[var_obj]
            )
        return state_record

    def get_coordination_reward_params_record(self):
        return self.last_coordination_reward_params_record.copy()

    def get_state_record(self):
        return self.last_coordination_state_record.copy()
