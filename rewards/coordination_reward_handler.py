"""
coordination_reward_handler.py

Responsabilidad:
Calcular el modulo de recompensa coordinativa por progreso global de tarea.

Implementa:
- potencial global Phi(k)
- progreso Delta Phi(k)
- reparto de credito por contribucion correctiva
- bonus coordinativo acotado por lazo
- decaimiento por error local
- progreso local hacia setpoint
- deuda local acumulada
"""

import math


class CoordinationRewardHandler:
    """
    Manejador del modulo coordinativo por progreso global.

    Consume:
    - reward_component: metricas agregadas del intervalo
    - extra_reward_component: series crudas por step
    """

    def __init__(self, config):
        """
        Inicializa el bloque coordinativo con configuracion declarativa.

        Args:
            config (dict): Configuracion de coordination_reward
        """
        self.config = config
        self.reward_mode = config['reward_mode']
        self.assign_mode = config['assign_mode']
        self.normalized_reward_mode = config.get('normalized_reward_mode', False)
        self.global_weight = config['global_weight']
        self.potential_features = config['potential_features']
        self.potential_weights = config['potential_weights']
        self.progress_positive_only = config['progress_positive_only']
        self.progress_clip_max = config['progress_clip_max']
        self.potential_weight_normalization = config.get('potential_weight_normalization', {})
        self.potential_weight_normalization_enabled = self.potential_weight_normalization.get('enabled', False)
        self.potential_weight_strict_sum = self.potential_weight_normalization.get('strict_weight_sum', False)
        self.potential_weight_tolerance = self.potential_weight_normalization.get('tolerance', 1.0e-9)
        self.coordination_bonus_clip = config.get('coordination_bonus_clip', {})
        self.coordination_bonus_clip_enabled = self.coordination_bonus_clip.get('enabled', True)
        self.coordination_bonus_progress_max = self.coordination_bonus_clip.get('progress_max', self.progress_clip_max)
        self.contribution_error_signal = config['contribution_error_signal']
        self.contribution_effort_signal = config['contribution_effort_signal']
        self.correction_signs = config['correction_signs']
        self.epsilon_credit = config['epsilon_credit']
        self.credit_reduction_mode = config['credit_reduction_mode']
        self.local_error_decay = config.get('local_error_decay', {})
        self.local_error_decay_enabled = self.local_error_decay.get('enabled', False)
        self.local_error_decay_method = self.local_error_decay.get('method', 'linear')
        self.local_error_decay_signal = self.local_error_decay.get('error_signal', 'error_{var_obj}')
        self.local_error_decay_reduction = self.local_error_decay.get('source_reduction', 'keep_last')
        self.local_error_thresholds = self.local_error_decay.get('thresholds', {})
        self.local_error_lambdas = self.local_error_decay.get('lambdas', {})
        self.local_error_hard_zero = self.local_error_decay.get('hard_zero_above_threshold', False)
        self.local_potential_weights = config.get('local_potential_weights', {'L_e': 1.0})
        self.local_progress = config.get('local_progress', {})
        self.local_progress_enabled = self.local_progress.get('enabled', False)
        self.local_progress_weight = self.local_progress.get('weight', 1.0)
        self.local_progress_positive_only = self.local_progress.get('positive_only', True)
        self.local_progress_condition_mode = self.local_progress.get('condition_mode', 'always')
        self.local_progress_clip_enabled = self.local_progress.get('clip_enabled', False)
        self.local_progress_clip_max = self.local_progress.get('clip_max', self.progress_clip_max)
        self.local_debt = config.get('local_debt', {})
        self.local_debt_enabled = self.local_debt.get('enabled', False)
        self.local_debt_persistence = self.local_debt.get('persistence', 0.0)
        self.local_debt_penalty_weight = self.local_debt.get('penalty_weight', 0.0)
        self.local_debt_clip_enabled = self.local_debt.get('clip_enabled', False)
        self.local_debt_clip_max = self.local_debt.get('clip_max', 1.0)

        self.var_objs = list(self.correction_signs.keys())
        self._compile_jobs()
        self.reset_episode()

    def _compile_jobs(self):
        """
        Precompila llaves y pesos para evitar armado dinamico en runtime.
        """
        potential_jobs_raw = []
        self.potential_weight_sum_raw = 0.0
        for feature_name in self.potential_features:
            feature_weights = self.potential_weights[feature_name]
            for var_obj in self.var_objs:
                flat_key = f'{feature_name}_{var_obj}'
                weight = feature_weights[var_obj]
                if weight < 0.0:
                    raise ValueError(f"Coordination potential weight must be non-negative: {flat_key}={weight}")
                potential_jobs_raw.append((flat_key, weight))
                self.potential_weight_sum_raw += weight

        self._validate_potential_weight_sum()
        normalizer = self.potential_weight_sum_raw if self.potential_weight_normalization_enabled else 1.0
        self.potential_jobs = []
        for flat_key, weight in potential_jobs_raw:
            self.potential_jobs.append((flat_key, weight / normalizer))
        self.potential_weight_sum_effective = sum(weight for _, weight in self.potential_jobs)

        self.contribution_jobs = []
        for var_obj in self.var_objs:
            error_key = self.contribution_error_signal.format(var_obj=var_obj)
            effort_key = self.contribution_effort_signal.format(var_obj=var_obj)
            correction_sign = self.correction_signs[var_obj]
            self.contribution_jobs.append((var_obj, error_key, effort_key, correction_sign))

        self.local_potential_jobs = self._compile_local_potential_jobs(self.local_potential_weights)

    def _validate_potential_weight_sum(self):
        """
        Valida o normaliza la composicion declarativa del potencial global.
        """
        if self.potential_weight_normalization_enabled and self.potential_weight_sum_raw <= 0.0:
            raise ValueError("Coordination potential weight sum must be positive when normalization is enabled")

        if self.potential_weight_strict_sum and abs(self.potential_weight_sum_raw - 1.0) > self.potential_weight_tolerance:
            raise ValueError(
                f"Coordination potential weights must sum 1.0, got {self.potential_weight_sum_raw}"
            )

    def _compile_local_potential_jobs(self, weight_config):
        """
        Precompila Phi_local por lazo desde pesos escalares o por variable.
        """
        jobs = {}
        for var_obj in self.var_objs:
            jobs[var_obj] = []
            for feature_name, weight_by_var in weight_config.items():
                weight = self._resolve_var_value(weight_by_var, var_obj, 0.0)
                if weight < 0.0:
                    raise ValueError(f"Local potential weight must be non-negative: {feature_name}_{var_obj}={weight}")
                flat_key = f'{feature_name}_{var_obj}'
                jobs[var_obj].append((flat_key, weight))
        return jobs

    def reset_episode(self):
        """
        Resetea la memoria del bloque al inicio de cada episodio.
        """
        self.previous_task_potential = None
        self.previous_local_potential = {var_obj: None for var_obj in self.var_objs}
        self.local_debt_accumulated = {var_obj: 0.0 for var_obj in self.var_objs}
        self.last_coordination_reward_params_record = {}
        self.last_coordination_state_record = self._build_state_record()

    def evaluate(self, reward_component, extra_reward_component):
        """
        Evalua el modulo coordinativo del intervalo actual.

        Args:
            reward_component (dict): Metricas agregadas del intervalo
            extra_reward_component (dict): Series crudas por step

        Returns:
            tuple:
                - coordination_reward_total (float): total del modulo coordinativo
                - flat_record (dict)
                - coordination_module_rewards_by_var (dict)
        """
        task_potential = self._compute_task_potential(reward_component)
        task_progress_delta = self._compute_task_progress_delta(task_potential)
        task_progress_rewardable = self._compute_rewardable_progress(task_progress_delta)
        task_progress_01 = self._compute_task_progress_01(task_progress_rewardable)
        coordination_bonus_total = self._compute_coordination_bonus_total(
            task_progress_rewardable,
            task_progress_01
        )

        credit_raw = self._compute_credit_raw(extra_reward_component)
        credit_share = self._compute_credit_share(credit_raw)
        error_abs_by_var = self._compute_local_error_abs(extra_reward_component)
        error_decay_by_var = self._compute_error_decay(error_abs_by_var)

        coordination_global_rewards_by_var = self._compute_coordination_rewards(
            credit_share,
            coordination_bonus_total,
            error_decay_by_var
        )
        coordination_rewards_by_var_01 = self._compute_coordination_rewards_01(
            credit_share,
            task_progress_01,
            error_decay_by_var
        )
        local_potential_by_var = self._compute_local_potential(reward_component)
        local_progress_delta_by_var = self._compute_local_progress_delta(local_potential_by_var)
        local_progress_reward_by_var = self._compute_local_progress_rewards(
            local_progress_delta_by_var,
            task_progress_delta
        )
        debt_penalty_by_var = self._compute_debt_penalties(local_potential_by_var)
        coordination_module_rewards_by_var = self._compute_module_rewards(
            coordination_global_rewards_by_var,
            local_progress_reward_by_var,
            debt_penalty_by_var
        )

        coordination_reward_total = 0.0
        coordination_global_reward_total = 0.0
        coordination_reward_total_01 = 0.0
        coordination_progress_reward_total = 0.0
        coordination_debt_penalty_total = 0.0
        for var_obj in self.var_objs:
            coordination_reward_total += coordination_module_rewards_by_var[var_obj]
            coordination_global_reward_total += coordination_global_rewards_by_var[var_obj]
            coordination_reward_total_01 += coordination_rewards_by_var_01[var_obj]
            coordination_progress_reward_total += local_progress_reward_by_var[var_obj]
            coordination_debt_penalty_total += debt_penalty_by_var[var_obj]

        flat_record = {
            'coordination_reward_normalized_mode': float(self.normalized_reward_mode),
            'coordination_reward_bound': self._compute_coordination_reward_bound(),
            'coordination_potential_weight_sum_raw': self.potential_weight_sum_raw,
            'coordination_potential_weight_sum_effective': self.potential_weight_sum_effective,
            'coordination_bonus_clip_enabled': float(self.coordination_bonus_clip_enabled),
            'coordination_bonus_total': coordination_bonus_total,
            'coordination_global_reward_total': coordination_global_reward_total,
            'coordination_task_potential': task_potential,
            'coordination_task_progress_delta': task_progress_delta,
            'coordination_task_progress_positive': task_progress_rewardable,
            'coordination_task_progress_01': task_progress_01,
            'coordination_reward_total_01': coordination_reward_total_01,
            'coordination_reward_total': coordination_reward_total,
            'coordination_module_reward_total': coordination_reward_total,
            'coordination_progress_reward_total': coordination_progress_reward_total,
            'coordination_debt_penalty_total': coordination_debt_penalty_total
        }

        for var_obj in self.var_objs:
            flat_record[f'coordination_credit_raw_{var_obj}'] = credit_raw[var_obj]
            flat_record[f'coordination_credit_share_{var_obj}'] = credit_share[var_obj]
            flat_record[f'coordination_error_abs_{var_obj}'] = error_abs_by_var[var_obj]
            flat_record[f'coordination_error_decay_{var_obj}'] = error_decay_by_var[var_obj]
            flat_record[f'coordination_reward_01_{var_obj}'] = coordination_rewards_by_var_01[var_obj]
            flat_record[f'coordination_global_reward_{var_obj}'] = coordination_global_rewards_by_var[var_obj]
            flat_record[f'coordination_local_potential_{var_obj}'] = local_potential_by_var[var_obj]
            flat_record[f'coordination_local_progress_delta_{var_obj}'] = local_progress_delta_by_var[var_obj]
            flat_record[f'coordination_progress_reward_{var_obj}'] = local_progress_reward_by_var[var_obj]
            flat_record[f'coordination_debt_{var_obj}'] = self.local_debt_accumulated[var_obj]
            flat_record[f'coordination_debt_penalty_{var_obj}'] = debt_penalty_by_var[var_obj]
            flat_record[f'coordination_module_reward_{var_obj}'] = coordination_module_rewards_by_var[var_obj]
            flat_record[f'extra_conditional_coordination_bonus_{var_obj}'] = coordination_global_rewards_by_var[var_obj]

        self.previous_task_potential = task_potential
        self.previous_local_potential = local_potential_by_var.copy()
        self.last_coordination_reward_params_record = flat_record
        self.last_coordination_state_record = self._build_state_record()

        return coordination_reward_total, flat_record, coordination_module_rewards_by_var

    def _compute_task_potential(self, reward_component):
        """
        Calcula Phi(k) como suma ponderada de features globales por lazo.
        """
        task_potential = 0.0
        for flat_key, weight in self.potential_jobs:
            task_potential += weight * reward_component[flat_key]
        return task_potential

    def _compute_task_progress_delta(self, current_task_potential):
        """
        Calcula Delta Phi(k) = Phi(k-1) - Phi(k).
        """
        if self.previous_task_potential is None:
            return 0.0
        return self.previous_task_potential - current_task_potential

    def _compute_rewardable_progress(self, task_progress_delta):
        """
        Obtiene el progreso premiable del intervalo.
        """
        if self.progress_positive_only:
            progress_value = max(0.0, task_progress_delta)
        else:
            progress_value = task_progress_delta

        if self.coordination_bonus_clip_enabled:
            if progress_value > self.coordination_bonus_progress_max:
                progress_value = self.coordination_bonus_progress_max
            if self.progress_positive_only and progress_value < 0.0:
                progress_value = 0.0

        return progress_value

    def _compute_task_progress_01(self, task_progress_rewardable):
        """
        Normaliza el progreso coordinativo por la cota declarada del bloque.
        """
        if self.coordination_bonus_progress_max <= 0.0:
            return 0.0
        return min(1.0, max(0.0, task_progress_rewardable / self.coordination_bonus_progress_max))

    def _compute_coordination_bonus_total(self, task_progress_rewardable, task_progress_01):
        """
        Separa la magnitud total del bonus de su reparto por credito.
        """
        if self.normalized_reward_mode:
            return task_progress_01
        return self.global_weight * task_progress_rewardable

    def _compute_credit_raw(self, extra_reward_component):
        """
        Calcula contribucion correctiva media por lazo.
        """
        credit_raw = {}

        for var_obj, error_key, effort_key, correction_sign in self.contribution_jobs:
            error_series = extra_reward_component[error_key]
            effort_series = extra_reward_component[effort_key]
            n_steps = min(len(error_series), len(effort_series))

            raw_credit_sum = 0.0
            for step_idx in range(n_steps):
                corrective_contribution = correction_sign * error_series[step_idx] * effort_series[step_idx]
                raw_credit_sum += max(0.0, corrective_contribution)

            if self.credit_reduction_mode == 'average' and n_steps > 0:
                credit_raw[var_obj] = raw_credit_sum / n_steps
            else:
                credit_raw[var_obj] = raw_credit_sum

        return credit_raw

    def _compute_credit_share(self, credit_raw):
        """
        Calcula la fraccion de credito coordinativo por lazo.
        """
        credit_raw_total = 0.0
        for var_obj in self.var_objs:
            credit_raw_total += credit_raw[var_obj]

        denominator = self.epsilon_credit + credit_raw_total
        credit_share = {}
        for var_obj in self.var_objs:
            credit_share[var_obj] = credit_raw[var_obj] / denominator

        return credit_share

    def _compute_local_error_abs(self, extra_reward_component):
        """
        Extrae el error local actual usado para modular el bonus coordinativo.
        """
        error_abs_by_var = {}
        for var_obj in self.var_objs:
            error_key = self.local_error_decay_signal.format(var_obj=var_obj)
            error_series = extra_reward_component.get(error_key, [])
            error_value = self._reduce_series(error_series, self.local_error_decay_reduction)
            error_abs_by_var[var_obj] = abs(error_value)
        return error_abs_by_var

    def _compute_error_decay(self, error_abs_by_var):
        """
        Calcula d_v(e) para cada lazo.
        """
        decay_by_var = {}
        for var_obj in self.var_objs:
            if not self.local_error_decay_enabled:
                decay_by_var[var_obj] = 1.0
                continue

            threshold = self._resolve_var_value(self.local_error_thresholds, var_obj, 1.0)
            if threshold <= 0.0:
                decay_by_var[var_obj] = 1.0 if error_abs_by_var[var_obj] <= 0.0 else 0.0
                continue

            ratio = error_abs_by_var[var_obj] / threshold
            if self.local_error_decay_method == 'linear':
                factor = max(0.0, 1.0 - ratio)
            elif self.local_error_decay_method == 'exponential':
                lambda_v = self._resolve_var_value(self.local_error_lambdas, var_obj, 1.0)
                factor = math.exp(-lambda_v * (ratio ** 2))
            elif self.local_error_decay_method == 'hard':
                factor = 1.0 if ratio <= 1.0 else 0.0
            else:
                raise ValueError(f"Unsupported coordination error decay method: {self.local_error_decay_method}")

            if self.local_error_hard_zero and ratio > 1.0:
                factor = 0.0

            decay_by_var[var_obj] = min(1.0, max(0.0, factor))

        return decay_by_var

    def _compute_coordination_rewards(self, credit_share, coordination_bonus_total, error_decay_by_var):
        """
        Calcula el bonus coordinativo final por lazo.
        """
        coordination_rewards_by_var = {}
        for var_obj in self.var_objs:
            coordination_rewards_by_var[var_obj] = (
                credit_share[var_obj]
                * coordination_bonus_total
                * error_decay_by_var[var_obj]
            )
        return coordination_rewards_by_var

    def _compute_coordination_rewards_01(self, credit_share, task_progress_01, error_decay_by_var):
        """
        Calcula la version acotada [0, 1] del bonus coordinativo.
        """
        coordination_rewards_by_var = {}
        for var_obj in self.var_objs:
            coordination_rewards_by_var[var_obj] = (
                credit_share[var_obj]
                * task_progress_01
                * error_decay_by_var[var_obj]
            )
        return coordination_rewards_by_var

    def _compute_local_potential(self, reward_component):
        """
        Calcula Phi_local_v(k) para progreso local y deuda.
        """
        local_potential_by_var = {}
        for var_obj in self.var_objs:
            potential = 0.0
            for flat_key, weight in self.local_potential_jobs[var_obj]:
                potential += weight * reward_component.get(flat_key, 0.0)
            local_potential_by_var[var_obj] = potential
        return local_potential_by_var

    def _compute_local_progress_delta(self, local_potential_by_var):
        """
        Calcula Delta Phi_local_v(k) = Phi_local_v(k-1) - Phi_local_v(k).
        """
        progress_delta_by_var = {}
        for var_obj in self.var_objs:
            previous = self.previous_local_potential[var_obj]
            if previous is None:
                progress_delta_by_var[var_obj] = 0.0
            else:
                progress_delta_by_var[var_obj] = previous - local_potential_by_var[var_obj]
        return progress_delta_by_var

    def _compute_local_progress_rewards(self, local_progress_delta_by_var, task_progress_delta):
        """
        Calcula R_progress_v, opcionalmente condicionado por mejora global.
        """
        condition_factor_by_var = self._compute_local_progress_condition(
            local_progress_delta_by_var,
            task_progress_delta
        )
        progress_reward_by_var = {}
        for var_obj in self.var_objs:
            if not self.local_progress_enabled:
                progress_reward_by_var[var_obj] = 0.0
                continue

            progress_value = local_progress_delta_by_var[var_obj]
            if self.local_progress_positive_only:
                progress_value = max(0.0, progress_value)

            if self.local_progress_clip_enabled and progress_value > self.local_progress_clip_max:
                progress_value = self.local_progress_clip_max

            weight = self._resolve_var_value(self.local_progress_weight, var_obj, 1.0)
            progress_reward_by_var[var_obj] = (
                weight
                * progress_value
                * condition_factor_by_var[var_obj]
            )
        return progress_reward_by_var

    def _compute_local_progress_condition(self, local_progress_delta_by_var, task_progress_delta):
        """
        Resuelve la condicion declarativa del progreso local.
        """
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

    def _compute_debt_penalties(self, local_potential_by_var):
        """
        Actualiza D_v(k) y calcula R_debt_v.
        """
        debt_penalty_by_var = {}
        for var_obj in self.var_objs:
            if self.local_debt_enabled:
                debt_value = (
                    self.local_debt_persistence * self.local_debt_accumulated[var_obj]
                    + local_potential_by_var[var_obj]
                )
                if self.local_debt_clip_enabled and debt_value > self.local_debt_clip_max:
                    debt_value = self.local_debt_clip_max
                self.local_debt_accumulated[var_obj] = max(0.0, debt_value)
                eta = self._resolve_var_value(self.local_debt_penalty_weight, var_obj, 0.0)
                debt_penalty_by_var[var_obj] = eta * self.local_debt_accumulated[var_obj]
            else:
                self.local_debt_accumulated[var_obj] = 0.0
                debt_penalty_by_var[var_obj] = 0.0
        return debt_penalty_by_var

    def _compute_module_rewards(self, coordination_rewards_by_var, progress_rewards_by_var, debt_penalty_by_var):
        """
        Calcula R_coord_module_v = R_coord_v + R_progress_v - R_debt_v.
        """
        module_rewards_by_var = {}
        for var_obj in self.var_objs:
            module_rewards_by_var[var_obj] = (
                coordination_rewards_by_var[var_obj]
                + progress_rewards_by_var[var_obj]
                - debt_penalty_by_var[var_obj]
            )
        return module_rewards_by_var

    def _compute_coordination_reward_bound(self):
        """
        Cota analitica usada para mapear el bloque coordinativo.
        """
        if self.normalized_reward_mode:
            return 1.0
        if self.coordination_bonus_clip_enabled:
            return max(0.0, self.global_weight * self.coordination_bonus_progress_max)
        return 0.0

    def _resolve_var_value(self, value_config, var_obj, default):
        """
        Resuelve parametros escalares o dict por variable.
        """
        if isinstance(value_config, dict):
            return value_config.get(var_obj, default)
        if value_config is None:
            return default
        return value_config

    def _reduce_series(self, values, reduction_mode):
        """
        Reduce una serie cruda a un escalar intervalar.
        """
        if not values:
            return 0.0

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
        """
        Expone variables internas observables para el estado del agente.
        """
        state_record = {}
        for var_obj in self.var_objs:
            state_record[f'coordination_debt_{var_obj}'] = self.local_debt_accumulated[var_obj]
        return state_record

    def get_coordination_reward_params_record(self):
        """
        Retorna el ultimo registro plano del bloque coordinativo.
        """
        return self.last_coordination_reward_params_record.copy()

    def get_state_record(self):
        """
        Retorna variables internas aptas para discretizacion declarativa.
        """
        return self.last_coordination_state_record.copy()
