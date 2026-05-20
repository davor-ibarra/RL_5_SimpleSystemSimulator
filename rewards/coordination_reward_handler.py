"""
coordination_reward_handler.py

Responsabilidad:
Calcular bonus coordinativo por progreso global de tarea.

Implementa:
- potencial global Phi(k)
- progreso Delta Phi(k)
- reparto de credito por contribucion correctiva
- bonus coordinativo por lazo
"""


class CoordinationRewardHandler:
    """
    Manejador del bonus coordinativo por progreso global.

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
        self.contribution_error_signal = config['contribution_error_signal']
        self.contribution_effort_signal = config['contribution_effort_signal']
        self.correction_signs = config['correction_signs']
        self.epsilon_credit = config['epsilon_credit']
        self.credit_reduction_mode = config['credit_reduction_mode']

        self.var_objs = list(self.correction_signs.keys())
        self._compile_jobs()
        self.reset_episode()

    def _compile_jobs(self):
        """
        Precompila llaves y pesos para evitar armado dinamico en runtime.
        """
        self.potential_jobs = []
        for feature_name in self.potential_features:
            feature_weights = self.potential_weights[feature_name]
            for var_obj in self.var_objs:
                flat_key = f'{feature_name}_{var_obj}'
                weight = feature_weights[var_obj]
                self.potential_jobs.append((flat_key, weight))

        self.contribution_jobs = []
        for var_obj in self.var_objs:
            error_key = self.contribution_error_signal.format(var_obj=var_obj)
            effort_key = self.contribution_effort_signal.format(var_obj=var_obj)
            correction_sign = self.correction_signs[var_obj]
            self.contribution_jobs.append((var_obj, error_key, effort_key, correction_sign))

    def reset_episode(self):
        """
        Resetea la memoria del bloque al inicio de cada episodio.
        """
        self.previous_task_potential = None
        self.last_coordination_reward_params_record = {}

    def evaluate(self, reward_component, extra_reward_component):
        """
        Evalua el bonus coordinativo del intervalo actual.

        Args:
            reward_component (dict): Metricas agregadas del intervalo
            extra_reward_component (dict): Series crudas por step

        Returns:
            tuple:
                - coordination_reward_total (float)
                - flat_record (dict)
                - coordination_rewards_by_var (dict)
        """
        task_potential = self._compute_task_potential(reward_component)
        task_progress_delta = self._compute_task_progress_delta(task_potential)
        task_progress_rewardable = self._compute_rewardable_progress(task_progress_delta)
        task_progress_01 = self._compute_task_progress_01(task_progress_rewardable)

        credit_raw = self._compute_credit_raw(extra_reward_component)
        credit_share = self._compute_credit_share(credit_raw)
        coordination_rewards_by_var = self._compute_coordination_rewards(
            credit_share,
            task_progress_rewardable,
            task_progress_01
        )
        coordination_rewards_by_var_01 = self._compute_coordination_rewards_01(
            credit_share,
            task_progress_01
        )

        coordination_reward_total = 0.0
        coordination_reward_total_01 = 0.0
        for var_obj in self.var_objs:
            coordination_reward_total += coordination_rewards_by_var[var_obj]
            coordination_reward_total_01 += coordination_rewards_by_var_01[var_obj]

        flat_record = {
            'coordination_reward_normalized_mode': float(self.normalized_reward_mode),
            'coordination_reward_bound': self._compute_coordination_reward_bound(),
            'coordination_task_potential': task_potential,
            'coordination_task_progress_delta': task_progress_delta,
            'coordination_task_progress_positive': task_progress_rewardable,
            'coordination_task_progress_01': task_progress_01,
            'coordination_reward_total_01': coordination_reward_total_01,
            'coordination_reward_total': coordination_reward_total
        }

        for var_obj in self.var_objs:
            flat_record[f'coordination_credit_raw_{var_obj}'] = credit_raw[var_obj]
            flat_record[f'coordination_credit_share_{var_obj}'] = credit_share[var_obj]
            flat_record[f'coordination_reward_01_{var_obj}'] = coordination_rewards_by_var_01[var_obj]
            flat_record[f'extra_conditional_coordination_bonus_{var_obj}'] = coordination_rewards_by_var[var_obj]

        self.previous_task_potential = task_potential
        self.last_coordination_reward_params_record = flat_record

        return coordination_reward_total, flat_record, coordination_rewards_by_var

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

        if progress_value > self.progress_clip_max:
            progress_value = self.progress_clip_max

        return progress_value

    def _compute_task_progress_01(self, task_progress_rewardable):
        """
        Normaliza el progreso coordinativo por la cota declarada del bloque.
        """
        if self.progress_clip_max <= 0.0:
            return 0.0
        return min(1.0, max(0.0, task_progress_rewardable / self.progress_clip_max))

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

    def _compute_coordination_rewards(self, credit_share, task_progress_rewardable, task_progress_01):
        """
        Calcula el bonus coordinativo final por lazo.
        """
        if self.normalized_reward_mode:
            return self._compute_coordination_rewards_01(credit_share, task_progress_01)

        coordination_rewards_by_var = {}
        for var_obj in self.var_objs:
            coordination_rewards_by_var[var_obj] = (
                self.global_weight
                * credit_share[var_obj]
                * task_progress_rewardable
            )
        return coordination_rewards_by_var

    def _compute_coordination_rewards_01(self, credit_share, task_progress_01):
        """
        Calcula la version acotada [0, 1] del bonus coordinativo.
        """
        coordination_rewards_by_var = {}
        for var_obj in self.var_objs:
            coordination_rewards_by_var[var_obj] = credit_share[var_obj] * task_progress_01
        return coordination_rewards_by_var

    def _compute_coordination_reward_bound(self):
        """
        Cota analitica legacy usada para mapear el bloque a [0, 1].
        """
        return max(0.0, self.global_weight * self.progress_clip_max)

    def get_coordination_reward_params_record(self):
        """
        Retorna el ultimo registro plano del bloque coordinativo.
        """
        return self.last_coordination_reward_params_record.copy()
