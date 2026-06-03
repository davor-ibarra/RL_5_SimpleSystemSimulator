"""
reward_composer.py

Responsabilidad:
Componer componentes firmados homogeneos y aplicar curriculum por regimen
cooperativo cuando esta declarado.
"""


class RewardComposer:
    """
    Compositor final en dominio signed_unit.
    """

    def __init__(self, config_main, var_objs):
        self.config_main = config_main
        self.var_objs = var_objs
        self.config = config_main['reward_base']['reward_config']['reward_composition']
        self.mode = self.config['mode']
        self.reward_domain = self.config['reward_domain']
        if self.reward_domain != 'signed_unit':
            raise ValueError(f"Unsupported reward_domain: {self.reward_domain}")

        self.components_config = self.config['components']
        self.component_names = [
            'local_quality',
            'local_marginal_quality',
            'cooperative_quality',
            'cooperative_marginal_quality',
            'coupling_penalty',
            'cooperative_regime'
        ]
        self.weights = self._compile_component_weights()
        self.regime_config = self.config['cooperative_regime']
        self.current_betas = {
            'syn': self.regime_config['beta_schedule']['beta_syn']['min'],
            'coop': self.regime_config['beta_schedule']['beta_coop']['max'],
            'self': self.regime_config['beta_schedule']['beta_self']['min'],
            'fail': self.regime_config['beta_schedule']['beta_fail']['min']
        }
        self.previous_betas = None
        self.episode_index = -1
        self.current_learning_rate = self.config_main['agent_base']['params']['learning_rate']
        self.current_maturity = 0.0
        self.curriculum_frozen = False
        self.last_record = {}

    def _compile_component_weights(self):
        weights = {}
        weight_sum = 0.0
        for component_name in self.component_names:
            weight = self.components_config[component_name]['weight']
            if weight < 0.0:
                raise ValueError(f"Reward component weight must be non-negative: {component_name}={weight}")
            weights[component_name] = weight
            weight_sum += weight

        if self.config['strict_weight_sum'] and abs(weight_sum - 1.0) > self.config['weight_tolerance']:
            raise ValueError(f"Reward component weights must sum 1.0, got {weight_sum}")
        if weight_sum <= 0.0:
            raise ValueError("Reward component weights must be positive")

        if self.config['strict_weight_sum']:
            return weights
        return {
            component_name: weights[component_name] / weight_sum
            for component_name in self.component_names
        }

    def reset_episode(self):
        self.episode_index += 1
        self._update_curriculum_state()
        self.last_record = {}

    def compose(self, local_components, cooperative_components, coupling_components):
        records = {
            'reward_composition_mode': self.mode,
            'reward_domain': self.reward_domain,
            'reward_component_weight_sum': sum(self.weights.values()),
            'reward_curriculum_episode_index': self.episode_index,
            'reward_curriculum_learning_rate': self.current_learning_rate,
            'reward_curriculum_maturity': self.current_maturity,
            'reward_curriculum_frozen': float(self.curriculum_frozen),
            'reward_beta_syn': self.current_betas['syn'],
            'reward_beta_coop': self.current_betas['coop'],
            'reward_beta_self': self.current_betas['self'],
            'reward_beta_fail': self.current_betas['fail']
        }
        rewards_by_var = {}

        for var_obj in self.var_objs:
            component_values = self._build_component_values(
                var_obj,
                local_components,
                cooperative_components,
                coupling_components
            )
            reward_signed = 0.0
            for component_name in self.component_names:
                reward_signed += self.weights[component_name] * component_values[component_name]
            reward_signed = self._clip_signed(reward_signed)
            reward_score = 0.5 * (reward_signed + 1.0)
            rewards_by_var[var_obj] = reward_signed

            records[f'reward_component_local_quality_signed_{var_obj}'] = component_values['local_quality']
            records[f'reward_component_local_marginal_quality_signed_{var_obj}'] = component_values['local_marginal_quality']
            records[f'reward_component_cooperative_quality_signed_{var_obj}'] = component_values['cooperative_quality']
            records[f'reward_component_cooperative_marginal_quality_signed_{var_obj}'] = component_values['cooperative_marginal_quality']
            records[f'reward_component_coupling_penalty_signed_{var_obj}'] = component_values['coupling_penalty']
            records[f'reward_component_cooperative_regime_signed_{var_obj}'] = component_values['cooperative_regime']

            for component_name in self.component_names:
                records[f'reward_weight_{component_name}_{var_obj}'] = self.weights[component_name]

            records[f'reward_signed_{var_obj}'] = reward_signed
            records[f'reward_score_01_{var_obj}'] = reward_score

        self.last_record = records
        return records, rewards_by_var

    def _build_component_values(
        self,
        var_obj,
        local_components,
        cooperative_components,
        coupling_components
    ):
        regime_values = self._compute_regime_values(
            local_components[var_obj]['local_cost_01'],
            cooperative_components[var_obj]['cooperative_cost_01']
        )
        return {
            'local_quality': local_components[var_obj]['local_quality_signed'],
            'local_marginal_quality': local_components[var_obj]['local_marginal_quality_signed'],
            'cooperative_quality': cooperative_components[var_obj]['cooperative_quality_signed'],
            'cooperative_marginal_quality': cooperative_components[var_obj]['cooperative_marginal_quality_signed'],
            'coupling_penalty': coupling_components[var_obj]['coupling_penalty_signed'],
            'cooperative_regime': regime_values['reward_signed']
        }

    def _compute_regime_values(self, local_cost, cooperative_cost):
        phi_syn = (1.0 - local_cost) * (1.0 - cooperative_cost)
        phi_coop = local_cost * (1.0 - cooperative_cost)
        phi_self = (1.0 - local_cost) * cooperative_cost
        phi_fail = local_cost * cooperative_cost
        reward_signed = self._clip_signed(
            self.current_betas['syn'] * phi_syn
            + self.current_betas['coop'] * phi_coop
            - self.current_betas['self'] * phi_self
            - self.current_betas['fail'] * phi_fail
        )
        return {
            'phi_syn': phi_syn,
            'phi_coop': phi_coop,
            'phi_self': phi_self,
            'phi_fail': phi_fail,
            'reward_signed': reward_signed
        }

    def build_regime_record(self, local_components, cooperative_components):
        records = {}
        for var_obj in self.var_objs:
            regime_values = self._compute_regime_values(
                local_components[var_obj]['local_cost_01'],
                cooperative_components[var_obj]['cooperative_cost_01']
            )
            records[f'regime_phi_syn_{var_obj}'] = regime_values['phi_syn']
            records[f'regime_phi_coop_{var_obj}'] = regime_values['phi_coop']
            records[f'regime_phi_self_{var_obj}'] = regime_values['phi_self']
            records[f'regime_phi_fail_{var_obj}'] = regime_values['phi_fail']
            records[f'regime_reward_signed_{var_obj}'] = regime_values['reward_signed']
            records[f'reward_beta_syn_{var_obj}'] = self.current_betas['syn']
            records[f'reward_beta_coop_{var_obj}'] = self.current_betas['coop']
            records[f'reward_beta_self_{var_obj}'] = self.current_betas['self']
            records[f'reward_beta_fail_{var_obj}'] = self.current_betas['fail']
        return records

    def _update_curriculum_state(self):
        if self.mode == 'composition':
            self.current_learning_rate = self._compute_learning_rate()
            self.current_maturity = 0.0
            self.curriculum_frozen = False
            return

        if self.mode != 'curriculum_regime':
            raise ValueError(f"Unsupported reward composition mode: {self.mode}")

        self.current_learning_rate = self._compute_learning_rate()
        self.current_maturity = self._compute_maturity(self.current_learning_rate)
        freeze_threshold = self.regime_config['beta_schedule']['alpha_freeze']

        if self.previous_betas is not None and self.current_learning_rate <= freeze_threshold:
            self.curriculum_frozen = True
            self.current_betas = self.previous_betas.copy()
            return

        self.curriculum_frozen = False
        next_betas = self._compute_curriculum_betas(self.current_maturity)
        self._validate_curriculum_transition(next_betas)
        self.current_betas = next_betas
        self.previous_betas = self.current_betas.copy()

    def _compute_learning_rate(self):
        agent_params = self.config_main['agent_base']['params']
        alpha_0 = agent_params['learning_rate']
        decay_config = agent_params['learning_rate_decay']
        if not decay_config['enabled']:
            return alpha_0
        decayed = alpha_0 * (decay_config['decay_factor'] ** self.episode_index)
        return max(decay_config['learning_rate_min'], decayed)

    def _compute_maturity(self, learning_rate):
        agent_params = self.config_main['agent_base']['params']
        alpha_0 = agent_params['learning_rate']
        alpha_min = agent_params['learning_rate_decay']['learning_rate_min']
        denominator = alpha_0 - alpha_min
        if denominator <= 0.0:
            raise ValueError("learning_rate must be greater than learning_rate_min for curriculum_regime")
        return self._clip_01(1.0 - ((learning_rate - alpha_min) / denominator))

    def _compute_curriculum_betas(self, maturity):
        schedule = self.regime_config['beta_schedule']
        beta_syn = self._increasing_schedule(schedule['beta_syn'], maturity)
        beta_coop = self._decreasing_schedule(schedule['beta_coop'], maturity)
        beta_self = self._increasing_schedule(schedule['beta_self'], maturity)
        beta_fail = self._increasing_schedule(schedule['beta_fail'], maturity)
        if beta_syn <= beta_coop:
            raise ValueError(
                f"curriculum_regime requires beta_syn > beta_coop, got {beta_syn} <= {beta_coop}"
            )
        return {
            'syn': beta_syn,
            'coop': beta_coop,
            'self': beta_self,
            'fail': beta_fail
        }

    def _increasing_schedule(self, limits, maturity):
        return limits['min'] + (limits['max'] - limits['min']) * maturity

    def _decreasing_schedule(self, limits, maturity):
        return limits['max'] - (limits['max'] - limits['min']) * maturity

    def _validate_curriculum_transition(self, next_betas):
        if self.previous_betas is None:
            return
        if not self.regime_config['beta_schedule']['enforce_eta_beta_lt_alpha']:
            return

        max_delta = 0.0
        for beta_name in next_betas:
            delta = abs(next_betas[beta_name] - self.previous_betas[beta_name])
            max_delta = max(max_delta, delta)
        if max_delta >= self.current_learning_rate:
            raise ValueError(
                f"Curriculum beta step must be lower than learning_rate: "
                f"eta_beta={max_delta}, alpha={self.current_learning_rate}"
            )

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))

    def _clip_signed(self, value):
        return min(1.0, max(-1.0, value))

    def get_state_record(self):
        return {
            'reward_curriculum_learning_rate': self.current_learning_rate,
            'reward_curriculum_maturity': self.current_maturity,
            'reward_curriculum_frozen': float(self.curriculum_frozen),
            'reward_beta_syn': self.current_betas['syn'],
            'reward_beta_coop': self.current_betas['coop'],
            'reward_beta_self': self.current_betas['self'],
            'reward_beta_fail': self.current_betas['fail']
        }
