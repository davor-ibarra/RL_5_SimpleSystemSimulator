"""
reward_composer.py

Responsabilidad:
Componer la recompensa estructurada A/C/S para el destinatario final y,
opcionalmente, aplicar el curriculum por regimen cooperativo acoplado.
"""


class RewardComposer:
    """
    Compositor final en dominio signed_unit.

    La formulacion base es:
    r = lambda_A R_A + lambda_C R_C + lambda_S R_S

    La formulacion curricular es:
    r_tilde = (1 - lambda_Q) r + lambda_Q R_Q

    Con techo estrategico habilitado:
    r_final = min(r_tilde, 1 - 2 C_strategy)
    """

    def __init__(self, config_main, var_objs):
        self.config_main = config_main
        self.var_objs = var_objs
        self.config = config_main['reward_base']['reward_config']['reward_composition']
        self.mode = self.config['mode']
        self.reward_domain = self.config['reward_domain']
        if self.reward_domain != 'signed_unit':
            raise ValueError(f"Unsupported reward_domain: {self.reward_domain}")

        self.structured_weights = self.config['structured_weights']
        self.mixing = self.config['mixing']
        self.curriculum_config = self.config['curriculum']
        self.risk_ceiling_config = self.config['risk_ceiling']
        self.curriculum_enabled = self.mode == 'curriculum_regime'

        self._validate_structured_weights()

        initial_betas = self._compute_curriculum_betas(0.0)
        self.current_betas = initial_betas.copy()
        self.previous_betas = None
        self.previous_lambda_curriculum = None
        self.previous_maturity = None

        self.episode_index = -1
        self.current_learning_rate = self.config_main['agent_base']['params']['learning_rate']
        self.current_maturity = 0.0
        self.current_lambda_curriculum = 0.0
        self.curriculum_frozen = False
        self.last_record = {}

    def _validate_structured_weights(self):
        lambda_sum = (
            self.structured_weights['autonomy']
            + self.structured_weights['cooperation']
            + self.structured_weights['strategy']
        )
        if self.config['strict_weight_sum'] and abs(lambda_sum - 1.0) > self.config['weight_tolerance']:
            raise ValueError(f"Structured reward weights must sum 1.0, got {lambda_sum}")
        if lambda_sum <= 0.0:
            raise ValueError("Structured reward weights must be positive")

    def reset_episode(self):
        self.episode_index += 1
        self._update_curriculum_state()
        self.last_record = {}

    def compose(self, local_components, cooperative_components, strategy_components):
        return self.compose_controller(local_components, cooperative_components, strategy_components)

    def compose_controller(self, local_components, cooperative_components, strategy_components):
        records = {
            'reward_curriculum_episode_index': self.episode_index,
            'reward_curriculum_maturity': self.current_maturity,
            'reward_lambda_curriculum': self.current_lambda_curriculum,
            'reward_curriculum_frozen': float(self.curriculum_frozen),
            'reward_beta_syn': self.current_betas['syn'],
            'reward_beta_coop': self.current_betas['coop'],
            'reward_beta_self': self.current_betas['self'],
            'reward_beta_fail': self.current_betas['fail']
        }
        rewards_by_var = {}
        regime_values_by_var = {}
        base_interval_sum = 0.0
        curriculum_interval_sum = 0.0
        pre_ceiling_interval_sum = 0.0
        risk_ceiling_interval_sum = 0.0
        ceiling_active_interval_sum = 0.0

        for var_obj in self.var_objs:
            autonomy_signed = self._compute_autonomy_reward(var_obj, local_components)
            cooperation_signed = self._compute_cooperation_reward(var_obj, cooperative_components)
            strategy_signed = strategy_components[var_obj]['strategy_reward_signed']
            strategy_risk_cost = strategy_components[var_obj]['strategy_risk_cost_01']
            strategy_decision_penalty_signed = strategy_components[var_obj][
                'strategy_decision_penalty_signed'
            ]

            autonomy_cost = self._clip_01((1.0 - autonomy_signed) / 2.0)
            cooperation_cost = self._clip_01((1.0 - cooperation_signed) / 2.0)
            regime_values = self._compute_regime_values(autonomy_cost, cooperation_cost)

            base_reward_signed = self._clip_signed(
                self.structured_weights['autonomy'] * autonomy_signed
                + self.structured_weights['cooperation'] * cooperation_signed
                + self.structured_weights['strategy'] * strategy_signed
            )
            curriculum_reward_signed = regime_values['reward_signed']
            pre_ceiling_reward_signed = self._clip_signed(
                (1.0 - self.current_lambda_curriculum) * base_reward_signed
                + self.current_lambda_curriculum * curriculum_reward_signed
            )
            (
                final_reward_signed,
                risk_ceiling_signed,
                ceiling_active
            ) = self._apply_risk_ceiling(pre_ceiling_reward_signed, strategy_risk_cost)
            reward_score = 0.5 * (final_reward_signed + 1.0)

            rewards_by_var[var_obj] = final_reward_signed
            regime_values_by_var[var_obj] = regime_values
            base_interval_sum += base_reward_signed
            curriculum_interval_sum += curriculum_reward_signed
            pre_ceiling_interval_sum += pre_ceiling_reward_signed
            risk_ceiling_interval_sum += risk_ceiling_signed
            ceiling_active_interval_sum += ceiling_active

            records[f'reward_autonomy_signed_{var_obj}'] = autonomy_signed
            records[f'reward_cooperation_signed_{var_obj}'] = cooperation_signed
            records[f'reward_strategy_signed_{var_obj}'] = strategy_signed
            records[f'reward_strategy_cost_01_{var_obj}'] = strategy_risk_cost
            records[f'reward_strategy_decision_penalty_signed_{var_obj}'] = (
                strategy_decision_penalty_signed
            )
            records[f'reward_autonomy_cost_01_{var_obj}'] = autonomy_cost
            records[f'reward_cooperation_cost_01_{var_obj}'] = cooperation_cost
            records[f'regime_phi_syn_{var_obj}'] = regime_values['phi_syn']
            records[f'regime_phi_coop_{var_obj}'] = regime_values['phi_coop']
            records[f'regime_phi_self_{var_obj}'] = regime_values['phi_self']
            records[f'regime_phi_fail_{var_obj}'] = regime_values['phi_fail']
            records[f'regime_reward_signed_{var_obj}'] = curriculum_reward_signed
            records[f'reward_base_signed_{var_obj}'] = base_reward_signed
            records[f'reward_curriculum_signed_{var_obj}'] = curriculum_reward_signed
            records[f'reward_pre_ceiling_signed_{var_obj}'] = pre_ceiling_reward_signed
            records[f'reward_risk_ceiling_signed_{var_obj}'] = risk_ceiling_signed
            records[f'reward_ceiling_active_{var_obj}'] = ceiling_active
            records[f'reward_signed_{var_obj}'] = final_reward_signed
            records[f'reward_score_01_{var_obj}'] = reward_score

        records['reward_base_interval_signed'] = base_interval_sum / len(self.var_objs)
        records['reward_curriculum_interval_signed'] = curriculum_interval_sum / len(self.var_objs)
        records['reward_pre_ceiling_interval_signed'] = pre_ceiling_interval_sum / len(self.var_objs)
        records['reward_risk_ceiling_interval_signed'] = risk_ceiling_interval_sum / len(self.var_objs)
        records['reward_ceiling_active_interval'] = ceiling_active_interval_sum / len(self.var_objs)

        self.last_record = records
        return records, rewards_by_var, regime_values_by_var

    def compose_agent(
        self,
        local_components,
        agent_cooperative_components,
        agent_strategy_components,
        agent_to_var_obj_map,
        var_obj_to_agent_names
    ):
        records = {
            'reward_curriculum_episode_index': self.episode_index,
            'reward_curriculum_maturity': self.current_maturity,
            'reward_lambda_curriculum': self.current_lambda_curriculum,
            'reward_curriculum_frozen': float(self.curriculum_frozen),
            'reward_beta_syn': self.current_betas['syn'],
            'reward_beta_coop': self.current_betas['coop'],
            'reward_beta_self': self.current_betas['self'],
            'reward_beta_fail': self.current_betas['fail']
        }
        rewards_by_agent = {}
        regime_values_by_var = {}
        aggregate_by_var = self._build_var_aggregates()
        base_interval_sum = 0.0
        curriculum_interval_sum = 0.0
        pre_ceiling_interval_sum = 0.0
        risk_ceiling_interval_sum = 0.0
        ceiling_active_interval_sum = 0.0

        for agent_name, var_obj in agent_to_var_obj_map.items():
            autonomy_signed = self._compute_autonomy_reward(var_obj, local_components)
            cooperation_signed = self._compute_cooperation_from_component(
                agent_cooperative_components[agent_name]
            )
            strategy_signed = agent_strategy_components[agent_name]['strategy_reward_signed']
            strategy_risk_cost = agent_strategy_components[agent_name]['strategy_risk_cost_01']
            strategy_decision_penalty_signed = agent_strategy_components[agent_name][
                'strategy_decision_penalty_signed'
            ]

            autonomy_cost = self._clip_01((1.0 - autonomy_signed) / 2.0)
            cooperation_cost = self._clip_01((1.0 - cooperation_signed) / 2.0)
            regime_values = self._compute_regime_values(autonomy_cost, cooperation_cost)

            base_reward_signed = self._clip_signed(
                self.structured_weights['autonomy'] * autonomy_signed
                + self.structured_weights['cooperation'] * cooperation_signed
                + self.structured_weights['strategy'] * strategy_signed
            )
            curriculum_reward_signed = regime_values['reward_signed']
            pre_ceiling_reward_signed = self._clip_signed(
                (1.0 - self.current_lambda_curriculum) * base_reward_signed
                + self.current_lambda_curriculum * curriculum_reward_signed
            )
            (
                final_reward_signed,
                risk_ceiling_signed,
                ceiling_active
            ) = self._apply_risk_ceiling(pre_ceiling_reward_signed, strategy_risk_cost)
            final_reward_signed = self._clip_signed(
                final_reward_signed + strategy_decision_penalty_signed
            )
            reward_score = 0.5 * (final_reward_signed + 1.0)

            rewards_by_agent[agent_name] = final_reward_signed
            base_interval_sum += base_reward_signed
            curriculum_interval_sum += curriculum_reward_signed
            pre_ceiling_interval_sum += pre_ceiling_reward_signed
            risk_ceiling_interval_sum += risk_ceiling_signed
            ceiling_active_interval_sum += ceiling_active
            self._accumulate_var_aggregate(
                aggregate_by_var[var_obj],
                autonomy_signed,
                cooperation_signed,
                strategy_signed,
                strategy_risk_cost,
                strategy_decision_penalty_signed,
                autonomy_cost,
                cooperation_cost,
                regime_values,
                base_reward_signed,
                curriculum_reward_signed,
                pre_ceiling_reward_signed,
                risk_ceiling_signed,
                ceiling_active,
                final_reward_signed,
                reward_score
            )

            records[f'reward_agent_autonomy_signed_{agent_name}'] = autonomy_signed
            records[f'reward_agent_cooperation_signed_{agent_name}'] = cooperation_signed
            records[f'reward_agent_strategy_signed_{agent_name}'] = strategy_signed
            records[f'reward_agent_strategy_cost_01_{agent_name}'] = strategy_risk_cost
            records[f'reward_agent_strategy_decision_penalty_signed_{agent_name}'] = (
                strategy_decision_penalty_signed
            )
            records[f'reward_agent_base_signed_{agent_name}'] = base_reward_signed
            records[f'reward_agent_curriculum_signed_{agent_name}'] = curriculum_reward_signed
            records[f'reward_agent_pre_ceiling_signed_{agent_name}'] = pre_ceiling_reward_signed
            records[f'reward_agent_risk_ceiling_signed_{agent_name}'] = risk_ceiling_signed
            records[f'reward_agent_ceiling_active_{agent_name}'] = ceiling_active
            records[f'reward_agent_signed_{agent_name}'] = final_reward_signed

        interval_denominator = len(agent_to_var_obj_map)
        records['reward_base_interval_signed'] = base_interval_sum / interval_denominator
        records['reward_curriculum_interval_signed'] = curriculum_interval_sum / interval_denominator
        records['reward_pre_ceiling_interval_signed'] = pre_ceiling_interval_sum / interval_denominator
        records['reward_risk_ceiling_interval_signed'] = risk_ceiling_interval_sum / interval_denominator
        records['reward_ceiling_active_interval'] = ceiling_active_interval_sum / interval_denominator

        self._write_var_aggregate_records(records, regime_values_by_var, aggregate_by_var, var_obj_to_agent_names)
        self.last_record = records
        return records, rewards_by_agent, regime_values_by_var

    def compose_common(self, local_components, common_cooperative_component, strategy_components):
        records = {
            'reward_curriculum_episode_index': self.episode_index,
            'reward_curriculum_maturity': self.current_maturity,
            'reward_lambda_curriculum': self.current_lambda_curriculum,
            'reward_curriculum_frozen': float(self.curriculum_frozen),
            'reward_beta_syn': self.current_betas['syn'],
            'reward_beta_coop': self.current_betas['coop'],
            'reward_beta_self': self.current_betas['self'],
            'reward_beta_fail': self.current_betas['fail']
        }

        autonomy_signed = self._compute_common_autonomy_reward(local_components)
        cooperation_signed = self._compute_cooperation_from_component(common_cooperative_component)
        strategy_risk_cost = self._compute_common_strategy_risk(strategy_components)
        strategy_signed = -strategy_risk_cost

        autonomy_cost = self._clip_01((1.0 - autonomy_signed) / 2.0)
        cooperation_cost = self._clip_01((1.0 - cooperation_signed) / 2.0)
        regime_values = self._compute_regime_values(autonomy_cost, cooperation_cost)

        base_reward_signed = self._clip_signed(
            self.structured_weights['autonomy'] * autonomy_signed
            + self.structured_weights['cooperation'] * cooperation_signed
            + self.structured_weights['strategy'] * strategy_signed
        )
        curriculum_reward_signed = regime_values['reward_signed']
        pre_ceiling_reward_signed = self._clip_signed(
            (1.0 - self.current_lambda_curriculum) * base_reward_signed
            + self.current_lambda_curriculum * curriculum_reward_signed
        )
        (
            final_reward_signed,
            risk_ceiling_signed,
            ceiling_active
        ) = self._apply_risk_ceiling(pre_ceiling_reward_signed, strategy_risk_cost)
        reward_score = 0.5 * (final_reward_signed + 1.0)

        records['reward_common_autonomy_signed'] = autonomy_signed
        records['reward_common_cooperation_signed'] = cooperation_signed
        records['reward_common_strategy_signed'] = strategy_signed
        records['reward_common_strategy_cost_01'] = strategy_risk_cost
        records['reward_common_base_signed'] = base_reward_signed
        records['reward_common_curriculum_signed'] = curriculum_reward_signed
        records['reward_common_pre_ceiling_signed'] = pre_ceiling_reward_signed
        records['reward_common_risk_ceiling_signed'] = risk_ceiling_signed
        records['reward_common_ceiling_active'] = ceiling_active
        records['reward_common_signed'] = final_reward_signed
        records['reward_common_score_01'] = reward_score
        records['reward_base_interval_signed'] = base_reward_signed
        records['reward_curriculum_interval_signed'] = curriculum_reward_signed
        records['reward_pre_ceiling_interval_signed'] = pre_ceiling_reward_signed
        records['reward_risk_ceiling_interval_signed'] = risk_ceiling_signed
        records['reward_ceiling_active_interval'] = ceiling_active

        regime_values_by_var = {}
        for var_obj in self.var_objs:
            records[f'reward_autonomy_signed_{var_obj}'] = autonomy_signed
            records[f'reward_cooperation_signed_{var_obj}'] = cooperation_signed
            records[f'reward_strategy_signed_{var_obj}'] = strategy_signed
            records[f'reward_strategy_cost_01_{var_obj}'] = strategy_risk_cost
            records[f'reward_strategy_decision_penalty_signed_{var_obj}'] = 0.0
            records[f'reward_autonomy_cost_01_{var_obj}'] = autonomy_cost
            records[f'reward_cooperation_cost_01_{var_obj}'] = cooperation_cost
            records[f'regime_phi_syn_{var_obj}'] = regime_values['phi_syn']
            records[f'regime_phi_coop_{var_obj}'] = regime_values['phi_coop']
            records[f'regime_phi_self_{var_obj}'] = regime_values['phi_self']
            records[f'regime_phi_fail_{var_obj}'] = regime_values['phi_fail']
            records[f'regime_reward_signed_{var_obj}'] = curriculum_reward_signed
            records[f'reward_base_signed_{var_obj}'] = base_reward_signed
            records[f'reward_curriculum_signed_{var_obj}'] = curriculum_reward_signed
            records[f'reward_pre_ceiling_signed_{var_obj}'] = pre_ceiling_reward_signed
            records[f'reward_risk_ceiling_signed_{var_obj}'] = risk_ceiling_signed
            records[f'reward_ceiling_active_{var_obj}'] = ceiling_active
            records[f'reward_signed_{var_obj}'] = final_reward_signed
            records[f'reward_score_01_{var_obj}'] = reward_score
            regime_values_by_var[var_obj] = regime_values

        self.last_record = records
        return records, final_reward_signed, regime_values_by_var

    def _compute_autonomy_reward(self, var_obj, local_components):
        return self._clip_signed(
            self.mixing['rho_local'] * local_components[var_obj]['local_quality_signed']
            + (1.0 - self.mixing['rho_local'])
            * local_components[var_obj]['local_marginal_quality_signed']
        )

    def _compute_cooperation_reward(self, var_obj, cooperative_components):
        return self._compute_cooperation_from_component(cooperative_components[var_obj])

    def _compute_cooperation_from_component(self, cooperative_component):
        return self._clip_signed(
            self.mixing['rho_global'] * cooperative_component['global_quality_signed']
            + (1.0 - self.mixing['rho_global'])
            * cooperative_component['cooperative_marginal_quality_signed']
        )

    def _compute_common_autonomy_reward(self, local_components):
        autonomy_sum = 0.0
        for var_obj in self.var_objs:
            autonomy_sum += self._compute_autonomy_reward(var_obj, local_components)
        return self._clip_signed(autonomy_sum / len(self.var_objs))

    def _compute_common_strategy_risk(self, strategy_components):
        strategy_risk_cost = 0.0
        for var_obj in self.var_objs:
            strategy_risk_cost = max(
                strategy_risk_cost,
                strategy_components[var_obj]['strategy_risk_cost_01']
            )
        return self._clip_01(strategy_risk_cost)

    def _build_var_aggregates(self):
        aggregates = {}
        for var_obj in self.var_objs:
            aggregates[var_obj] = {
                'count': 0.0,
                'autonomy_signed': 0.0,
                'cooperation_signed': 0.0,
                'strategy_signed': 0.0,
                'strategy_risk_cost': 0.0,
                'strategy_decision_penalty_signed': 0.0,
                'autonomy_cost': 0.0,
                'cooperation_cost': 0.0,
                'phi_syn': 0.0,
                'phi_coop': 0.0,
                'phi_self': 0.0,
                'phi_fail': 0.0,
                'regime_reward_signed': 0.0,
                'base_reward_signed': 0.0,
                'curriculum_reward_signed': 0.0,
                'pre_ceiling_reward_signed': 0.0,
                'risk_ceiling_signed': 0.0,
                'ceiling_active': 0.0,
                'final_reward_signed': 0.0,
                'reward_score': 0.0
            }
        return aggregates

    def _accumulate_var_aggregate(
        self,
        aggregate,
        autonomy_signed,
        cooperation_signed,
        strategy_signed,
        strategy_risk_cost,
        strategy_decision_penalty_signed,
        autonomy_cost,
        cooperation_cost,
        regime_values,
        base_reward_signed,
        curriculum_reward_signed,
        pre_ceiling_reward_signed,
        risk_ceiling_signed,
        ceiling_active,
        final_reward_signed,
        reward_score
    ):
        aggregate['count'] += 1.0
        aggregate['autonomy_signed'] += autonomy_signed
        aggregate['cooperation_signed'] += cooperation_signed
        aggregate['strategy_signed'] += strategy_signed
        aggregate['strategy_risk_cost'] += strategy_risk_cost
        aggregate['strategy_decision_penalty_signed'] += strategy_decision_penalty_signed
        aggregate['autonomy_cost'] += autonomy_cost
        aggregate['cooperation_cost'] += cooperation_cost
        aggregate['phi_syn'] += regime_values['phi_syn']
        aggregate['phi_coop'] += regime_values['phi_coop']
        aggregate['phi_self'] += regime_values['phi_self']
        aggregate['phi_fail'] += regime_values['phi_fail']
        aggregate['regime_reward_signed'] += regime_values['reward_signed']
        aggregate['base_reward_signed'] += base_reward_signed
        aggregate['curriculum_reward_signed'] += curriculum_reward_signed
        aggregate['pre_ceiling_reward_signed'] += pre_ceiling_reward_signed
        aggregate['risk_ceiling_signed'] += risk_ceiling_signed
        aggregate['ceiling_active'] += ceiling_active
        aggregate['final_reward_signed'] += final_reward_signed
        aggregate['reward_score'] += reward_score

    def _write_var_aggregate_records(
        self,
        records,
        regime_values_by_var,
        aggregate_by_var,
        var_obj_to_agent_names
    ):
        for var_obj in self.var_objs:
            agent_count = len(var_obj_to_agent_names[var_obj])
            if agent_count <= 0:
                raise ValueError(f"Cannot aggregate agent rewards without agents for {var_obj}")

            denominator = aggregate_by_var[var_obj]['count']
            records[f'reward_autonomy_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['autonomy_signed'] / denominator
            )
            records[f'reward_cooperation_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['cooperation_signed'] / denominator
            )
            records[f'reward_strategy_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['strategy_signed'] / denominator
            )
            records[f'reward_strategy_cost_01_{var_obj}'] = (
                aggregate_by_var[var_obj]['strategy_risk_cost'] / denominator
            )
            records[f'reward_strategy_decision_penalty_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['strategy_decision_penalty_signed'] / denominator
            )
            records[f'reward_autonomy_cost_01_{var_obj}'] = (
                aggregate_by_var[var_obj]['autonomy_cost'] / denominator
            )
            records[f'reward_cooperation_cost_01_{var_obj}'] = (
                aggregate_by_var[var_obj]['cooperation_cost'] / denominator
            )
            records[f'regime_phi_syn_{var_obj}'] = (
                aggregate_by_var[var_obj]['phi_syn'] / denominator
            )
            records[f'regime_phi_coop_{var_obj}'] = (
                aggregate_by_var[var_obj]['phi_coop'] / denominator
            )
            records[f'regime_phi_self_{var_obj}'] = (
                aggregate_by_var[var_obj]['phi_self'] / denominator
            )
            records[f'regime_phi_fail_{var_obj}'] = (
                aggregate_by_var[var_obj]['phi_fail'] / denominator
            )
            records[f'regime_reward_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['regime_reward_signed'] / denominator
            )
            records[f'reward_base_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['base_reward_signed'] / denominator
            )
            records[f'reward_curriculum_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['curriculum_reward_signed'] / denominator
            )
            records[f'reward_pre_ceiling_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['pre_ceiling_reward_signed'] / denominator
            )
            records[f'reward_risk_ceiling_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['risk_ceiling_signed'] / denominator
            )
            records[f'reward_ceiling_active_{var_obj}'] = (
                aggregate_by_var[var_obj]['ceiling_active'] / denominator
            )
            records[f'reward_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['final_reward_signed'] / denominator
            )
            records[f'reward_score_01_{var_obj}'] = (
                aggregate_by_var[var_obj]['reward_score'] / denominator
            )
            records[f'reward_diagnostic_controller_aggregate_autonomy_signed_{var_obj}'] = (
                records[f'reward_autonomy_signed_{var_obj}']
            )
            records[f'reward_diagnostic_controller_aggregate_cooperation_signed_{var_obj}'] = (
                records[f'reward_cooperation_signed_{var_obj}']
            )
            records[f'reward_diagnostic_controller_aggregate_strategy_signed_{var_obj}'] = (
                records[f'reward_strategy_signed_{var_obj}']
            )
            records[f'reward_diagnostic_controller_aggregate_strategy_decision_penalty_signed_{var_obj}'] = (
                aggregate_by_var[var_obj]['strategy_decision_penalty_signed'] / denominator
            )
            records[f'reward_diagnostic_controller_aggregate_signed_{var_obj}'] = (
                records[f'reward_signed_{var_obj}']
            )
            regime_values_by_var[var_obj] = {
                'phi_syn': records[f'regime_phi_syn_{var_obj}'],
                'phi_coop': records[f'regime_phi_coop_{var_obj}'],
                'phi_self': records[f'regime_phi_self_{var_obj}'],
                'phi_fail': records[f'regime_phi_fail_{var_obj}']
            }

    def _compute_regime_values(self, autonomy_cost, cooperation_cost):
        phi_syn = (1.0 - autonomy_cost) * (1.0 - cooperation_cost)
        phi_coop = autonomy_cost * (1.0 - cooperation_cost)
        phi_self = (1.0 - autonomy_cost) * cooperation_cost
        phi_fail = autonomy_cost * cooperation_cost
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

    def _apply_risk_ceiling(self, reward_signed, strategy_risk_cost):
        risk_ceiling_signed = self._compute_risk_ceiling(strategy_risk_cost)
        if self.risk_ceiling_config['enabled'] and reward_signed > risk_ceiling_signed:
            return self._clip_signed(risk_ceiling_signed), risk_ceiling_signed, 1.0
        return reward_signed, risk_ceiling_signed, 0.0

    def _compute_risk_ceiling(self, strategy_risk_cost):
        if not self.risk_ceiling_config['enabled']:
            return 1.0

        mode = self.risk_ceiling_config['mode']
        if mode == 'linear_signed_upper_bound':
            return self._clip_signed(
                self.risk_ceiling_config['intercept']
                - self.risk_ceiling_config['slope'] * strategy_risk_cost
            )

        raise ValueError(f"Unsupported reward risk ceiling mode: {mode}")

    def _update_curriculum_state(self):
        self.current_learning_rate = self._compute_learning_rate()

        if self.mode == 'composition':
            self.current_maturity = 0.0
            self.current_lambda_curriculum = 0.0
            self.curriculum_frozen = False
            self.current_betas = self._compute_curriculum_betas(0.0)
            return

        if self.mode != 'curriculum_regime':
            raise ValueError(f"Unsupported reward composition mode: {self.mode}")

        next_maturity = self._compute_maturity(self.current_learning_rate)
        freeze_threshold = self.curriculum_config['beta_schedule']['alpha_freeze']

        if self.previous_betas is not None and self.current_learning_rate <= freeze_threshold:
            self.curriculum_frozen = True
            self.current_betas = self.previous_betas.copy()
            self.current_lambda_curriculum = self.previous_lambda_curriculum
            self.current_maturity = self.previous_maturity
            return

        self.curriculum_frozen = False
        next_betas = self._compute_curriculum_betas(next_maturity)
        self._validate_curriculum_transition(next_betas)
        self.current_maturity = next_maturity
        self.current_lambda_curriculum = self._schedule_value(
            self.curriculum_config['lambda_schedule'],
            self.current_maturity
        )
        self.current_betas = next_betas
        self.previous_betas = self.current_betas.copy()
        self.previous_lambda_curriculum = self.current_lambda_curriculum
        self.previous_maturity = self.current_maturity

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
        schedule = self.curriculum_config['beta_schedule']
        beta_syn = self._schedule_value(schedule['beta_syn'], maturity)
        beta_coop = self._schedule_value(schedule['beta_coop'], maturity)
        beta_self = self._schedule_value(schedule['beta_self'], maturity)
        beta_fail = self._schedule_value(schedule['beta_fail'], maturity)
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

    def _schedule_value(self, schedule, maturity):
        return schedule['initial'] + (schedule['final'] - schedule['initial']) * maturity

    def _validate_curriculum_transition(self, next_betas):
        if self.previous_betas is None:
            return
        if not self.curriculum_config['beta_schedule']['enforce_delta_beta_lt_kappa_alpha']:
            return

        max_delta = 0.0
        for beta_name in next_betas:
            delta = abs(next_betas[beta_name] - self.previous_betas[beta_name])
            max_delta = max(max_delta, delta)

        limit = self.curriculum_config['beta_schedule']['kappa'] * self.current_learning_rate
        if max_delta >= limit:
            raise ValueError(
                f"Curriculum beta step must be lower than kappa*learning_rate: "
                f"delta_beta={max_delta}, limit={limit}"
            )

    def _clip_01(self, value):
        return min(1.0, max(0.0, value))

    def _clip_signed(self, value):
        return min(1.0, max(-1.0, value))

    def get_state_record(self):
        return {
            'reward_curriculum_maturity': self.current_maturity,
            'reward_lambda_curriculum': self.current_lambda_curriculum,
            'reward_curriculum_frozen': float(self.curriculum_frozen),
            'reward_beta_syn': self.current_betas['syn'],
            'reward_beta_coop': self.current_betas['coop'],
            'reward_beta_self': self.current_betas['self'],
            'reward_beta_fail': self.current_betas['fail']
        }

    def get_episode_summary_record(self):
        return {
            'reward_curriculum_maturity': self.current_maturity,
            'reward_lambda_curriculum': self.current_lambda_curriculum,
            'reward_curriculum_frozen': float(self.curriculum_frozen),
            'reward_beta_syn': self.current_betas['syn'],
            'reward_beta_coop': self.current_betas['coop'],
            'reward_beta_self': self.current_betas['self'],
            'reward_beta_fail': self.current_betas['fail']
        }
