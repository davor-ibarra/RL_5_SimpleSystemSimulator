"""
reward_calculator_base.py

Responsabilidad:
Orquestar la recompensa declarativa del sistema.

Flujo:
LocalControlQualityReward -> CooperativeTransitionEvaluator -> CreditAllocator
-> InternalRiskPenaltyReward -> RewardComposer -> reward_for_learning.

Cada bloque expone registros planos y falla por acceso directo si la
configuracion no declara lo que el calculo necesita.
"""

import importlib

from rewards.reward_composer import RewardComposer


class RewardCalculatorBase:
    """
    Orquestador de recompensa.

    Mantiene una sola capa de asignacion cooperativa activa por corrida,
    determinada por reward_config.reward_approach.
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.reward_base_config = config_main['reward_base']
        self.reward_config = self.reward_base_config['reward_config']
        self.reward_calculation_config = self.reward_base_config['reward_calculation']
        self.reward_approach = self.reward_config['reward_approach']

        self.var_objs = self._extract_var_objs()
        self.agent_to_var_obj_map, self.agent_to_gain_type_map = self._build_agent_maps()
        self.var_obj_to_agent_names = self._build_var_obj_to_agent_names()

        self.local_control_quality_reward = self._instantiate_reward_module('local_control_quality')
        self.cooperative_transition_evaluator = self._instantiate_reward_module(
            'cooperative_transition_evaluator'
        )
        self.credit_allocator = self._instantiate_reward_module('credit_allocator')
        self.internal_risk_penalty_reward = self._instantiate_reward_module('internal_risk_penalty')
        self.reward_composer = RewardComposer(config_main, self.var_objs)

        self._last_local_record = {}
        self._last_cooperative_record = {}
        self._last_credit_record = {}
        self._last_strategy_record = {}
        self._last_reward_composition_record = {}
        self._last_global_interval_reward = 0.0
        self._last_assign_internal_reward_dict = {}
        self._last_controller_rewards = {var_obj: 0.0 for var_obj in self.var_objs}
        self._reset_episode_accumulators()

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

            controller_name = var_obj_to_controller[var_obj]
            controller_cfg = controllers_config[controller_name]
            if gain_type not in controller_cfg['initial_conditions']:
                raise ValueError(
                    f"Enabled agent '{agent_name}' references gain '{gain_type}', "
                    f"but controller '{controller_name}' does not declare it"
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

    def _instantiate_reward_module(self, config_key):
        module_config = self.reward_calculation_config[config_key]
        if not module_config['enabled']:
            raise ValueError(f"Required reward module is disabled: {config_key}")

        module = importlib.import_module(module_config['module_path'])
        reward_class = getattr(module, module_config['class_name'])
        return reward_class(self.config_main)

    def reset_episode(self):
        self._last_local_record = {}
        self._last_cooperative_record = {}
        self._last_credit_record = {}
        self._last_strategy_record = {}
        self._last_reward_composition_record = {}
        self._last_global_interval_reward = 0.0
        self._last_assign_internal_reward_dict = {}
        self._last_controller_rewards = {var_obj: 0.0 for var_obj in self.var_objs}

        self.local_control_quality_reward.reset_episode()
        self.cooperative_transition_evaluator.reset_episode()
        self.credit_allocator.reset_episode()
        self.internal_risk_penalty_reward.reset_episode()
        self.reward_composer.reset_episode()
        self._reset_episode_accumulators()

    def calculate(self, processed_metrics_dict, termination_flag='unknown', current_time_sec=0.0, actions_dict=None):
        reward_component = processed_metrics_dict['reward_component']
        extra_reward_component = processed_metrics_dict['extra_reward_component']

        local_records, local_components = self.local_control_quality_reward.evaluate(
            reward_component,
            extra_reward_component
        )
        cooperative_records, cooperative_components = self.cooperative_transition_evaluator.evaluate(
            reward_component,
            extra_reward_component,
            local_records
        )
        (
            credit_records,
            controller_cooperative_components,
            agent_cooperative_components,
            common_cooperative_component
        ) = self.credit_allocator.allocate(
            cooperative_components,
            extra_reward_component,
            actions_dict
        )
        (
            strategy_records,
            controller_strategy_components,
            agent_strategy_components
        ) = self.internal_risk_penalty_reward.evaluate(
            local_records,
            extra_reward_component,
            actions_dict,
            self.agent_to_var_obj_map
        )

        (
            composition_records,
            controller_rewards,
            agent_rewards,
            regime_values_by_var
        ) = self._compose_for_active_approach(
            local_components,
            controller_cooperative_components,
            agent_cooperative_components,
            common_cooperative_component,
            controller_strategy_components,
            agent_strategy_components
        )

        self._last_local_record = local_records
        self._last_cooperative_record = cooperative_records
        self._last_credit_record = credit_records
        self._last_strategy_record = strategy_records
        self._last_reward_composition_record = composition_records
        self._last_controller_rewards = controller_rewards
        self._last_assign_internal_reward_dict = self._assign_rewards(controller_rewards, agent_rewards)
        self._last_global_interval_reward = self._compute_assigned_interval_reward(
            self._last_assign_internal_reward_dict
        )
        self._accumulate_episode_records(composition_records, controller_rewards, regime_values_by_var)

        return self._last_assign_internal_reward_dict

    def _compose_for_active_approach(
        self,
        local_components,
        controller_cooperative_components,
        agent_cooperative_components,
        common_cooperative_component,
        controller_strategy_components,
        agent_strategy_components
    ):
        if self.reward_approach == 'controller_reward':
            composition_records, controller_rewards, regime_values_by_var = (
                self.reward_composer.compose_controller(
                    local_components,
                    controller_cooperative_components,
                    controller_strategy_components
                )
            )
            agent_rewards = {}
            return composition_records, controller_rewards, agent_rewards, regime_values_by_var

        if self.reward_approach == 'agent_reward':
            composition_records, agent_rewards, regime_values_by_var = (
                self.reward_composer.compose_agent(
                    local_components,
                    agent_cooperative_components,
                    agent_strategy_components,
                    self.agent_to_var_obj_map,
                    self.var_obj_to_agent_names
                )
            )
            controller_rewards = self._aggregate_agent_rewards_by_var(agent_rewards)
            return composition_records, controller_rewards, agent_rewards, regime_values_by_var

        if self.reward_approach == 'unique_for_all_reward':
            composition_records, common_reward, regime_values_by_var = (
                self.reward_composer.compose_common(
                    local_components,
                    common_cooperative_component,
                    controller_strategy_components
                )
            )
            controller_rewards = {
                var_obj: common_reward
                for var_obj in self.var_objs
            }
            agent_rewards = {
                agent_name: common_reward
                for agent_name in self.agent_to_var_obj_map
            }
            return composition_records, controller_rewards, agent_rewards, regime_values_by_var

        raise ValueError(f"Unsupported reward_approach: {self.reward_approach}")

    def _assign_rewards(self, controller_rewards, agent_rewards):
        if self.reward_approach == 'controller_reward':
            return self._assign_controller_rewards(controller_rewards)

        if self.reward_approach == 'agent_reward':
            return self._assign_agent_rewards(agent_rewards)

        if self.reward_approach == 'unique_for_all_reward':
            return self._assign_agent_rewards(agent_rewards)

        raise ValueError(f"Unsupported reward_approach: {self.reward_approach}")

    def _assign_controller_rewards(self, controller_rewards):
        assign_dict = {}
        for agent_name, var_obj in self.agent_to_var_obj_map.items():
            assign_dict[agent_name] = controller_rewards[var_obj]
        return assign_dict

    def _assign_agent_rewards(self, agent_rewards):
        assign_dict = {}
        for agent_name in self.agent_to_var_obj_map:
            assign_dict[agent_name] = agent_rewards[agent_name]
        return assign_dict

    def _aggregate_agent_rewards_by_var(self, agent_rewards):
        controller_rewards = {}
        for var_obj in self.var_objs:
            reward_sum = 0.0
            agent_count = len(self.var_obj_to_agent_names[var_obj])
            if agent_count <= 0:
                raise ValueError(f"Cannot aggregate agent rewards without agents for {var_obj}")
            for agent_name in self.var_obj_to_agent_names[var_obj]:
                reward_sum += agent_rewards[agent_name]
            controller_rewards[var_obj] = reward_sum / agent_count
        return controller_rewards

    def _compute_assigned_interval_reward(self, assigned_rewards):
        if not assigned_rewards:
            return 0.0
        total_reward = 0.0
        for agent_name in self.agent_to_var_obj_map:
            total_reward += assigned_rewards[agent_name]
        return total_reward / len(self.agent_to_var_obj_map)

    def get_records(self):
        records = {
            'global_interval_reward': self._last_global_interval_reward
        }

        for agent_name, reward_val in self._last_assign_internal_reward_dict.items():
            records[f'reward_{agent_name}'] = reward_val
            records[f'reward_for_learning_{agent_name}'] = reward_val

        records.update(self._last_local_record)
        records.update(self._last_cooperative_record)
        records.update(self._last_credit_record)
        records.update(self._last_strategy_record)
        records.update(self._last_reward_composition_record)
        return records

    def get_agent_state_records(self):
        records = {}
        records.update(self.local_control_quality_reward.get_state_record())
        records.update(self.cooperative_transition_evaluator.get_state_record())
        records.update(self.reward_composer.get_state_record())
        return records

    def get_episode_summary_rewards(self):
        summary = self.reward_composer.get_episode_summary_record()

        interval_count = self.episode_interval_count
        if interval_count <= 0:
            return summary

        for var_obj in self.var_objs:
            summary[f'episode_mean_regime_phi_syn_{var_obj}'] = (
                self.episode_regime_sums[var_obj]['syn'] / interval_count
            )
            summary[f'episode_mean_regime_phi_coop_{var_obj}'] = (
                self.episode_regime_sums[var_obj]['coop'] / interval_count
            )
            summary[f'episode_mean_regime_phi_self_{var_obj}'] = (
                self.episode_regime_sums[var_obj]['self'] / interval_count
            )
            summary[f'episode_mean_regime_phi_fail_{var_obj}'] = (
                self.episode_regime_sums[var_obj]['fail'] / interval_count
            )
            summary[f'episode_mean_reward_base_signed_{var_obj}'] = (
                self.episode_base_reward_sums[var_obj] / interval_count
            )
            summary[f'episode_mean_reward_curriculum_signed_{var_obj}'] = (
                self.episode_curriculum_reward_sums[var_obj] / interval_count
            )
            summary[f'episode_mean_reward_pre_ceiling_signed_{var_obj}'] = (
                self.episode_pre_ceiling_reward_sums[var_obj] / interval_count
            )
            summary[f'episode_mean_reward_risk_ceiling_signed_{var_obj}'] = (
                self.episode_risk_ceiling_sums[var_obj] / interval_count
            )
            summary[f'episode_mean_reward_ceiling_active_{var_obj}'] = (
                self.episode_ceiling_active_sums[var_obj] / interval_count
            )
            summary[f'episode_mean_reward_signed_{var_obj}'] = (
                self.episode_reward_sums[var_obj] / interval_count
            )

        return summary

    def _reset_episode_accumulators(self):
        self.episode_interval_count = 0
        self.episode_regime_sums = {
            var_obj: {
                'syn': 0.0,
                'coop': 0.0,
                'self': 0.0,
                'fail': 0.0
            }
            for var_obj in self.var_objs
        }
        self.episode_base_reward_sums = {
            var_obj: 0.0
            for var_obj in self.var_objs
        }
        self.episode_curriculum_reward_sums = {
            var_obj: 0.0
            for var_obj in self.var_objs
        }
        self.episode_pre_ceiling_reward_sums = {
            var_obj: 0.0
            for var_obj in self.var_objs
        }
        self.episode_risk_ceiling_sums = {
            var_obj: 0.0
            for var_obj in self.var_objs
        }
        self.episode_ceiling_active_sums = {
            var_obj: 0.0
            for var_obj in self.var_objs
        }
        self.episode_reward_sums = {
            var_obj: 0.0
            for var_obj in self.var_objs
        }

    def _accumulate_episode_records(self, composition_records, controller_rewards, regime_values_by_var):
        self.episode_interval_count += 1
        for var_obj in self.var_objs:
            self.episode_regime_sums[var_obj]['syn'] += regime_values_by_var[var_obj]['phi_syn']
            self.episode_regime_sums[var_obj]['coop'] += regime_values_by_var[var_obj]['phi_coop']
            self.episode_regime_sums[var_obj]['self'] += regime_values_by_var[var_obj]['phi_self']
            self.episode_regime_sums[var_obj]['fail'] += regime_values_by_var[var_obj]['phi_fail']
            self.episode_base_reward_sums[var_obj] += composition_records[f'reward_base_signed_{var_obj}']
            self.episode_curriculum_reward_sums[var_obj] += composition_records[
                f'reward_curriculum_signed_{var_obj}'
            ]
            self.episode_pre_ceiling_reward_sums[var_obj] += composition_records[
                f'reward_pre_ceiling_signed_{var_obj}'
            ]
            self.episode_risk_ceiling_sums[var_obj] += composition_records[
                f'reward_risk_ceiling_signed_{var_obj}'
            ]
            self.episode_ceiling_active_sums[var_obj] += composition_records[
                f'reward_ceiling_active_{var_obj}'
            ]
            self.episode_reward_sums[var_obj] += controller_rewards[var_obj]
