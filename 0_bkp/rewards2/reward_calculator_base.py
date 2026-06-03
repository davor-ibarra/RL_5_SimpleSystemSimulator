"""
reward_calculator_base.py

Responsabilidad:
Orquestador del sistema de recompensas.

Instancia modulos declarados, coordina sus registros planos, compone en dominio
firmado y asigna una recompensa por controlador a los agentes asociados.
"""

import importlib

from rewards.reward_composer import RewardComposer


class RewardCalculatorBase:
    """
    Orquestador de recompensa.

    Mantiene el contrato del proyecto:
    config declarativa -> modulos plug and play -> registros planos ->
    asignacion por controller_reward.
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.reward_base_config = config_main['reward_base']
        self.reward_config = self.reward_base_config['reward_config']
        self.reward_calculation_config = self.reward_base_config['reward_calculation']
        self.reward_approach = self.reward_config['reward_approach']

        self.var_objs = self._extract_var_objs()
        self.agent_to_var_obj_map = self._build_agent_to_var_obj_map()

        self.local_control_quality_reward = self._instantiate_reward_module('local_control_quality')
        self.cooperative_control_quality_reward = self._instantiate_reward_module('cooperative_control_quality')
        self.coupling_penalty_reward = self._instantiate_reward_module('coupling_penalty')
        self.reward_composer = RewardComposer(config_main, self.var_objs)

        self._last_local_record = {}
        self._last_cooperative_record = {}
        self._last_coupling_record = {}
        self._last_regime_record = {}
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

    def _build_agent_to_var_obj_map(self):
        mapping = {}
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
        self._last_coupling_record = {}
        self._last_regime_record = {}
        self._last_reward_composition_record = {}
        self._last_global_interval_reward = 0.0
        self._last_assign_internal_reward_dict = {}
        self._last_controller_rewards = {var_obj: 0.0 for var_obj in self.var_objs}

        self.local_control_quality_reward.reset_episode()
        self.cooperative_control_quality_reward.reset_episode()
        self.coupling_penalty_reward.reset_episode()
        self.reward_composer.reset_episode()
        self._reset_episode_accumulators()

    def calculate(self, processed_metrics_dict, termination_flag='unknown', current_time_sec=0.0, actions_dict=None):
        reward_component = processed_metrics_dict['reward_component']
        extra_reward_component = processed_metrics_dict['extra_reward_component']

        local_records, local_components = self.local_control_quality_reward.evaluate(
            reward_component,
            extra_reward_component
        )
        cooperative_records, cooperative_components = self.cooperative_control_quality_reward.evaluate(
            reward_component,
            extra_reward_component,
            local_records
        )
        coupling_records, coupling_components = self.coupling_penalty_reward.evaluate(
            extra_reward_component,
            local_records
        )
        regime_records = self.reward_composer.build_regime_record(
            local_components,
            cooperative_components
        )
        composition_records, controller_rewards = self.reward_composer.compose(
            local_components,
            cooperative_components,
            coupling_components
        )

        self._last_local_record = local_records
        self._last_cooperative_record = cooperative_records
        self._last_coupling_record = coupling_records
        self._last_regime_record = regime_records
        self._last_reward_composition_record = composition_records
        self._last_controller_rewards = controller_rewards
        self._last_global_interval_reward = self._compute_global_interval_reward(controller_rewards)
        self._last_reward_composition_record['reward_base_interval_reward'] = self._last_global_interval_reward
        self._last_assign_internal_reward_dict = self._assign_rewards(controller_rewards)
        self._accumulate_episode_regimes(regime_records, controller_rewards)

        return self._last_assign_internal_reward_dict

    def _compute_global_interval_reward(self, controller_rewards):
        if not controller_rewards:
            return 0.0
        return sum(controller_rewards[var_obj] for var_obj in self.var_objs) / len(self.var_objs)

    def _assign_rewards(self, controller_rewards):
        assign_dict = {}

        if self.reward_approach == 'controller_reward':
            for agent_name, var_obj in self.agent_to_var_obj_map.items():
                assign_dict[agent_name] = controller_rewards[var_obj]
            return assign_dict

        if self.reward_approach == 'unique_for_all_reward':
            global_reward = self._compute_global_interval_reward(controller_rewards)
            for agent_name in self.agent_to_var_obj_map:
                assign_dict[agent_name] = global_reward
            return assign_dict

        raise ValueError(f"Unsupported reward_approach for refactored reward: {self.reward_approach}")

    def get_records(self):
        records = {
            'global_interval_reward': self._last_global_interval_reward
        }

        for agent_name, reward_val in self._last_assign_internal_reward_dict.items():
            records[f'reward_{agent_name}'] = reward_val

        records.update(self._last_local_record)
        records.update(self._last_cooperative_record)
        records.update(self._last_coupling_record)
        records.update(self._last_regime_record)
        records.update(self._last_reward_composition_record)
        return records

    def get_agent_state_records(self):
        records = {}
        records.update(self.local_control_quality_reward.get_state_record())
        records.update(self.cooperative_control_quality_reward.get_state_record())
        records.update(self.reward_composer.get_state_record())
        return records

    def get_episode_summary_rewards(self):
        summary = {
            'reward_curriculum_learning_rate': self.reward_composer.current_learning_rate,
            'reward_curriculum_maturity': self.reward_composer.current_maturity,
            'reward_curriculum_frozen': float(self.reward_composer.curriculum_frozen),
            'reward_beta_syn': self.reward_composer.current_betas['syn'],
            'reward_beta_coop': self.reward_composer.current_betas['coop'],
            'reward_beta_self': self.reward_composer.current_betas['self'],
            'reward_beta_fail': self.reward_composer.current_betas['fail']
        }

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
        self.episode_reward_sums = {
            var_obj: 0.0
            for var_obj in self.var_objs
        }

    def _accumulate_episode_regimes(self, regime_records, controller_rewards):
        self.episode_interval_count += 1
        for var_obj in self.var_objs:
            self.episode_regime_sums[var_obj]['syn'] += regime_records[f'regime_phi_syn_{var_obj}']
            self.episode_regime_sums[var_obj]['coop'] += regime_records[f'regime_phi_coop_{var_obj}']
            self.episode_regime_sums[var_obj]['self'] += regime_records[f'regime_phi_self_{var_obj}']
            self.episode_regime_sums[var_obj]['fail'] += regime_records[f'regime_phi_fail_{var_obj}']
            self.episode_reward_sums[var_obj] += controller_rewards[var_obj]
