"""
reward_calculator_base.py

Responsabilidad:
Orquestador del sistema de recompensas.
- Instancia dinamicamente principal_reward_impl y extra_rewards_handler.
- Orquesta: principal + extras + coordinacion + extension local.
- Asigna recompensas a agentes segun reward_approach.
"""

import importlib


class RewardCalculatorBase:
    """
    Orquestador del sistema de recompensas.

    Instancia dinamicamente:
    - principal_reward_impl (e.g., LagrangeRewardCalculator)
    - extra_rewards_handler (ExtraRewardsHandler)
    - coordination_reward_handler (CoordinationRewardHandler)
    - local_reward_extension_handler (LocalRewardExtensionHandler)
    """

    def __init__(self, config_main):
        self.config_main = config_main
        self.reward_base_config = config_main['reward_base']
        self.reward_config = self.reward_base_config['reward_config']
        self.reward_calculation_config = self.reward_base_config['reward_calculation']

        self.reward_approach = self.reward_config['reward_approach']
        self.reward_composition_config = self.reward_config['reward_composition']
        self.reward_composition_enabled = self.reward_composition_config['enabled']
        self.normalized_composition_mode = self.reward_composition_config['normalized_composition_mode']
        self.reward_composition_strict = self.reward_composition_config['strict_weight_sum']

        self.agent_to_var_obj_map = self._build_agent_to_var_obj_map()

        self.principal_reward_impl = self._instantiate_principal_reward()
        self.extra_rewards_handler = self._instantiate_extra_rewards()
        self.coordination_reward_handler = self._instantiate_coordination_reward()
        self.local_reward_extension_handler = self._instantiate_local_reward_extension()
        self.reward_block_weights = self._build_reward_block_weights()
        self.agent_individual_weights = self._build_agent_individual_weights()

        self._last_principal_record = {}
        self._last_extra_record = {}
        self._last_coordination_record = {}
        self._last_local_extension_record = {}
        self._last_reward_composition_record = {}
        self._last_global_interval_reward = 0.0
        self._last_assign_internal_reward_dict = {}

    def _build_agent_individual_weights(self):
        """
        Extrae pesos individuales por agente para acceso O(1) en ejecucion.
        """
        weights = {}
        if self.reward_approach != 'individual_reward':
            return weights

        if not self.principal_reward_impl:
            raise ValueError("individual_reward requires principal_reward to be enabled")

        method = self.principal_reward_impl.method
        individual_params = self.reward_config['individual_reward_params']

        if method == 'weighted_exponential':
            params = individual_params['weighted_exponential_params']
        else:
            params = individual_params['lineal_combination_params']

        for agent_name, var_obj in self.agent_to_var_obj_map.items():
            weights[agent_name] = params[var_obj][agent_name]

        return weights

    def _build_agent_to_var_obj_map(self):
        """
        Construye agent_name -> var_obj usando la convencion canonica:
            {gain_type}_{var_obj}

        El var_obj se valida contra los controladores declarados.
        """
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

    def _instantiate_principal_reward(self):
        principal_config = self.reward_calculation_config['principal_reward']
        if not principal_config['enabled']:
            return None

        module = importlib.import_module(principal_config['module_path'])
        reward_class = getattr(module, principal_config['class_name'])
        return reward_class(self.config_main)

    def _instantiate_extra_rewards(self):
        extra_config = self.reward_calculation_config['extra_rewards']
        if not extra_config['enabled']:
            return None

        module = importlib.import_module(extra_config['module_path'])
        handler_class = getattr(module, extra_config['class_name'])
        return handler_class(extra_config)

    def _instantiate_coordination_reward(self):
        coordination_config = self.reward_calculation_config['coordination_reward']
        if not coordination_config['enabled']:
            return None

        module = importlib.import_module(coordination_config['module_path'])
        handler_class = getattr(module, coordination_config['class_name'])
        return handler_class(coordination_config)

    def _instantiate_local_reward_extension(self):
        local_extension_config = self.reward_calculation_config['local_reward_extension']
        if not local_extension_config['enabled']:
            return None

        module = importlib.import_module(local_extension_config['module_path'])
        handler_class = getattr(module, local_extension_config['class_name'])
        return handler_class(self.config_main)

    def _build_reward_block_weights(self):
        """
        Construye pesos solo cuando reward_composition esta habilitada.
        Si esta apagada, no existe composicion conceptual ni pesos implicitos.
        """
        if not self.reward_composition_enabled:
            return {}

        raw_weights = self.reward_composition_config['block_weights']
        weights = {
            'principal_reward': raw_weights['principal_reward'],
            'extra_rewards': raw_weights['extra_rewards'],
            'coordination_reward': raw_weights['coordination_reward'],
            'local_reward_extension': raw_weights['local_reward_extension']
        }

        for block_name, weight in weights.items():
            if weight < 0.0:
                raise ValueError(f"Reward block weight must be non-negative: {block_name}={weight}")
            if weight > 0.0 and not self._is_reward_block_enabled(block_name):
                raise ValueError(f"Reward block {block_name} is disabled but has weight {weight}")

        weight_sum = sum(weights.values())
        if self.reward_composition_strict and abs(weight_sum - 1.0) > 1.0e-9:
            raise ValueError(f"Reward block weights must sum 1.0, got {weight_sum}")

        if weight_sum > 0.0 and not self.reward_composition_strict:
            weights = {
                block_name: weight / weight_sum
                for block_name, weight in weights.items()
            }

        return weights

    def _is_reward_block_enabled(self, block_name):
        if block_name == 'principal_reward':
            return self.reward_calculation_config['principal_reward']['enabled']
        if block_name == 'extra_rewards':
            return self.reward_calculation_config['extra_rewards']['enabled']
        if block_name == 'coordination_reward':
            return self.reward_calculation_config['coordination_reward']['enabled']
        if block_name == 'local_reward_extension':
            return self.reward_calculation_config['local_reward_extension']['enabled']
        raise ValueError(f"Unknown reward block: {block_name}")

    def _is_reward_block_normalized(self, block_name):
        if block_name == 'principal_reward':
            return self.reward_calculation_config['principal_reward']['normalized_reward_mode']
        if block_name == 'extra_rewards':
            return self.reward_calculation_config['extra_rewards']['normalized_reward_mode']
        if block_name == 'coordination_reward':
            return self.reward_calculation_config['coordination_reward']['normalized_reward_mode']
        if block_name == 'local_reward_extension':
            return self.reward_calculation_config['local_reward_extension']['normalized_reward_mode']
        raise ValueError(f"Unknown reward block: {block_name}")

    def _compose_interval_reward(self, principal_reward, extra_reward, coordination_reward, local_extension_reward):
        """
        Compone con pesos declarativos si la composicion esta activa.
        Con composicion apagada, agrega valores nativos sin registrar pesos falsos.
        """
        raw_values = {
            'principal_reward': principal_reward,
            'extra_rewards': extra_reward,
            'coordination_reward': coordination_reward,
            'local_reward_extension': local_extension_reward
        }

        if not self.reward_composition_enabled:
            total_reward = sum(raw_values.values())
            self._last_reward_composition_record = {
                'reward_composition_enabled': 0.0,
                'reward_normalized_composition_mode': float(self.normalized_composition_mode),
                'reward_raw_principal_reward': principal_reward,
                'reward_raw_extra_rewards': extra_reward,
                'reward_raw_coordination_reward': coordination_reward,
                'reward_raw_local_reward_extension': local_extension_reward,
                'reward_raw_aggregate_reward': total_reward
            }
            return total_reward

        if self.normalized_composition_mode:
            return self._compose_interval_reward_from_scores(raw_values)

        return self._compose_interval_reward_from_raw(raw_values)

    def _compose_interval_reward_from_raw(self, raw_values):
        weighted_values = {}
        for block_name, raw_value in raw_values.items():
            block_weight = self.reward_block_weights[block_name]
            weighted_values[block_name] = block_weight * raw_value

        self._last_reward_composition_record = {
            'reward_composition_enabled': 1.0,
            'reward_normalized_composition_mode': float(self.normalized_composition_mode),
            'reward_block_weight_sum': sum(self.reward_block_weights.values()),
            'reward_block_weight_principal_reward': self.reward_block_weights['principal_reward'],
            'reward_block_weight_extra_rewards': self.reward_block_weights['extra_rewards'],
            'reward_block_weight_coordination_reward': self.reward_block_weights['coordination_reward'],
            'reward_block_weight_local_reward_extension': self.reward_block_weights['local_reward_extension'],
            'reward_raw_principal_reward': raw_values['principal_reward'],
            'reward_raw_extra_rewards': raw_values['extra_rewards'],
            'reward_raw_coordination_reward': raw_values['coordination_reward'],
            'reward_raw_local_reward_extension': raw_values['local_reward_extension'],
            'reward_weighted_principal_reward': weighted_values['principal_reward'],
            'reward_weighted_extra_rewards': weighted_values['extra_rewards'],
            'reward_weighted_coordination_reward': weighted_values['coordination_reward'],
            'reward_weighted_local_reward_extension': weighted_values['local_reward_extension'],
            'reward_composition_mode': 'raw'
        }
        return sum(weighted_values.values())

    def _compose_interval_reward_from_scores(self, raw_values):
        reward_form = self._resolve_composition_reward_form()
        block_scores = {
            block_name: self._get_block_composition_score(block_name)
            for block_name in raw_values.keys()
        }
        weighted_scores = {
            block_name: self.reward_block_weights[block_name] * block_score
            for block_name, block_score in block_scores.items()
        }
        composition_score_01 = self._validate_score_01(
            sum(weighted_scores.values()),
            'reward_composition_score_01'
        )
        composition_reward = self._score_to_reward(composition_score_01, reward_form)
        weighted_rewards = {
            block_name: self.reward_block_weights[block_name] * self._score_to_reward(block_score, reward_form)
            for block_name, block_score in block_scores.items()
        }

        self._last_reward_composition_record = {
            'reward_composition_enabled': 1.0,
            'reward_normalized_composition_mode': float(self.normalized_composition_mode),
            'reward_composition_mode': 'score_01',
            'reward_composition_reward_form': reward_form,
            'reward_block_weight_sum': sum(self.reward_block_weights.values()),
            'reward_block_weight_principal_reward': self.reward_block_weights['principal_reward'],
            'reward_block_weight_extra_rewards': self.reward_block_weights['extra_rewards'],
            'reward_block_weight_coordination_reward': self.reward_block_weights['coordination_reward'],
            'reward_block_weight_local_reward_extension': self.reward_block_weights['local_reward_extension'],
            'reward_raw_principal_reward': raw_values['principal_reward'],
            'reward_raw_extra_rewards': raw_values['extra_rewards'],
            'reward_raw_coordination_reward': raw_values['coordination_reward'],
            'reward_raw_local_reward_extension': raw_values['local_reward_extension'],
            'reward_composition_input_principal_reward_score_01': block_scores['principal_reward'],
            'reward_composition_input_extra_rewards_score_01': block_scores['extra_rewards'],
            'reward_composition_input_coordination_reward_score_01': block_scores['coordination_reward'],
            'reward_composition_input_local_reward_extension_score_01': block_scores['local_reward_extension'],
            'reward_composition_weighted_principal_reward_score_01': weighted_scores['principal_reward'],
            'reward_composition_weighted_extra_rewards_score_01': weighted_scores['extra_rewards'],
            'reward_composition_weighted_coordination_reward_score_01': weighted_scores['coordination_reward'],
            'reward_composition_weighted_local_reward_extension_score_01': weighted_scores['local_reward_extension'],
            'reward_composition_score_01': composition_score_01,
            'reward_composition_cost_01': 1.0 - composition_score_01,
            'reward_weighted_principal_reward': weighted_rewards['principal_reward'],
            'reward_weighted_extra_rewards': weighted_rewards['extra_rewards'],
            'reward_weighted_coordination_reward': weighted_rewards['coordination_reward'],
            'reward_weighted_local_reward_extension': weighted_rewards['local_reward_extension']
        }
        return composition_reward

    def _compose_loop_reward(
        self,
        principal_reward,
        extra_reward,
        coordination_reward,
        local_extension_reward,
        agent_name=None,
        var_obj=None,
        principal_score_override=None
    ):
        if not self.reward_composition_enabled:
            return (
                principal_reward
                + extra_reward
                + coordination_reward
                + local_extension_reward
            )

        if self.normalized_composition_mode:
            reward_form = self._resolve_composition_reward_form()
            block_scores = {
                'principal_reward': (
                    principal_score_override
                    if principal_score_override is not None
                    else self._get_block_composition_score('principal_reward', var_obj=var_obj)
                ),
                'extra_rewards': self._get_block_composition_score('extra_rewards', var_obj=var_obj),
                'coordination_reward': self._get_block_composition_score('coordination_reward', var_obj=var_obj),
                'local_reward_extension': self._get_block_composition_score(
                    'local_reward_extension',
                    agent_name=agent_name,
                    var_obj=var_obj
                )
            }
            block_scores = {
                block_name: self._validate_score_01(
                    block_score,
                    f'{block_name}_composition_score_01_{agent_name or var_obj or "loop"}'
                )
                for block_name, block_score in block_scores.items()
            }
            weighted_scores = {
                block_name: self.reward_block_weights[block_name] * block_score
                for block_name, block_score in block_scores.items()
            }
            composition_score_01 = self._validate_score_01(
                sum(weighted_scores.values()),
                f'reward_composition_score_01_{agent_name or var_obj or "loop"}'
            )

            if agent_name:
                self._last_reward_composition_record[f'reward_composition_score_01_{agent_name}'] = composition_score_01
                self._last_reward_composition_record[f'reward_composition_cost_01_{agent_name}'] = 1.0 - composition_score_01
                for block_name, block_score in block_scores.items():
                    key_prefix = f'reward_composition_input_{block_name}_score_01_{agent_name}'
                    self._last_reward_composition_record[key_prefix] = block_score

            return self._score_to_reward(composition_score_01, reward_form)

        return (
            self.reward_block_weights['principal_reward'] * principal_reward
            + self.reward_block_weights['extra_rewards'] * extra_reward
            + self.reward_block_weights['coordination_reward'] * coordination_reward
            + self.reward_block_weights['local_reward_extension'] * local_extension_reward
        )

    def _get_block_record(self, block_name):
        if block_name == 'principal_reward':
            return self._last_principal_record
        if block_name == 'extra_rewards':
            return self._last_extra_record
        if block_name == 'coordination_reward':
            return self._last_coordination_record
        if block_name == 'local_reward_extension':
            return self._last_local_extension_record
        raise ValueError(f"Unknown reward block: {block_name}")

    def _get_block_composition_score(self, block_name, agent_name=None, var_obj=None):
        if self.reward_block_weights.get(block_name, 0.0) <= 0.0:
            return 0.0

        record = self._get_block_record(block_name)
        candidate_keys = self._build_composition_score_candidate_keys(block_name, agent_name, var_obj)
        for key in candidate_keys:
            if key in record:
                return self._validate_score_01(record[key], key)

        raise ValueError(
            f"Reward block {block_name} has positive composition weight but does not expose "
            f"a composition score in [0, 1]. Tried: {candidate_keys}"
        )

    def _build_composition_score_candidate_keys(self, block_name, agent_name=None, var_obj=None):
        keys = []

        if block_name == 'principal_reward':
            if var_obj:
                keys.extend([
                    f'principal_composition_score_01_{var_obj}',
                    f'principal_local_score_reward_{var_obj}',
                    f'principal_score_01_{var_obj}'
                ])
            keys.extend(['principal_composition_score_01', 'principal_score_01'])
            return keys

        if block_name == 'extra_rewards':
            if var_obj:
                keys.append(f'extra_composition_score_01_{var_obj}')
            keys.append('extra_composition_score_01')
            return keys

        if block_name == 'coordination_reward':
            if var_obj:
                keys.extend([
                    f'coordination_composition_score_01_{var_obj}',
                    f'coordination_reward_01_{var_obj}'
                ])
            keys.extend(['coordination_composition_score_01', 'coordination_reward_total_01'])
            return keys

        if block_name == 'local_reward_extension':
            if agent_name:
                keys.append(f'local_extension_composition_score_01_{agent_name}')
            if var_obj:
                keys.append(f'local_extension_composition_score_01_{var_obj}')
            keys.append('local_extension_composition_score_01')
            return keys

        raise ValueError(f"Unknown reward block: {block_name}")

    def _validate_score_01(self, value, key):
        if not isinstance(value, (int, float, bool)):
            raise ValueError(f"Composition score {key} must be numeric, got {type(value).__name__}")
        value = float(value)
        tolerance = 1.0e-9
        if value < -tolerance or value > 1.0 + tolerance:
            raise ValueError(f"Composition score {key} must be in [0, 1], got {value}")
        return min(1.0, max(0.0, value))

    def _resolve_composition_reward_form(self):
        reward_form = self._last_principal_record.get('principal_nonlinear_reward_form', 'cost')
        if reward_form not in ('cost', 'score'):
            return 'cost'
        return reward_form

    def _score_to_reward(self, score_01, reward_form):
        score_01 = self._validate_score_01(score_01, 'composition_score_01')
        if reward_form == 'score':
            return score_01
        return score_01 - 1.0

    def reset_episode(self):
        self._last_principal_record = {}
        self._last_extra_record = {}
        self._last_coordination_record = {}
        self._last_local_extension_record = {}
        self._last_reward_composition_record = {}
        self._last_global_interval_reward = 0.0
        self._last_assign_internal_reward_dict = {}

        if self.principal_reward_impl:
            self.principal_reward_impl.reset_episode()
        if self.extra_rewards_handler:
            self.extra_rewards_handler.reset_episode()
        if self.coordination_reward_handler:
            self.coordination_reward_handler.reset_episode()
        if self.local_reward_extension_handler:
            self.local_reward_extension_handler.reset_episode()

    def calculate(self, processed_metrics_dict, termination_flag='unknown', current_time_sec=0.0, actions_dict=None):
        reward_component = processed_metrics_dict['reward_component']
        extra_reward_component = processed_metrics_dict['extra_reward_component']
        active_var_objs = set(self.agent_to_var_obj_map.values())

        principal_reward = 0.0
        controller_rewards = {var_obj: 0.0 for var_obj in active_var_objs}
        self._last_principal_record = {}

        extra_reward = 0.0
        extra_rewards_by_var = {var_obj: 0.0 for var_obj in active_var_objs}
        self._last_extra_record = {}

        coordination_reward = 0.0
        coordination_rewards_by_var = {var_obj: 0.0 for var_obj in active_var_objs}
        self._last_coordination_record = {}

        local_extension_reward = 0.0
        local_extension_rewards_by_agent = {
            agent_name: 0.0
            for agent_name in self.agent_to_var_obj_map.keys()
        }
        self._last_local_extension_record = {}

        if self.principal_reward_impl:
            principal_reward, self._last_principal_record, controller_rewards = (
                self.principal_reward_impl.compute_reward(reward_component)
            )

        if self.extra_rewards_handler:
            extra_reward, self._last_extra_record = self.extra_rewards_handler.evaluate(
                extra_reward_component,
                reward_component,
                termination_flag,
                current_time_sec
            )
            for var_obj in active_var_objs:
                extra_rewards_by_var[var_obj] = self._last_extra_record[f'extra_total_{var_obj}']

        if self.coordination_reward_handler:
            coordination_reward, self._last_coordination_record, coordination_rewards_by_var = (
                self.coordination_reward_handler.evaluate(reward_component, extra_reward_component)
            )

        if self.local_reward_extension_handler:
            (
                local_extension_reward,
                self._last_local_extension_record,
                _,
                local_extension_rewards_by_agent
            ) = self.local_reward_extension_handler.evaluate(
                reward_component,
                extra_reward_component,
                actions_dict,
                self.agent_to_var_obj_map
            )

        base_interval_reward = self._compose_interval_reward(
            principal_reward,
            extra_reward,
            coordination_reward,
            local_extension_reward
        )
        self._last_global_interval_reward = base_interval_reward
        self._last_reward_composition_record['reward_base_interval_reward'] = base_interval_reward

        self._last_assign_internal_reward_dict = self._assign_rewards(
            self._last_global_interval_reward,
            controller_rewards,
            extra_rewards_by_var,
            coordination_rewards_by_var,
            reward_component,
            local_extension_rewards_by_agent
        )

        return self._last_assign_internal_reward_dict

    def get_records(self):
        records = {}
        records['global_interval_reward'] = self._last_global_interval_reward

        for agent_name, reward_val in self._last_assign_internal_reward_dict.items():
            records[f'reward_{agent_name}'] = reward_val

        if self._last_principal_record:
            for key, value in self._last_principal_record.items():
                if key == 'controller_rewards':
                    for var_obj, loss_val in value.items():
                        records[f'L_{var_obj}'] = loss_val
                elif isinstance(value, (int, float, bool, str)):
                    records[key] = value

        records.update(self._last_extra_record)
        records.update(self._last_coordination_record)
        records.update(self._last_local_extension_record)
        records.update(self._last_reward_composition_record)

        return records

    def get_agent_state_records(self):
        records = {}

        if self.coordination_reward_handler and hasattr(self.coordination_reward_handler, 'get_state_record'):
            records.update(self.coordination_reward_handler.get_state_record())

        if self.local_reward_extension_handler and hasattr(self.local_reward_extension_handler, 'get_state_record'):
            records.update(self.local_reward_extension_handler.get_state_record())

        return records

    def get_episode_summary_rewards(self):
        accumulated_band_bonus = 0.0
        accumulated_band_bonus_by_var = {}
        goal_bonus = 0.0

        if self.extra_rewards_handler:
            for var_obj, bonus_val in self.extra_rewards_handler.accumulated_band_bonus.items():
                accumulated_band_bonus_by_var[f'accumulated_band_bonus_{var_obj}'] = bonus_val
                accumulated_band_bonus += bonus_val
            last_record = self.extra_rewards_handler.last_extra_reward_params_record
            for key, value in last_record.items():
                if key.startswith('extra_bonus_goal_'):
                    goal_bonus = value
                    break

        summary = {
            'accumulated_band_bonus': accumulated_band_bonus,
            'goal_bonus': goal_bonus
        }
        summary.update(accumulated_band_bonus_by_var)

        return summary

    def _assign_rewards(
        self,
        global_reward,
        controller_rewards,
        extra_rewards_by_var,
        coordination_rewards_by_var,
        flat_reward_component,
        local_extension_rewards_by_agent=None
    ):
        assign_dict = {}
        local_extension_rewards_by_agent = local_extension_rewards_by_agent or {}

        if self.reward_approach == 'unique_for_all_reward':
            for agent_name in self.agent_to_var_obj_map.keys():
                assign_dict[agent_name] = global_reward

        elif self.reward_approach == 'controller_reward':
            for agent_name, var_obj in self.agent_to_var_obj_map.items():
                assign_dict[agent_name] = self._compose_loop_reward(
                    controller_rewards[var_obj],
                    extra_rewards_by_var[var_obj],
                    coordination_rewards_by_var[var_obj],
                    local_extension_rewards_by_agent[agent_name],
                    agent_name=agent_name,
                    var_obj=var_obj
                )

        elif self.reward_approach == 'individual_reward':
            for agent_name, var_obj in self.agent_to_var_obj_map.items():
                agent_weights = self.agent_individual_weights[agent_name]
                individual_principal_reward = self.principal_reward_impl.compute_agent_individual_reward(
                    flat_reward_component,
                    var_obj,
                    agent_weights
                )
                individual_principal_score = None
                if self.normalized_composition_mode:
                    individual_principal_score = (
                        self.principal_reward_impl.compute_agent_individual_composition_score(
                            flat_reward_component,
                            var_obj,
                            agent_weights
                        )
                    )
                assign_dict[agent_name] = self._compose_loop_reward(
                    individual_principal_reward,
                    extra_rewards_by_var[var_obj],
                    coordination_rewards_by_var[var_obj],
                    local_extension_rewards_by_agent[agent_name],
                    agent_name=agent_name,
                    var_obj=var_obj,
                    principal_score_override=individual_principal_score
                )

        else:
            raise ValueError(f"Unsupported reward_approach: {self.reward_approach}")

        return assign_dict
