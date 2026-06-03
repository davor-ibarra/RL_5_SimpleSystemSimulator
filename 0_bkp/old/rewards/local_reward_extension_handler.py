"""
local_reward_extension_handler.py

Responsabilidad:
Calcular correcciones locales opcionales para enriquecer la senal de aprendizaje.

Implementa:
- costo local no lineal J_i
- baseline marginal inmediata o suavizada
- penalizacion por movimiento improductivo por agente
- termino sinergico unico por lazo
"""

import math


class LocalRewardExtensionHandler:
    """
    Manejador de extension local de recompensa.

    Consume:
    - reward_component: metricas agregadas del intervalo
    - extra_reward_component: series crudas por step
    - actions_dict: acciones discretas ejecutadas por agente
    """

    def __init__(self, config_main):
        """
        Inicializa el bloque con configuracion declarativa.

        Args:
            config_main (dict): Configuracion principal completa
        """
        self.config_main = config_main
        reward_calculation = config_main['reward_base']['reward_calculation']
        self.config = reward_calculation['local_reward_extension']
        self.enabled = self.config['enabled']
        self.normalized_reward_mode = self.config['normalized_reward_mode']
        self.component_weights = self.config['component_weights']

        self.cost_config = self.config['cost']
        self.cost_enabled = self.cost_config['enabled']
        self.marginal_config = self.config['marginal_baseline']
        self.marginal_enabled = self.marginal_config['enabled']
        self.movement_config = self.config['movement_penalty']
        self.movement_enabled = self.movement_config['enabled']
        self.synergy_config = self.config['synergy']
        self.synergy_enabled = self.synergy_config['enabled']

        actions_space = config_main['agent_base']['agent_config']['actions']['actions_space']
        self.maintain_action_ids = self._extract_action_ids(actions_space, 'maintain')

        controllers_config = config_main['controller_base']['controllers']
        self.var_objs = [
            ctrl_cfg['params']['name_objective_var']
            for ctrl_cfg in controllers_config.values()
        ]

        coordination_config = reward_calculation['coordination_reward']
        self.default_correction_signs = coordination_config['correction_signs']

        self.reset_episode()

    def reset_episode(self):
        """
        Resetea memoria temporal al inicio de episodio.
        """
        self.previous_cost = {var_obj: None for var_obj in self.var_objs}
        self.baseline_cost = {var_obj: None for var_obj in self.var_objs}
        self.last_record = self._build_zero_record({})
        self.last_state_record = self._build_state_record()

    def evaluate(self, reward_component, extra_reward_component, actions_dict, agent_to_var_obj_map):
        """
        Evalua la extension local del intervalo.

        Returns:
            tuple:
                - local_extension_reward_total (float)
                - flat_record (dict)
                - local_extension_rewards_by_var (dict)
                - local_extension_rewards_by_agent (dict)
        """
        if not self.enabled:
            flat_record = self._build_zero_record(agent_to_var_obj_map)
            rewards_by_var = {var_obj: 0.0 for var_obj in self.var_objs}
            rewards_by_agent = {agent_name: 0.0 for agent_name in agent_to_var_obj_map.keys()}
            self.last_record = flat_record
            self.last_state_record = self._build_state_record()
            return 0.0, flat_record, rewards_by_var, rewards_by_agent

        cost_terms_by_var = self._compute_cost_terms_by_var(reward_component)
        component_weights_by_var = self._compute_component_weights_by_var()
        marginal_by_var, marginal_record, move_reference_by_var = self._compute_marginal_rewards(cost_terms_by_var)
        synergy_by_var, synergy_record = self._compute_synergy_rewards(extra_reward_component)

        rewards_by_var = {}
        for var_obj in self.var_objs:
            if self.normalized_reward_mode:
                rewards_by_var[var_obj] = (
                    component_weights_by_var[var_obj]['marginal_baseline'] * marginal_by_var[var_obj]
                    + component_weights_by_var[var_obj]['synergy'] * synergy_by_var[var_obj]['reward']
                )
            else:
                rewards_by_var[var_obj] = marginal_by_var[var_obj] + synergy_by_var[var_obj]['reward']

        move_penalty_by_agent = self._compute_move_penalties(
            actions_dict,
            agent_to_var_obj_map,
            move_reference_by_var
        )
        rewards_by_agent = {}
        composition_scores_by_agent = {}
        for agent_name, var_obj in agent_to_var_obj_map.items():
            if self.normalized_reward_mode:
                rewards_by_agent[agent_name] = (
                    rewards_by_var[var_obj]
                    - component_weights_by_var[var_obj]['movement_penalty'] * move_penalty_by_agent[agent_name]
                )
            else:
                rewards_by_agent[agent_name] = rewards_by_var[var_obj] - move_penalty_by_agent[agent_name]
            composition_scores_by_agent[agent_name] = self._signed_unit_to_score(rewards_by_agent[agent_name])

        total_reward = self._compute_total_reward(rewards_by_agent)
        composition_score_total = (
            sum(composition_scores_by_agent.values()) / len(composition_scores_by_agent)
            if composition_scores_by_agent else 0.5
        )
        flat_record = self._build_base_record(total_reward)
        flat_record['local_extension_composition_score_01'] = composition_score_total
        flat_record['local_extension_composition_cost_01'] = 1.0 - composition_score_total
        flat_record.update(marginal_record)
        flat_record.update(synergy_record)

        for var_obj in self.var_objs:
            terms = cost_terms_by_var[var_obj]
            var_composition_score = self._signed_unit_to_score(rewards_by_var[var_obj])
            flat_record[f'local_component_weight_marginal_baseline_{var_obj}'] = (
                component_weights_by_var[var_obj]['marginal_baseline']
            )
            flat_record[f'local_component_weight_movement_penalty_{var_obj}'] = (
                component_weights_by_var[var_obj]['movement_penalty']
            )
            flat_record[f'local_component_weight_synergy_{var_obj}'] = (
                component_weights_by_var[var_obj]['synergy']
            )
            flat_record[f'local_error_potential_{var_obj}'] = terms['E']
            flat_record[f'local_effort_gate_{var_obj}'] = terms['g_u']
            flat_record[f'local_cost_J_{var_obj}'] = terms['J']
            flat_record[f'local_marginal_signal_{var_obj}'] = marginal_by_var[var_obj]
            flat_record[f'local_marginal_reward_{var_obj}'] = (
                component_weights_by_var[var_obj]['marginal_baseline'] * marginal_by_var[var_obj]
                if self.normalized_reward_mode
                else marginal_by_var[var_obj]
            )
            flat_record[f'local_synergy_reward_{var_obj}'] = (
                component_weights_by_var[var_obj]['synergy'] * synergy_by_var[var_obj]['reward']
                if self.normalized_reward_mode
                else synergy_by_var[var_obj]['reward']
            )
            flat_record[f'local_extension_reward_{var_obj}'] = rewards_by_var[var_obj]
            flat_record[f'local_reward_extension_{var_obj}'] = rewards_by_var[var_obj]
            flat_record[f'local_extension_composition_score_01_{var_obj}'] = var_composition_score
            flat_record[f'local_extension_composition_cost_01_{var_obj}'] = 1.0 - var_composition_score

        for agent_name in agent_to_var_obj_map.keys():
            var_obj = agent_to_var_obj_map[agent_name]
            flat_record[f'local_move_penalty_signal_{agent_name}'] = move_penalty_by_agent[agent_name]
            flat_record[f'local_move_penalty_{agent_name}'] = (
                component_weights_by_var[var_obj]['movement_penalty'] * move_penalty_by_agent[agent_name]
                if self.normalized_reward_mode
                else move_penalty_by_agent[agent_name]
            )
            flat_record[f'local_extension_reward_{agent_name}'] = rewards_by_agent[agent_name]
            flat_record[f'local_extension_composition_score_01_{agent_name}'] = composition_scores_by_agent[agent_name]
            flat_record[f'local_extension_composition_cost_01_{agent_name}'] = 1.0 - composition_scores_by_agent[agent_name]

        self.previous_cost = {
            var_obj: cost_terms_by_var[var_obj]['J']
            for var_obj in self.var_objs
        }
        self.last_record = flat_record
        self.last_state_record = self._build_state_record()
        return total_reward, flat_record, rewards_by_var, rewards_by_agent

    def _compute_cost_terms_by_var(self, reward_component):
        terms_by_var = {}
        for var_obj in self.var_objs:
            alpha_e = self._resolve_var_value(self.cost_config['alpha_e'], var_obj)
            effort_weight = self._resolve_var_value(self.cost_config['effort_weight'], var_obj)
            effort_gate_kappa = self._resolve_var_value(
                self.cost_config['effort_gate_kappa'],
                var_obj
            )

            L_e = self._local_cost_input(reward_component[f'L_e_{var_obj}'])
            L_edot = self._local_cost_input(reward_component[f'L_edot_{var_obj}'])
            L_u = self._local_cost_input(reward_component[f'L_u_{var_obj}'])
            E = alpha_e * L_e + (1.0 - alpha_e) * L_edot
            g_u = math.exp(-effort_gate_kappa * E)
            J = (E + effort_weight * g_u * L_u) / (1.0 + effort_weight)

            terms_by_var[var_obj] = {
                'L_e': L_e,
                'L_edot': L_edot,
                'L_u': L_u,
                'E': E,
                'g_u': g_u,
                'J': J
            }

        return terms_by_var

    def _compute_marginal_rewards(self, cost_terms_by_var):
        mode = self.marginal_config['mode']
        marginal_by_var = {}
        move_reference_by_var = {}
        flat = {}

        for var_obj in self.var_objs:
            current_cost = cost_terms_by_var[var_obj]['J']
            previous_cost = self.previous_cost[var_obj]
            delta_j = 0.0 if previous_cost is None else previous_cost - current_cost

            previous_baseline = self.baseline_cost[var_obj]
            advantage = 0.0 if previous_baseline is None else previous_baseline - current_cost
            tau = self._resolve_var_value(self.marginal_config['baseline_tau'], var_obj)
            if previous_baseline is None:
                next_baseline = current_cost
            else:
                next_baseline = (1.0 - tau) * previous_baseline + tau * current_cost
            self.baseline_cost[var_obj] = next_baseline

            if not self.marginal_enabled:
                marginal_reward = 0.0
                move_reference = advantage if mode == 'smoothed_baseline' else delta_j
            elif mode == 'immediate_delta':
                if self.normalized_reward_mode:
                    marginal_reward = self._clip_signed_01(delta_j)
                else:
                    beta = self._resolve_var_value(self.marginal_config['beta_immediate'], var_obj)
                    chi = self._resolve_var_value(
                        self.marginal_config['deterioration_weight_immediate'],
                        var_obj
                    )
                    marginal_reward = beta * max(0.0, delta_j) - chi * max(0.0, -delta_j)
                move_reference = delta_j
            else:
                if self.normalized_reward_mode:
                    marginal_reward = self._clip_signed_01(advantage)
                else:
                    beta = self._resolve_var_value(self.marginal_config['beta_smooth'], var_obj)
                    chi = self._resolve_var_value(
                        self.marginal_config['deterioration_weight_smooth'],
                        var_obj
                    )
                    marginal_reward = beta * max(0.0, advantage) - chi * max(0.0, -advantage)
                move_reference = advantage

            marginal_by_var[var_obj] = marginal_reward
            move_reference_by_var[var_obj] = move_reference
            flat[f'local_cost_delta_{var_obj}'] = delta_j
            flat[f'local_baseline_B_{var_obj}'] = next_baseline
            flat[f'local_advantage_A_{var_obj}'] = advantage
            flat[f'local_marginal_reward_{var_obj}'] = marginal_reward

        return marginal_by_var, flat, move_reference_by_var

    def _compute_move_penalties(self, actions_dict, agent_to_var_obj_map, move_reference_by_var):
        penalties = {}
        action_values = actions_dict['vars_decision']

        for agent_name, var_obj in agent_to_var_obj_map.items():
            action_key = f'action_{agent_name}'
            action_value = action_values[action_key]
            moved = action_value is not None and action_value not in self.maintain_action_ids
            no_improvement = move_reference_by_var[var_obj] <= 0.0
            if self.movement_enabled and moved and no_improvement:
                if self.normalized_reward_mode:
                    penalties[agent_name] = 1.0
                else:
                    penalties[agent_name] = self._resolve_agent_value(
                        self.movement_config['penalty_weight'],
                        agent_name,
                        var_obj,
                    )
            else:
                penalties[agent_name] = 0.0

        return penalties

    def _compute_synergy_rewards(self, extra_reward_component):
        synergy_by_var = {}
        flat = {}

        for var_obj in self.var_objs:
            if not self.synergy_enabled:
                synergy_by_var[var_obj] = {'reward': 0.0, 'synergy': 0.0}
                flat[f'local_synergy_error_derivative_{var_obj}'] = 0.0
                flat[f'local_synergy_error_effort_{var_obj}'] = 0.0
                flat[f'local_synergy_{var_obj}'] = 0.0
                flat[f'local_synergy_reward_{var_obj}'] = 0.0
                continue

            error_key = self.synergy_config['error_signal'].format(var_obj=var_obj)
            edot_key = self.synergy_config['derivative_error_signal'].format(var_obj=var_obj)
            effort_key = self.synergy_config['effort_signal'].format(var_obj=var_obj)

            error_series = extra_reward_component[error_key]
            edot_series = extra_reward_component[edot_key]
            effort_series = extra_reward_component[effort_key]
            n_steps = min(len(error_series), len(edot_series), len(effort_series))

            if n_steps <= 0:
                synergy_by_var[var_obj] = {'reward': 0.0, 'synergy': 0.0}
                flat[f'local_synergy_error_derivative_{var_obj}'] = 0.0
                flat[f'local_synergy_error_effort_{var_obj}'] = 0.0
                flat[f'local_synergy_{var_obj}'] = 0.0
                flat[f'local_synergy_reward_{var_obj}'] = 0.0
                continue

            gamma_e = self._resolve_var_value(self.synergy_config['gamma_e'], var_obj)
            gamma_edot = self._resolve_var_value(self.synergy_config['gamma_edot'], var_obj)
            gamma_u = self._resolve_var_value(self.synergy_config['gamma_u'], var_obj)
            weight = self._resolve_var_value(self.synergy_config['weight'], var_obj)
            correction_sign = self._resolve_var_value(
                self.synergy_config['correction_signs'],
                var_obj
            )

            sedot_sum = 0.0
            seu_sum = 0.0
            synergy_sum = 0.0
            for idx in range(n_steps):
                error_value = error_series[idx]
                edot_value = edot_series[idx]
                effort_value = effort_series[idx]
                error_gate = math.tanh(gamma_e * abs(error_value))
                sedot = math.tanh(gamma_edot * (-error_value * edot_value))
                seu = math.tanh(gamma_u * correction_sign * error_value * effort_value)
                sedot_sum += sedot
                seu_sum += seu
                synergy_sum += error_gate * sedot * seu

            sedot_interval = sedot_sum / n_steps
            seu_interval = seu_sum / n_steps
            synergy = synergy_sum / n_steps
            if self.normalized_reward_mode:
                reward = synergy
            else:
                reward = weight * synergy
            synergy_by_var[var_obj] = {'reward': reward, 'synergy': synergy}
            flat[f'local_synergy_error_derivative_{var_obj}'] = sedot_interval
            flat[f'local_synergy_error_effort_{var_obj}'] = seu_interval
            flat[f'local_synergy_{var_obj}'] = synergy
            flat[f'local_synergy_reward_{var_obj}'] = reward

        return synergy_by_var, flat

    def _build_base_record(self, total_reward):
        return {
            'local_extension_enabled': float(self.enabled),
            'local_extension_normalized_mode': float(self.normalized_reward_mode),
            'local_extension_cost_enabled': float(self.cost_enabled),
            'local_extension_marginal_enabled': float(self.marginal_enabled),
            'local_extension_movement_penalty_enabled': float(self.movement_enabled),
            'local_extension_synergy_enabled': float(self.synergy_enabled),
            'local_extension_reward_total': total_reward
        }

    def _build_zero_record(self, agent_to_var_obj_map):
        flat = self._build_base_record(0.0)
        flat['local_extension_composition_score_01'] = 0.5
        flat['local_extension_composition_cost_01'] = 0.5
        for var_obj in self.var_objs:
            flat[f'local_component_weight_marginal_baseline_{var_obj}'] = 0.0
            flat[f'local_component_weight_movement_penalty_{var_obj}'] = 0.0
            flat[f'local_component_weight_synergy_{var_obj}'] = 0.0
            flat[f'local_error_potential_{var_obj}'] = 0.0
            flat[f'local_effort_gate_{var_obj}'] = 0.0
            flat[f'local_cost_J_{var_obj}'] = 0.0
            flat[f'local_cost_delta_{var_obj}'] = 0.0
            flat[f'local_baseline_B_{var_obj}'] = 0.0
            flat[f'local_advantage_A_{var_obj}'] = 0.0
            flat[f'local_marginal_signal_{var_obj}'] = 0.0
            flat[f'local_marginal_reward_{var_obj}'] = 0.0
            flat[f'local_synergy_error_derivative_{var_obj}'] = 0.0
            flat[f'local_synergy_error_effort_{var_obj}'] = 0.0
            flat[f'local_synergy_{var_obj}'] = 0.0
            flat[f'local_synergy_reward_{var_obj}'] = 0.0
            flat[f'local_extension_reward_{var_obj}'] = 0.0
            flat[f'local_reward_extension_{var_obj}'] = 0.0
            flat[f'local_extension_composition_score_01_{var_obj}'] = 0.5
            flat[f'local_extension_composition_cost_01_{var_obj}'] = 0.5

        for agent_name in agent_to_var_obj_map.keys():
            flat[f'local_move_penalty_signal_{agent_name}'] = 0.0
            flat[f'local_move_penalty_{agent_name}'] = 0.0
            flat[f'local_extension_reward_{agent_name}'] = 0.0
            flat[f'local_extension_composition_score_01_{agent_name}'] = 0.5
            flat[f'local_extension_composition_cost_01_{agent_name}'] = 0.5

        return flat

    def _compute_total_reward(self, rewards_by_agent):
        if not rewards_by_agent:
            return 0.0
        if self.normalized_reward_mode:
            return sum(rewards_by_agent.values()) / len(rewards_by_agent)
        return sum(rewards_by_agent.values())

    def _local_cost_input(self, value):
        if self.normalized_reward_mode:
            return min(1.0, max(0.0, abs(value)))
        return abs(value)

    def _compute_component_weights_by_var(self):
        weights_by_var = {}
        for var_obj in self.var_objs:
            marginal_weight = self._resolve_var_value(
                self.component_weights['marginal_baseline'],
                var_obj
            )
            movement_weight = self._resolve_var_value(
                self.component_weights['movement_penalty'],
                var_obj
            )
            synergy_weight = self._resolve_var_value(
                self.component_weights['synergy'],
                var_obj
            )

            if not self.marginal_enabled:
                marginal_weight = 0.0
            if not self.movement_enabled:
                movement_weight = 0.0
            if not self.synergy_enabled:
                synergy_weight = 0.0

            if self.normalized_reward_mode:
                weight_sum = marginal_weight + movement_weight + synergy_weight
                if weight_sum > 0.0:
                    marginal_weight = marginal_weight / weight_sum
                    movement_weight = movement_weight / weight_sum
                    synergy_weight = synergy_weight / weight_sum

            weights_by_var[var_obj] = {
                'marginal_baseline': marginal_weight,
                'movement_penalty': movement_weight,
                'synergy': synergy_weight
            }

        return weights_by_var

    def _clip_signed_01(self, value):
        return min(1.0, max(-1.0, value))

    def _signed_unit_to_score(self, value):
        return 0.5 * (self._clip_signed_01(value) + 1.0)

    def _extract_action_ids(self, actions_space, action_name):
        action_ids = set()
        for action_id, value in actions_space.items():
            if value == action_name:
                action_ids.add(action_id)
                try:
                    action_ids.add(int(action_id))
                except (TypeError, ValueError):
                    pass
        return action_ids

    def _resolve_var_value(self, value_config, var_obj):
        if isinstance(value_config, dict):
            return value_config[var_obj]
        return value_config

    def _resolve_agent_value(self, value_config, agent_name, var_obj):
        if isinstance(value_config, dict):
            if agent_name in value_config:
                return value_config[agent_name]
            if var_obj in value_config:
                return value_config[var_obj]
        return value_config

    def _build_state_record(self):
        state_record = {}
        for var_obj in self.var_objs:
            baseline_value = self.baseline_cost[var_obj]
            previous_value = self.previous_cost[var_obj]
            state_record[f'local_baseline_B_{var_obj}'] = 0.0 if baseline_value is None else baseline_value
            state_record[f'local_previous_cost_J_{var_obj}'] = 0.0 if previous_value is None else previous_value
        return state_record

    def get_local_reward_extension_params_record(self):
        """
        Retorna el ultimo registro plano del bloque local.
        """
        return self.last_record.copy()

    def get_state_record(self):
        """
        Retorna variables internas aptas para discretizacion declarativa.
        """
        return self.last_state_record.copy()
