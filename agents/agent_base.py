"""
agent_base.py

Responsabilidad:
Orquestador agnostico del agente.
Define la interfaz estandar de comunicacion con el simulador.
Delega toda la logica especifica al agente concreto.
"""

import importlib


class AgentBase:
    """
    Orquestador agnostico del agente.
    Define metodos estandar y delega al agente especifico.
    """

    def __init__(self, config_main):
        """
        Inicializa el orquestador con el agente especifico.

        Args:
            config_main (dict): Configuracion principal
        """
        self.config_main = config_main
        self.config_agent = config_main['agent_base']

        module_path = self.config_agent['module_path']
        class_name = self.config_agent['class_name']

        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        self.agent_impl = agent_class(config_main)

        self.agent_names = self.agent_impl.agent_names
        self.agent_gain_steps = self.agent_impl.agent_gain_steps

    def reset_episode(self):
        """
        Resetea el agente al inicio de un episodio.
        """
        self.agent_impl.reset_episode()

    def build_agent_state(self, agent_context):
        """
        Entrega el contexto plano al agente especifico para que construya su estado.

        Args:
            agent_context (dict): Contexto plano de senales disponibles.

        Returns:
            dict: Estado del agente.
        """
        return self.agent_impl.build_agent_state(agent_context)

    def select_action(self, agent_state):
        """
        Selecciona acciones basado en el estado.

        Args:
            agent_state (dict): Estado del agente

        Returns:
            dict: actions_dict con formato canonico
        """
        return self.agent_impl.select_action(agent_state)

    def learn(self, prev_agent_state, next_agent_state, actions_dict, reward_for_learning, terminated):
        """
        Actualiza el agente basado en la experiencia.

        Args:
            prev_agent_state (dict): Estado previo
            next_agent_state (dict): Estado siguiente
            actions_dict (dict): Acciones tomadas
            reward_for_learning (dict): {agent_name: reward}
            terminated (bool): Si el episodio termino

        Returns:
            dict: learn_info con metricas de aprendizaje
        """
        return self.agent_impl.learn(
            prev_agent_state, next_agent_state, actions_dict, reward_for_learning, terminated
        )

    def get_records(self):
        """
        Retorna dict plano con parametros interval-level del agente.

        Returns:
            dict: Registro plano interval-level del agente
        """
        return self.agent_impl.get_records()

    def get_agent_params_records(self):
        """
        Retorna dict plano con parametros episode-level del agente.

        Returns:
            dict: Registro plano episode-level del agente
        """
        return self.agent_impl.get_end_episode_records()

    def get_agent_state_learn_dict(self):
        """
        Expone el estado completo del agente para persistencia/heatmaps.

        Returns:
            dict: Q-tables, visit_counts y metadata de discretizacion
        """
        return self.agent_impl.get_agent_state_learn_dict()
