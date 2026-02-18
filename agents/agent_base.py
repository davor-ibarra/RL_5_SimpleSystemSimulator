"""
agent_base.py

Responsabilidad:
Orquestador agnóstico del agente.
Define la interfaz estándar de comunicación con el simulador.
Delega toda la lógica específica al agente concreto.
"""

import importlib


class AgentBase:
    """
    Orquestador agnóstico del agente.
    Define métodos estándar y delega al agente específico.
    """
    
    def __init__(self, config_main):
        """
        Inicializa el orquestador con el agente específico.
        
        Args:
            config_main (dict): Configuración principal
        """
        self.config_main = config_main
        self.config_agent = config_main['agent_base']
        
        # Instanciación dinámica desde config (module_path + class_name)
        module_path = self.config_agent['module_path']
        class_name = self.config_agent['class_name']
        
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        self.agent_impl = agent_class(config_main)
        
        # Exponer atributos del específico para acceso desde simulador
        self.agent_names = self.agent_impl.agent_names
        self.agent_gain_steps = self.agent_impl.agent_gain_steps
    
    def reset_episode(self):
        """
        Resetea el agente al inicio de un episodio.
        """
        self.agent_impl.reset_episode()
    
    def build_agent_state(self, dynamic_state_dict, controller_gains_dict):
        """
        Construye el estado del agente.
        
        Args:
            dynamic_state_dict (dict): Estado del sistema dinámico
            controller_gains_dict (dict): Ganancias actuales {agent_name: value}
            
        Returns:
            dict: Estado del agente
        """
        return self.agent_impl.build_agent_state(dynamic_state_dict, controller_gains_dict)
    
    def select_action(self, agent_state):
        """
        Selecciona acciones basado en el estado.
        
        Args:
            agent_state (dict): Estado del agente
            
        Returns:
            dict: actions_dict con formato canónico
        """
        return self.agent_impl.select_action(agent_state)
    
    def learn(self, prev_agent_state, next_agent_state, actions_dict, reward_for_learning, terminated):
        """
        Actualiza el agente basado en la experiencia.
        
        Args:
            prev_agent_state (dict): Estado previo
            next_agent_state (dict): Estado siguiente
            actions_dict (dict): Acciones tomadas
            reward_for_learning (dict): {agent_name: reward} — recompensas por agente
            terminated (bool): Si el episodio terminó
            
        Returns:
            dict: learn_info con métricas de aprendizaje
        """
        return self.agent_impl.learn(
            prev_agent_state, next_agent_state, actions_dict, reward_for_learning, terminated
        )
    
    def get_records(self):
        """
        Retorna dict plano con parámetros interval-level del agente para el MetricCollector.
        Solo estadísticas Q-table y métricas que cambian por intervalo.
        
        Returns:
            dict: Registro plano interval-level del agente
        """
        return self.agent_impl.get_records()
    
    def get_agent_params_records(self):
        """
        Retorna dict plano con parámetros episode-level del agente.
        Incluye: epsilon, learning_rate, y cualquier otro parámetro
        que cambie solo por episodio.
        
        Returns:
            dict: Registro plano episode-level del agente
        """
        return self.agent_impl.get_end_episode_records()
    
    def get_agent_state_learn_dict(self):
        """
        Expone el estado completo del agente para persistencia/heatmaps.
        
        Returns:
            dict: Q-tables, visit_counts y metadata de discretización
        """
        return self.agent_impl.get_agent_state_learn_dict()
