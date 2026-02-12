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
        self.gain_step = self.agent_impl.gain_step
    
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
    
    def learn(self, prev_agent_state, next_agent_state, actions_dict, reward_info, terminated):
        """
        Actualiza el agente basado en la experiencia.
        
        Args:
            prev_agent_state (dict): Estado previo
            next_agent_state (dict): Estado siguiente
            actions_dict (dict): Acciones tomadas
            reward_info (dict): Información de recompensa
            terminated (bool): Si el episodio terminó
            
        Returns:
            dict: learn_info con métricas de aprendizaje
        """
        return self.agent_impl.learn(
            prev_agent_state, next_agent_state, actions_dict, reward_info, terminated
        )
    
    def get_params_dict(self):
        """
        Expone parámetros de telemetría ligera del agente (estadísticas agregadas).
        
        Returns:
            dict: Parámetros actuales del agente (epsilon, learning_rate, estadísticas Q)
        """
        return self.agent_impl.get_params_dict()
    
    def get_agent_state_learn_dict(self):
        """
        Expone el estado completo del agente para persistencia/heatmaps.
        
        Returns:
            dict: Q-tables, visit_counts y metadata de discretización
        """
        return self.agent_impl.get_agent_state_learn_dict()
