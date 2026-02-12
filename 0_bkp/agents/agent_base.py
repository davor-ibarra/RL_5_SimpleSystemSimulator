"""
agent_base.py

Responsabilidad:
Definir la interfaz base para agentes de decisión.
Construir estado de decisión desde observables del sistema dinámico y bins de gains actuales.
Producir actions_dict con forma fija.
Aplicar aprendizaje con reward_info.
Exponer estado interno y métricas de aprendizaje para logging.
"""


class AgentBase:
    """
    Clase base abstracta para agentes de decisión.
    Define el contrato público que todos los agentes deben cumplir.
    """
    
    def __init__(self, config):
        """
        Inicializa el agente con configuración directa.
        
        Args:
            config (dict): Configuración del agente
        """
        pass
    
    def reset_episode(self):
        """
        Resetea el estado del agente al inicio de un episodio.
        """
        pass
    
    def build_agent_state(self, dynamic_state_dict):
        """
        Construye estado de decisión desde observables del sistema dinámico
        y bins de gains actuales (descubiertos desde controladores).
        
        Args:
            dynamic_state_dict (dict): Estado actual del sistema dinámico
            
        Returns:
            Estado del agente (tipo específico según implementación)
        """
        pass
    
    def select_action(self, agent_state):
        """
        Selecciona acciones y produce actions_dict con forma fija.
        
        Args:
            agent_state: Estado del agente
            
        Returns:
            dict: actions_dict con estructura:
                - vars_values: {kp_<var_obj>, ki_<var_obj>, kd_<var_obj>}
                - vars_decision: {action_kp_<var_obj>, action_ki_<var_obj>, action_kd_<var_obj>}
                - vars_delta: {delta_gain}
        """
        pass
    
    def learn(self, prev_state, next_state, actions_dict, reward_info, terminated):
        """
        Aplica aprendizaje con reward_info.
        
        Args:
            prev_state: Estado previo del agente
            next_state: Estado siguiente del agente
            actions_dict (dict): Acciones tomadas
            reward_info (dict): Información de recompensa
            terminated (bool): Si el episodio terminó
            
        Returns:
            dict: learn_info con métricas de aprendizaje
        """
        pass
    
    def get_learning_metrics(self):
        """
        Expone estado interno y métricas de aprendizaje para logging.
        
        Returns:
            dict: Métricas de aprendizaje
        """
        pass
