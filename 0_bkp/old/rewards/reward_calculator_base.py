"""
reward_calculator_base.py

Responsabilidad:
Definir la interfaz base para calculadores de recompensa.
Calcular la recompensa global como función fija de reward_component.
Calcular extra rewards de forma declarativa a partir de extra_reward_component y metrics_info.
Construir siempre assign_internal_reward_dict con nombres de agente estables.
Incluir métodos públicos estándar para formalizar el traspaso de información de recompensa.
"""


class RewardCalculatorBase:
    """
    Clase base abstracta para calculadores de recompensa.
    Define el contrato público que todos los calculadores deben cumplir.
    """
    
    def __init__(self, config):
        """
        Inicializa el calculador de recompensa con configuración directa.
        
        Args:
            config (dict): Configuración del calculador de recompensa
        """
        pass
    
    def reset_episode(self):
        """
        Resetea el estado del calculador al inicio de un episodio.
        """
        pass
    
    def calculate(self, processed_metrics_dict, actions_dict):
        """
        Calcula la recompensa del intervalo.
        
        Args:
            processed_metrics_dict (dict): Métricas procesadas del intervalo con:
                - reward_component
                - extra_reward_component
                - metrics_info
            actions_dict (dict): Acciones aplicadas en el intervalo
            
        Returns:
            dict: reward_info con estructura:
                - global_interval_reward: float
                - assign_internal_reward_dict: dict {agent_name -> float}
        """
        pass
    
    def _calculate_base_reward(self, reward_component):
        """
        Calcula la recompensa base desde reward_component.
        
        Args:
            reward_component (dict): Componentes de recompensa por controlador
            
        Returns:
            float: Recompensa base calculada
        """
        pass
    
    def _calculate_extra_rewards(self, extra_reward_component, metrics_info):
        """
        Calcula extra rewards de forma declarativa usando la configuración.
        Descubre variables requeridas para extra rewards desde config.
        Evalúa reglas declarativas de rango/banda sin hardcoding de nombres.
        
        Args:
            extra_reward_component (dict): Componentes extra para bonus/penalties
            metrics_info (dict): Información de métricas
            
        Returns:
            float: Suma de extra rewards (bonus/penalties)
        """
        pass
    
    def _build_assign_internal_reward_dict(self, global_reward, agent_names):
        """
        Construye el diccionario de asignación interna de recompensa
        con nombres de agente estables.
        
        Args:
            global_reward (float): Recompensa global del intervalo
            agent_names (list): Lista de nombres de agentes
            
        Returns:
            dict: Diccionario {agent_name -> reward}
        """
        pass
    
    def get_reward_components(self):
        """
        Expone los componentes de recompensa para logging/análisis.
        
        Returns:
            dict: Componentes de recompensa detallados
        """
        pass
