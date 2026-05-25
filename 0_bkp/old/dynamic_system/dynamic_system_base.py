"""
dynamic_system_base.py

Responsabilidad:
Definir una interfaz pública mínima para:
- Reset de episodio
- Step de integración bajo u_total
- Exposición del estado como dynamic_system_state_dict
- Evaluación de terminación
"""


class DynamicSystemBase:
    """
    Clase base abstracta para sistemas dinámicos.
    Define el contrato público que todos los sistemas dinámicos deben cumplir.
    """
    
    def __init__(self, config):
        """
        Inicializa el sistema dinámico con configuración directa.
        
        Args:
            config (dict): Configuración del sistema dinámico
        """
        pass
    
    def reset_episode(self, episode_id, config):
        """
        Resetea el sistema dinámico al inicio de un episodio.
        
        Args:
            episode_id (int): Identificador del episodio
            config (dict): Configuración para el reset
        """
        pass
    
    def step(self, u_total, dt_sec):
        """
        Integra el sistema dinámico un paso con u_total.
        
        Args:
            u_total (float): Acción de control total
            dt_sec (float): Paso de tiempo en segundos
        """
        pass
    
    def get_current_state_dict(self):
        """
        Expone el estado actual como dynamic_system_state_dict.
        Mapea el vector interno a llaves canónicas del sistema.
        
        Returns:
            dict: Estado actual del sistema dinámico
        """
        pass

