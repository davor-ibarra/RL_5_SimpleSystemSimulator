"""
cart_pole_dynamic_system.py

Responsabilidad:
Implementar el sistema dinámico cart-pole específico.
Mapear el vector interno a llaves canónicas del sistema.
Incluir variables derivadas necesarias para logging (p. ej., cart_force).
"""

from dynamic_system.dynamic_system_base import DynamicSystemBase


class CartPoleDynamicSystem(DynamicSystemBase):
    """
    Implementación específica del sistema cart-pole.
    
    Llaves canónicas del estado:
    - cart_position
    - cart_velocity
    - pendulum_angle
    - pendulum_velocity
    - cart_force
    """
    
    def __init__(self, config):
        """
        Inicializa el sistema cart-pole con configuración directa.
        
        Args:
            config (dict): Configuración del sistema cart-pole
        """
        super().__init__(config)
    
    def reset_episode(self, episode_id, config):
        """
        Resetea el sistema cart-pole al inicio de un episodio.
        
        Args:
            episode_id (int): Identificador del episodio
            config (dict): Configuración para el reset
        """
        pass
    
    def step(self, u_total, dt_sec):
        """
        Integra el sistema cart-pole un paso con u_total.
        
        Args:
            u_total (float): Fuerza aplicada al carro
            dt_sec (float): Paso de tiempo en segundos
        """
        pass
    
    def get_current_state_dict(self):
        """
        Expone el estado actual como dynamic_system_state_dict.
        
        Returns:
            dict: Estado actual con llaves canónicas:
                - cart_position
                - cart_velocity
                - pendulum_angle
                - pendulum_velocity
                - cart_force
        """
        pass