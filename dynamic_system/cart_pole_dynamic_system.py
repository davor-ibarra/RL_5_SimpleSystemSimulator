"""
cart_pole_dynamic_system.py

Responsabilidad:
Implementar las ecuaciones de movimiento del sistema cart-pole.
Manejar conversión de acción de control a fuerza (actuador específico).
Implementa: ecuaciones dinámicas, normalización de ángulo, y mapeo de estados.
"""

import numpy as np


class CartPoleDynamicSystem:
    """
    Implementación específica del sistema cart-pole invertido.
    
    Llaves canónicas del estado:
    - cart_position
    - cart_velocity
    - pendulum_angle (0 = vertical hacia arriba)
    - pendulum_velocity
    - cart_force
    """
    
    def __init__(self, config_main):
        """
        Inicializa el sistema cart-pole con configuración directa.
        
        Args:
            config_main (dict): Configuración del sistema cart-pole
        """
        self.config_main = config_main
        self.dynamic_system_config = config_main['dynamic_system']
        self.initial_conditions = self.dynamic_system_config['initial_conditions']
        self.state_name_idx_dict = {key: idx for idx, key in enumerate(self.initial_conditions.keys())}

        # Parámetros físicos (alineados con nombres del config YAML)
        self.m1 = self.dynamic_system_config['params']['mass_cart_kg']
        self.m2 = self.dynamic_system_config['params']['mass_pendulum_kg']
        self.l_bar = self.dynamic_system_config['params']['l']
        self.g_accel = self.dynamic_system_config['params']['g']
        self.cr_friction = self.dynamic_system_config['params']['cart_friction_coef']
        self.ca_friction = self.dynamic_system_config['params']['pivot_friction_coef']
        
        # Parámetros del actuador
        self.max_torque_nm = self.dynamic_system_config['params']['max_torque_nm']
        self.gear_ratio = self.dynamic_system_config['params']['gear_ratio']
        self.pinion_radius_m = self.dynamic_system_config['params']['pinion_radius_m']
        
        # Configuración de normalización de estado
        self.normalization_config = self.dynamic_system_config['state_normalization']
        self.normalization_enabled = self.normalization_config['enabled']
        self.output_limits = self.normalization_config['output_limits']
        self.ranges_params = self.normalization_config['ranges_params']
        
        # Estado del actuador
        self.cart_force = 0.0
        
        # Validación
        if not (self.m1 > 0 and self.m2 > 0 and self.l_bar > 0 and self.g_accel > 0):
            raise ValueError("Masas, longitud y gravedad deben ser positivas.")
    
    def reset_episode(self):
        """
        Resetea el estado interno del sistema cart-pole.
        
        Returns:
            np.array: Estado inicial del sistema cart-pole
        """
        self.cart_force = 0.0
        return self._build_initial_state(self.initial_conditions)

    def _build_initial_state(self, initial_state_dict):
        """
        Construye el estado inicial desde un diccionario.
        
        Args:
            initial_state_dict (dict): Diccionario con condiciones iniciales
            
        Returns:
            np.array: Estado inicial [cart_position, cart_velocity, pendulum_angle, pendulum_velocity]
        """
        state = np.array([
            initial_state_dict['cart_position'],
            initial_state_dict['cart_velocity'],
            self.reference_system_correction(initial_state_dict['pendulum_angle']),
            initial_state_dict['pendulum_velocity']
        ], dtype=float)
        
        return state
    
    def apply_u_total(self, u_total):
        """
        Convierte acción de control total normalizada a fuerza del actuador.
        
        Args:
            u_total (float): Acción de control total normalizada [-1, 1]
            
        Returns:
            float: Fuerza aplicada al carro
        """
        # Convertir acción normalizada a torque y luego a fuerza
        tau = u_total * self.max_torque_nm
        self.cart_force = (self.gear_ratio * tau) / self.pinion_radius_m
        return self.cart_force
    
    def dynamics(self, state_vector, time_t, force):
        """
        Ecuaciones de movimiento del cart-pole (para odeint).
        
        Args:
            state_vector (np.array): [cart_position, cart_velocity, pendulum_angle, pendulum_velocity]
            time_t (float): Tiempo actual (requerido por odeint, no usado)
            force (float): Fuerza aplicada al carro
            
        Returns:
            list: Derivadas [dx1_dt, dx2_dt, dx3_dt, dx4_dt]
        """
        x1, x2, x3, x4 = state_vector
        
        # Precalcular seno y coseno
        sin_x3 = np.sin(x3)
        cos_x3 = np.cos(x3)
        
        # Fuerza neta en el carro (fuerza - fricción)
        net_force = force - self.cr_friction * x2
        
        # Denominador común
        common_denom = self.m1 + self.m2 * sin_x3**2
        
        # dx1/dt = velocidad del carro
        dx1_dt = x2
        
        # dx2/dt = aceleración del carro
        dx2_dt = (
            net_force +
            self.m2 * self.l_bar * (x4**2) * sin_x3 -
            self.m2 * self.g_accel * sin_x3 * cos_x3
        ) / common_denom
        
        # dx3/dt = velocidad angular
        dx3_dt = x4
        
        # dx4/dt = aceleración angular del péndulo
        numerator_dx4 = (
            -net_force * cos_x3 +
            (self.m1 + self.m2) * self.g_accel * sin_x3 -
            self.m2 * self.l_bar * (x4**2) * sin_x3 * cos_x3 -
            self.ca_friction * x4
        )
        dx4_dt = numerator_dx4 / (self.l_bar * common_denom)
        
        return [dx1_dt, dx2_dt, dx3_dt, dx4_dt]
    
    def reference_system_correction(self, angle_values):
        """
        Normaliza el ángulo del péndulo a [-π, π].
        
        Args:
            angle_values (np.array or float): Valor(es) de ángulo a corregir
            
        Returns:
            np.array or float: Valor(es) corregido(s)
        """
        return (angle_values + np.pi) % (2 * np.pi) - np.pi
    
    def normalize_state(self, state_arr):
        """
        Normaliza el vector de estado según ranges_params y output_limits del config.
        
        Args:
            state_arr (np.array): Estado físico [cart_position, cart_velocity, pendulum_angle, pendulum_velocity]
            
        Returns:
            np.array: Estado normalizado según output_limits (ej: [-1, 1])
        """
        if not self.normalization_enabled:
            return state_arr.copy()
        
        normalized_arr = np.zeros_like(state_arr)
        out_min, out_max = self.output_limits
        
        for var_name, idx in self.state_name_idx_dict.items():
            value = state_arr[idx]
            
            if var_name in self.ranges_params:
                in_min, in_max = self.ranges_params[var_name]
                # Normalización lineal: [in_min, in_max] -> [out_min, out_max]
                normalized_arr[idx] = out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
            else:
                # Si no hay rango definido, mantener valor original
                normalized_arr[idx] = value
        
        return normalized_arr
    
    def get_params_dict(self):
        """
        Retorna parámetros adicionales del sistema (no incluidos en state_arr).
        
        Returns:
            dict: Diccionario con parámetros adicionales del sistema
        """
        return {
            'cart_force': self.cart_force
        }