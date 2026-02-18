"""
pid_controller.py

Responsabilidad:
Implementar un PID directo por var_obj.
Calcular y exponer sus variables internas (error, derivada, integral, acción, delta acción).
Aceptar correcciones anti-windup desde la lógica global de ControllerBase.
"""


class PIDController:
    """
    Controlador PID para una variable objetivo específica.
    
    Llaves generadas por convención usando var_obj:
    - error_<var_obj>
    - derivative_error_<var_obj>
    - integral_error_<var_obj>
    - control_action_<var_obj>
    - delta_control_action_<var_obj>
    - is_saturated_<var_obj>
    - saturation_proportion_<var_obj>
    """
    
    def __init__(self, controller_name, config_main):
        """
        Inicializa el controlador PID con configuración directa.
        
        Args:
            controller_name (str): Nombre del controlador
            config_main (dict): Configuración del controlador PID
        """
        # Configuración principal
        self.config_main = config_main
        self.controller_name = controller_name
        self.controller_config = config_main['controller_base']['controllers'][controller_name]
        
        # Objetivo controlado
        self.var_obj = self.controller_config['params']['name_objective_var']
        self.setpoint_raw = self.controller_config['params']['setpoint']
        self.error_is_setpoint_minus_pv = self.controller_config['params']['error_is_setpoint_minus_pv']
        
        # Normalización de setpoint (usa config del dynamic_system si está habilitado)
        self.setpoint_normalized = self._normalize_setpoint()
        
        # Ganancias iniciales
        self.kp = self.controller_config['initial_conditions']['kp']
        self.ki = self.controller_config['initial_conditions']['ki']
        self.kd = self.controller_config['initial_conditions']['kd']
        
        # Estado interno del PID
        self.error = 0.0
        self.derivative_error = 0.0
        self.integral_error = 0.0
        self.prev_error = 0.0

        # First step flag
        self.first_step = True
        
        # Acción de control
        self.control_action = 0.0
        self.prev_control_action = 0.0
        self.delta_control_action = 0.0
        
        # Estado de saturación (será actualizado por controller_base)
        self.is_saturated = False
        self.u_eff = 0.0
        self.prev_u_eff = 0.0
        self.delta_u_eff = 0.0

        # Anti-windup
        self.antiwindup_enabled = self.controller_config['params']['anti_windup']['enabled']
        self.antiwindup_method = self.controller_config['params']['anti_windup']['method']
        self.back_calculation_betha = self.controller_config['params']['anti_windup']['back_calculation_betha']
    
    def _normalize_setpoint(self):
        """
        Normaliza el setpoint usando la configuración de normalización del dynamic_system.
        Si la normalización está habilitada, transforma el setpoint de escala física a [-1, 1].
        
        Returns:
            float: Setpoint normalizado (o raw si normalización deshabilitada)
        """
        normalization_config = self.config_main['dynamic_system']['state_normalization']
        
        if not normalization_config['enabled']:
            return self.setpoint_raw
        
        ranges_params = normalization_config['ranges_params']
        output_limits = normalization_config['output_limits']
        
        if self.var_obj not in ranges_params:
            return self.setpoint_raw
        
        in_min, in_max = ranges_params[self.var_obj]
        out_min, out_max = output_limits
        
        # Normalización lineal: [in_min, in_max] -> [out_min, out_max]
        normalized = out_min + (self.setpoint_raw - in_min) * (out_max - out_min) / (in_max - in_min)
        return normalized
    
    def reset_episode(self):
        """
        Resetea el estado del controlador PID al inicio de un episodio.
        """
        self.kp = self.controller_config['initial_conditions']['kp']
        self.ki = self.controller_config['initial_conditions']['ki']
        self.kd = self.controller_config['initial_conditions']['kd']
        self.error = 0.0
        self.derivative_error = 0.0
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.control_action = 0.0
        self.prev_control_action = 0.0
        self.delta_control_action = 0.0
        self.is_saturated = False
        self.saturation_proportion = 0.0
        self.first_step = True
    
    def compute(self, dynamic_state_dict, dt_sec):
        """
        Calcula la acción de control PID.
        La saturación y anti-windup se manejan a nivel global en ControllerBase.
        
        Args:
            dynamic_state_dict (dict): Estado actual del sistema dinámico (normalizado)
            dt_sec (float): Paso de tiempo en segundos
            
        Returns:
            float: Acción de control
        """
        # Obtener valor actual de la variable objetivo (normalizado)
        current_value = dynamic_state_dict[self.var_obj]
        
        # Calcular error usando setpoint normalizado
        self.error = self._calculate_error(current_value)
        
        # Calcular derivada del error
        if not self.first_step:
            self.derivative_error = (self.error - self.prev_error) / dt_sec
        else:
            self.derivative_error = 0.0
        
        # Calcular integral del error (acumulación)
        self.integral_error += self.error * dt_sec
        
        # Calcular acción de control PID (sin saturar)
        self.prev_control_action = self.control_action
        self.control_action = (
            self.kp * self.error +
            self.ki * self.integral_error +
            self.kd * self.derivative_error
        )
        
        # Calcular delta de acción de control
        self.delta_control_action = self.control_action - self.prev_control_action
        
        # Actualizar error previo para próxima iteración
        self.prev_error = self.error
        
        return self.control_action
    
    def _calculate_error(self, current_value):
        """
        Calcula el error usando setpoint normalizado.
        
        Args:
            current_value (float): Valor actual de la variable objetivo (normalizado)
            
        Returns:
            float: Error
        """
        if self.error_is_setpoint_minus_pv:
            return current_value - self.setpoint_normalized
        else:
            return self.setpoint_normalized - current_value

    def _apply_antiwindup_correction(self, correction, dt_sec):
        """
        Acepta correcciones anti-windup globales desde ControllerBase.
        Se aplica cuando hay saturación global y se reparte proporcionalmente.
        
        Args:
            correction (float): Corrección anti-windup (error de saturación ponderado)
            dt_sec (float): Paso de tiempo (para métodos back_calculation futuros)
        """
        if self.antiwindup_method == 'conditional':
            self._apply_conditional_antiwindup_correction(correction)
    
    def _apply_conditional_antiwindup_correction(self, e_sat):
        """
        Aplica corrección antiwindup al integrador.
        La corrección viene en unidades de [control_action]; se divide por ki
        para convertir a unidades de [error × tiempo].
        
        Args:
            e_sat (float): Error de saturación (u_raw - u_sat)
        """
        if self.ki > 0:
            integral_correction = e_sat / self.ki
            self.integral_error -= integral_correction
    
    
    # --- Public methods ---

    def update_gains(self, kp, ki, kd):
        """
        Actualiza las ganancias del controlador PID.
        
        Args:
            kp (float): Ganancia proporcional
            ki (float): Ganancia integral
            kd (float): Ganancia derivativa
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def get_current_controller_gains(self):
        """
        Retorna las ganancias del controlador PID.
        
        Returns:
            dict: {'kp': kp, 'ki': ki, 'kd': kd}
        """
        return {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}

    def return_state_to_controller(self, u_total_saturated, u_total_raw, is_saturated_global, dt_sec):
        """
        Retorna el estado del controlador base a cada controlador.
        """
        self.is_saturated = is_saturated_global
        
        # Calcular acción de control efectiva
        if u_total_raw != 0:
            scaling_factor = u_total_saturated / u_total_raw
        else:
            scaling_factor = 1.0
            
        self.u_eff = scaling_factor * self.control_action
        self.delta_u_eff = self.u_eff - self.prev_u_eff
        self.prev_u_eff = self.u_eff
        
        # Calcular error de saturación
        e_sat = self.u_eff - self.control_action

        # Aplicar corrección antiwindup
        if self.is_saturated and self.antiwindup_enabled:
            self._apply_antiwindup_correction(e_sat, dt_sec)
    
    def get_controller_record(self):
        """
        Expone el estado interno del controlador.
        
        Returns:
            dict: Registro de estado con llaves canónicas por var_obj
        """
        return {
            f'error_{self.var_obj}': self.error,
            f'prev_error_{self.var_obj}': self.prev_error,
            f'delta_error_{self.var_obj}': self.error - self.prev_error,
            f'derivative_error_{self.var_obj}': self.derivative_error,
            f'integral_error_{self.var_obj}': self.integral_error,
            f'control_action_{self.var_obj}': self.control_action,
            f'prev_control_action_{self.var_obj}': self.prev_control_action,
            f'delta_control_action_{self.var_obj}': self.delta_control_action,
            f'u_eff_{self.var_obj}': self.u_eff,
            f'prev_u_eff_{self.var_obj}': self.prev_u_eff,
            f'delta_u_eff_{self.var_obj}': self.delta_u_eff,
            f'kp_{self.var_obj}': self.kp,
            f'ki_{self.var_obj}': self.ki,
            f'kd_{self.var_obj}': self.kd
        }
