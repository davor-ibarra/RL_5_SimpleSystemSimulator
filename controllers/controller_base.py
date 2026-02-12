"""
controller_base.py

Responsabilidad:
Establecer el contrato común de controladores y contener la lógica para:
- Sumar múltiples controladores en un u_total
- Aplicar saturación global de actuador
- Informar a cada controlador su corrección anti-windup (método Conditional)
"""

import importlib


class ControllerBase:
    """
    Clase base que gestiona la suma de controladores y anti-windup global.
    
    Responsabilidades:
    - Suma de múltiples controladores en u_total
    - Saturación global de actuador
    - Distribución de correcciones anti-windup (solo si hay saturación)
    """
    
    def __init__(self, config_main):
        """
        Inicializa el controlador base con configuración directa.
        
        Args:
            config_main (dict): Configuración principal
        """
        self.config_main = config_main
        
        # Parámetros de simulación
        self.dt_sec = config_main['simulation']['dt_sec']
        
        # Instanciar controladores específicos (instanciación dinámica desde config)
        self.controllers = {}
        controllers_config = config_main['controller_base']['controllers']
        for controller_name, controller_config in controllers_config.items():
            module_path = controller_config['module_path']
            class_name = controller_config['class_name']
            
            module = importlib.import_module(module_path)
            controller_class = getattr(module, class_name)
            self.controllers[controller_name] = controller_class(controller_name, self.config_main)
        
        # Extraer límites de saturación global
        self.global_actuator_enabled = config_main['controller_base']['global_actuator_enabled']
        self.u_min_global = config_main['controller_base']['global_actuator_limits'][0]
        self.u_max_global = config_main['controller_base']['global_actuator_limits'][1]
        
        # Estado interno
        self.u_total = 0.0
        self.u_total_raw = 0.0
        self.u_total_saturated = 0.0
        self.prev_u_total = 0.0
        self.delta_u_total = 0.0
        self.is_saturated_global = False
    
    def reset_episode(self):
        """
        Resetea el estado del controlador base al inicio de un episodio.
        """
        self.u_total = 0.0
        self.u_total_raw = 0.0
        self.u_total_saturated = 0.0
        self.prev_u_total = 0.0
        self.delta_u_total = 0.0
        self.is_saturated_global = False
        for controller_instance in self.controllers.values():
            controller_instance.reset_episode()
    
    def compute_control(self, dynamic_state_dict):
        """
        Calcula la acción total de control y el snapshot step-level del controlador.
        
        Args:
            dynamic_state_dict (dict): Estado actual del sistema dinámico (normalizado)
            
        Returns:
            tuple: (u_total: float, controller_state_record: dict)
        """
        # 1. Calcular acción de control de cada controlador
        individual_actions = {}
        for controller_name in self.controllers.keys():
            controller = self.controllers[controller_name]
            action = controller.compute(dynamic_state_dict, self.dt_sec)
            individual_actions[controller_name] = action
        
        # 2. Sumar todas las acciones para obtener u_total sin saturar
        u_total_raw = sum(individual_actions.values())
        
        # 3. Aplicar saturación global (solo si está habilitada)
        if self.global_actuator_enabled:
            u_total_saturated = max(self.u_min_global, min(self.u_max_global, u_total_raw))
        else:
            u_total_saturated = u_total_raw
        
        # 4. Calcular delta de u_total y guardar estados para record
        self.prev_u_total = self.u_total
        self.u_total = u_total_saturated
        self.u_total_raw = u_total_raw
        self.u_total_saturated = u_total_saturated
        self.delta_u_total = self.u_total - self.prev_u_total
        
        # 5. Anti-windup Conditional: solo si hay saturación Y está habilitado
        self.is_saturated_global = self.global_actuator_enabled and (u_total_raw != u_total_saturated)
        if self.is_saturated_global:
            saturation_error = u_total_raw - u_total_saturated
            self._apply_antiwindup_corrections(individual_actions, saturation_error)
        
        # 6. Actualizar estado de saturación en cada controlador
        for controller_name in self.controllers.keys():
            self.controllers[controller_name].set_saturation_status(self.is_saturated_global, u_total_raw, u_total_saturated)
        
        # 7. Construir registro del estado del controlador
        controller_state_record = self._build_controller_state_record(
            u_total_saturated, individual_actions, self.controllers, self.controllers.keys()
        )
        
        return u_total_saturated, controller_state_record
    
    def _apply_antiwindup_corrections(self, individual_actions, saturation_error):
        """
        Distribuye correcciones anti-windup a cada controlador.
        Corrección proporcional a la contribución de cada controlador.
        Solo aplica si el controlador tiene anti-windup habilitado.
        
        Args:
            individual_actions (dict): Acciones individuales por controlador
            saturation_error (float): Error de saturación (u_raw - u_saturated)
        """
        total_contribution = sum(abs(individual_actions[cn]) for cn in self.controllers.keys())
        
        if total_contribution == 0:
            return
        
        for controller_name in self.controllers.keys():
            controller = self.controllers[controller_name]
            
            # Solo aplicar si el controlador tiene anti-windup habilitado
            if not controller.antiwindup_enabled:
                continue
            
            contribution_ratio = abs(individual_actions[controller_name]) / total_contribution
            correction = saturation_error * contribution_ratio
            controller.apply_antiwindup_correction(correction, self.dt_sec)
    
    def _build_controller_state_record(self, u_total, individual_actions, 
                                         controllers, controller_names):
        """
        Construye el registro del estado del controlador para el step actual.
        
        Args:
            u_total (float): Acción de control total
            individual_actions (dict): Acciones individuales por controlador
            controllers (dict): Diccionario {controller_name: Controller}
            controller_names (list): Lista ordenada de nombres de controladores
            
        Returns:
            dict: Registro del estado del controlador
        """
        # Construir bloque global con todas las llaves del template
        global_controller = {
            'u_total': u_total,
            'u_total_raw': self.u_total_raw,
            'u_total_saturated': self.u_total_saturated,
            'is_saturated_global': self.is_saturated_global,
            'prev_u_total': self.prev_u_total,
            'delta_u_total': self.delta_u_total
        }
        
        # Añadir contribuciones por controlador
        for controller_name in controller_names:
            global_controller[f'u_contrib_{controller_name}'] = individual_actions[controller_name]
        
        # Construir record completo
        record = {'global_controller': global_controller}
        
        # Agregar estado de cada controlador individual
        for controller_name in controller_names:
            controller = controllers[controller_name]
            record[controller_name] = controller.get_controller_record()
        
        return record
