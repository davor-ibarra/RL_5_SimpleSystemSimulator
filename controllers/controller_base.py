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
        self.saturation_excess = 0.0
    
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
        self.saturation_excess = 0.0
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
        self.u_total_raw = sum(individual_actions.values())
        
        # 3. Aplicar saturación global (solo si está habilitada)
        if self.global_actuator_enabled:
            self.u_total_saturated = max(self.u_min_global, min(self.u_max_global, self.u_total_raw))
        else:
            self.u_total_saturated = self.u_total_raw
        
        # 4. Calcular, guardar y actualizar estados para record
        self.prev_u_total = self.u_total
        self.u_total = self.u_total_saturated
        self.delta_u_total = self.u_total - self.prev_u_total
        self.is_saturated_global = self.global_actuator_enabled and (self.u_total_raw != self.u_total_saturated) and (self.u_total_raw != 0.0)
        self.saturation_excess = abs(self.u_total_raw - self.u_total_saturated)
        self._return_state_to_controllers()
        
        return self.u_total_saturated
    
    def _return_state_to_controllers(self):
        """
        Retorna el estado del controlador base a cada controlador.
        Calcula señales locales de control desde la acción global saturada.
        """
        if self.u_total_raw != 0.0:
            scaling_factor = self.u_total_saturated / self.u_total_raw
        else:
            scaling_factor = 1.0

        if self.u_total_raw > 0.0:
            u_raw_sign = 1.0
        elif self.u_total_raw < 0.0:
            u_raw_sign = -1.0
        else:
            u_raw_sign = 0.0

        if self.u_total_saturated > 0.0:
            u_saturated_sign = 1.0
        elif self.u_total_saturated < 0.0:
            u_saturated_sign = -1.0
        else:
            u_saturated_sign = 0.0

        support_total = 0.0
        if u_saturated_sign != 0.0:
            for controller in self.controllers.values():
                support_total += max(0.0, u_saturated_sign * controller.control_action)

        saturation_support_total = 0.0
        if self.is_saturated_global and u_raw_sign != 0.0:
            for controller in self.controllers.values():
                saturation_support_total += max(0.0, u_raw_sign * controller.control_action)

        positive_demand = 0.0
        negative_demand = 0.0
        for controller in self.controllers.values():
            positive_demand += max(0.0, controller.control_action)
            negative_demand += max(0.0, -controller.control_action)
        cancelled_demand = min(positive_demand, negative_demand)

        for controller in self.controllers.values():
            control_action_eff = scaling_factor * controller.control_action

            if support_total > 0.0:
                support = max(0.0, u_saturated_sign * controller.control_action)
                u_alloc = self.u_total_saturated * support / support_total
            else:
                u_alloc = 0.0

            u_conflict = self._compute_cancelled_conflict(
                controller.control_action,
                positive_demand,
                negative_demand,
                cancelled_demand
            )
            saturation_proportion = self._compute_local_saturation_proportion(
                controller.control_action,
                u_raw_sign,
                saturation_support_total
            )
            is_saturated_local = saturation_proportion > 0.0

            controller.return_state_to_controller(
                control_action_eff,
                u_alloc,
                u_conflict,
                is_saturated_local,
                saturation_proportion,
                self.dt_sec
            )

    def _compute_cancelled_conflict(self, control_action, positive_demand, negative_demand, cancelled_demand):
        if cancelled_demand <= 0.0 or control_action == 0.0:
            return 0.0

        if control_action > 0.0 and positive_demand > 0.0:
            return cancelled_demand * control_action / positive_demand

        if control_action < 0.0 and negative_demand > 0.0:
            return cancelled_demand * (-control_action) / negative_demand

        return 0.0

    def _compute_local_saturation_proportion(self, control_action, u_raw_sign, saturation_support_total):
        if not self.is_saturated_global or u_raw_sign == 0.0 or saturation_support_total <= 0.0:
            return 0.0

        support = max(0.0, u_raw_sign * control_action)
        if support <= 0.0:
            return 0.0

        return min(1.0, self.saturation_excess / saturation_support_total)
    
    def get_records(self):
        """
        Retorna dict plano con llaves canónicas para el MetricCollector.
        Fusiona señales globales + señales per-controller en un solo dict.
        
        Returns:
            dict: Registro plano step-level del controlador
        """
        records = {}
        
        # Señales globales
        records['u_total'] = self.u_total
        records['u_total_raw'] = self.u_total_raw
        records['u_total_saturated'] = self.u_total_saturated
        records['is_saturated_global'] = self.is_saturated_global
        records['saturation_excess'] = self.saturation_excess
        records['prev_u_total'] = self.prev_u_total
        records['delta_u_total'] = self.delta_u_total
        
        # Contribuciones y records per-controller (usando var_obj como sufijo)
        for controller_name, controller in self.controllers.items():
            records[f'u_contrib_{controller.var_obj}'] = controller.control_action
            
            # get_controller_record() ya retorna llaves planas con var_obj
            controller_record = controller.get_controller_record()
            records.update(controller_record)
        
        return records
