"""
dynamic_system_base.py

Responsabilidad:
Orquestador del sistema dinámico que maneja:
- Reset de episodio (delega a implementación específica)
- Integración del sistema dinámico específico (odeint)
- Evaluación de terminación (flags desde config)
- Exposición del estado canónico

El sistema específico implementa: ecuaciones de movimiento, conversión de 
acción de control a fuerza/actuador, normalización de estado, y mapeo a dict.
"""

import importlib
import numpy as np
from scipy.integrate import odeint


class DynamicSystemBase:
    """
    Orquestador del sistema dinámico.
    Maneja orquestación de reset, integración, terminación y exposición de estado.
    """
    
    def __init__(self, config_main):
        """
        Inicializa el orquestador con sistema específico.
        Args:
            config_main (dict): Configuración del sistema
        """
        self.config_main = config_main

        # Instanciar sistema específico (instanciación dinámica desde config)
        dynamic_system_config = self.config_main['dynamic_system']
        module_path = dynamic_system_config['module_path']
        class_name = dynamic_system_config['class_name']
        module = importlib.import_module(module_path)
        system_class = getattr(module, class_name)
        self.dynamic_system_impl = system_class(config_main)
        
        # Estado actual: físico (array y dict) y normalizado (dict)
        self.current_state_arr = None
        self.current_state_dict = None
        self.current_state_norm_dict = None
        self.dynamic_system_state_name_idx_dict = self.dynamic_system_impl.state_name_idx_dict
        
        # Tiempo actual
        self.current_time = 0.0
        
        # Extraer configuración de terminación
        self.termination_config = self.config_main['dynamic_system']['termination_conditions']
        self.var_name_boundary_limits = self.termination_config['boundary_constraint']
        self.var_name_stabilization_limits = self.termination_config['stabilization_criteria']

        # Flags de terminación
        self.terminated = False
        self.termination_reason = ""

        # Configuración de corrección de sistema de referencia
        self.reference_system_correction_config = self.config_main['dynamic_system']['reference_system_correction']
    
    def reset_episode(self):
        """
        Resetea el sistema dinámico al inicio de un episodio.
        
        Returns:
            None (actualiza estado interno)
        """
        # Resetear tiempo y flags de terminación
        self.current_time = 0.0
        self.terminated = False
        self.termination_reason = ""
        
        # Resetear estado interno del sistema específico (retorna np.array)
        self.current_state_arr = self.dynamic_system_impl.reset_episode()
        
        # Convertir array físico a dict
        self.current_state_dict = self.get_dynamic_system_state_to_dict(self.current_state_arr)

        # Normalizar estado y convertir a dict
        current_state_norm_arr = self.dynamic_system_impl.normalize_state(self.current_state_arr)
        self.current_state_norm_dict = self.get_dynamic_system_state_to_dict(current_state_norm_arr)
    
    def step(self, u_total, dt_sec):
        """
        Integra el sistema dinámico un paso con u_total.
        
        Args:
            u_total (float): Acción de control total normalizada [-1, 1]
            dt_sec (float): Paso de tiempo en segundos
        """
        dynamic_system_params_record = {}
        # Delegar conversión de acción a fuerza/actuador al sistema específico
        force = self.dynamic_system_impl.apply_u_total(u_total)
        
        # Puntos de integración
        time_points = [self.current_time, self.current_time + dt_sec]
        
        # Integrar usando odeint (delega ecuaciones al sistema específico)
        next_state_arr = odeint(
            self.dynamic_system_impl.dynamics,
            self.current_state_arr,
            time_points,
            args=(force,)
        )[-1]
        
        # Corregir sistema de referencia (delega al sistema específico)
        if self.reference_system_correction_config['enabled']:
            vars_correction = self.reference_system_correction_config['vars_correction']
            vars_correction_idx = [self.dynamic_system_state_name_idx_dict[var_name] for var_name in vars_correction]
            next_state_arr[vars_correction_idx] = self.dynamic_system_impl.reference_system_correction(next_state_arr[vars_correction_idx])
        
        # Normalizar estado (delega al sistema específico)
        current_state_norm_arr = self.dynamic_system_impl.normalize_state(next_state_arr)
        
        # Actualizar estado y tiempo
        self.current_state_arr = next_state_arr
        self.current_time += dt_sec

        # Actualizar estado físico (dict)
        self.current_state_dict = self.get_dynamic_system_state_to_dict(self.current_state_arr)
        
        # Actualizar estado normalizado (dict)
        self.current_state_norm_dict = self.get_dynamic_system_state_to_dict(current_state_norm_arr)
        
        # Construir registro con llaves claras: raw_ para físico, norm_ implícito en retorno principal
        dynamic_system_params_record['raw_state'] = self.current_state_dict
        dynamic_system_params_record['params'] = self.get_params_dict()

        return self.current_state_norm_dict, dynamic_system_params_record
    
    def check_termination(self):
        """
        Evalúa condición de terminación usando valores físicos internos.
        
        Returns:
            tuple: (terminated: bool, termination_reason: str)
        """
        # Usar estado físico actual para evaluar terminación
        physical_state = self.current_state_dict
        
        # 1) Boundary: basta que UNA variable salga
        for var_name, limits in self.var_name_boundary_limits.items():
            value = physical_state[var_name]
            if value < limits[0] or value > limits[1]:
                self.terminated = True
                self.termination_reason = "limit_exceeded"
                return True, self.termination_reason

        # 2) Stabilization: TODAS deben cumplir
        for var_name, limits in self.var_name_stabilization_limits.items():
            value = physical_state[var_name]
            if value < limits[0] or value > limits[1]:
                return False, ""

        # Si llega aquí, todas están estabilizadas
        if self.var_name_stabilization_limits:
            self.terminated = True
            self.termination_reason = "stabilization_success"
            return True, self.termination_reason

        return False, ""

    def get_dynamic_system_state_to_dict(self, current_state_arr):
        """
        Expone el estado actual como dynamic_system_state_dict.
        
        Returns:
            dict: Estado actual del sistema dinámico en formato dict
        """
        dynamic_system_state_dict = {}
        for var_name, idx in self.dynamic_system_state_name_idx_dict.items():
            dynamic_system_state_dict[var_name] = current_state_arr[idx]
        return dynamic_system_state_dict

    def get_params_dict(self):
        """
        Expone parámetros adicionales del sistema (fuerza, etc.).
        Delega al sistema específico para mapear estado a dict.
        
        Returns:
            dict: Parámetros del sistema dinámico
        """
        return self.dynamic_system_impl.get_params_dict()