"""
metric_processing.py

Responsabilidad:
Procesar series de steps en processed_metrics_dict del intervalo.
Recibe flat_step_records (lista de dicts planos con llaves canónicas)
y produce métricas agregadas y normalizadas según config declarativa.

Entrada:
flat_step_records = [
    {error_<var_obj>: val, control_action_<var_obj>: val, ...},  # step 0
    {error_<var_obj>: val, control_action_<var_obj>: val, ...},  # step 1
    ...
]

Salida (3 secciones fijas):
processed_metrics_dict = {
    "reward_component": {
        "<var_obj>": {L_e, L_edot, L_I, L_u, L_delta_u},
        "global_vars": {...}  # Solo si está en config
    },
    "extra_reward_component": {
        "error_<var_obj>": [serie completa],
        "control_action_<var_obj>": [serie completa],
    },
    "metrics_info": {}  # Placeholder para métricas futuras
}
"""

import numpy as np


class MetricProcessing:
    """
    Procesa series de pasos en métricas agregadas de intervalo.
    Aplica normalización declarativa desde config.
    """
    
    def __init__(self, config_main):
        """
        Inicializa el procesador de métricas.
        
        Args:
            config_main (dict): Configuración principal
        """
        self.config_main = config_main
        
        # Precomputar mapping var_obj → controller_name desde config
        self.var_obj_to_controller = self._build_var_obj_mapping()
        
        # Extraer config de normalización
        self.normalization_config = self._extract_normalization_config()
    
    def _build_var_obj_mapping(self):
        """
        Construye mapping var_obj → controller_name desde config.
        
        Returns:
            dict: {var_obj: controller_name}
        """
        mapping = {}
        controllers_config = self.config_main['controller_base']['controllers']
        
        for controller_name, controller_cfg in controllers_config.items():
            var_obj = controller_cfg['params']['name_objective_var']
            mapping[var_obj] = controller_name
        
        return mapping
    
    def _extract_normalization_config(self):
        """
        Extrae configuración de normalización desde config.
        
        Returns:
            dict: Config de normalización con enabled, output_limits, params
        """
        reward_calculation = self.config_main['reward_base']['reward_calculation']
        metric_processing_config = reward_calculation['metric_processing']
        norm_config = metric_processing_config['normalization']
        
        return {
            'enabled': norm_config['enabled'],
            'output_limits': norm_config['output_limits'],
            'params': norm_config['params']
        }
    
    def reset_episode(self):
        """
        Resetea el procesador al inicio de un episodio.
        """
        pass
    
    def process_interval_metrics(self):
        """
        [PLACEHOLDER] -> Debería traer los step_records del intervalo desde el collector con método get_step_records()
        Procesa las series de pasos del intervalo en métricas normalizadas.

            
        Returns:
            dict: processed_metrics_dict con estructura de 3 secciones
                - reward_component: {var_obj: {L_e, L_edot, ...}}
                - extra_reward_component: {error_<var_obj>: [serie], ...}
                - metrics_info: {}
        """
    
    def _process_var_obj_metrics(self, step_records, var_obj):
        """
        Procesa métricas para un var_obj específico.
        Mapea señales del PID a features (L_e, L_edot, L_I, L_u, L_delta_u).
        Aplica normalización solo si enabled=true en config.
        
        Args:
            step_records (list[dict]): Lista de dicts planos por step
            var_obj (str): Variable objetivo
            
        Returns:
            dict: Features {L_e, L_edot, L_I, L_u, L_delta_u}
        """
        if not step_records:
            return {}
        
        # Mapeo de feature_key → signal_key (llaves canónicas del PID)
        signal_mapping = {
            'e': f'error_{var_obj}',
            'edot': f'derivative_error_{var_obj}',
            'I': f'integral_error_{var_obj}',
            'u': f'control_action_{var_obj}',
            'delta_u': f'delta_control_action_{var_obj}'
        }
        
        features = {}
        normalization_enabled = self.normalization_config['enabled']
        norm_params = self.normalization_config['params']
        output_limits = self.normalization_config['output_limits']
        
        # Config de normalización para este var_obj (puede no existir)
        var_norm_params = norm_params[var_obj] if var_obj in norm_params else {}
        
        for feature_key, signal_key in signal_mapping.items():
            # Extraer valores de la serie
            values = self._extract_signal_series(step_records, signal_key)
            
            # Si no hay valores, continuar sin agregar esta feature
            if not values:
                continue
            
            # Obtener config de normalización para esta feature (puede no existir)
            if feature_key in var_norm_params:
                feature_norm_config = var_norm_params[feature_key]
                method = feature_norm_config['method']
                value_range = feature_norm_config['range']
            else:
                # Default si no hay config específica
                method = 'mean_squared'
                value_range = [-1.0, 1.0]
            
            # Agregar según método
            aggregated = self._aggregate_with_method(values, method)
            
            # Normalizar solo si está habilitado
            if normalization_enabled:
                normalized = self._normalize_value(aggregated, value_range, output_limits, method)
                features[f'L_{feature_key}'] = normalized
            else:
                features[f'L_{feature_key}'] = aggregated
        
        return features
    
    def _process_global_vars(self, step_records, global_vars_config):
        """
        Procesa variables globales SOLO si están definidas en config.
        
        Args:
            step_records (list[dict]): Lista de dicts planos por step
            global_vars_config (dict): Config de variables globales
            
        Returns:
            dict: Features globales
        """
        if not step_records:
            return {}
        
        global_metrics = {}
        normalization_enabled = self.normalization_config['enabled']
        output_limits = self.normalization_config['output_limits']
        
        for var_name, var_config in global_vars_config.items():
            # Extraer serie directamente por llave plana
            values = self._extract_signal_series(step_records, var_name)
            
            if not values:
                continue
            
            method = var_config['method']
            value_range = var_config['range']
            
            aggregated = self._aggregate_with_method(values, method)
            
            if normalization_enabled:
                normalized = self._normalize_value(aggregated, value_range, output_limits, method)
                global_metrics[f'L_{var_name}'] = normalized
            else:
                global_metrics[f'L_{var_name}'] = aggregated
        
        return global_metrics
    
    def _extract_raw_series_for_extras(self, step_records):
        """
        Extrae series crudas planas por var_obj para extra_reward_component.
        Incluye la serie completa del intervalo para que bonus/penalty puedan
        evaluar si la condición se cumplió durante el intervalo.
        
        Args:
            step_records (list[dict]): Lista de dicts planos por step
            
        Returns:
            dict: Crudos planos {error_<var_obj>: [lista], ...}
        """
        extras = {}
        
        for var_obj in self.var_obj_to_controller:
            # Extraer error crudo (señal principal para bonus/penalty)
            error_key = f'error_{var_obj}'
            error_series = self._extract_signal_series(step_records, error_key)
            extras[error_key] = error_series
            
            # Extraer control_action crudo (para penalty de esfuerzo)
            action_key = f'control_action_{var_obj}'
            action_series = self._extract_signal_series(step_records, action_key)
            extras[action_key] = action_series
        
        return extras
    
    def _extract_signal_series(self, step_records, signal_name):
        """
        Extrae una serie de valores de una señal por llave plana.
        
        Args:
            step_records (list[dict]): Lista de dicts planos por step
            signal_name (str): Llave canónica de la señal (e.g. error_pendulum_angle)
            
        Returns:
            list: Lista de valores (solo numéricos finitos)
        """
        values = []
        for record in step_records:
            if signal_name in record:
                val = record[signal_name]
                if isinstance(val, (int, float)) and np.isfinite(val):
                    values.append(val)
        return values
    
    def _aggregate_with_method(self, values, method):
        """
        Agrega una lista de valores según el método especificado.
        Método estándar reutilizable para cualquier tipo de variable.
        
        Args:
            values (list): Lista de valores
            method (str): Método de agregación
            
        Returns:
            float: Valor agregado
        """
        arr = np.array(values, dtype=float)
        
        if method == 'mean':
            return float(np.mean(arr))
        elif method == 'mean_squared':
            return float(np.mean(arr ** 2))
        elif method == 'rms':
            return float(np.sqrt(np.mean(arr ** 2)))
        elif method == 'abs':
            return float(np.mean(np.abs(arr)))
        elif method == 'accum':
            return float(np.sum(arr))
        elif method == 'proportion':
            return float(np.mean(arr > 0))
        elif method == 'keep_last':
            return float(arr[-1])
        else:
            # Default: mean_squared
            return float(np.mean(arr ** 2))
    
    def _normalize_value(self, value, value_range, output_limits, method):
        """
        Normaliza un valor al rango de salida especificado.
        
        Args:
            value (float): Valor a normalizar
            value_range (list): [min, max] del rango de entrada
            output_limits (list): [min, max] del rango de salida
            method (str): Método de agregación usado
            
        Returns:
            float: Valor normalizado
        """
        range_min, range_max = value_range
        out_min, out_max = output_limits
        
        # Calcular span según el método
        if method == 'mean_squared':
            # Para mean_squared, el rango de entrada es [0, max(|min|, |max|)^2]
            max_abs = max(abs(range_min), abs(range_max))
            range_span = max_abs ** 2
        elif method == 'rms':
            # Para rms, el rango de entrada es [0, max(|min|, |max|)]
            range_span = max(abs(range_min), abs(range_max))
        elif method == 'abs':
            # Para abs, el rango de entrada es [0, max(|min|, |max|)]
            range_span = max(abs(range_min), abs(range_max))
        elif method == 'proportion':
            # Para proportion, el rango es [0, 1]
            range_span = 1.0
        else:
            # Para mean, accum, keep_last: rango lineal
            range_span = range_max - range_min if range_max != range_min else 1.0
            # Ajustar value a rango [0, span]
            value = abs(value - range_min)
        
        # Normalizar a [0, 1]
        if range_span > 0:
            normalized_01 = min(1.0, abs(value) / range_span)
        else:
            normalized_01 = 0.0
        
        # Mapear a output_limits
        return out_min + normalized_01 * (out_max - out_min)
