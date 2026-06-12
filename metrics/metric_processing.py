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
        metric_processing_config = self.config_main['reward_base']['reward_calculation']['metric_processing']
        
        # Extraer config de normalización
        self.normalization_config = self._extract_normalization_config()
        self.aggregation_config = metric_processing_config['aggregation']

        # Extraer config de global_vars (puede no existir)
        feature_params = self.aggregation_config['features_params']
        self.global_vars_config = feature_params['global_vars']
        
        # ---------------------------------------------------------
        # Pre-compilar trabajos estáticos para evitar chequeos continuos
        # ---------------------------------------------------------
        self.primitive_signal_normalization_jobs = (
            self._compile_primitive_signal_normalization_jobs()
        )
        self.feature_jobs = self._compile_feature_jobs()
        self.global_jobs = self._compile_global_jobs()
        
        # Estado interno: resultado del último process_interval_metrics
        self._last_processed_metrics = None
    
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
            dict: Config de normalización con enabled, output_limits, signals_params
        """
        reward_calculation = self.config_main['reward_base']['reward_calculation']
        metric_processing_config = reward_calculation['metric_processing']
        norm_config = metric_processing_config['normalization']
        
        return {
            'enabled': norm_config['enabled'],
            'output_limits': norm_config['output_limits'],
            'signals_params': norm_config['signals_params']
        }

    def _compile_primitive_signal_normalization_jobs(self):
        """
        Pre-compila las normalizaciones step-level declaradas en config.
        """
        jobs = []
        if not self.normalization_config['enabled']:
            return jobs

        for var_obj in self.var_obj_to_controller:
            var_config = self.normalization_config['signals_params'][var_obj]
            for destination_key, signal_config in var_config.items():
                jobs.append((
                    signal_config['signal'].format(var_obj=var_obj),
                    destination_key.format(var_obj=var_obj),
                    signal_config['range']
                ))

        return jobs
    
    def _compile_feature_jobs(self):
        """
        Pre-compila la lista de operaciones de agregación y normalización por var_obj 
        para no re-evaluar los diccionarios de configuración en vivo.
        """
        jobs = {}
        feature_params = self.aggregation_config['features_params']
        
        for var_obj in self.var_obj_to_controller:
            jobs[var_obj] = []
            var_norm = feature_params[var_obj]
            
            for f_key, cfg in var_norm.items():
                sig_key = cfg['root_var'].format(var_obj=var_obj)
                
                method = cfg['method']
                v_range = cfg['range']
                dest_agg = cfg['record'].format(var_obj=var_obj, method=method)
                
                # Tupla super rápida: (f_key, sig_key, dest_agg, dest_norm, method, v_range)
                jobs[var_obj].append((
                    f_key, 
                    sig_key, 
                    dest_agg,
                    f'L_{f_key}_{var_obj}', 
                    method, 
                    v_range
                ))
        return jobs
        
    def _compile_global_jobs(self):
        """
        Pre-compila la lista de operaciones de agregación para variables globales.
        """
        jobs = []
        for var_name, var_config in self.global_vars_config.items():
            sig_key = var_config['root_var']
            method = var_config['method']
            v_range = var_config['range']
            jobs.append((
                sig_key,
                f'{sig_key}_{method}',
                f'L_{sig_key}',
                method,
                v_range
            ))
        return jobs
        
    def reset_episode(self):
        """
        Resetea el procesador al inicio de un episodio.
        """
        self._last_processed_metrics = None

    def normalize_step_record(self, step_record):
        """
        Agrega señales primitivas normalizadas al registro step-level.
        """
        if not self.normalization_config['enabled']:
            return step_record

        output_limits = self.normalization_config['output_limits']
        for source_key, destination_key, value_range in self.primitive_signal_normalization_jobs:
            step_record[destination_key] = self._normalize_signed_sample(
                step_record[source_key],
                value_range,
                output_limits
            )

        return step_record
    
    def process_interval_metrics(self, step_records):
        """
        Procesa las series de pasos del intervalo en métricas normalizadas.
        
        Args:
            step_records (list[dict]): Lista de dicts planos por step
                con llaves canónicas (error_<var_obj>, control_action_<var_obj>, ...)
            
        Returns:
            dict: processed_metrics_dict con estructura de 3 secciones
                - reward_component: {L_e_<var_obj>: val, ...} (Aplanado)
                - extra_reward_component: {error_<var_obj>: [serie], ...}
                - metrics_info: {}
        """
        for step_record in step_records:
            self.normalize_step_record(step_record)

        # Centralizar transposición a formato columnar O(N)
        columnar_data = {}
        if step_records:
            keys = step_records[0].keys()
            for key in keys:
                columnar_data[key] = [
                    record[key] for record in step_records
                    if isinstance(record[key], (int, float)) and np.isfinite(record[key])
                ]
        
        # 1. Reward component: métricas agregadas y normalizadas en un solo nivel
        reward_component = {}
        for var_obj in self.var_obj_to_controller:
            var_metrics = self._process_var_obj_metrics(columnar_data, var_obj)
            reward_component.update(var_metrics)
        
        # Global vars (solo si hay config en init)
        if self.global_jobs:
            global_metrics = self._process_global_vars(columnar_data)
            reward_component.update(global_metrics)
        
        # 2. Extra reward component: series crudas y primitivas normalizadas step-level
        extra_reward_component = self._extract_raw_series_for_extras(columnar_data)
        
        # 3. Metrics info (placeholder para métricas futuras)
        metrics_info = {}
        
        # Empaquetar resultado
        processed_metrics_dict = {
            'reward_component': reward_component,
            'extra_reward_component': extra_reward_component,
            'metrics_info': metrics_info
        }
        
        # Almacenar para get_records()
        self._last_processed_metrics = processed_metrics_dict
        
        return processed_metrics_dict
    
    def get_records(self):
        """
        Retorna dict plano con las métricas procesadas del intervalo.
        Al estar previamente aplanadas, solo requiere un update y se exponen directo.
        
        Returns:
            dict: Registro plano interval-level de métricas procesadas
        """
        records = {}
        if self._last_processed_metrics:
            records.update(self._last_processed_metrics['reward_component'])
            
        return records
    
    def _process_var_obj_metrics(self, columnar_data, var_obj):
        """
        Procesa métricas para un var_obj específico desde la data columnar.
        Usa jobs pre-compilados para O(1).
        """
        features = {}
        normalization_enabled = self.aggregation_config['normalizate_agg_too']
        output_limits = self.aggregation_config['output_limits']
        
        # Extraer operaciones pre-compiladas estáticas
        jobs = self.feature_jobs[var_obj]
        
        for (f_key, sig_key, dest_agg, dest_norm, method, v_range) in jobs:
            values = columnar_data[sig_key]
            
            if not values:
                raise ValueError(f"No finite values available for metric signal {sig_key}")
            else:
                aggregated = self._aggregate_with_method(values, method)
                
            features[dest_agg] = aggregated
            if f_key == 'u':
                features[f'u_eff_{var_obj}_{method}'] = aggregated
            elif f_key == 'delta_u':
                features[f'delta_u_eff_{var_obj}_{method}'] = aggregated
            
            if normalization_enabled:
                features[dest_norm] = self._normalize_value(aggregated, v_range, output_limits, method)
            else:
                features[dest_norm] = aggregated
                
        return features
    
    def _process_global_vars(self, columnar_data):
        """
        Procesa variables globales usando los jobs pre-compilados O(1).
        """
        global_metrics = {}
        normalization_enabled = self.aggregation_config['normalizate_agg_too']
        output_limits = self.aggregation_config['output_limits']
        
        for (sig_key, dest_agg, dest_norm, method, v_range) in self.global_jobs:
            values = columnar_data[sig_key]
            
            if not values:
                raise ValueError(f"No finite values available for global metric signal {sig_key}")
                
            aggregated = self._aggregate_with_method(values, method)
            global_metrics[dest_agg] = aggregated
            
            if normalization_enabled:
                global_metrics[dest_norm] = self._normalize_value(aggregated, v_range, output_limits, method)
            else:
                global_metrics[dest_norm] = aggregated
                
        return global_metrics
    
    def _extract_raw_series_for_extras(self, columnar_data):
        """
        Extrae series crudas de la estructura columnar per-var_obj 
        para la matriz de extra rewards.
        
        Args:
            columnar_data (dict): Data transpuesta O(N).
            
        Returns:
            dict: Crudos planos {error_<var_obj>: [lista], ...}.
        """
        extras = {}
        if not columnar_data:
            return extras
            
        signal_prefixes = [
            'error',
            'derivative_error',
            'integral_error',
            'control_action',
            'delta_control_action',
            'control_action_eff',
            'delta_control_action_eff',
            'u_alloc',
            'delta_u_alloc',
            'u_conflict',
            'delta_u_conflict',
            'is_saturated',
            'saturation_proportion',
            'u_eff',
            'delta_u_eff'
        ]

        for var_obj in self.var_obj_to_controller:
            for signal_prefix in signal_prefixes:
                signal_key = f'{signal_prefix}_{var_obj}'
                extras[signal_key] = columnar_data[signal_key]

        for _, destination_key, _ in self.primitive_signal_normalization_jobs:
            extras[destination_key] = columnar_data[destination_key]
            
        # Exponer variables crudas del sistema (e.g. pendulum_velocity_raw)
        # para shaping firmado tipo notebook sin acoplarlas al reward principal.
        for signal_name, signal_values in columnar_data.items():
            if signal_name.endswith('_raw'):
                extras[signal_name] = signal_values

        return extras
    
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
            raise ValueError(f"Unsupported metric aggregation method: {method}")
    
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
        if range_max == range_min:
            raise ValueError(f"Invalid normalization range with equal limits: {value_range}")
        
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
            range_span = range_max - range_min
            # Ajustar value a rango [0, span]
            value = abs(value - range_min)
        
        # Normalizar a [0, 1]
        if range_span > 0:
            normalized_01 = min(1.0, abs(value) / range_span)
        else:
            normalized_01 = 0.0
        
        # Mapear a output_limits
        return out_min + normalized_01 * (out_max - out_min)

    def _normalize_signed_sample(self, value, value_range, output_limits):
        """
        Normaliza una muestra preservando signo y clipeando a los limites de salida.
        """
        range_min, range_max = value_range
        out_min, out_max = output_limits
        bound = max(abs(range_min), abs(range_max))
        signed_unit = value / bound
        signed_unit = min(1.0, max(-1.0, signed_unit))
        normalized_01 = (signed_unit + 1.0) / 2.0
        return out_min + normalized_01 * (out_max - out_min)
