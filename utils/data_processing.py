"""
data_processing.py

Responsabilidad:
Definir cálculo/síntesis de datos, no persistencia.
Contiene funciones de utilidad para:
- Calcular resúmenes de episodios
- Transformaciones auxiliares de datos
- Agregaciones y consolidaciones en tablas
"""

import numpy as np


def calculate_episode_summary(episode_data, config_data_summary=None):
    """
    Calcula el resumen de un episodio completo de forma estrictamente declarativa según config.
    Produce las columnas definidas en data_first_cols + estadísticas de data_stats.
    NO usa valores por defecto ni .get() para evitar datos espurios o silenciosos.
    
    Args:
        episode_data (dict): Datos del episodio en formato canónico
        config_data_summary (dict): Configuración de resumen (obligatorio)
        
    Returns:
        dict: Resumen del episodio
    """
    if not config_data_summary:
         return {}

    summary = {}
    config_root = config_data_summary['data_summary']
    first_cols = config_root['data_first_cols']
    stats_cols = config_root['data_stats']
    
    # 1. Unificar fuentes de datos disponibles en un solo lookup plano
    # Prioridad de lectura (conceptualmente, aunque aquí es unificación plana):
    # step_data -> interval_data -> end_episode_data
    # Se usa update secuencial.
    data_sources = {}
    
    if 'step_data' in episode_data:
        data_sources.update(episode_data['step_data'])
        
    if 'interval_data' in episode_data:
        data_sources.update(episode_data['interval_data'])
    
    if 'end_episode_data' in episode_data:
        data_sources.update(episode_data['end_episode_data'])
    
    # Agregar episode_id manualmente si no está en end_episode_data
    if 'episode_id' not in data_sources:
        data_sources['episode_id'] = episode_data['episode_id']
    
    # 2. Construir columnas principales (data_first_cols)
    for col in first_cols:
        
        # Caso Especial: Performance (Cálculo derivado)
        if col == 'performance':
            # Requiere total_reward y t_sec (final)
            # Solo calcular si ambos existen
            if 'total_reward' in data_sources and 't_sec' in data_sources:
                 total_reward = data_sources['total_reward']
                 t_sec_list = data_sources['t_sec']
                 
                 # Validar que t_sec_list sea una lista no vacía y el último valor > 0
                 if isinstance(t_sec_list, list) and len(t_sec_list) > 0:
                     duration = t_sec_list[-1]
                     if duration > 0:
                         summary['performance'] = total_reward / duration
                     else:
                         summary['performance'] = 0.0
                 else:
                     summary['performance'] = 0.0
            else:
                 summary['performance'] = 0.0
            continue

        # Caso Especial: Columnas 'final_*' (Extraction de último valor)
        if col.startswith('final_'):
            # Determinar llave original
            if col == 'final_t_sec':
                original_key = 't_sec'
            else:
                original_key = col[6:] 
            
            if original_key in data_sources:
                values = data_sources[original_key]
                if isinstance(values, list) and values:
                    summary[col] = values[-1]
                else:
                    # Si no es lista, quizas es escalar ya calculado y presente
                    summary[col] = values
            # Si no está en datasources, no se agrega
            continue
            
        # Caso General: Mapeo directo
        if col in data_sources:
            values = data_sources[col]
            if isinstance(values, list):
                 if values:
                     summary[col] = values[-1]
            else:
                 summary[col] = values

    # 2.1. Columnas dinamicas de reward por controlador.
    # Preserva controladores adicionales aunque no esten listados explicitamente.
    for key, values in data_sources.items():
        if not key.startswith('total_reward_controller_') or key in summary:
            continue
        if isinstance(values, list):
            if values:
                summary[key] = values[-1]
        else:
            summary[key] = values

    # 3. Calcular estadísticas (data_stats)
    for key in stats_cols:
        if key in data_sources:
            values = data_sources[key]
            # Solo calcular si es lista de números y no está vacía
            if isinstance(values, list) and values and isinstance(values[0], (int, float)):
                stats = compute_statistics(values)
                for stat_name, stat_val in stats.items():
                     summary[f'{key}_{stat_name}'] = stat_val
            elif isinstance(values, (int, float, np.number)) and not isinstance(values, bool):
                stats = compute_statistics([values])
                for stat_name, stat_val in stats.items():
                     summary[f'{key}_{stat_name}'] = stat_val

    return summary


def aggregate_metrics(interval_data_list, metric_names):
    """
    Agrega métricas de múltiples intervalos.
    
    Args:
        interval_data_list (list): Lista de datos de intervalos
        metric_names (list): Nombres de métricas a agregar
        
    Returns:
        dict: Métricas agregadas (mean, std, min, max por métrica)
    """
    aggregated = {}
    
    for metric_name in metric_names:
        values = []
        for interval_data in interval_data_list:
            if metric_name in interval_data:
                values.append(interval_data[metric_name])
        
        if values:
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)
            aggregated[f'{metric_name}_min'] = np.min(values)
            aggregated[f'{metric_name}_max'] = np.max(values)
    
    return aggregated


def compute_statistics(values):
    """
    Calcula estadísticas descriptivas de una serie de valores.
    
    Args:
        values (list/array): Valores para calcular estadísticas
        
    Returns:
        dict: Estadísticas (mean, std, min, max, percentiles)
    """
    if not values or len(values) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'p25': 0.0,
            'p50': 0.0,
            'p75': 0.0,
            'max': 0.0
        }
    
    # Asegurar tipo numérico (float) para evitar errores con booleanos en np.percentile
    arr = np.array(values, dtype=float)
    
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'max': float(np.max(arr))
    }


def normalize_data(values, min_val, max_val):
    """
    Normaliza valores a un rango específico.
    
    Args:
        values (list/array): Valores a normalizar
        min_val (float): Valor mínimo del rango
        max_val (float): Valor máximo del rango
        
    Returns:
        array: Valores normalizados entre 0 y 1
    """
    arr = np.array(values)
    range_val = max_val - min_val
    
    if range_val == 0:
        return np.zeros_like(arr)
    
    return (arr - min_val) / range_val


def consolidate_controller_data(controller_state_series, controller_names):
    """
    Consolida datos de múltiples controladores en una tabla.
    
    Args:
        controller_state_series (list): Serie de estados de controladores
        controller_names (list): Nombres de controladores
        
    Returns:
        dict: Datos consolidados por controlador
    """
    consolidated = {name: [] for name in controller_names}
    
    for step_state in controller_state_series:
        for controller_name in controller_names:
            if controller_name in step_state:
                consolidated[controller_name].append(step_state[controller_name])
    
    return consolidated


def filter_data_by_config(data, save_config):
    """
    Filtra datos según configuración de guardado.
    
    Args:
        data (dict): Datos completos
        save_config (dict): Configuración de qué guardar
        
    Returns:
        dict: Datos filtrados según configuración
    """
    # Si no hay configuración, retornar todo
    if not save_config:
        return data
    
    # Extraer variables habilitadas
    enabled_params = set()
    components = {}
    if 'data_save' in save_config and 'components' in save_config['data_save']:
        components = save_config['data_save']['components']
    
    for component_name, param_list in components.items():
        for param_config in param_list:
            param_name = param_config['params']
            enabled_params.add(param_name)
    
    # Filtrar datos
    filtered = {}
    for key, value in data.items():
        if key in enabled_params:
            filtered[key] = value
    
    return filtered


def compute_termination_statistics(episodes_data):
    """
    Calcula estadísticas de terminación de episodios.
    
    Args:
        episodes_data (list): Lista de datos de episodios en formato canónico
        
    Returns:
        dict: Estadísticas de terminación (conteos por razón)
    """
    counts = {}
    
    for episode in episodes_data:
        end_data = episode['end_episode_data']
        reason = end_data['end_termination_reason']
        if reason not in counts:
            counts[reason] = 0
        counts[reason] = counts[reason] + 1
    
    total = len(episodes_data)
    
    return {
        'counts': counts,
        'total': total,
        'percentages': {k: v / total * 100 for k, v in counts.items()} if total > 0 else {}
    }
