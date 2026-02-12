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


def calculate_episode_summary(episode_data):
    """
    Calcula el resumen de un episodio completo.
    Produce las columnas esperadas por summary_first_cols + estadísticas de interval_data.
    
    Args:
        episode_data (dict): Datos del episodio en formato canónico con:
            - episode_id: int
            - step_data: dict de listas
            - interval_data: dict de listas
            - end_episode_data: dict
        
    Returns:
        dict: Resumen del episodio con estadísticas agregadas
    """
    episode_id = episode_data['episode_id']
    end_data = episode_data['end_episode_data']
    interval_data = episode_data['interval_data']
    
    # Campos base desde end_episode_data
    summary = {
        'episode': episode_id,
        'episode_wall_time_sec': end_data['episode_wall_time_sec'],
        'total_agent_decisions': end_data['total_agent_decisions'],
        'termination_reason': end_data['end_termination_reason'],
        'total_reward': end_data['total_reward'],
        'accumulated_band_bonus': end_data['accumulated_band_bonus'],
        'goal_bonus': end_data['goal_bonus'],
    }
    
    # Estadísticas de rewards desde interval_data columnar
    global_rewards = interval_data['global_interval_reward']
    if global_rewards:
        summary['avg_reward'] = float(np.mean(global_rewards))
        summary['std_reward'] = float(np.std(global_rewards))
    
    # Estadísticas de métricas numéricas del interval_data
    # Iterar llaves numéricas (excluir metadata y strings)
    skip_keys = {'n_interval', 'interval_id', 'interval_start_step_idx',
                 'interval_end_step_idx', 'termination_reason'}
    
    for key, values in interval_data.items():
        if key in skip_keys:
            continue
        if not isinstance(values, list) or not values:
            continue
        # Solo procesar series numéricas
        if isinstance(values[0], (int, float)):
            stats = compute_statistics(values)
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
            'max': 0.0,
            'p25': 0.0,
            'p50': 0.0,
            'p75': 0.0
        }
    
    arr = np.array(values)
    
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75))
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
    components = save_config.get('data_save', {}).get('components', {})
    
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