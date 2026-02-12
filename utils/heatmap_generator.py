"""
heatmap_generator.py

Responsabilidad:
Etapa de preparación para visualizaciones tipo heatmap.
Lee el detalle persistido, combina/alinea en una estructura tabular
y devuelve el insumo para el generador de gráficos.
"""

import os
import json
import numpy as np

# Importar pandas opcionalmente
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class HeatmapGenerator:
    """
    Generador de datos para heatmaps.
    
    Responsabilidad:
    - Leer datos detallados persistidos
    - Combinar y alinear en estructura tabular
    - Preparar insumos para visualización
    """
    
    def __init__(self, output_dir):
        """
        Inicializa el generador de heatmaps.
        
        Args:
            output_dir (str): Directorio de salida con resultados
        """
        self.output_dir = output_dir
        self.episodes_data = None
    
    def prepare_heatmap_data(self, metric_name, aggregation_type='mean'):
        """
        Prepara datos para generar un heatmap de una métrica específica.
        
        Args:
            metric_name (str): Nombre de la métrica
            aggregation_type (str): Tipo de agregación ('mean', 'max', 'min', 'count')
            
        Returns:
            dict: Datos estructurados para heatmap
        """
        if self.episodes_data is None:
            self.load_all_episodes()
        
        # Extraer valores de la métrica
        values = []
        for episode in self.episodes_data:
            for interval_record in episode['intervals']:
                interval_data = interval_record['interval_data']
                if metric_name in interval_data:
                    values.append(interval_data[metric_name])
        
        return {
            'metric_name': metric_name,
            'aggregation_type': aggregation_type,
            'values': values
        }
    
    def load_all_episodes(self):
        """
        Carga todos los episodios desde archivos JSON persistidos.
        
        Returns:
            list: Lista de datos de episodios
        """
        chunks_dir = os.path.join(self.output_dir, 'chunks')
        self.episodes_data = []
        
        if not os.path.exists(chunks_dir):
            return self.episodes_data
        
        # Listar archivos de chunks ordenados
        chunk_files = sorted([
            f for f in os.listdir(chunks_dir) 
            if f.startswith('episode_chunk_') and f.endswith('.json')
        ])
        
        for chunk_file in chunk_files:
            filepath = os.path.join(chunks_dir, chunk_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                episodes = json.load(f)
                self.episodes_data.extend(episodes)
        
        return self.episodes_data
    
    def load_episode_details(self, episode_ids=None):
        """
        Carga detalles de episodios desde archivos JSON persistidos.
        
        Args:
            episode_ids (list): Lista de IDs de episodios a cargar (None = todos)
            
        Returns:
            list: Lista de datos de episodios
        """
        if self.episodes_data is None:
            self.load_all_episodes()
        
        if episode_ids is None:
            return self.episodes_data
        
        # Filtrar por IDs
        return [ep for ep in self.episodes_data if ep['episode_id'] in episode_ids]
    
    def flatten_to_step_records(self, episodes=None, filter_termination_reason=None):
        """
        Aplana episodios a registros por step para visualización.
        
        Args:
            episodes (list): Lista de episodios (None = cargar todos)
            filter_termination_reason (list): Filtrar por reason (None = todos)
            
        Returns:
            list: Lista de dicts, uno por step
        """
        if episodes is None:
            if self.episodes_data is None:
                self.load_all_episodes()
            episodes = self.episodes_data
        
        # Filtrar por termination_reason si se especifica
        if filter_termination_reason:
            episodes = [ep for ep in episodes 
                       if ep['termination_reason'] in filter_termination_reason]
        
        step_records = []
        
        for episode in episodes:
            episode_id = episode['episode_id']
            termination_reason = episode['termination_reason']
            
            for interval_idx, interval_record in enumerate(episode['intervals']):
                interval_data = interval_record['interval_data']
                
                # Series step-level
                dynamic_series = interval_data.get('dynamic_system_state_dict', [])
                controller_series = interval_data.get('controller_state_dict', [])
                
                n_steps = len(dynamic_series)
                
                for step_idx in range(n_steps):
                    record = {
                        'episode_id': episode_id,
                        'interval_id': interval_idx,
                        'step_id': step_idx,
                        'termination_reason': termination_reason
                    }
                    
                    # Añadir estado dinámico
                    if step_idx < len(dynamic_series):
                        for key, value in dynamic_series[step_idx].items():
                            record[key] = value
                    
                    # Añadir estado de controlador (aplanar anidado)
                    if step_idx < len(controller_series):
                        controller_state = controller_series[step_idx]
                        for section_name, section_data in controller_state.items():
                            if isinstance(section_data, dict):
                                for key, value in section_data.items():
                                    record[key] = value
                            else:
                                record[section_name] = section_data
                    
                    step_records.append(record)
        
        return step_records
    
    def align_data_by_state_action(self, episodes_data):
        """
        Alinea datos por estados y acciones para crear estructura tabular.
        
        Args:
            episodes_data (list): Datos de episodios
            
        Returns:
            dict: Datos alineados en estructura tabular
        """
        return self.flatten_to_step_records(episodes_data)
    
    def aggregate_by_bins(self, aligned_data, metric_name, aggregation_type):
        """
        Agrega datos por bins de estado-acción.
        
        Args:
            aligned_data (dict): Datos alineados
            metric_name (str): Métrica a agregar
            aggregation_type (str): Tipo de agregación
            
        Returns:
            array: Matriz agregada para heatmap
        """
        # Extraer valores de la métrica
        values = [record.get(metric_name, 0) for record in aligned_data]
        
        if aggregation_type == 'mean':
            return np.mean(values) if values else 0.0
        elif aggregation_type == 'sum':
            return np.sum(values) if values else 0.0
        elif aggregation_type == 'count':
            return len(values)
        elif aggregation_type == 'max':
            return np.max(values) if values else 0.0
        elif aggregation_type == 'min':
            return np.min(values) if values else 0.0
        else:
            return np.mean(values) if values else 0.0
    
    def generate_histogram2d_data(self, x_variable, y_variable, 
                                   filter_termination_reason=None, bins=100):
        """
        Genera datos para histograma 2D / heatmap.
        
        Args:
            x_variable (str): Variable para eje X
            y_variable (str): Variable para eje Y
            filter_termination_reason (list): Filtrar por reason
            bins (int): Número de bins
            
        Returns:
            tuple: (H, xedges, yedges) para histogram2d
        """
        step_records = self.flatten_to_step_records(
            filter_termination_reason=filter_termination_reason
        )
        
        x_values = [r.get(x_variable) for r in step_records if x_variable in r]
        y_values = [r.get(y_variable) for r in step_records if y_variable in r]
        
        if not x_values or not y_values:
            return None, None, None
        
        # Asegurar misma longitud
        min_len = min(len(x_values), len(y_values))
        x_values = x_values[:min_len]
        y_values = y_values[:min_len]
        
        H, xedges, yedges = np.histogram2d(x_values, y_values, bins=bins)
        
        return H, xedges, yedges
    
    def generate(self, output_root_path_gen, heatmap_configs_list, output_excel_target_filepath):
        """
        Genera datos pre-calculados para heatmaps y los guarda en Excel.
        
        Args:
            output_root_path_gen (str): Directorio raíz de resultados
            heatmap_configs_list (list): Lista de configuraciones de heatmap
            output_excel_target_filepath (str): Ruta del archivo Excel de salida
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not HAS_PANDAS:
            logger.error("[HeatmapGenerator] pandas no disponible. No se pueden generar datos de heatmap.")
            return
        
        logger.info(f"[HeatmapGenerator] Generando datos para {len(heatmap_configs_list)} heatmaps...")
        
        # Cargar datos detallados
        step_records = self._load_detailed_step_records(output_root_path_gen)
        
        if not step_records:
            logger.warning("[HeatmapGenerator] No hay datos detallados para generar heatmaps.")
            return
        
        # Crear ExcelWriter
        try:
            with pd.ExcelWriter(output_excel_target_filepath, engine='openpyxl') as writer:
                for heatmap_cfg in heatmap_configs_list:
                    plot_name = heatmap_cfg.get('name')
                    plot_index = heatmap_cfg.get('_internal_plot_index', '?')
                    
                    # Determinar nombre de hoja
                    if plot_name:
                        sheet_name = str(plot_name)[:31]  # Excel limita a 31 caracteres
                    else:
                        sheet_name = f"heatmap_idx_{plot_index}"[:31]
                    
                    x_var = heatmap_cfg.get('x_variable', 'time')
                    y_var = heatmap_cfg.get('y_variable', 'state')
                    config = heatmap_cfg.get('config', {})
                    bins = config.get('bins', 100)
                    filter_reason = config.get('filter_termination_reason')
                    
                    logger.info(f"[HeatmapGenerator] Procesando heatmap '{sheet_name}': {x_var} vs {y_var}")
                    
                    # Filtrar registros
                    filtered_records = step_records
                    if filter_reason:
                        filtered_records = [r for r in step_records 
                                           if r.get('termination_reason') in filter_reason]
                    
                    # Extraer valores
                    x_values = [r.get(x_var) for r in filtered_records if x_var in r and r.get(x_var) is not None]
                    y_values = [r.get(y_var) for r in filtered_records if y_var in r and r.get(y_var) is not None]
                    
                    if not x_values or not y_values:
                        logger.warning(f"[HeatmapGenerator] No hay datos para heatmap '{sheet_name}'")
                        continue
                    
                    # Asegurar misma longitud
                    min_len = min(len(x_values), len(y_values))
                    x_values = np.array(x_values[:min_len])
                    y_values = np.array(y_values[:min_len])
                    
                    # Generar histogram2d
                    H, xedges, yedges = np.histogram2d(x_values, y_values, bins=bins)
                    
                    # Calcular centros de bins
                    x_centers = (xedges[:-1] + xedges[1:]) / 2
                    y_centers = (yedges[:-1] + yedges[1:]) / 2
                    
                    # Crear DataFrame con centros como índices
                    df_grid = pd.DataFrame(H.T, index=y_centers, columns=x_centers)
                    df_grid.index.name = y_var
                    df_grid.columns.name = x_var
                    
                    # Escribir a Excel
                    df_grid.to_excel(writer, sheet_name=sheet_name)
                    logger.info(f"[HeatmapGenerator] Hoja '{sheet_name}' escrita con shape {df_grid.shape}")
            
            logger.info(f"[HeatmapGenerator] Archivo Excel generado: {output_excel_target_filepath}")
            
        except Exception as e:
            logger.error(f"[HeatmapGenerator] Error generando archivo Excel: {e}", exc_info=True)
    
    def _load_detailed_step_records(self, output_root_path):
        """
        Carga datos detallados desde archivos JSON.
        
        Args:
            output_root_path (str): Directorio de resultados
            
        Returns:
            list: Lista de registros por step
        """
        import logging
        logger = logging.getLogger(__name__)
        
        all_records = []
        
        # Buscar archivos simulation_data_ep_*.json
        try:
            files = [f for f in os.listdir(output_root_path) 
                    if f.startswith("simulation_data_ep_") and f.endswith(".json")]
            
            if not files:
                # Intentar con chunks si no hay archivos de simulación directos
                return self.flatten_to_step_records()
            
            # Ordenar por número de episodio
            try:
                files.sort(key=lambda name: int(name.split('_ep_')[-1].split('_to_')[0]))
            except (ValueError, IndexError):
                pass
            
            for filename in files:
                filepath = os.path.join(output_root_path, filename)
                try:
                    import json
                    with open(filepath, 'r', encoding='utf-8') as f:
                        episodes_in_file = json.load(f)
                    
                    if isinstance(episodes_in_file, list):
                        for episode_dict in episodes_in_file:
                            if not isinstance(episode_dict, dict):
                                continue
                            
                            time_values = episode_dict.get('time', [])
                            episode_id = episode_dict.get('episode', [0])[0] if isinstance(episode_dict.get('episode'), list) else episode_dict.get('episode', 0)
                            term_reason = episode_dict.get('termination_reason', ['unknown'])[0] if isinstance(episode_dict.get('termination_reason'), list) else episode_dict.get('termination_reason', 'unknown')
                            
                            num_steps = len(time_values)
                            
                            for step_idx in range(num_steps):
                                record = {
                                    'episode_id': episode_id,
                                    'step_id': step_idx,
                                    'termination_reason': term_reason
                                }
                                
                                # Añadir cada métrica
                                for key, values in episode_dict.items():
                                    if isinstance(values, list) and len(values) > step_idx:
                                        record[key] = values[step_idx]
                                
                                all_records.append(record)
                                
                except Exception as e:
                    logger.error(f"[HeatmapGenerator] Error cargando {filename}: {e}")
            
            logger.info(f"[HeatmapGenerator] Cargados {len(all_records)} registros detallados")
            return all_records
            
        except Exception as e:
            logger.error(f"[HeatmapGenerator] Error buscando archivos: {e}")
            return []
