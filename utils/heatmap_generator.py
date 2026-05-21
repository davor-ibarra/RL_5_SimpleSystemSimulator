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
        Adapta la estructura columnar 'interval_data'.
        """
        if self.episodes_data is None:
            self.load_all_episodes()
        
        # Extraer valores de la métrica
        values = []
        for episode in self.episodes_data:
            interval_data = episode.get('interval_data', {})
            if metric_name in interval_data:
                metric_values = interval_data[metric_name]
                if isinstance(metric_values, list):
                    values.extend(metric_values)
        
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
        # No usar subcarpeta chunks, los archivos están en la raíz según ResultHandler
        chunks_dir = self.output_dir
        self.episodes_data = []
        
        if not os.path.exists(chunks_dir):
            return self.episodes_data
        
        # Listar archivos de chunks ordenados
        # Patrón correcto: episodes_chunks_*.json (plural)
        chunk_files = sorted([
            f for f in os.listdir(chunks_dir) 
            if f.startswith('episodes_chunks_') and f.endswith('.json')
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
        Aplana episodios a registros por step para visualización, usando step_data columnar.
        
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
            filtered_episodes = []
            for ep in episodes:
                reason = None
                # Prioridad: top-level -> end_episode_data
                if 'termination_reason' in ep:
                    reason = ep['termination_reason']
                elif 'end_episode_data' in ep and 'end_termination_reason' in ep['end_episode_data']:
                    reason = ep['end_episode_data']['end_termination_reason']
                
                if reason in filter_termination_reason:
                    filtered_episodes.append(ep)
            episodes = filtered_episodes
        
        step_records = []
        
        for episode in episodes:
            episode_id = episode['episode_id']
            
            # Obtener termination reason
            termination_reason = None
            if 'termination_reason' in episode:
                termination_reason = episode['termination_reason']
            elif 'end_episode_data' in episode and 'end_termination_reason' in episode['end_episode_data']:
                termination_reason = episode['end_episode_data']['end_termination_reason']
            
            # Procesar step_data (formato columnar)
            if 'step_data' not in episode:
                continue
                
            step_data = episode['step_data']
            
            # Determinar longitud de series
            n_steps = 0
            keys_to_process = []
            
            if 'n_step' in step_data:
                 n_steps = step_data['n_step']
            
            for key, val in step_data.items():
                if isinstance(val, list):
                    if n_steps == 0 and len(val) > 0:
                        n_steps = len(val)
                    keys_to_process.append(key)
            
            if n_steps == 0:
                continue
                
            # Iterar y construir registros
            for i in range(n_steps):
                record = {
                    'episode_id': episode_id,
                    'step_id': i,
                    'termination_reason': termination_reason
                }
                
                for key in keys_to_process:
                    val_list = step_data[key]
                    if i < len(val_list):
                        val = val_list[i]
                        record[key] = val
                
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
        values = [record[metric_name] for record in aligned_data if metric_name in record]
        
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
        
        x_values = [r[x_variable] for r in step_records if x_variable in r]
        y_values = [r[y_variable] for r in step_records if y_variable in r]
        
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
        
        skipped_heatmaps = []
        written_sheets = 0

        # Crear ExcelWriter
        try:
            with pd.ExcelWriter(output_excel_target_filepath, engine='openpyxl') as writer:
                for heatmap_cfg in heatmap_configs_list:
                    # Configs de visualización pueden usar get() porque son opcionales/configuración
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
                        filtered_records = [
                            r for r in step_records 
                            if 'termination_reason' in r and r['termination_reason'] in filter_reason
                        ]
                    
                    # Extraer valores usando acceso directo seguro (verificando existencia)
                    x_values = [r[x_var] for r in filtered_records if x_var in r and r[x_var] is not None]
                    y_values = [r[y_var] for r in filtered_records if y_var in r and r[y_var] is not None]
                    
                    
                    if not x_values or not y_values:
                        logger.warning(f"[HeatmapGenerator] No hay datos para heatmap '{sheet_name}'")
                        skipped_heatmaps.append({
                            'sheet_name': sheet_name,
                            'x_variable': x_var,
                            'y_variable': y_var,
                            'filter_termination_reason': str(filter_reason),
                            'reason': 'empty_after_filter'
                        })
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
                    written_sheets += 1
                    logger.info(f"[HeatmapGenerator] Hoja '{sheet_name}' escrita con shape {df_grid.shape}")

                if skipped_heatmaps:
                    pd.DataFrame(skipped_heatmaps).to_excel(writer, sheet_name='_skipped_heatmaps', index=False)
                    logger.info(f"[HeatmapGenerator] Heatmaps omitidos registrados: {len(skipped_heatmaps)}")

                if written_sheets == 0 and not skipped_heatmaps:
                    pd.DataFrame([{'message': 'no_heatmap_data'}]).to_excel(writer, sheet_name='_empty_heatmaps', index=False)
            
            logger.info(f"[HeatmapGenerator] Archivo Excel generado: {output_excel_target_filepath}")
            
        except Exception as e:
            logger.error(f"[HeatmapGenerator] Error generando archivo Excel: {e}", exc_info=True)

    def _load_detailed_step_records(self, output_root_path):
        """
        Carga datos detallados usando la lógica centralizada.
        
        Args:
            output_root_path (str): Directorio de resultados (no se usa si self.output_dir ya está set)
            
        Returns:
            list: Lista de registros por step
        """
        # Reutilizar la lógica ya corregida en flatten_to_step_records
        # que a su vez llama a load_all_episodes (que ya busca en chunks)
        return self.flatten_to_step_records()
