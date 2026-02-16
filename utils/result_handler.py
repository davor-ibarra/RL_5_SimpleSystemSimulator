"""
result_handler.py

Responsabilidad:
Frontera única de persistencia. Escribe datos en estructura filtrada/eficiente.
Incluye lógica de chunking para corridas largas.
Genera summary rows automáticamente via calculate_episode_summary al guardar
cada episodio, conectando el pipeline save_episode → append_summary_row → flush.

Outputs:
- metadata.json: Snapshot de configuración
- chunks/episode_chunk_*.json: Datos detallados por chunks
- summary.xlsx: Fila por episodio (auto-generada desde end_episode_data + interval_data)
- agent_state/agent_state_ep_*.xlsx: Estado del agente (Excel con hojas por agente)
"""

import os
import json
from utils.numpy_encoder import NumpyEncoder, sanitize_for_json
from utils.data_processing import calculate_episode_summary

# Importar pandas opcionalmente para Excel
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class ResultHandler:
    """
    Manejador único de persistencia de resultados.
    
    Responsabilidades:
    - Administrar carpeta de corrida y rutas de salida
    - Escribir metadata como snapshot directo de configuración
    - Escribir chunks JSON con episodios acumulados
    - Escribir/actualizar Excel de resumen
    - Guardar estado del agente
    """
    
    def __init__(self, output_dir, config_data_summary=None):
        """
        Inicializa el manejador de resultados.
        
        Args:
            output_dir (str): Directorio de salida para la corrida
            config_data_summary (dict): Configuración de resumen de datos (opcional)
        """
        self.output_dir = output_dir
        self.config_data_save = config_data_summary if config_data_summary else {}
        
        # Configuración de chunking
        self.episodes_per_chunk = 100
        
        # Buffer de episodios para chunking
        self.episode_buffer = []
        self.current_chunk_id = 0
        
        # Buffer de filas de summary
        self.summary_rows = []
        
        # Asegurar la creación del directorio
        self._ensure_output_structure()
    
    def _ensure_output_structure(self):
        """
        Crea el directorio necesario para la corrida.
        """
        # Crear directorio principal si no existe
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_metadata(self, metadata_dict):
        """
        Escribe metadata como snapshot directo de la configuración usada.
        
        Args:
            metadata_dict (dict): Metadata a guardar
        """
        filepath = os.path.join(self.output_dir, 'metadata.json')
        sanitized = sanitize_for_json(metadata_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sanitized, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    def save_episode(self, episode_data):
        """
        Acumula episodio en buffer y escribe chunk cuando corresponda.
        Genera summary row automáticamente via calculate_episode_summary.
        
        Args:
            episode_data (dict): Datos del episodio completo
        """
        # Añadir al buffer
        self.episode_buffer.append(episode_data)
        
        # Generar y acumular summary row
        summary_row = calculate_episode_summary(episode_data, self.config_data_save)
        self.append_summary_row(summary_row)
        
        # Si el buffer alcanza el tamaño de chunk, escribir
        if len(self.episode_buffer) >= self.episodes_per_chunk:
            self._flush_episode_buffer()
    
    def _flush_episode_buffer(self):
        """
        Escribe el buffer actual como chunk JSON y limpia el buffer.
        """
        if not self.episode_buffer:
            return
        
        # Escribir chunk
        self.save_episode_chunk(self.episode_buffer, self.current_chunk_id)
        
        # Incrementar chunk_id y limpiar buffer
        self.current_chunk_id += 1
        self.episode_buffer = []
    
    def save_episode_chunk(self, episodes_data, chunk_id):
        """
        Escribe un chunk JSON de episodios.
        
        Args:
            episodes_data (list): Lista de episodios procesados
            chunk_id (int): Identificador del chunk
        """
        filepath = self.get_chunk_filepath(chunk_id)
        sanitized = sanitize_for_json(episodes_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sanitized, f, cls=NumpyEncoder, indent=None, ensure_ascii=False)
    
    def append_summary_row(self, summary_row):
        """
        Acumula fila de resumen para escritura posterior.
        
        Args:
            summary_row (dict): Fila de resumen del episodio
        """
        self.summary_rows.append(summary_row)
    
    def _flush_summary(self):
        """
        Escribe todas las filas de resumen acumuladas a Excel.
        """
        if not self.summary_rows or not HAS_PANDAS:
            return
        
        filepath = self.get_summary_filepath()
        
        # Crear DataFrame desde filas
        df = pd.DataFrame(self.summary_rows)
        
        # Ordenar columnas según configuración si existe
        # Nota: La llave raíz en el nuevo yaml es 'data_summary', no 'data_save'
        config_root = self.config_data_save['data_summary']
        first_cols = config_root['data_first_cols']
        
        if first_cols:
            # Reordenar columnas: primero las especificadas, luego el resto
            existing_first = [c for c in first_cols if c in df.columns]
            remaining = [c for c in df.columns if c not in existing_first]
            df = df[existing_first + remaining]
        
        # Escribir a Excel
        df.to_excel(filepath, index=False)
    
    def save_agent_state_learn_dict(self, state_dict, episode_id):
        """
        Guarda estado del agente (Q-tables, visit counts) en Excel.
        Crea una hoja por agente con sus tablas Q y de visitas.
        
        Args:
            state_dict (dict): Estado serializable del agente (q_tables, visit_counts)
            episode_id (int): Identificador del episodio
        """
        if not HAS_PANDAS:
            return
        
        filepath = self.get_agent_state_filepath(episode_id)
        
        q_tables = state_dict['q_tables']
        visit_counts = state_dict['visit_counts']
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for agent_name in q_tables:
                q_table_data = q_tables[agent_name]
                visit_data = visit_counts[agent_name]
                
                # Convertir a DataFrames
                df_q = pd.DataFrame(q_table_data)
                df_visits = pd.DataFrame(visit_data)
                
                # Escribir Q-Table con título
                pd.DataFrame(["Q-Table"]).to_excel(writer, sheet_name=agent_name, startrow=0, startcol=0, index=False, header=False)
                df_q.to_excel(writer, sheet_name=agent_name, startrow=1, startcol=0)
                
                # Determinar posición para Visit Counts
                start_row_visits = len(df_q) + 4
                
                # Escribir Visit Counts con título
                pd.DataFrame(["Visit Counts"]).to_excel(writer, sheet_name=agent_name, startrow=start_row_visits, startcol=0, index=False, header=False)
                df_visits.to_excel(writer, sheet_name=agent_name, startrow=start_row_visits + 1, startcol=0)
    
    def finalize_run(self):
        """
        Finaliza la corrida: flush de episodios y summary pendientes.
        """
        # Flush de episodios pendientes
        self._flush_episode_buffer()
        
        # Flush de summary
        self._flush_summary()
    
    def get_chunk_filepath(self, chunk_id):
        """
        Genera la ruta del archivo para un chunk específico.
        
        Args:
            chunk_id (int): Identificador del chunk
            
        Returns:
            str: Ruta completa del archivo del chunk
        """
        return os.path.join(self.output_dir, f'episodes_chunks_{chunk_id:04d}.json')
    
    def get_summary_filepath(self):
        """
        Genera la ruta del archivo de resumen Excel.
        
        Returns:
            str: Ruta completa del archivo de resumen
        """
        return os.path.join(self.output_dir, 'summary.xlsx')
    
    def get_agent_state_filepath(self, episode_id):
        """
        Genera la ruta del archivo de estado del agente.
        
        Args:
            episode_id (int): Identificador del episodio
            
        Returns:
            str: Ruta completa del archivo de estado del agente
        """
        return os.path.join(self.output_dir, f'agent_state_ep_{episode_id}.xlsx')
    
    # ==================== MÉTODOS DE LECTURA (para visualización) ====================
    
    def load_all_chunks(self):
        """
        Carga todos los chunks de episodios.
        
        Returns:
            list: Lista de todos los episodios
        """
        chunks_dir = os.path.join(self.output_dir, 'chunks')
        all_episodes = []
        
        if not os.path.exists(chunks_dir):
            return all_episodes
        
        # Listar archivos de chunks ordenados
        chunk_files = sorted([
            f for f in os.listdir(chunks_dir) 
            if f.startswith('episode_chunk_') and f.endswith('.json')
        ])
        
        for chunk_file in chunk_files:
            filepath = os.path.join(chunks_dir, chunk_file)
            with open(filepath, 'r', encoding='utf-8') as f:
                episodes = json.load(f)
                all_episodes.extend(episodes)
        
        return all_episodes
    
    def load_summary(self):
        """
        Carga el archivo de resumen como DataFrame.
        
        Returns:
            pd.DataFrame or None: DataFrame con resumen o None si no existe
        """
        if not HAS_PANDAS:
            return None
        
        filepath = self.get_summary_filepath()
        
        if not os.path.exists(filepath):
            return None
        
        return pd.read_excel(filepath)
    
    def flatten_episode_to_steps(self, episode_data):
        """
        Aplana un episodio a registros por step para visualización.
        Trabaja con la estructura columnar: episode_data['step_data'] = {key: [values]}.
        
        Args:
            episode_data (dict): Datos del episodio en formato canónico
            
        Returns:
            list: Lista de dicts, uno por step
        """
        step_records = []
        episode_id = episode_data['episode_id']
        step_data = episode_data['step_data']
        n_steps = step_data['n_step']
        
        # Obtener todas las llaves excepto n_step
        data_keys = [k for k in step_data.keys() if k != 'n_step']
        
        for step_idx in range(n_steps):
            record = {
                'episode_id': episode_id,
                'step_id': step_idx
            }
            
            for key in data_keys:
                values = step_data[key]
                if step_idx < len(values):
                    record[key] = values[step_idx]
            
            step_records.append(record)
        
        return step_records