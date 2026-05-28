"""
heatmap_generator.py

Responsabilidad:
Preparar datos agregados para heatmaps desde los chunks detallados de episodios.
La generacion principal trabaja en streaming por archivo y por episodio para no
materializar todos los steps de una corrida grande en memoria.
"""

import gc
import json
import logging
import os

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """
    Generador de datos para heatmaps.

    Responsabilidad:
    - Leer datos detallados persistidos.
    - Acumular histogramas 2D por chunks.
    - Persistir grillas tabulares para el generador Matplotlib.
    """

    def __init__(self, output_dir):
        """
        Inicializa el generador de heatmaps.

        Args:
            output_dir (str): Directorio de salida con resultados.
        """
        self.output_dir = output_dir
        self.episodes_data = None

    def prepare_heatmap_data(self, metric_name, aggregation_type='mean'):
        """
        Prepara datos para generar un heatmap de una metrica especifica.
        Metodo de compatibilidad para analisis antiguos sobre interval_data.
        """
        if self.episodes_data is None:
            self.load_all_episodes()

        values = []
        for episode in self.episodes_data:
            if 'interval_data' not in episode:
                continue

            interval_data = episode['interval_data']
            if metric_name in interval_data and isinstance(interval_data[metric_name], list):
                values.extend(interval_data[metric_name])

        return {
            'metric_name': metric_name,
            'aggregation_type': aggregation_type,
            'values': values
        }

    def load_all_episodes(self):
        """
        Carga todos los episodios desde archivos JSON persistidos.

        Returns:
            list: Lista de datos de episodios.
        """
        self.episodes_data = []

        if not os.path.exists(self.output_dir):
            return self.episodes_data

        for chunk_filepath in self._chunk_filepaths(self.output_dir):
            with open(chunk_filepath, 'r', encoding='utf-8') as file:
                episodes = json.load(file)
            self.episodes_data.extend(episodes)

        return self.episodes_data

    def load_episode_details(self, episode_ids=None):
        """
        Carga detalles de episodios desde archivos JSON persistidos.

        Args:
            episode_ids (list): Lista de IDs de episodios a cargar (None = todos).

        Returns:
            list: Lista de datos de episodios.
        """
        if self.episodes_data is None:
            self.load_all_episodes()

        if episode_ids is None:
            return self.episodes_data

        return [ep for ep in self.episodes_data if ep['episode_id'] in episode_ids]

    def flatten_to_step_records(self, episodes=None, filter_termination_reason=None):
        """
        Aplana episodios a registros por step para visualizacion.
        Se conserva como compatibilidad; no se usa en la ruta grande de heatmaps.

        Args:
            episodes (list): Lista de episodios (None = cargar todos).
            filter_termination_reason (list): Filtrar por reason (None = todos).

        Returns:
            list: Lista de dicts, uno por step.
        """
        if episodes is None:
            if self.episodes_data is None:
                self.load_all_episodes()
            episodes = self.episodes_data

        step_records = []
        for episode in episodes:
            if filter_termination_reason and not self._episode_matches_filter(
                episode,
                filter_termination_reason
            ):
                continue

            episode_id = episode['episode_id']
            termination_reason = self._termination_reason(episode)

            if 'step_data' not in episode:
                continue

            step_data = episode['step_data']
            n_steps = 0
            keys_to_process = []

            if 'n_step' in step_data and isinstance(step_data['n_step'], int):
                n_steps = step_data['n_step']

            for key, val in step_data.items():
                if isinstance(val, list):
                    if n_steps == 0 and len(val) > 0:
                        n_steps = len(val)
                    keys_to_process.append(key)

            if n_steps == 0:
                continue

            for i in range(n_steps):
                record = {
                    'episode_id': episode_id,
                    'step_id': i,
                    'termination_reason': termination_reason
                }

                for key in keys_to_process:
                    val_list = step_data[key]
                    if i < len(val_list):
                        record[key] = val_list[i]

                step_records.append(record)

        return step_records

    def align_data_by_state_action(self, episodes_data):
        """
        Alinea datos por estados y acciones para crear estructura tabular.

        Args:
            episodes_data (list): Datos de episodios.

        Returns:
            dict: Datos alineados en estructura tabular.
        """
        return self.flatten_to_step_records(episodes_data)

    def aggregate_by_bins(self, aligned_data, metric_name, aggregation_type):
        """
        Agrega datos por bins de estado-accion.

        Args:
            aligned_data (dict): Datos alineados.
            metric_name (str): Metrica a agregar.
            aggregation_type (str): Tipo de agregacion.

        Returns:
            array: Matriz agregada para heatmap.
        """
        values = [record[metric_name] for record in aligned_data if metric_name in record]

        if aggregation_type == 'mean':
            return np.mean(values) if values else 0.0
        if aggregation_type == 'sum':
            return np.sum(values) if values else 0.0
        if aggregation_type == 'count':
            return len(values)
        if aggregation_type == 'max':
            return np.max(values) if values else 0.0
        if aggregation_type == 'min':
            return np.min(values) if values else 0.0

        return np.mean(values) if values else 0.0

    def generate_histogram2d_data(
        self,
        x_variable,
        y_variable,
        filter_termination_reason=None,
        bins=100
    ):
        """
        Genera datos para histograma 2D / heatmap.
        Metodo de compatibilidad para uso interactivo.

        Args:
            x_variable (str): Variable para eje X.
            y_variable (str): Variable para eje Y.
            filter_termination_reason (list): Filtrar por reason.
            bins (int): Numero de bins.

        Returns:
            tuple: (H, xedges, yedges) para histogram2d.
        """
        step_records = self.flatten_to_step_records(
            filter_termination_reason=filter_termination_reason
        )

        x_values = [r[x_variable] for r in step_records if x_variable in r]
        y_values = [r[y_variable] for r in step_records if y_variable in r]

        if not x_values or not y_values:
            return None, None, None

        min_len = min(len(x_values), len(y_values))
        x_values = x_values[:min_len]
        y_values = y_values[:min_len]

        return np.histogram2d(x_values, y_values, bins=bins)

    def generate(self, output_root_path_gen, heatmap_configs_list, output_excel_target_filepath):
        """
        Genera datos pre-calculados para heatmaps y los guarda en Excel.

        Args:
            output_root_path_gen (str): Directorio raiz de resultados.
            heatmap_configs_list (list): Lista de configuraciones de heatmap.
            output_excel_target_filepath (str): Ruta del archivo Excel de salida.
        """
        if not HAS_PANDAS:
            logger.error("[HeatmapGenerator] pandas no disponible. No se pueden generar datos de heatmap.")
            return

        heatmap_jobs = self._build_heatmap_jobs(heatmap_configs_list)
        chunk_filepaths = self._chunk_filepaths(output_root_path_gen)

        logger.info(
            "[HeatmapGenerator] Generando datos para %s heatmaps desde %s chunks.",
            len(heatmap_jobs),
            len(chunk_filepaths)
        )

        if not heatmap_jobs:
            self._write_empty_workbook(output_excel_target_filepath, 'no_heatmap_configs')
            return

        if not chunk_filepaths:
            logger.warning("[HeatmapGenerator] No se encontraron chunks de episodios.")
            self._write_empty_workbook(output_excel_target_filepath, 'no_episode_chunks')
            return

        for chunk_index, chunk_filepath in enumerate(chunk_filepaths, start=1):
            logger.info(
                "[HeatmapGenerator] Procesando chunk %s/%s: %s",
                chunk_index,
                len(chunk_filepaths),
                os.path.basename(chunk_filepath)
            )

            chunk_episode_count = 0
            for episode in self._iter_chunk_episodes(chunk_filepath):
                chunk_episode_count += 1
                self._accumulate_episode(heatmap_jobs, episode)

            logger.info(
                "[HeatmapGenerator] Chunk %s procesado con %s episodios.",
                os.path.basename(chunk_filepath),
                chunk_episode_count
            )
            gc.collect()

        self._write_heatmap_workbook(heatmap_jobs, output_excel_target_filepath)

    def _build_heatmap_jobs(self, heatmap_configs_list):
        """Construye los acumuladores de histograma desde la configuracion."""
        used_sheet_names = set()
        heatmap_jobs = []

        for heatmap_cfg in heatmap_configs_list:
            plot_index = heatmap_cfg['_internal_plot_index'] if '_internal_plot_index' in heatmap_cfg else '?'
            sheet_name = self._sheet_name(heatmap_cfg, plot_index, used_sheet_names)
            x_variable = heatmap_cfg['x_variable']
            y_variable = heatmap_cfg['y_variable']
            config = heatmap_cfg['config']
            bins = int(config['bins'])
            x_edges = self._axis_edges(config, 'x', bins, sheet_name)
            y_edges = self._axis_edges(config, 'y', bins, sheet_name)

            heatmap_jobs.append({
                'sheet_name': sheet_name,
                'x_variable': x_variable,
                'y_variable': y_variable,
                'filter_termination_reason': config['filter_termination_reason'],
                'x_edges': x_edges,
                'y_edges': y_edges,
                'x_centers': (x_edges[:-1] + x_edges[1:]) / 2,
                'y_centers': (y_edges[:-1] + y_edges[1:]) / 2,
                'histogram': np.zeros((bins, bins), dtype=np.float64),
                'matched_episode_count': 0,
                'observed_sample_count': 0,
                'in_range_sample_count': 0
            })

            logger.info(
                "[HeatmapGenerator] Heatmap '%s' configurado: %s vs %s, bins=%s.",
                sheet_name,
                x_variable,
                y_variable,
                bins
            )

        return heatmap_jobs

    def _sheet_name(self, heatmap_cfg, plot_index, used_sheet_names):
        """Resuelve un nombre de hoja valido y unico para Excel."""
        if 'name' in heatmap_cfg and heatmap_cfg['name']:
            base_name = str(heatmap_cfg['name'])[:31]
        else:
            base_name = f"heatmap_idx_{plot_index}"[:31]

        sheet_name = base_name
        suffix_index = 1
        while sheet_name in used_sheet_names:
            suffix = f"_{suffix_index}"
            sheet_name = f"{base_name[:31 - len(suffix)]}{suffix}"
            suffix_index += 1

        used_sheet_names.add(sheet_name)
        return sheet_name

    def _axis_edges(self, config, axis_name, bins, sheet_name):
        """Construye bordes deterministas de bin para un eje."""
        min_key = f'{axis_name}min'
        max_key = f'{axis_name}max'
        axis_min = config[min_key]
        axis_max = config[max_key]

        if axis_min is None or axis_max is None:
            raise ValueError(
                f"Heatmap '{sheet_name}' requiere {min_key}/{max_key} para generacion streaming."
            )

        if bins <= 0:
            raise ValueError(f"Heatmap '{sheet_name}' requiere bins > 0.")

        axis_min = float(axis_min)
        axis_max = float(axis_max)
        if axis_max <= axis_min:
            raise ValueError(
                f"Heatmap '{sheet_name}' tiene rango invalido {min_key}={axis_min}, {max_key}={axis_max}."
            )

        return np.linspace(axis_min, axis_max, bins + 1)

    def _chunk_filepaths(self, output_dir):
        """Lista los chunks de episodios en orden estable."""
        if not os.path.exists(output_dir):
            return []

        chunk_filenames = sorted([
            filename for filename in os.listdir(output_dir)
            if filename.startswith('episodes_chunks_') and filename.endswith('.json')
        ])
        return [os.path.join(output_dir, filename) for filename in chunk_filenames]

    def _iter_chunk_episodes(self, chunk_filepath):
        """
        Itera episodios de un archivo JSON array sin cargar el chunk completo.
        Cada objeto de episodio si se materializa, pero se libera antes del siguiente.
        """
        decoder = json.JSONDecoder()
        read_size = 4 * 1024 * 1024
        buffer = ''
        array_open = False
        reached_eof = False

        with open(chunk_filepath, 'r', encoding='utf-8') as file:
            while True:
                if not reached_eof:
                    chunk = file.read(read_size)
                    if chunk:
                        buffer += chunk
                    else:
                        reached_eof = True

                while True:
                    buffer = buffer.lstrip()
                    if not array_open:
                        if not buffer:
                            break
                        if buffer[0] != '[':
                            raise ValueError(f"Chunk JSON invalido: {chunk_filepath}")
                        buffer = buffer[1:]
                        array_open = True
                        continue

                    buffer = buffer.lstrip()
                    if not buffer:
                        break
                    if buffer[0] == ']':
                        return
                    if buffer[0] == ',':
                        buffer = buffer[1:]
                        continue

                    try:
                        episode, offset = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        if reached_eof:
                            raise
                        break

                    yield episode
                    buffer = buffer[offset:]

                if reached_eof:
                    if not buffer.strip() or buffer.strip() == ']':
                        return
                    raise ValueError(f"Chunk JSON incompleto: {chunk_filepath}")

    def _accumulate_episode(self, heatmap_jobs, episode):
        """Acumula un episodio en todos los heatmaps aplicables."""
        step_data = episode['step_data']
        numeric_cache = {}

        for job in heatmap_jobs:
            filter_termination_reason = job['filter_termination_reason']
            if filter_termination_reason and not self._episode_matches_filter(
                episode,
                filter_termination_reason
            ):
                continue

            x_values = self._numeric_step_series(step_data, job['x_variable'], numeric_cache)
            y_values = self._numeric_step_series(step_data, job['y_variable'], numeric_cache)
            n_values = min(x_values.size, y_values.size)

            if n_values == 0:
                continue

            x_values = x_values[:n_values]
            y_values = y_values[:n_values]
            finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
            observed_count = int(np.count_nonzero(finite_mask))

            if observed_count == 0:
                continue

            H, _, _ = np.histogram2d(
                x_values[finite_mask],
                y_values[finite_mask],
                bins=[job['x_edges'], job['y_edges']]
            )

            job['histogram'] += H
            job['matched_episode_count'] += 1
            job['observed_sample_count'] += observed_count
            job['in_range_sample_count'] += int(H.sum())

    def _numeric_step_series(self, step_data, variable_name, numeric_cache):
        """Convierte una serie de step_data a ndarray numerico y la cachea por episodio."""
        if variable_name not in numeric_cache:
            values = step_data[variable_name]
            if not isinstance(values, list):
                raise TypeError(f"step_data['{variable_name}'] debe ser una lista.")
            numeric_cache[variable_name] = np.asarray(values, dtype=float)

        return numeric_cache[variable_name]

    def _episode_matches_filter(self, episode, filter_termination_reason):
        """Evalua si un episodio cumple el filtro de termino."""
        return self._termination_reason(episode) in filter_termination_reason

    def _termination_reason(self, episode):
        """Obtiene la razon de termino persistida por episodio."""
        if 'termination_reason' in episode:
            return episode['termination_reason']

        if 'end_episode_data' in episode and 'end_termination_reason' in episode['end_episode_data']:
            return episode['end_episode_data']['end_termination_reason']

        return None

    def _write_heatmap_workbook(self, heatmap_jobs, output_excel_target_filepath):
        """Escribe grillas de heatmap en un xlsx temporal y luego reemplaza el destino."""
        skipped_heatmaps = []
        written_sheets = 0
        temp_filepath = f"{output_excel_target_filepath}.tmp.xlsx"

        try:
            with pd.ExcelWriter(temp_filepath, engine='openpyxl') as writer:
                for job in heatmap_jobs:
                    if job['in_range_sample_count'] == 0:
                        skipped_heatmaps.append({
                            'sheet_name': job['sheet_name'],
                            'x_variable': job['x_variable'],
                            'y_variable': job['y_variable'],
                            'filter_termination_reason': str(job['filter_termination_reason']),
                            'matched_episode_count': job['matched_episode_count'],
                            'observed_sample_count': job['observed_sample_count'],
                            'reason': 'empty_after_filter_or_axis_range'
                        })
                        logger.warning(
                            "[HeatmapGenerator] No hay datos en rango para heatmap '%s'.",
                            job['sheet_name']
                        )
                        continue

                    df_grid = pd.DataFrame(
                        job['histogram'].T,
                        index=job['y_centers'],
                        columns=job['x_centers']
                    )
                    df_grid.index.name = job['y_variable']
                    df_grid.columns.name = job['x_variable']
                    df_grid.to_excel(writer, sheet_name=job['sheet_name'])
                    written_sheets += 1
                    logger.info(
                        "[HeatmapGenerator] Hoja '%s' escrita con shape %s y %s muestras en rango.",
                        job['sheet_name'],
                        df_grid.shape,
                        job['in_range_sample_count']
                    )

                if skipped_heatmaps:
                    pd.DataFrame(skipped_heatmaps).to_excel(
                        writer,
                        sheet_name='_skipped_heatmaps',
                        index=False
                    )

                if written_sheets == 0 and not skipped_heatmaps:
                    pd.DataFrame([{'message': 'no_heatmap_data'}]).to_excel(
                        writer,
                        sheet_name='_empty_heatmaps',
                        index=False
                    )

            os.replace(temp_filepath, output_excel_target_filepath)
            logger.info("[HeatmapGenerator] Archivo Excel generado: %s", output_excel_target_filepath)

        except Exception:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            raise

    def _write_empty_workbook(self, output_excel_target_filepath, message):
        """Escribe un workbook valido cuando no hay datos para heatmaps."""
        temp_filepath = f"{output_excel_target_filepath}.tmp.xlsx"
        try:
            with pd.ExcelWriter(temp_filepath, engine='openpyxl') as writer:
                pd.DataFrame([{'message': message}]).to_excel(
                    writer,
                    sheet_name='_empty_heatmaps',
                    index=False
                )
            os.replace(temp_filepath, output_excel_target_filepath)
            logger.info("[HeatmapGenerator] Archivo Excel vacio generado: %s", output_excel_target_filepath)
        except Exception:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            raise
