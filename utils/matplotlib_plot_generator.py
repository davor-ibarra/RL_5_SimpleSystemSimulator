import logging
import os
import json
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import numpy as np
import seaborn as sns
import gc

from interfaces.plot_generator import PlotGenerator

# 1.1: Logger a nivel de módulo
logger = logging.getLogger(__name__)

class MatplotlibPlotGenerator(PlotGenerator):
    """
    Generates plots using Matplotlib.
    Loads pre-calculated heatmap data from an Excel file.
    Applies styling based on plot configuration using a common helper method.
    """
    def __init__(self):
        logger.info("MatplotlibPlotGenerator instance created.")

    # --- Método Helper Centralizado para Estilos ---

    def _apply_common_mpl_styles(self, ax: plt.Axes, fig: plt.Figure, plot_config_data: Dict[str, Any]):
        """
        Aplica estilos comunes de Matplotlib a los ejes y figura dados,
        leyendo la configuración desde plot_config['config'].
        Aplica límites y número de ticks de forma estricta.
        """
        # 1. Obtener sub-diccionario de configuración de estilo
        cfg = plot_config_data.get('config', {})
        plot_name = plot_config_data.get("name", f"plot_{plot_config_data.get('type', 'unknown')}_{plot_config_data.get('_internal_plot_index','?')}") # Para logging

        # 2. --- Títulos y Etiquetas ---
        if cfg.get('show_title', True):
            ax.set_title(cfg.get('title', plot_name), fontsize=cfg.get('title_fontsize', 14), pad=cfg.get('title_pad', 5))
        # Obtener texto de etiqueta actual o usar variable de config como fallback
        current_xlabel_text = ax.get_xlabel()
        current_ylabel_text = ax.get_ylabel()
        default_xlabel = current_xlabel_text or plot_config_data.get('x_variable', '')
        default_ylabel = current_ylabel_text or plot_config_data.get('y_variable', '')
        # Formatear defaults si son strings no vacíos
        final_xlabel = cfg.get('xlabel', default_xlabel.replace('_', ' ').title() if isinstance(default_xlabel, str) and default_xlabel else default_xlabel)
        final_ylabel = cfg.get('ylabel', default_ylabel.replace('_', ' ').title() if isinstance(default_ylabel, str) and default_ylabel else default_ylabel)
        ax.set_xlabel(final_xlabel, fontsize=cfg.get('xlabel_fontsize', 12), labelpad=cfg.get('xlabel_pad', 5))
        ax.set_ylabel(final_ylabel, fontsize=cfg.get('ylabel_fontsize', 12), labelpad=cfg.get('ylabel_pad', 5))

        # 3. --- Límites de Ejes (Aplicar estrictamente desde config) ---
        # Estos se aplican DESPUÉS de que el plot se haya dibujado (incluyendo imshow con extent)
        xmin, xmax = cfg.get('xmin'), cfg.get('xmax')
        ymin, ymax = cfg.get('ymin'), cfg.get('ymax')
        # Obtener límites actuales por si solo se especifica uno (min o max)
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        final_xmin = xmin if xmin is not None else current_xlim[0]
        final_xmax = xmax if xmax is not None else current_xlim[1]
        final_ymin = ymin if ymin is not None else current_ylim[0]
        final_ymax = ymax if ymax is not None else current_ylim[1]
        # Solo aplicar si los límites finales son válidos (evitar min > max)
        if final_xmax > final_xmin:
            ax.set_xlim(left=final_xmin, right=final_xmax)
        if final_ymax > final_ymin:
            ax.set_ylim(bottom=final_ymin, top=final_ymax)

        # 4. --- Ticks ---
        tick_fs = cfg.get('tick_fontsize', 10)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        ax.tick_params(axis='both', which='minor', labelsize=tick_fs * 0.8)

        # Configuración de Localización de Ticks (Número EXACTO)
        num_xticks_cfg = cfg.get('num_xticks')
        num_yticks_cfg = cfg.get('num_yticks')

        # Usar LinearLocator para número exacto de ticks si se especifica
        if isinstance(num_xticks_cfg, (int, float)) and num_xticks_cfg > 0:
            # LinearLocator necesita un entero
            num_xticks = int(num_xticks_cfg)
            ax.xaxis.set_major_locator(mticker.LinearLocator(numticks=num_xticks))
        # Formato de etiquetas X (evitar notación científica)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False)) # useMathText=False para formato simple
        ax.ticklabel_format(style='plain', axis='x', useOffset=False)

        if isinstance(num_yticks_cfg, (int, float)) and num_yticks_cfg > 0:
            num_yticks = int(num_yticks_cfg)
            ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=num_yticks))
        # Formato de etiquetas Y (evitar notación científica)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        
        # Decimales de ticks
        dec = cfg.get('tick_decimals')
        if dec is not None:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.{dec}f}"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: f"{y:.{dec}f}"))

        # Rotación de Etiquetas del Eje X
        xtick_rotation = cfg.get('xtick_rotation', 0)
        if xtick_rotation != 0:
            plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha="right", rotation_mode="anchor")
        else:
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # 5. --- Grid ---
        grid_visible = cfg.get('grid_on', False)
        # No mostrar grid en heatmaps para evitar desorden visual
        is_heatmap = plot_config_data.get('type') == 'heatmap'
        ax.grid(visible=(grid_visible and not is_heatmap), linestyle='--', linewidth=0.5, alpha=0.6)

        # 6. --- Leyenda ---
        show_legend_cfg = cfg.get('show_legend', False)
        legend_pos = cfg.get('legend_pos', 'best') # Obtener posición aquí
        handles, labels = ax.get_legend_handles_labels()
        if show_legend_cfg and handles:
            legend_fs = cfg.get('legend_fontsize', 'small')
            legend_title = cfg.get('legend_title', None)
            legend_transparency = cfg.get('legend_transparency', None)
            if legend_pos == 'outside':
                ax.legend(handles, labels, title=legend_title, bbox_to_anchor=(1.04, 1), loc='upper left', 
                          fontsize=legend_fs, title_fontsize=legend_fs, framealpha=legend_transparency)
            else:
                ax.legend(handles, labels, title=legend_title, loc=legend_pos, 
                          fontsize=legend_fs, title_fontsize=legend_fs, framealpha=legend_transparency)
        elif ax.get_legend() is not None:
             ax.get_legend().remove()

        # 7. --- Ajuste Final del Layout ---
        # Ajustar para leyenda exterior si es necesario
        rect_right = 0.88 if legend_pos == 'outside' and show_legend_cfg and handles else 1.0
        try:
            fig.tight_layout(rect=[0, 0, rect_right, 1])
        except Exception as e_tl:
            logger.warning(f"Advertencia al aplicar fig.tight_layout() para plot '{plot_name}': {e_tl}")


    # --- Métodos Privados para Cargar Datos (sin cambios) ---
    def _load_summary_data(self, output_root_path_plot: str) -> Optional[pd.DataFrame]:
        summary_path = os.path.join(output_root_path_plot, 'episodes_summary_data.xlsx')
        if not os.path.exists(summary_path): logger.error(f"Archivo de resumen no encontrado: {summary_path}"); return None
        try:
            df = pd.read_excel(summary_path, engine='openpyxl' if self._check_openpyxl() else None)
            logger.info(f"Datos de resumen cargados desde {summary_path} ({len(df)} filas).")
            return df
        except Exception as e: logger.error(f"Error cargando archivo de resumen '{summary_path}': {e}", exc_info=True); return None

    def _load_detailed_data(self, output_root_path_plot: str) -> Optional[pd.DataFrame]:
        all_aligned_data: List[pd.DataFrame] = []
        logger.info(f"Buscando archivos simulation_data_ep_*.json en: {output_root_path_plot} para datos detallados...")
        try:
            files = [f for f in os.listdir(output_root_path_plot) if f.startswith("simulation_data_ep_") and f.endswith(".json")]
            if not files: logger.warning("No se encontraron archivos de datos detallados."); return None
            try: files.sort(key=lambda name: int(name.split('_ep_')[-1].split('_to_')[0]))
            except (ValueError, IndexError): logger.warning("No se pudo ordenar los archivos de datos detallados.")
            raw_episode_list = []
            for filename in files:
                filepath = os.path.join(output_root_path_plot, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f: episodes_in_file = json.load(f)
                    if isinstance(episodes_in_file, list): raw_episode_list.extend([ep for ep in episodes_in_file if isinstance(ep, dict)])
                except Exception as e_load: logger.error(f"Error cargando o procesando {filename}: {e_load}. Saltando.")
            if not raw_episode_list: return None
            for i, episode_dict in enumerate(raw_episode_list):
                time_values = episode_dict.get('time')
                episode_id_val = episode_dict.get('episode', [f'ep_idx_{i}'])[0]
                if not isinstance(time_values, list) or not time_values: continue
                num_steps = len(time_values); ref_index = pd.RangeIndex(num_steps)
                temp_data = {'time': pd.Series(time_values, index=ref_index)}
                for metric, values in episode_dict.items():
                    if metric == 'time': continue
                    if isinstance(values, list):
                        s = pd.Series(index=ref_index, dtype=object); valid_len = min(len(values), num_steps)
                        s.iloc[:valid_len] = values[:valid_len]
                        temp_data[metric] = s
                    elif values is not None: temp_data[metric] = pd.Series([values] * num_steps, index=ref_index)
                try: episode_df = pd.DataFrame(temp_data); episode_df['episode'] = episode_id_val; all_aligned_data.append(episode_df)
                except Exception as e_df: logger.error(f"Error en DataFrame ep {episode_id_val}: {e_df}")
            if not all_aligned_data: return None
            df_combined = pd.concat(all_aligned_data, ignore_index=True); logger.info(f"Datos detallados cargados y aplanados: {len(df_combined)} timesteps."); return df_combined
        except Exception as e_main: logger.error(f"Error inesperado cargando datos detallados: {e_main}", exc_info=True); return None

    # --- Métodos Privados para Cargar Datos ---

    def _load_summary_data(self, output_root_path_plot: str) -> Optional[pd.DataFrame]:
        """Carga los datos de resumen desde episodes_summary.xlsx."""
        summary_path = os.path.join(output_root_path_plot, 'episodes_summary_data.xlsx')
        if not os.path.exists(summary_path):
            logger.error(f"Archivo de resumen no encontrado: {summary_path}")
            return None
        try:
            df = pd.read_excel(summary_path, engine='openpyxl' if self._check_openpyxl() else None)
            logger.info(f"Datos de resumen cargados desde {summary_path} ({len(df)} filas).")
            return df
        except Exception as e:
            logger.error(f"Error cargando archivo de resumen '{summary_path}': {e}", exc_info=True)
            return None
    
    def _load_detailed_data(self, output_root_path_plot: str) -> Optional[pd.DataFrame]:
        """
        Carga y combina datos detallados de archivos simulation_data_ep_*.json.
        (Mantenemos la lógica anterior aquí, aunque HeatmapGenerator tiene una versión similar).
        """
        all_aligned_data: List[pd.DataFrame] = []
        logger.info(f"Buscando archivos simulation_data_ep_*.json en: {output_root_path_plot} para datos detallados...")
        try:
            files = [f for f in os.listdir(output_root_path_plot) if f.startswith("simulation_data_ep_") and f.endswith(".json")]
            if not files:
                logger.warning("No se encontraron archivos de datos detallados para plots que los requieran.")
                return None
            try:
                files.sort(key=lambda name: int(name.split('_ep_')[-1].split('_to_')[0]))
            except (ValueError, IndexError):
                logger.warning("No se pudo ordenar los archivos de datos detallados por número de episodio.")

            raw_episode_list = []
            for filename in files:
                filepath = os.path.join(output_root_path_plot, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        episodes_in_file = json.load(f)
                    if isinstance(episodes_in_file, list):
                        raw_episode_list.extend([ep for ep in episodes_in_file if isinstance(ep, dict)])
                except Exception as e_load:
                    logger.error(f"Error cargando o procesando {filename}: {e_load}. Saltando.")

            if not raw_episode_list: return None

            for i, episode_dict in enumerate(raw_episode_list):
                time_values = episode_dict.get('time')
                episode_id_val = episode_dict.get('episode', [f'ep_idx_{i}'])[0] # Default ID
                if not isinstance(time_values, list) or not time_values: continue
                num_steps = len(time_values)
                ref_index = pd.RangeIndex(num_steps)
                temp_data = {'time': pd.Series(time_values, index=ref_index)}
                for metric, values in episode_dict.items():
                    if metric == 'time': continue
                    if isinstance(values, list):
                        s = pd.Series(index=ref_index, dtype=object)
                        valid_len = min(len(values), num_steps); s.iloc[:valid_len] = values[:valid_len]
                        temp_data[metric] = s
                    elif values is not None:
                        temp_data[metric] = pd.Series([values] * num_steps, index=ref_index)

                try:
                    episode_df = pd.DataFrame(temp_data); episode_df['episode'] = episode_id_val
                    all_aligned_data.append(episode_df)
                except Exception as e_df: logger.error(f"Error en DataFrame ep {episode_id_val}: {e_df}")

            if not all_aligned_data: return None
            df_combined = pd.concat(all_aligned_data, ignore_index=True)
            logger.info(f"Datos detallados cargados y aplanados: {len(df_combined)} timesteps.")
            return df_combined
        except Exception as e_main:
            logger.error(f"Error inesperado cargando datos detallados: {e_main}", exc_info=True)
            return None

    def _load_heatmap_data(self, output_root_path_plot: str, plot_config_data: Dict) -> Optional[pd.DataFrame]:
        """Carga los datos pre-calculados para un heatmap específico desde data_heatmaps.xlsx."""
        heatmap_data_path = os.path.join(output_root_path_plot, 'data_heatmaps.xlsx')
        plot_name_cfg = plot_config_data.get('name')
        plot_index = plot_config_data.get('_internal_plot_index', '?') # Usar índice interno

        # 2.1: Determinar nombre de hoja consistentemente con HeatmapGenerator
        if plot_name_cfg:
            sheet_name = str(plot_name_cfg)[:31] # Limitar a 31 caracteres
        else:
            # Usar nombre por defecto si 'name' no está en config
            sheet_name = f"heatmap_idx_{plot_index}"[:31]
            # La advertencia sobre falta de 'name' ya la dio HeatmapGenerator

        log_name_ref = f"name='{plot_name_cfg}'" if plot_name_cfg else f"index={plot_index}"

        if not os.path.exists(heatmap_data_path):
            logger.error(f"Archivo de datos para heatmaps no encontrado: {heatmap_data_path}. El heatmap '{log_name_ref}' no se puede generar.")
            return None
        try:
            logger.info(f"Cargando datos para heatmap ({log_name_ref}) desde hoja '{sheet_name}' en {heatmap_data_path}...")
            # 2.2: Leer Excel especificando la columna 0 como índice (centros de bins Y)
            #      Usar openpyxl si está disponible para compatibilidad .xlsx
            df_grid = pd.read_excel(heatmap_data_path,
                                    sheet_name=sheet_name,
                                    index_col=0,
                                    engine='openpyxl' if self._check_openpyxl() else None)
            logger.info(f"Datos cargados para hoja '{sheet_name}' ({df_grid.shape[0]}x{df_grid.shape[1]}).")

            # 2.3: Asegurar que los índices y columnas sean flotantes (centros de bins)
            #      El archivo Excel debería tenerlos ya como números si HeatmapGenerator funcionó bien.
            try:
                # Las columnas leídas de Excel pueden ser strings si la primera fila no era numérica
                df_grid.columns = pd.to_numeric(df_grid.columns)
                # El índice (index_col=0) debería ser numérico directamente si la columna 0 lo era
                df_grid.index = pd.to_numeric(df_grid.index)
            except ValueError as e_conv:
                logger.warning(f"Heatmap ('{log_name_ref}'): No se pudieron convertir índices/columnas de la hoja '{sheet_name}' a números. Verificar el archivo Excel. Error: {e_conv}")
                # Devolver None porque los ejes serán incorrectos
                return None
            except Exception as e_idx:
                logger.warning(f"Heatmap ('{log_name_ref}'): Error procesando índices/columnas de la hoja '{sheet_name}': {e_idx}. Verificar el archivo Excel.")
                return None

            # 2.4: Verificar que los nombres de los ejes (que son las variables originales) se cargaron
            if df_grid.index.name is None or df_grid.columns.name is None:
                logger.warning(f"Heatmap ('{log_name_ref}'): No se pudieron determinar los nombres de los ejes (variables originales) desde la hoja '{sheet_name}'. Se usarán x/y variables de la config.")
                # Intentar asignar desde config si faltan (deben coincidir con cómo guardó HeatmapGenerator)
                df_grid.index.name = df_grid.index.name or plot_config_data.get('y_variable', 'Y')
                df_grid.columns.name = df_grid.columns.name or plot_config_data.get('x_variable', 'X')

            return df_grid
        except ValueError as e_sheet: # Específicamente para error de nombre de hoja
            if 'Worksheet named' in str(e_sheet) or 'No sheet named' in str(e_sheet):
                logger.error(f"Heatmap ('{log_name_ref}'): No se encontró la hoja '{sheet_name}' en {heatmap_data_path}. Verificar 'name' en config YAML y si HeatmapGenerator creó la hoja.")
            else:
                logger.error(f"Error de valor leyendo hoja '{sheet_name}' para heatmap '{log_name_ref}': {e_sheet}", exc_info=True)
            return None
        except FileNotFoundError: # Ya chequeado arriba, pero por si acaso
            logger.error(f"Archivo heatmap {heatmap_data_path} no encontrado (re-check).")
            return None
        except Exception as e: # Otros errores (permisos, archivo corrupto, etc.)
            logger.error(f"Error inesperado cargando datos para heatmap '{log_name_ref}' desde hoja '{sheet_name}': {e}", exc_info=True)
            return None

    def _check_openpyxl(self) -> bool:
        """Verifica si openpyxl está instalado."""
        try:
            import openpyxl # type: ignore
            return True
        except ImportError:
            return False

    # --- Método Principal de Generación ---

    def generate_plot(self, plot_config_data: Dict[str, Any], output_root_path_plot: str):
        """Genera un plot basado en la configuración dada."""
        plot_type = plot_config_data.get("type")
        plot_index = plot_config_data.get('_internal_plot_index','?') # Índice para logging/defaults
        plot_name_cfg = plot_config_data.get("name")
        log_name_ref = f"name='{plot_name_cfg}'" if plot_name_cfg else f"index={plot_index}"

        # 2.5: Usar nombre base más informativo si 'name' falta
        plot_name_default = f"plot_{plot_type}_{plot_index}"
        output_filename_default = f"{plot_name_default}.png"
        output_filename = plot_config_data.get("output_filename", output_filename_default)
        output_path = os.path.join(output_root_path_plot, output_filename)

        style_config = plot_config_data.get("config", {}) # Sub-diccionario de estilo

        if not plot_type:
            logger.error(f"Plot config ({log_name_ref}) no tiene 'type'. Saltando.")
            return

        logger.info(f"Generando plot ({log_name_ref}) (Tipo: {plot_type}) -> {output_filename}")

        # --- Carga de Datos Específica por Tipo ---
        data_loaded: Optional[pd.DataFrame] = None
        data_source_type = plot_config_data.get("source") # summary, detailed

        if plot_type == "heatmap":
            # Heatmap carga sus datos DENTRO de _generate_heatmap_plot
            pass
        elif data_source_type == "summary":
            data_loaded = self._load_summary_data(output_root_path_plot)
            if data_loaded is None or data_loaded.empty:
                logger.warning(f"No se pudieron cargar datos de resumen para plot ({log_name_ref}). Se generará un plot vacío si es posible.")
                # No retornar aún, permitir que la función de ploteo maneje data=None
        elif data_source_type == "detailed":
            data_loaded = self._load_detailed_data(output_root_path_plot)
            if data_loaded is None or data_loaded.empty:
                logger.warning(f"No se pudieron cargar datos detallados para plot ({log_name_ref}). Se generará un plot vacío si es posible.")
                # No retornar aún
        else:
            # Para plots que no son heatmap, la fuente es requerida
            logger.error(f"Plot config ({log_name_ref}) tiene tipo '{plot_type}' pero falta 'source' ('summary' o 'detailed'). Saltando.")
            return

        # --- Creación de Figura y Ejes ---
        figsize = (style_config.get('figsize_w', 12), style_config.get('figsize_h', 6))
        fig, ax = plt.subplots(figsize=figsize)

        try:
            # --- Dispatching a función de ploteo específica ---
            plot_generated = False # Flag para saber si se llamó a una función de ploteo
            if plot_type == "line":
                self._generate_line_plot(ax, plot_config_data, data_loaded)
                plot_generated = True
            elif plot_type == "scatter":
                self._generate_scatter_plot(ax, plot_config_data, data_loaded)
                plot_generated = True
            elif plot_type == "histogram":
                self._generate_histogram(ax, plot_config_data, data_loaded)
                plot_generated = True
            elif plot_type == "bar":
                self._generate_bar_plot(ax, plot_config_data, data_loaded)
                plot_generated = True
            elif plot_type == "heatmap":
                # Pasar fig también porque heatmap necesita añadir colorbar
                self._generate_heatmap_plot(ax, fig, plot_config_data, output_root_path_plot)
                plot_generated = True
            else:
                # Este caso no debería ocurrir si type se valida antes, pero por seguridad
                plt.close(fig) # Cerrar figura no usada
                raise NotImplementedError(f"Tipo de plot desconocido o no implementado: '{plot_type}'")

            # --- Aplicar Estilos Comunes ---
            # 2.6: Llamar al helper de estilos DESPUÉS de que el plot principal se haya dibujado
            self._apply_common_mpl_styles(ax, fig, plot_config_data)

            # --- Guardar Plot (solo si se generó algo) ---
            if plot_generated:
                # Usar tight_layout ANTES de guardar para ajustar
                try: fig.tight_layout()
                except Exception as e_tl: 
                    logger.warning(f"Warning al aplicar fig.tight_layout() para {output_filename}: {e_tl}")

                plt.savefig(output_path, bbox_inches='tight') # bbox_inches='tight' es importante
                logger.info(f"Plot guardado en: {output_path}")
            else:
                logger.warning(f"No se generó contenido para el plot ({log_name_ref}). No se guardará archivo.")

        except NotImplementedError as nie:
            logger.error(f"Error en plot ({log_name_ref}): {nie}")
        except FileNotFoundError as fnfe: # Errores de carga de datos específicos
            logger.error(f"Error en plot ({log_name_ref}): Archivo de datos no encontrado - {fnfe}")
        except ValueError as ve: # Errores de datos o configuración inválidos
            logger.error(f"Error en plot ({log_name_ref}): Datos o configuración inválidos - {ve}")
        except KeyError as ke: # Errores por claves faltantes en config o datos
            logger.error(f"Error en plot ({log_name_ref}): Clave faltante - {ke}")
        except Exception as e: # Capturar cualquier otro error inesperado
            logger.error(f"Error inesperado generando plot ({log_name_ref}): {e}", exc_info=True)
        finally:
            # 2.7: Asegurar limpieza de datos y que la figura se cierre siempre
            if 'data_loaded' in locals() and data_loaded is not None:
                del data_loaded
            plt.close(fig)

    # --- Métodos Privados para cada tipo de Plot ---

    def _generate_line_plot(self, ax, plot_config_data: Dict, data: Optional[pd.DataFrame]):
        """Genera un gráfico de línea."""
        style_config = plot_config_data.get('config', {})
        plot_name = plot_config_data.get("name", "line_plot")
        if data is None or data.empty:
            logger.warning(f"Datos vacíos para line plot '{plot_name}'. Se generará plot vacío.")
            ax.text(0.5, 0.5, 'Datos no disponibles', **{'ha':'center', 'va':'center', 'color':'gray'})
            # Aplicar título incluso si está vacío
            ax.set_title(style_config.get('title', plot_name), fontsize=style_config.get('title_fontsize', 14))
            return

        x_var = plot_config_data.get('x_variable')
        y_var = plot_config_data.get('y_variable')
        if not x_var or not y_var: raise ValueError(f"Line plot '{plot_name}': Faltan 'x_variable' o 'y_variable'.")
        if x_var not in data.columns or y_var not in data.columns: raise ValueError(f"Line plot '{plot_name}': Columnas '{x_var}' o '{y_var}' no encontradas.")

        df_filtered = data.copy()
        filter_reason = style_config.get('filter_termination_reason')
        if filter_reason and isinstance(filter_reason, list) and 'termination_reason' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['termination_reason'].isin(filter_reason)]
            if df_filtered.empty: logger.warning(f"Line plot '{plot_name}': No quedan datos después de filtrar por {filter_reason}. Plot vacío.")

        if not df_filtered.empty:
            x_data = pd.to_numeric(df_filtered[x_var], errors='coerce')
            y_data = pd.to_numeric(df_filtered[y_var], errors='coerce')
            valid_idx = x_data.notna() & y_data.notna()
            if valid_idx.any():
                x_plot, y_plot = x_data[valid_idx], y_data[valid_idx]
                # Ordenar por X para que la línea tenga sentido
                sort_order = x_plot.argsort()
                ax.plot(x_plot.iloc[sort_order], y_plot.iloc[sort_order],
                        color=style_config.get('line_color', '#1f77b4'),
                        linewidth=style_config.get('line_width', 1.0),
                        marker=style_config.get('marker_style', ''),
                        markersize=style_config.get('marker_size', 3),
                        markerfacecolor=style_config.get('marker_color', '#ff7f0e'),
                        markeredgecolor=style_config.get('marker_color', '#ff7f0e'),
                        label=y_var) # Añadir label para leyenda automática
            else: logger.warning(f"Line plot '{plot_name}': No quedan datos numéricos válidos después de coerción/filtrado.")

    def _generate_scatter_plot(self, ax, plot_config_data: Dict, data: Optional[pd.DataFrame]):
        """Genera un gráfico de dispersión."""
        style_config = plot_config_data.get('config', {})
        plot_name = plot_config_data.get("name", "scatter_plot")
        if data is None or data.empty:
            logger.warning(f"Datos vacíos para scatter plot '{plot_name}'. Se generará plot vacío.")
            ax.text(0.5, 0.5, 'Datos no disponibles', **{'ha':'center', 'va':'center', 'color':'gray'})
            ax.set_title(style_config.get('title', plot_name), fontsize=style_config.get('title_fontsize', 14))
            return

        x_var = plot_config_data.get('x_variable')
        y_var = plot_config_data.get('y_variable')
        if not x_var or not y_var: raise ValueError(f"Scatter plot '{plot_name}': Faltan 'x_variable' o 'y_variable'.")
        if x_var not in data.columns or y_var not in data.columns: raise ValueError(f"Scatter plot '{plot_name}': Columnas '{x_var}' o '{y_var}' no encontradas.")

        df_filtered = data.copy() # Añadir lógica de filtrado

        if not df_filtered.empty:
            x_data = pd.to_numeric(df_filtered[x_var], errors='coerce')
            y_data = pd.to_numeric(df_filtered[y_var], errors='coerce')
            valid_idx = x_data.notna() & y_data.notna()
            if valid_idx.any():
                ax.scatter(x_data[valid_idx], y_data[valid_idx],
                            s=style_config.get('marker_size', 20),
                            c=style_config.get('marker_color', '#1f77b4'),
                            marker=style_config.get('marker_style', 'o'),
                            alpha=style_config.get('alpha', 0.6),
                            label=f"{y_var} vs {x_var}") # Label para leyenda
            else: logger.warning(f"Scatter plot '{plot_name}': No quedan datos numéricos válidos.")

    def _generate_histogram(self, ax, plot_config_data: Dict, data: Optional[pd.DataFrame]):
        """Genera un histograma."""
        style_config = plot_config_data.get('config', {})
        plot_name = plot_config_data.get("name", "histogram")
        if data is None or data.empty:
            logger.warning(f"Datos vacíos para histogram '{plot_name}'. Se generará plot vacío.")
            ax.text(0.5, 0.5, 'Datos no disponibles', **{'ha':'center', 'va':'center', 'color':'gray'})
            ax.set_title(style_config.get('title', plot_name), fontsize=style_config.get('title_fontsize', 14))
            return

        variable = plot_config_data.get('variable')
        if not variable: raise ValueError(f"Histogram '{plot_name}': Falta 'variable'.")
        if variable not in data.columns: raise ValueError(f"Histogram '{plot_name}': Columna '{variable}' no encontrada.")

        values = pd.to_numeric(data[variable], errors='coerce').dropna()
        if values.empty:
            logger.warning(f"Histogram '{plot_name}': No quedan datos numéricos válidos para variable '{variable}'.")
            return

        ax.hist(values,
                bins=style_config.get('bins', 'auto'),
                color=style_config.get('color', '#1f77b4'),
                alpha=style_config.get('alpha', 0.7),
                density=style_config.get('density', False))

    def _generate_bar_plot(self, ax, plot_config_data: Dict, data: Optional[pd.DataFrame]):
        """Genera un gráfico de barras."""
        style_config = plot_config_data.get('config', {})
        plot_name = plot_config_data.get("name", "bar_plot")
        if data is None or data.empty:
            logger.warning(f"Datos vacíos para bar plot '{plot_name}'. Se generará plot vacío.")
            ax.text(0.5, 0.5, 'Datos no disponibles', **{'ha':'center', 'va':'center', 'color':'gray'})
            ax.set_title(style_config.get('title', plot_name), fontsize=style_config.get('title_fontsize', 14))
            return

        variable = plot_config_data.get('variable') # Variable categórica
        if not variable: raise ValueError(f"Bar plot '{plot_name}': Falta 'variable'.")
        if variable not in data.columns: raise ValueError(f"Bar plot '{plot_name}': Columna '{variable}' no encontrada.")

        df = data.copy()
        group_size = style_config.get('group_size')
        counts = None

        # Preparar datos para contar/agrupar
        try:
            if group_size and 'episode' in df.columns:
                df['episode'] = pd.to_numeric(df['episode'], errors='coerce').dropna().astype(int)
                if not df.empty:
                    df['episode_group'] = (df['episode'] // group_size) * group_size
                    # Contar ocurrencias de 'variable' dentro de cada grupo
                    counts = df.groupby('episode_group')[variable].value_counts().unstack(fill_value=0)
                    # 2.7: Asignar título de leyenda explícitamente para barras apiladas
                    plot_config_data['config']['legend_title'] = variable.replace('_',' ').title()
                else: logger.warning(f"Bar plot '{plot_name}': No hay episodios válidos para agrupar.")
            else:
                # Conteo simple de la variable categórica
                counts = df[variable].value_counts().sort_index()
        except Exception as e_prep:
            raise ValueError(f"Error preparando datos para bar plot '{plot_name}': {e_prep}")

        # Colormap y colores
        if counts is None or counts.empty:
            logger.warning(f"Bar plot '{plot_name}': No hay datos para contar/agrupar variable '{variable}'.")
            return # No se puede plotear


        legend_var_cfg = style_config.get("legend_var")
        custom_color_map, custom_label_map = {}, {}
        if isinstance(legend_var_cfg, list):
            for entry in legend_var_cfg:
                v = entry.get("value")
                if not v:
                    continue
                c = entry.get("color")
                l = entry.get("label")
                if c:
                    custom_color_map[v] = c
                if l:
                    custom_label_map[v] = l
        
        # Plotear barras
        if isinstance(counts, pd.DataFrame):  # Stacked bar
            # 1) DataFrame con etiquetas de leyenda renombradas
            counts_to_plot = counts.copy()
            if custom_label_map:
                counts_to_plot.columns = [custom_label_map.get(col, col) for col in counts.columns]

            # 2) Colores alineados al orden original de columnas
            if custom_color_map:
                colors = [custom_color_map.get(col, None) for col in counts.columns]
            else:
                cmap_name = style_config.get('cmap', 'tab10')
                try:
                    cmap = plt.get_cmap(cmap_name)
                except ValueError:
                    cmap = plt.get_cmap('tab10')
                colors = [cmap(i % cmap.N) for i in range(len(counts.columns))]

            counts_to_plot.plot(kind='bar', stacked=True, ax=ax,
                                width=style_config.get('bar_width', 0.8),
                                color=colors)

            # Forzar leyenda para stacked
            plot_config_data['config']['show_legend'] = True

            # Etiquetas X si hay agrupación
            if group_size:
                try:
                    ax.set_xticklabels([f"{int(i)}-{int(i+group_size-1)}" for i in counts.index])
                except ValueError:
                    ax.set_xticklabels(counts.index)
            else:
                ax.set_xticklabels(counts.index)

        else:  # Simple bar
            # 1) Serie con etiquetas renombradas
            counts_to_plot = counts.copy()
            if custom_label_map:
                counts_to_plot.index = [custom_label_map.get(idx, idx) for idx in counts.index]

            # 2) Colores
            if custom_color_map:
                colors = [custom_color_map.get(idx, None) for idx in counts.index]
            else:
                cmap_name = style_config.get('cmap', 'tab10')
                try:
                    cmap = plt.get_cmap(cmap_name)
                except ValueError:
                    cmap = plt.get_cmap('tab10')
                colors = [cmap(i % cmap.N) for i in range(len(counts))]

            counts_to_plot.plot(kind='bar', ax=ax,
                                width=style_config.get('bar_width', 0.8),
                                color=colors)
            ax.set_xticklabels(counts_to_plot.index)

    def _generate_heatmap_plot(self, ax: plt.Axes, fig: plt.Figure, plot_config_data: Dict, output_root_path_plot: str):
        """Genera un gráfico de heatmap. Usa extent y aplica límites estrictos."""
        style_config = plot_config_data.get('config', {})
        plot_name_cfg = plot_config_data.get('name'); plot_index = plot_config_data.get('_internal_plot_index', '?')
        log_name_ref = f"name='{plot_name_cfg}'" if plot_name_cfg else f"index={plot_index}"

        grid_df = self._load_heatmap_data(output_root_path_plot, plot_config_data)
        if grid_df is None or grid_df.empty:
            logger.warning(f"Datos vacíos para heatmap ({log_name_ref}). Plot vacío.")
            ax.text(0.5, 0.5, 'Datos Heatmap No Disponibles', **{'ha':'center', 'va':'center', 'color':'red', 'fontsize':12})
            return

        data_values = grid_df.values
        y_bin_centers = grid_df.index.to_numpy()
        x_bin_centers = grid_df.columns.to_numpy()

        # Reemplazar NaNs con 0
        finite_data_values = np.nan_to_num(data_values, nan=0.0)

        # Calcular extent (límites exteriores de los píxeles)
        plot_extent = None
        try:
            if len(x_bin_centers) > 1: x_width = np.mean(np.diff(x_bin_centers)); extent_x_min = x_bin_centers[0]-x_width/2.0; extent_x_max = x_bin_centers[-1]+x_width/2.0
            else: x_width = 1.0; extent_x_min = x_bin_centers[0]-x_width/2.0; extent_x_max = x_bin_centers[0]+x_width/2.0
            if len(y_bin_centers) > 1: y_width = np.mean(np.diff(y_bin_centers)); extent_y_min = y_bin_centers[0]-y_width/2.0; extent_y_max = y_bin_centers[-1]+y_width/2.0
            else: y_width = 1.0; extent_y_min = y_bin_centers[0]-y_width/2.0; extent_y_max = y_bin_centers[0]+y_width/2.0
            plot_extent = [extent_x_min, extent_x_max, extent_y_min, extent_y_max]
            #logger.debug(f"Heatmap ({log_name_ref}): Extent calculado: {plot_extent}")
        except Exception as e_extent: logger.warning(f"Heatmap ({log_name_ref}): No se pudo calcular 'extent': {e_extent}."); plot_extent = None

        cmap_name = style_config.get('cmap', 'viridis')
        cmin_cfg, cmax_cfg = style_config.get('cmin'), style_config.get('cmax')
        clabel = style_config.get('clabel', '')
        log_scale = style_config.get('log_scale', False)

        # Configurar normalización
        norm, vmin, vmax = None, cmin_cfg, cmax_cfg
        if log_scale:
            valid_data_for_log = finite_data_values[finite_data_values > 0]
            if valid_data_for_log.size > 0:
                 log_vmin = vmin if vmin is not None and vmin > 0 else np.min(valid_data_for_log)
                 log_vmax = vmax if vmax is not None else np.max(valid_data_for_log)
                 if log_vmin > 0 and log_vmax > log_vmin: norm = mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax); vmin, vmax = log_vmin, log_vmax
                 else: logger.warning(f"Heatmap ({log_name_ref}): No se pudo establecer LogNorm. Usando lineal."); log_scale = False
            else: logger.warning(f"Heatmap ({log_name_ref}): No hay datos positivos para LogNorm. Usando lineal."); log_scale = False
        if not log_scale: norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Ploteo con imshow y extent
        im = ax.imshow(finite_data_values, cmap=cmap_name, aspect='auto', interpolation='nearest', origin='lower', norm=norm, extent=plot_extent)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(clabel + (' (Log Scale)' if log_scale else ''), fontsize=style_config.get('clabel_fontsize', 10))
        cbar.ax.tick_params(labelsize=style_config.get('tick_fontsize', 10))

        # Asignar nombres de ejes (para que _apply_common los use)
        ax.set_xlabel(grid_df.columns.name)
        ax.set_ylabel(grid_df.index.name)