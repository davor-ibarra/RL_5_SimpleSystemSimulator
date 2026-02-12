"""
metric_collector.py

Responsabilidad:
Único punto de recolección del episodio. Mantiene buffers columnar vivos
(dict-de-listas) que crecen incrementalmente por append en cada evento.

Produce un output canónico con tres secciones:
  - step_data:    series por step (n_step listas de igual largo)
  - interval_data: series por intervalo (n_interval listas de igual largo)
  - end_episode_data: resumen y snapshots del episodio

El collector no contiene lógica de negocio: solo recibe dicts crudos
de los componentes upstream, aplana las llaves necesarias, hace append
sobre las activas según el template, y empaqueta al cierre.
"""

import copy


class MetricCollector:
    """
    Recolecta métricas durante la simulación con append incremental.
    
    Usa el template de output para determinar qué llaves capturar.
    Delega persistencia a ResultHandler.
    """
    
    def __init__(self, result_handler, config_template_output):
        """
        Inicializa el colector de métricas.
        
        Args:
            result_handler: Instancia de ResultHandler para persistencia
            config_template_output (dict): Template de output (sub_config_template_output)
        """
        self.result_handler = result_handler
        
        # Extraer llaves activas del template
        episode_template = config_template_output['episode_data']
        
        step_template = episode_template['step_data']
        interval_template = episode_template['interval_data']
        
        # Llaves activas: todas las que tienen [] como valor (listas vacías)
        # Excluir contadores (n_step, n_interval)
        self.step_keys = self._extract_list_keys(step_template, exclude={'n_step'})
        self.interval_keys = self._extract_list_keys(interval_template, exclude={'n_interval'})
        
        # Template de end_episode_data (se usa como referencia, no como filtro)
        self.end_episode_template = episode_template['end_episode_data']
        
        # Buffers del episodio actual
        self.current_episode_id = None
        self.step_data = {}
        self.interval_data = {}
        self.step_count = 0
        self.interval_count = 0
    
    def _extract_list_keys(self, template_section, exclude=None):
        """
        Extrae las llaves activas de una sección del template.
        Una llave activa es aquella cuyo valor es una lista vacía [].
        
        Args:
            template_section (dict): Sección del template
            exclude (set): Llaves a excluir
            
        Returns:
            list: Lista ordenada de llaves activas
        """
        if exclude is None:
            exclude = set()
        
        active_keys = []
        for key, value in template_section.items():
            if key in exclude:
                continue
            if isinstance(value, list):
                active_keys.append(key)
        return active_keys
    
    def on_episode_start(self, episode_id):
        """
        Inicializa buffers columnar para un nuevo episodio.
        Cada llave activa comienza como lista vacía.
        
        Args:
            episode_id (int): Identificador del episodio
        """
        self.current_episode_id = episode_id
        self.step_data = {key: [] for key in self.step_keys}
        self.interval_data = {key: [] for key in self.interval_keys}
        self.step_count = 0
        self.interval_count = 0
    
    def on_step(self, dynamic_system_state_norm_dict, dynamic_system_params_record,
                controller_state_record, current_time):
        """
        Registra un step: aplana los dicts crudos y hace append sobre las llaves activas.
        
        Args:
            dynamic_system_state_norm_dict (dict): Estado normalizado {var: val}
            dynamic_system_params_record (dict): Record del sistema {raw_state: {...}, params: {...}}
            controller_state_record (dict): Record del controlador {global_controller: {...}, <ctrl>: {...}}
            current_time (float): Tiempo actual en segundos
        """
        # Aplanar todos los dicts crudos en un único dict plano
        flat = self._flatten_step_data(
            dynamic_system_state_norm_dict,
            dynamic_system_params_record,
            controller_state_record,
            current_time
        )
        
        # Append sobre cada llave activa
        for key in self.step_keys:
            self.step_data[key].append(flat.get(key))
        
        self.step_count += 1
    
    def on_interval_end(self, interval_flat_data):
        """
        Registra el cierre de un intervalo: hace append sobre las llaves activas de interval_data.
        
        Args:
            interval_flat_data (dict): Dict plano con todas las llaves interval-level
                ya aplanadas por SimulationManager (interval_id, terminated, 
                termination_reason, L_e_*, rewards, actions, learn_info, etc.)
        """
        for key in self.interval_keys:
            self.interval_data[key].append(interval_flat_data.get(key))
        
        self.interval_count += 1
    
    def on_episode_end(self, episode_id, end_episode_data):
        """
        Cierra el episodio y delega a ResultHandler para persistencia.
        Empaqueta datos en la estructura canónica del template.
        
        Args:
            episode_id (int): Identificador del episodio
            end_episode_data (dict): Datos de cierre del episodio
                (end_terminated, end_termination_reason, q_tables, etc.)
        """
        episode_output = {
            'episode_id': episode_id,
            'step_data': {
                'n_step': self.step_count,
                **self.step_data
            },
            'interval_data': {
                'n_interval': self.interval_count,
                **self.interval_data
            },
            'end_episode_data': end_episode_data
        }
        
        # Delegar persistencia a ResultHandler
        self.result_handler.save_episode(episode_output)
        
        # Limpiar buffers
        self.step_data = {}
        self.interval_data = {}
        self.step_count = 0
        self.interval_count = 0
        self.current_episode_id = None
    
    def finalize_run(self):
        """
        Finaliza la corrida y fuerza flush de datos pendientes.
        """
        self.result_handler.finalize_run()
    
    # ── helpers de aplanamiento ──────────────────────────────────────────
    
    def _flatten_step_data(self, dynamic_system_state_norm_dict,
                           dynamic_system_params_record,
                           controller_state_record, current_time):
        """
        Aplana los dicts crudos de un step en un dict plano con las llaves del template.
        
        Convenciones de mapeo:
        - t_sec ← current_time
        - <var>_raw ← dynamic_system_params_record['raw_state'][var]
        - <var>_norm ← dynamic_system_state_norm_dict[var]
        - <param> ← dynamic_system_params_record['params'][param]
        - Llaves globales del controlador ← controller_state_record['global_controller']
        - Llaves per-controller ← controller_state_record[<controller_name>]
        
        Args:
            dynamic_system_state_norm_dict (dict): Estado normalizado
            dynamic_system_params_record (dict): Record del sistema dinámico
            controller_state_record (dict): Record del controlador
            current_time (float): Tiempo actual
            
        Returns:
            dict: Dict plano con todas las llaves disponibles
        """
        flat = {}
        
        # 1. Tiempo
        flat['t_sec'] = current_time
        
        # 2. Estado dinámico raw (cada var → <var>_raw)
        raw_state = dynamic_system_params_record.get('raw_state', {})
        for var_name, value in raw_state.items():
            flat[f'{var_name}_raw'] = value
        
        # 3. Estado dinámico normalizado (cada var → <var>_norm)
        for var_name, value in dynamic_system_state_norm_dict.items():
            flat[f'{var_name}_norm'] = value
        
        # 4. Parámetros del sistema (cart_force, etc.) — llaves planas
        params = dynamic_system_params_record.get('params', {})
        flat.update(params)
        
        # 5. Controlador global — llaves planas directas
        global_ctrl = controller_state_record.get('global_controller', {})
        flat.update(global_ctrl)
        
        # 6. Controladores individuales — llaves ya canónicas (error_<var>, kp_<var>, etc.)
        for section_name, section_data in controller_state_record.items():
            if section_name == 'global_controller':
                continue
            if isinstance(section_data, dict):
                flat.update(section_data)
        
        return flat
