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
        
        # Extraer llaves configuradas del template
        episode_template = config_template_output['episode_data']
        
        step_template = episode_template['step_data']
        interval_template = episode_template['interval_data']
        
        # Llaves configuradas: todas las que tienen [] como valor (listas vacías)
        # Excluir contadores (n_step, n_interval)
        self.config_step_keys = self._extract_list_keys(step_template, exclude={'n_step'})
        self.config_interval_keys = self._extract_list_keys(interval_template, exclude={'n_interval'})
        self.step_keys = []
        self.interval_keys = []
        
        # Template de end_episode_data (se usa como referencia, no como filtro)
        self.end_episode_template = episode_template['end_episode_data']
        
        # Buffers del episodio actual
        self.current_episode_id = None
        self.step_data = {}
        self.interval_data = {}
        self.step_count = 0
        self.interval_count = 0
        self.step_keys_initialized = False
        self.interval_keys_initialized = False
    
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
        self.step_keys = []
        self.interval_keys = []
        self.step_data = {}
        self.interval_data = {}
        self.step_count = 0
        self.interval_count = 0
        self.step_keys_initialized = False
        self.interval_keys_initialized = False

    def _initialize_step_keys(self, step_flat_data):
        self.step_keys = self._existing_configured_keys(self.config_step_keys, step_flat_data)
        self.step_data = {key: [] for key in self.step_keys}
        self.step_keys_initialized = True

    def _initialize_interval_keys(self, interval_flat_data):
        self.interval_keys = self._existing_configured_keys(self.config_interval_keys, interval_flat_data)
        self.interval_data = {key: [] for key in self.interval_keys}
        self.interval_keys_initialized = True

    def _existing_configured_keys(self, configured_keys, flat_data):
        active_keys = []
        for key in configured_keys:
            if key in flat_data:
                active_keys.append(key)
        return active_keys
    
    def on_step(self, step_flat_data):
        """
        Registra un step: hace append sobre las llaves activas.
        Recibe un dict plano ya construido con get_records() de cada componente.
        
        Args:
            step_flat_data (dict): Dict plano con llaves canónicas step-level
                (t_sec, <var>_raw, <var>_norm, cart_force, error_<var_obj>, ...)
        """
        if not self.step_keys_initialized:
            self._initialize_step_keys(step_flat_data)

        # Append sobre cada llave activa real
        for key in self.step_keys:
            self.step_data[key].append(step_flat_data[key])
        
        self.step_count += 1
    
    def on_interval_end(self, interval_flat_data):
        """
        Registra el cierre de un intervalo: hace append sobre las llaves activas de interval_data.
        
        Args:
            interval_flat_data (dict): Dict plano con todas las llaves interval-level
                ya aplanadas por SimulationManager (interval_id, terminated, 
                termination_reason, L_e_*, rewards, actions, learn_info, etc.)
        """
        if not self.interval_keys_initialized:
            self._initialize_interval_keys(interval_flat_data)

        for key in self.interval_keys:
            self.interval_data[key].append(interval_flat_data[key])
        
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
        self.step_keys = []
        self.interval_keys = []
        self.step_keys_initialized = False
        self.interval_keys_initialized = False
        self.current_episode_id = None
    
    def finalize_run(self):
        """
        Finaliza la corrida y fuerza flush de datos pendientes.
        """
        self.result_handler.finalize_run()
    
