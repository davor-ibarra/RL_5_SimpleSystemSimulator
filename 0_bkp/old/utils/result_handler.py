"""
result_handler.py

Responsabilidad:
Frontera única de persistencia. Compatible con el layout actual de resultados.
No procesa datos: recibe datos ya finales desde MetricCollector y los escribe.

Intención funcional:
- Persistir sin reinterpretación: metadata, episodios detallados agrupados en chunks,
  resumen global en Excel, estado del agente, y conversiones auxiliares existentes.
- Mantener convenciones de nombres/rutas para que la visualización y herramientas
  actuales sigan funcionando.
"""


class ResultHandler:
    """
    Manejador único de persistencia de resultados.
    
    Responsabilidades:
    - Administrar carpeta de corrida y rutas de salida
    - Escribir metadata como snapshot directo de configuración
    - Escribir chunks JSON tal como llegan
    - Escribir/actualizar Excel de resumen con filas
    - Guardar estado del agente cuando se solicite
    """
    
    def __init__(self, output_dir):
        """
        Inicializa el manejador de resultados.
        
        Args:
            output_dir (str): Directorio de salida para la corrida
        """
        pass
    
    def save_metadata(self, metadata_dict):
        """
        Escribe metadata como snapshot directo de la configuración usada.
        
        Args:
            metadata_dict (dict): Metadata a guardar
        """
        pass
    
    def save_episode_chunk(self, episodes_data, chunk_id):
        """
        Escribe chunks JSON tal como llegan (episodios ya procesados por el colector).
        
        Args:
            episodes_data (list): Lista de episodios procesados
            chunk_id (int): Identificador del chunk
        """
        pass
    
    def append_summary_row(self, summary_row):
        """
        Escribe/actualiza el Excel de resumen con una fila tal como llega.
        
        Args:
            summary_row (dict): Fila de resumen del episodio
        """
        pass
    
    def save_agent_state(self, agent, episode_id):
        """
        Guarda estado del agente cuando el orquestador lo solicite.
        
        Args:
            agent: Instancia del agente
            episode_id (int): Identificador del episodio
        """
        pass
    
    def ensure_output_structure(self):
        """
        Crea la estructura de directorios necesaria para la corrida.
        """
        pass
    
    def get_chunk_filepath(self, chunk_id):
        """
        Genera la ruta del archivo para un chunk específico.
        
        Args:
            chunk_id (int): Identificador del chunk
            
        Returns:
            str: Ruta completa del archivo del chunk
        """
        pass
    
    def get_summary_filepath(self):
        """
        Genera la ruta del archivo de resumen Excel.
        
        Returns:
            str: Ruta completa del archivo de resumen
        """
        pass
    
    def get_agent_state_filepath(self, episode_id):
        """
        Genera la ruta del archivo de estado del agente.
        
        Args:
            episode_id (int): Identificador del episodio
            
        Returns:
            str: Ruta completa del archivo de estado del agente
        """
        pass