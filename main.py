"""
main.py

Responsabilidad:
Orquestar la macro-ejecución:
1. Cargar YAML (3 dicts)
2. Construir instancias explícitamente (sin DI)
3. Guardar metadata de corrida (JSON de parámetros del YAML)
4. Ejecutar simulation_manager.run_simulation()
5. Generar plots (post-run)
"""

import yaml
import os
import logging
from datetime import datetime

# Importaciones de componentes
from dynamic_system.dynamic_system_base import DynamicSystemBase
from controllers.controller_base import ControllerBase
from agents.agent_base import AgentBase
from metrics.metric_processing import MetricProcessing
from metrics.metric_collector import MetricCollector
from rewards.reward_calculator_base import RewardCalculatorBase
from utils.result_handler import ResultHandler
from utils.matplotlib_plot_generator import MatplotlibPlotGenerator
from utils.heatmap_generator import HeatmapGenerator
from simulation_manager import SimulationManager
from visualization_manager import VisualizationManager

# Configurar logger a nivel de módulo
logger = logging.getLogger(__name__)


def main():
    """
    Ejecuta el flujo completo y único: carga configs → prepara contexto de corrida → 
    instancia componentes → guarda metadata → corre simulación → ejecuta visualización post-run.
    """
    # 1. Cargar configuraciones (acceso directo, sin validación)
    config_main = load_config('config/config_CartPole.yaml')
    config_data_save = load_config('config/sub_config_data_save_CartPole.yaml')
    config_visualization = load_config('config/sub_config_visualization_CartPole.yaml')
    config_template_output = load_config('config/sub_config_template_output_CartPole.yaml')
    
    # 2. Construir run_id y output_dir
    run_id, timestamp = _build_run_id()
    base_dir = config_main['data_handling']['output_root']
    system_id = config_main['dynamic_system']['system_name']
    output_dir = _build_output_dir(base_dir, system_id, timestamp)
    
    # 3. Construir metadata
    metadata_dict = _build_metadata(run_id, timestamp, config_main, config_data_save, config_visualization)
    
    # 4. Instanciar componentes
    components = _build_components(config_main, config_data_save, config_template_output, output_dir)
    
    # 5. Guardar metadata
    _save_metadata(output_dir, metadata_dict, components['result_handler'])
    
    # 6. Ejecutar simulación
    _run_simulation(components['simulation_manager'])
    
    # 7. Ejecutar visualización post-run
    _run_visualization(components['visualization_manager'])
    
    print(f"[MAIN] Corrida completada. Resultados en: {output_dir}")


def load_config(path):
    """
    Lee YAML y retorna dict tal cual, sin validación ni defaults.
    
    Args:
        path (str): Ruta al archivo YAML
        
    Returns:
        dict: Configuración cargada
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def _build_run_id():
    """
    Genera identificador único de corrida (timestamp/uuid) para nombrar carpeta y artefactos.
    
    Returns:
        tuple: (run_id: str, timestamp: str)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"run_{timestamp}"
    return run_id, timestamp


def _build_output_dir(base_dir, system_id, run_id):
    """
    Construye la carpeta de salida de la corrida de forma determinista.
    
    Args:
        base_dir (str): Directorio base
        system_id (str): Identificador del sistema
        run_id (str): Identificador de corrida
        
    Returns:
        str: Ruta completa del directorio de salida
    """
    output_dir = os.path.join(base_dir, system_id, run_id)
    return output_dir


def _build_metadata(run_id, timestamp, config_main, config_data_save, config_visualization):
    """
    Construye un snapshot declarativo de parámetros usados (lo que se ejecutó).
    
    Args:
        run_id (str): Identificador de corrida
        config_main (dict): Configuración principal
        config_data_save (dict): Configuración de guardado de datos
        config_visualization (dict): Configuración de visualización
        
    Returns:
        dict: Metadata de la corrida
    """
    metadata = {
        'run_id': run_id,
        'timestamp': timestamp,
        'config_main': config_main,
        'config_data_save': config_data_save,
        'config_visualization': config_visualization
    }
    return metadata


def _save_metadata(output_dir, metadata_dict, result_handler):
    """
    Delegación explícita a ResultHandler para persistir metadata.
    
    Args:
        output_dir (str): Directorio de salida
        metadata_dict (dict): Metadata a guardar
        result_handler: Instancia de ResultHandler
    """
    result_handler.save_metadata(metadata_dict)


def _build_components(config_main, config_data_save, config_template_output, output_dir):
    """
    Instancia explícitamente cada componente con toda la configuración
    y retorna un paquete único con:
    - sistema dinámico
    - controlador base + controladores específicos
    - agente
    - procesamiento de métricas
    - cálculo de recompensa
    - ResultHandler
    - MetricCollector (recibe ResultHandler + template para commits)
    - SimulationManager
    - VisualizationManager
    
    Args:
        config_main (dict): Configuración principal
        config_data_save (dict): Configuración de guardado de datos
        config_template_output (dict): Template de output para MetricCollector
        output_dir (str): Directorio de salida
        
    Returns:
        dict: Diccionario con todas las instancias de componentes
    """
    # 1. Sistema dinámico base
    dynamic_system = DynamicSystemBase(config_main)
    
    # 2. Controlador base
    controller_base = ControllerBase(config_main)
    
    # 3. Agente
    agent = AgentBase(config_main)
    
    # 4. Procesamiento de métricas
    metric_processing = MetricProcessing(config_main)
    
    # 5. Calculador de recompensa
    reward_calculator = RewardCalculatorBase(config_main)
    
    # 6. ResultHandler
    result_handler = ResultHandler(output_dir)
    
    # 7. MetricCollector (recibe ResultHandler + template de output)
    metric_collector = MetricCollector(result_handler, config_template_output)
    
    # 8. SimulationManager (recibe result_handler para save_agent_state directo)
    simulation_manager = SimulationManager(
        dynamic_system_base=dynamic_system,
        controller_base=controller_base,
        agent_base=agent,
        metric_processing=metric_processing,
        reward_calculator=reward_calculator,
        metric_collector=metric_collector,
        result_handler=result_handler,
        config_main=config_main,
        config_data_save=config_data_save,
        output_dir=output_dir
    )
    
    # 10. VisualizationManager (con DI: logger, PlotGenerator, HeatmapGenerator)
    plot_generator = MatplotlibPlotGenerator()
    heatmap_generator = HeatmapGenerator(output_dir)
    visualization_manager = VisualizationManager(
        logger_instance=logger,
        plot_generator=plot_generator,
        heatmap_generator=heatmap_generator,
        vis_config_data=config_visualization,
        results_folder_path=output_dir
    )
    
    # Retornar paquete completo
    return {
        'dynamic_system': dynamic_system,
        'controller_base': controller_base,
        'agent': agent,
        'metric_processing': metric_processing,
        'reward_calculator': reward_calculator,
        'result_handler': result_handler,
        'metric_collector': metric_collector,
        'simulation_manager': simulation_manager,
        'visualization_manager': visualization_manager
    }


def _run_simulation(simulation_manager):
    """
    Llama una sola vez simulation_manager.run_simulation().
    
    Args:
        simulation_manager: Instancia de SimulationManager
    """
    simulation_manager.run_simulation()


def _run_visualization(visualization_manager):
    """
    Ejecuta visualization_manager.run() solo post-run.
    VisualizationManager ya tiene vis_config_data y results_folder_path desde __init__.
    
    Args:
        visualization_manager: Instancia de VisualizationManager
    """
    visualization_manager.run()


if __name__ == "__main__":
    main()
