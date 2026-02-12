"""
numpy_encoder.py

Responsabilidad:
Codificador JSON personalizado para manejar tipos de NumPy (ndarray, float, int, bool).
Convierte NaN/inf a None para compatibilidad JSON.
"""

import json
import numpy as np

# Importar pandas opcionalmente
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class NumpyEncoder(json.JSONEncoder):
    """
    Codificador JSON personalizado para manejar tipos de NumPy.
    Convierte NaN/inf a None para compatibilidad JSON.
    """
    
    def default(self, obj_to_encode):
        """
        Convierte tipos no serializables a tipos Python nativos.
        
        Args:
            obj_to_encode: Objeto a codificar
            
        Returns:
            Representación serializable del objeto
        """
        # Arrays de NumPy a listas Python
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        
        # Tipos flotantes de NumPy (float16, float32, float64)
        if isinstance(obj_to_encode, np.floating):
            if not np.isfinite(obj_to_encode):
                return None
            return float(obj_to_encode)
        
        # Tipos enteros de NumPy
        if isinstance(obj_to_encode, np.integer):
            return int(obj_to_encode)
        
        # Booleanos de NumPy
        if isinstance(obj_to_encode, np.bool_):
            return bool(obj_to_encode)
        
        # Timestamps de Pandas (solo si pandas está disponible)
        if HAS_PANDAS and isinstance(obj_to_encode, pd.Timestamp):
            return obj_to_encode.isoformat()
        
        # Dejar que el codificador base maneje otros tipos
        return super(NumpyEncoder, self).default(obj_to_encode)


def sanitize_for_json(data):
    """
    Sanitiza recursivamente un diccionario/lista para serialización JSON.
    Convierte NaN/inf a None y tipos numpy a nativos.
    
    Args:
        data: Diccionario, lista o valor a sanitizar
        
    Returns:
        Datos sanitizados
    """
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist())
    elif isinstance(data, np.floating):
        if not np.isfinite(data):
            return None
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, float):
        if not np.isfinite(data):
            return None
        return data
    else:
        return data