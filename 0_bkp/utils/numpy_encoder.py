"""
numpy_encoder.py

Responsabilidad:
Codificador JSON personalizado para manejar tipos de NumPy (ndarray, float, int, bool)
y Timestamps de Pandas. Convierte NaN/inf a None para compatibilidad JSON.
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Codificador JSON personalizado para manejar tipos de NumPy (ndarray, float, int, bool)
    y Timestamps de Pandas. Convierte NaN/inf a None para compatibilidad JSON.
    """
    def default(self, obj_to_encode): # Renombrado para claridad
        # Arrays de NumPy a listas Python
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        
        # Tipos flotantes de NumPy (float16, float32, float64)
        # Usar np.floating para cubrir todos los tipos flotantes de NumPy
        if isinstance(obj_to_encode, np.floating):
            # Convertir NaN/inf a None para JSON válido, sino a float nativo
            return None if pd.isna(obj_to_encode) or not np.isfinite(obj_to_encode) else float(obj_to_encode)
            
        # Tipos enteros de NumPy
        # Usar np.integer para cubrir todos los tipos enteros de NumPy
        if isinstance(obj_to_encode, np.integer):
            return int(obj_to_encode)
            
        # Booleanos de NumPy
        if isinstance(obj_to_encode, np.bool_):
            return bool(obj_to_encode)
            
        # Timestamps de Pandas (si se usan y necesitan serializarse)
        if isinstance(obj_to_encode, pd.Timestamp):
            return obj_to_encode.isoformat() # Convertir a string ISO 8601

        # Dejar que el codificador base maneje otros tipos o lance el TypeError apropiado
        return super(NumpyEncoder, self).default(obj_to_encode)