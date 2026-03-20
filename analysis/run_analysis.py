"""
run_analysis.py — Script de análisis reutilizable para simulaciones RL CartPole.

Uso:
    python analysis/run_analysis.py <sim_id>
    
Ejemplo:
    python analysis/run_analysis.py 20260227_1310

Genera un Excel con múltiples hojas de resumen en: analysis/<sim_id>/analysis_results.xlsx
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# Configuración de rutas
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_BASE = os.path.join(PROJECT_ROOT, "results_history_CartPole", "cart_pole")
ANALYSIS_BASE = SCRIPT_DIR  # analysis/ folder


def get_sim_paths(sim_id):
    """Retorna las rutas de la simulación."""
    sim_dir = os.path.join(RESULTS_BASE, sim_id)
    return {
        "sim_dir": sim_dir,
        "metadata": os.path.join(sim_dir, "metadata.json"),
        "summary": os.path.join(sim_dir, "summary.xlsx"),
        "output_dir": os.path.join(ANALYSIS_BASE, sim_id),
        "output_xlsx": os.path.join(ANALYSIS_BASE, sim_id, "analysis_results.xlsx"),
    }


# ============================================================================
# Carga de datos
# ============================================================================

def load_data(paths):
    """Carga metadata y summary."""
    print(f"  Cargando metadata...")
    with open(paths["metadata"], 'r') as f:
        meta = json.load(f)
    
    print(f"  Cargando summary.xlsx (esto puede tardar)...")
    df = pd.read_excel(paths["summary"])
    print(f"  Cargado: {len(df)} episodios, {len(df.columns)} columnas")
    
    return meta, df


def detect_termination_col(df):
    """Detecta la columna de razón de terminación."""
    for candidate in ["termination_reason", "end_termination_reason"]:
        if candidate in df.columns:
            return candidate
    return None


# ============================================================================
# Hojas del Excel
# ============================================================================

def build_general_stats(df, meta, tr_col):
    """Hoja 1: Estadísticas generales."""
    rows = []
    
    # Info básica
    sim_cfg = meta["config_main"]["simulation"]
    rows.append({"Parámetro": "Total episodios", "Valor": len(df)})
    rows.append({"Parámetro": "Duración episodio (s)", "Valor": sim_cfg["episode_duration_sec"]})
    rows.append({"Parámetro": "dt (s)", "Valor": sim_cfg["dt_sec"]})
    rows.append({"Parámetro": "Intervalo decisión (s)", "Valor": sim_cfg["decision_interval_sec"]})
    rows.append({"Parámetro": "Steps por decisión", "Valor": int(sim_cfg["decision_interval_sec"] / sim_cfg["dt_sec"])})
    rows.append({"Parámetro": "Max decisiones/ep", "Valor": int(sim_cfg["episode_duration_sec"] / sim_cfg["decision_interval_sec"])})
    rows.append({"Parámetro": "", "Valor": ""})
    
    # Terminaciones
    if tr_col:
        counts = df[tr_col].value_counts()
        for reason, count in counts.items():
            rows.append({"Parámetro": f"Terminación: {reason}", "Valor": f"{count} ({count/len(df)*100:.1f}%)"})
    rows.append({"Parámetro": "", "Valor": ""})
    
    # Performance & Reward
    for col in ["performance", "total_reward"]:
        if col in df.columns:
            rows.append({"Parámetro": f"{col} — Min", "Valor": round(df[col].min(), 6)})
            rows.append({"Parámetro": f"{col} — Max", "Valor": round(df[col].max(), 6)})
            rows.append({"Parámetro": f"{col} — Mean", "Valor": round(df[col].mean(), 6)})
            rows.append({"Parámetro": f"{col} — Std", "Valor": round(df[col].std(), 6)})
            rows.append({"Parámetro": f"{col} — Median", "Valor": round(df[col].median(), 6)})
            rows.append({"Parámetro": "", "Valor": ""})
    
    # Últimos 500
    last = df.tail(500)
    rows.append({"Parámetro": "--- Últimos 500 episodios ---", "Valor": ""})
    if tr_col:
        for reason, count in last[tr_col].value_counts().items():
            rows.append({"Parámetro": f"  Terminación (last500): {reason}", "Valor": f"{count} ({count/500*100:.1f}%)"})
    for col in ["performance", "total_reward"]:
        if col in last.columns:
            rows.append({"Parámetro": f"  {col} (last500) Mean", "Valor": round(last[col].mean(), 6)})
            rows.append({"Parámetro": f"  {col} (last500) Std", "Valor": round(last[col].std(), 6)})
    
    return pd.DataFrame(rows)


def build_component_stats(df, prefix, sheet_label):
    """Construye tabla de stats para columnas con un prefijo dado."""
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    if not cols:
        return pd.DataFrame({"Info": [f"No se encontraron columnas con prefijo '{prefix}'"]})
    
    rows = []
    for col in cols:
        s = df[col].dropna()
        rows.append({
            "Componente": col,
            "Min": round(s.min(), 6),
            "Max": round(s.max(), 6),
            "Mean": round(s.mean(), 6),
            "Std": round(s.std(), 6),
            "Median": round(s.median(), 6),
            "p5": round(s.quantile(0.05), 6),
            "p95": round(s.quantile(0.95), 6),
        })
    return pd.DataFrame(rows)


def build_normalization_audit(df, meta):
    """Hoja: Auditoría de normalización — rangos config vs rangos reales."""
    reward_cfg = meta["config_main"]["reward_base"]["reward_calculation"]
    norm_cfg = reward_cfg["metric_processing"]["normalization"]["params"]
    
    # Mapeo: config key → columna summary (mean_squared)
    metric_map = {
        "e": "error_{var}_mean_squared",
        "edot": "derivative_error_{var}_mean_squared",
        "I": "integral_error_{var}_mean_squared",
        "u": "u_eff_{var}",
        "delta_u": "delta_u_eff_{var}_mean_squared",
    }
    
    rows = []
    
    for var_obj in ["pendulum_angle", "cart_position"]:
        var_cfg = norm_cfg.get(var_obj, {})
        for metric_key, col_template in metric_map.items():
            cfg_entry = var_cfg.get(metric_key, {})
            cfg_range = cfg_entry.get("range", None)
            method = cfg_entry.get("method", "mean_squared")
            
            col_name = col_template.replace("{var}", var_obj)
            
            # Para mean_squared, buscar la columna con sufijo _mean y _std también
            col_mean = f"{col_name.replace('_mean_squared', '')}_mean" if "_mean_squared" in col_name else f"{col_name}_mean"
            
            if cfg_range:
                range_min, range_max = cfg_range
                max_abs = max(abs(range_min), abs(range_max))
                if method == "mean_squared":
                    sq_max = max_abs ** 2
                else:
                    sq_max = max_abs
            else:
                sq_max = None
            
            # Buscar columna mean_squared en summary
            if col_name in df.columns:
                actual_col = col_name
            elif col_mean in df.columns:
                actual_col = col_mean
            else:
                # Buscar cualquier columna que contenga el patrón
                candidates = [c for c in df.columns if metric_key in c and var_obj in c and "mean" in c.lower()]
                actual_col = candidates[0] if candidates else None
            
            if actual_col and actual_col in df.columns:
                s = df[actual_col].dropna()
                actual_min = round(s.min(), 6)
                actual_max = round(s.max(), 6)
                actual_mean = round(s.mean(), 6)
                coverage = round(actual_max / sq_max * 100, 1) if sq_max and sq_max > 0 else None
            else:
                actual_min = actual_max = actual_mean = coverage = "N/A"
            
            # L_ values
            l_col = f"L_{metric_key}_{var_obj}_mean"
            if l_col in df.columns:
                l_s = df[l_col].dropna()
                l_min = round(l_s.min(), 6)
                l_max = round(l_s.max(), 6)
                l_mean = round(l_s.mean(), 6)
                l_std = round(l_s.std(), 6)
            else:
                l_min = l_max = l_mean = l_std = "N/A"
            
            rows.append({
                "Variable": var_obj,
                "Métrica": metric_key,
                "Método": method,
                "Rango Config": str(cfg_range) if cfg_range else "N/A",
                "max_abs²": round(sq_max, 6) if sq_max else "N/A",
                "Actual Min": actual_min,
                "Actual Max": actual_max,
                "Actual Mean": actual_mean,
                "Cobertura %": coverage if coverage else "N/A",
                "L_* Min": l_min,
                "L_* Max": l_max,
                "L_* Mean": l_mean,
                "L_* Std": l_std,
            })
    
    # Global vars
    global_cfg = norm_cfg.get("global_vars", {})
    for metric_key, cfg_entry in global_cfg.items():
        cfg_range = cfg_entry.get("range", None)
        method = cfg_entry.get("method", "mean_squared")
        if cfg_range:
            max_abs = max(abs(cfg_range[0]), abs(cfg_range[1]))
            sq_max = max_abs ** 2 if method == "mean_squared" else max_abs
        else:
            sq_max = None
        
        rows.append({
            "Variable": "global",
            "Métrica": metric_key,
            "Método": method,
            "Rango Config": str(cfg_range) if cfg_range else "N/A",
            "max_abs²": round(sq_max, 6) if sq_max else "N/A",
            "Actual Min": "—",
            "Actual Max": "—",
            "Actual Mean": "—",
            "Cobertura %": "—",
            "L_* Min": "—",
            "L_* Max": "—",
            "L_* Mean": "—",
            "L_* Std": "—",
        })
    
    return pd.DataFrame(rows)


def build_control_signals(df):
    """Hoja: Señales de control (errores, derivadas, integrales, acciones)."""
    prefixes = [
        "error_", "derivative_error_", "integral_error_",
        "control_action_", "u_eff_", "delta_u_eff_",
    ]
    
    rows = []
    for prefix in prefixes:
        # Solo columnas _mean (no _std, _min, etc. para mantener la tabla concisa)
        cols = sorted([c for c in df.columns 
                      if c.startswith(prefix) 
                      and not c.endswith("_mean_squared")
                      and c.endswith("_mean")])
        if not cols:
            # Intentar sin sufijo _mean
            cols = sorted([c for c in df.columns 
                          if c.startswith(prefix)
                          and not c.endswith("_mean_squared")
                          and "_std" not in c and "_min" not in c 
                          and "_max" not in c and "_p25" not in c 
                          and "_p50" not in c and "_p75" not in c])
        
        for col in cols:
            s = df[col].dropna()
            rows.append({
                "Señal": col,
                "Min": round(s.min(), 6),
                "Max": round(s.max(), 6),
                "Mean": round(s.mean(), 6),
                "Std": round(s.std(), 6),
            })
    
    # Mean squared
    ms_cols = sorted([c for c in df.columns if c.endswith("_mean_squared")])
    for col in ms_cols:
        s = df[col].dropna()
        rows.append({
            "Señal": col,
            "Min": round(s.min(), 6),
            "Max": round(s.max(), 6),
            "Mean": round(s.mean(), 6),
            "Std": round(s.std(), 6),
        })
    
    return pd.DataFrame(rows) if rows else pd.DataFrame({"Info": ["No se encontraron señales de control"]})


def build_final_gains(df, tr_col):
    """Hoja: Ganancias PID finales."""
    gain_cols = sorted([c for c in df.columns if c.startswith("final_k")])
    if not gain_cols:
        return pd.DataFrame({"Info": ["No se encontraron columnas de ganancias finales"]})
    
    rows = []
    
    # Global
    for col in gain_cols:
        s = df[col].dropna()
        rows.append({
            "Ganancia": col,
            "Subset": "Todos los episodios",
            "Min": round(s.min(), 4),
            "Max": round(s.max(), 4),
            "Mean": round(s.mean(), 4),
            "Std": round(s.std(), 4),
        })
    
    # Estabilizados
    if tr_col:
        stab_df = df[df[tr_col] == "stabilization_success"]
        if len(stab_df) > 0:
            for col in gain_cols:
                s = stab_df[col].dropna()
                rows.append({
                    "Ganancia": col,
                    "Subset": f"Estabilizados ({len(stab_df)} eps)",
                    "Min": round(s.min(), 4),
                    "Max": round(s.max(), 4),
                    "Mean": round(s.mean(), 4),
                    "Std": round(s.std(), 4),
                })
    
    # Últimos 500
    last = df.tail(500)
    for col in gain_cols:
        s = last[col].dropna()
        rows.append({
            "Ganancia": col,
            "Subset": "Últimos 500 episodios",
            "Min": round(s.min(), 4),
            "Max": round(s.max(), 4),
            "Mean": round(s.mean(), 4),
            "Std": round(s.std(), 4),
        })
    
    # Tiempo de estabilización
    if tr_col and "final_t_sec" in df.columns:
        stab_df = df[df[tr_col] == "stabilization_success"]
        if len(stab_df) > 0:
            s = stab_df["final_t_sec"].dropna()
            rows.append({
                "Ganancia": "final_t_sec",
                "Subset": f"Estabilizados ({len(stab_df)} eps)",
                "Min": round(s.min(), 4),
                "Max": round(s.max(), 4),
                "Mean": round(s.mean(), 4),
                "Std": round(s.std(), 4),
            })
    
    return pd.DataFrame(rows)


def build_reward_scale_analysis(df, meta):
    """Hoja: Análisis de escala de recompensas (contribución ponderada)."""
    reward_cfg = meta["config_main"]["reward_base"]["reward_calculation"]
    lagrange_cfg = reward_cfg["principal_reward"]
    method = lagrange_cfg.get("method", "lineal_combination")
    
    rows = []
    
    # Pesos Lagrange
    if method == "lineal_combination":
        features = lagrange_cfg.get("lineal_combination_params", {}).get("features", {})
    else:
        features = lagrange_cfg.get("weighted_exponential_params", {}).get("features", {})
    
    rows.append({"Aspecto": "--- Pesos Lagrange ---", "Valor": "", "Detalle": ""})
    total_weight = 0
    for feat, cfg in features.items():
        w = cfg.get("weight", 0)
        total_weight += w
        rows.append({"Aspecto": f"  {feat}", "Valor": w, "Detalle": f"weight={w}"})
    rows.append({"Aspecto": "  Σ weights", "Valor": total_weight, "Detalle": "Debe ser 1.0"})
    
    rows.append({"Aspecto": "", "Valor": "", "Detalle": ""})
    rows.append({"Aspecto": "--- Contribución media ponderada ---", "Valor": "", "Detalle": ""})
    
    # Calcular contribución real
    lagrange_cols = sorted([c for c in df.columns if c.startswith("L_") and c.endswith("_mean")])
    for feat, cfg in features.items():
        w = cfg.get("weight", 0)
        matching = [c for c in lagrange_cols if c.startswith(feat)]
        if matching:
            total_L = sum(df[c].mean() for c in matching)
            weighted = w * total_L
            rows.append({
                "Aspecto": f"  {feat} (w={w})",
                "Valor": round(weighted, 6),
                "Detalle": f"Σ L_mean={round(total_L, 6)}, weighted={round(weighted, 6)}"
            })
    
    # Extra rewards
    rows.append({"Aspecto": "", "Valor": "", "Detalle": ""})
    rows.append({"Aspecto": "--- Extra Rewards Config ---", "Valor": "", "Detalle": ""})
    
    extra_cfg = reward_cfg.get("extra_rewards", {})
    
    # Bandwidth bonus
    bb = extra_cfg.get("bonus_approach", {}).get("bandwidth_bonus", {})
    rows.append({"Aspecto": "  bandwidth_bonus enabled", "Valor": bb.get("enabled", False), "Detalle": ""})
    rows.append({"Aspecto": "  per_step_band_bonus", "Valor": bb.get("per_step_band_bonus", "N/A"), "Detalle": ""})
    rows.append({"Aspecto": "  max_total_band_bonus", "Valor": bb.get("max_total_band_bonus", "N/A"), "Detalle": ""})
    for var, rng in bb.get("ranges", {}).items():
        rows.append({"Aspecto": f"  band range: {var}", "Valor": str(rng), "Detalle": ""})
    
    # Dynamic penalty
    dp = extra_cfg.get("conditional_approach", {}).get("dynamic_penalty", {})
    rows.append({"Aspecto": "  dynamic_penalty enabled", "Valor": dp.get("enabled", False), "Detalle": ""})
    for var, cfg_dp in dp.get("dynamic_penalty_params", {}).items():
        cond = cfg_dp.get("condition", {})
        rows.append({
            "Aspecto": f"  dp: {var}",
            "Valor": f"w={cfg_dp.get('weight')}, scaled={cond.get('scaled')}",
            "Detalle": f"type={cond.get('type')}, feature={cond.get('feature')}"
        })
    
    # Escala comparativa
    rows.append({"Aspecto": "", "Valor": "", "Detalle": ""})
    rows.append({"Aspecto": "--- Escala comparativa (abs_mean) ---", "Valor": "", "Detalle": ""})
    
    if "total_reward" in df.columns:
        tr_mean = abs(df["total_reward"].mean())
        rows.append({"Aspecto": "  |total_reward| mean", "Valor": round(tr_mean, 4), "Detalle": "Referencia"})
        
        extra_cols = sorted([c for c in df.columns if c.startswith("extra_")])
        for col in extra_cols:
            s = df[col].dropna()
            abs_mean = s.abs().mean()
            if abs_mean > 0:
                ratio = abs_mean / tr_mean * 100
                rows.append({
                    "Aspecto": f"  {col}",
                    "Valor": round(abs_mean, 6),
                    "Detalle": f"{round(ratio, 2)}% del total_reward"
                })
    
    return pd.DataFrame(rows)


def build_state_vars(df):
    """Hoja: Variables de estado (raw y normalized)."""
    state_vars = ["pendulum_angle", "pendulum_velocity", "cart_position", "cart_velocity"]
    rows = []
    
    for var in state_vars:
        for suffix in ["_raw", "_norm"]:
            for stat in ["_mean", "_std", "_min", "_p25", "_p50", "_p75", "_max"]:
                col = f"{var}{suffix}{stat}"
                if col in df.columns:
                    s = df[col].dropna()
                    rows.append({
                        "Variable": var,
                        "Tipo": suffix.strip("_"),
                        "Estadístico": stat.strip("_"),
                        "Min": round(s.min(), 6),
                        "Max": round(s.max(), 6),
                        "Mean": round(s.mean(), 6),
                        "Std": round(s.std(), 6),
                    })
    
    return pd.DataFrame(rows) if rows else pd.DataFrame({"Info": ["No se encontraron variables de estado"]})


# ============================================================================
# Main
# ============================================================================

def run_analysis(sim_id):
    """Ejecuta el análisis completo y genera el Excel."""
    print(f"\n{'='*60}")
    print(f"  ANÁLISIS DE SIMULACIÓN: {sim_id}")
    print(f"{'='*60}\n")
    
    paths = get_sim_paths(sim_id)
    
    # Validar existencia
    if not os.path.exists(paths["metadata"]):
        print(f"ERROR: No se encontró metadata en {paths['metadata']}")
        sys.exit(1)
    if not os.path.exists(paths["summary"]):
        print(f"ERROR: No se encontró summary en {paths['summary']}")
        sys.exit(1)
    
    # Cargar datos
    meta, df = load_data(paths)
    tr_col = detect_termination_col(df)
    
    # Crear carpeta de salida
    os.makedirs(paths["output_dir"], exist_ok=True)
    
    # Construir hojas
    print("\n  Generando hojas de análisis...")
    
    sheets = {
        "general_stats": build_general_stats(df, meta, tr_col),
        "lagrange_components": build_component_stats(df, "L_", "Lagrange"),
        "agent_rewards": build_component_stats(df, "reward_", "Agent Rewards"),
        "extra_rewards": build_component_stats(df, "extra_", "Extra Rewards"),
        "normalization_audit": build_normalization_audit(df, meta),
        "control_signals": build_control_signals(df),
        "final_gains": build_final_gains(df, tr_col),
        "reward_scale": build_reward_scale_analysis(df, meta),
        "state_vars": build_state_vars(df),
    }
    
    # Escribir Excel
    print(f"  Escribiendo Excel en: {paths['output_xlsx']}")
    with pd.ExcelWriter(paths["output_xlsx"], engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Resumen en terminal
    print(f"\n{'='*60}")
    print(f"  RESUMEN RÁPIDO")
    print(f"{'='*60}")
    print(f"  Episodios: {len(df)}")
    if tr_col:
        stab = len(df[df[tr_col] == "stabilization_success"])
        print(f"  Estabilizaciones: {stab} ({stab/len(df)*100:.1f}%)")
    if "total_reward" in df.columns:
        print(f"  Total reward: mean={df['total_reward'].mean():.4f}, min={df['total_reward'].min():.4f}, max={df['total_reward'].max():.4f}")
    if "performance" in df.columns:
        print(f"  Performance: mean={df['performance'].mean():.4f}")
    
    print(f"\n  ✅ Excel generado: {paths['output_xlsx']}")
    print(f"  📂 Carpeta de análisis: {paths['output_dir']}")
    

def main():
    if len(sys.argv) < 2:
        print("Uso: python analysis/run_analysis.py <sim_id>")
        print("Ejemplo: python analysis/run_analysis.py 20260227_1310")
        sys.exit(1)
    
    sim_id = sys.argv[1]
    run_analysis(sim_id)


if __name__ == "__main__":
    main()
