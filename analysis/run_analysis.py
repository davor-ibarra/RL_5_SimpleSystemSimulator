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
HISTORY_WINDOW_EPISODES = 500


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


def list_chunk_files(paths):
    """Retorna la lista ordenada de chunks JSON."""
    chunk_files = [
        f for f in os.listdir(paths["sim_dir"])
        if f.startswith("episodes_chunks_") and f.endswith(".json")
    ]
    chunk_files.sort()
    return [os.path.join(paths["sim_dir"], f) for f in chunk_files]


def load_history_window(paths, window_episodes=HISTORY_WINDOW_EPISODES):
    """
    Carga una ventana de episodios desde los chunks JSON históricos.
    Se lee desde el final para capturar la política más reciente sin
    cargar toda la corrida detallada en memoria.
    """
    chunk_files = list_chunk_files(paths)
    selected_episodes = []
    
    for chunk_path in reversed(chunk_files):
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_episodes = json.load(f)
        
        selected_episodes = chunk_episodes + selected_episodes
        if len(selected_episodes) >= window_episodes:
            selected_episodes = selected_episodes[-window_episodes:]
            break
    
    if not selected_episodes:
        return None
    
    interval_example = selected_episodes[-1]["interval_data"]
    action_keys = [k for k in interval_example.keys() if k.startswith("action_")]
    interval_signal_keys = [
        k for k in interval_example.keys()
        if k == "global_interval_reward"
        or k.startswith("reward_")
        or k.startswith("L_")
        or k.startswith("extra_")
    ]
    
    action_counts = {key: {0: 0, 1: 0, 2: 0} for key in action_keys}
    interval_signal_values = {key: [] for key in interval_signal_keys}
    episode_rows = []
    termination_counts = {}
    
    for episode in selected_episodes:
        end_episode = episode["end_episode_data"]
        termination_reason = end_episode["end_termination_reason"]
        if termination_reason not in termination_counts:
            termination_counts[termination_reason] = 0
        termination_counts[termination_reason] += 1
        
        step_data = episode["step_data"]
        interval_data = episode["interval_data"]
        
        n_step = step_data["n_step"]
        n_interval = interval_data["n_interval"]
        sat_count = int(np.sum(np.array(step_data["is_saturated_global"], dtype=int)))
        sat_ratio = sat_count / n_step if n_step > 0 else 0.0
        mean_abs_u_total = float(np.mean(np.abs(np.array(step_data["u_total"], dtype=float)))) if n_step > 0 else 0.0
        mean_abs_u_total_raw = float(np.mean(np.abs(np.array(step_data["u_total_raw"], dtype=float)))) if n_step > 0 else 0.0
        max_abs_u_total_raw = float(np.max(np.abs(np.array(step_data["u_total_raw"], dtype=float)))) if n_step > 0 else 0.0
        final_t_sec = step_data["t_sec"][-1] if n_step > 0 else 0.0
        
        episode_rows.append({
            "episode_id": episode["episode_id"],
            "termination_reason": termination_reason,
            "n_step": n_step,
            "n_interval": n_interval,
            "final_t_sec": final_t_sec,
            "sat_ratio": sat_ratio,
            "sat_steps": sat_count,
            "mean_abs_u_total": mean_abs_u_total,
            "mean_abs_u_total_raw": mean_abs_u_total_raw,
            "max_abs_u_total_raw": max_abs_u_total_raw,
            "total_reward": end_episode["total_reward"],
            "total_agent_decisions": end_episode["total_agent_decisions"],
            "epsilon": end_episode["epsilon"],
            "learning_rate": end_episode["learning_rate"],
        })
        
        for key in action_keys:
            for action_value in interval_data[key]:
                action_counts[key][int(action_value)] += 1
        
        for key in interval_signal_keys:
            interval_signal_values[key].extend(interval_data[key])
    
    return {
        "window_episodes": len(selected_episodes),
        "selected_episodes": selected_episodes,
        "action_counts": action_counts,
        "interval_signal_values": interval_signal_values,
        "episode_rows": episode_rows,
        "termination_counts": termination_counts,
        "last_episode": selected_episodes[-1],
    }


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
        "u": "u_eff_{var}_mean_squared",
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
            
            col_mean = f"{col_name}_mean"
            
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
                actual_col = None
            
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
        cond_type = cond.get("type")
        cond_gain = cond.get("strength", cond.get("scaled"))
        cond_setpoint = cond.get("setpoint", cond.get("x_sp"))
        gain_label = "strength" if cond_type == "tanh" else "scaled"
        rows.append({
            "Aspecto": f"  dp: {var}",
            "Valor": f"w={cfg_dp.get('weight')}, {gain_label}={cond_gain}, sp={cond_setpoint}",
            "Detalle": f"type={cond_type}, feature={cond.get('feature')}"
        })
    
    # Escala comparativa
    rows.append({"Aspecto": "", "Valor": "", "Detalle": ""})
    rows.append({"Aspecto": "--- Escala comparativa (abs_mean) ---", "Valor": "", "Detalle": ""})
    
    if "total_reward" in df.columns:
        tr_mean = abs(df["total_reward"].mean())
        rows.append({"Aspecto": "  |total_reward| mean", "Valor": round(tr_mean, 4), "Detalle": "Referencia"})
        
        extra_cols = sorted([c for c in df.columns if c.startswith("extra_") and c.endswith("_mean")])
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


def build_history_window_overview(history_window):
    """Hoja: Resumen de la ventana histórica detallada."""
    rows = []
    episode_df = pd.DataFrame(history_window["episode_rows"])
    
    rows.append({"Métrica": "Episodios detallados cargados", "Valor": history_window["window_episodes"]})
    rows.append({"Métrica": "Episode ID inicial", "Valor": int(episode_df["episode_id"].min())})
    rows.append({"Métrica": "Episode ID final", "Valor": int(episode_df["episode_id"].max())})
    rows.append({"Métrica": "", "Valor": ""})
    
    for reason, count in history_window["termination_counts"].items():
        rows.append({"Métrica": f"Terminación ventana: {reason}", "Valor": f"{count} ({count/history_window['window_episodes']*100:.1f}%)"})
    
    rows.append({"Métrica": "", "Valor": ""})
    rows.append({"Métrica": "total_reward mean", "Valor": round(episode_df["total_reward"].mean(), 6)})
    rows.append({"Métrica": "total_reward std", "Valor": round(episode_df["total_reward"].std(), 6)})
    rows.append({"Métrica": "final_t_sec mean", "Valor": round(episode_df["final_t_sec"].mean(), 6)})
    rows.append({"Métrica": "sat_ratio mean", "Valor": round(episode_df["sat_ratio"].mean(), 6)})
    rows.append({"Métrica": "sat_ratio p95", "Valor": round(episode_df["sat_ratio"].quantile(0.95), 6)})
    rows.append({"Métrica": "mean_abs_u_total_raw mean", "Valor": round(episode_df["mean_abs_u_total_raw"].mean(), 6)})
    rows.append({"Métrica": "max_abs_u_total_raw mean", "Valor": round(episode_df["max_abs_u_total_raw"].mean(), 6)})
    
    return pd.DataFrame(rows)


def build_action_usage(history_window):
    """Hoja: Distribución de acciones por agente en la ventana detallada."""
    rows = []
    
    for action_key, counts in history_window["action_counts"].items():
        total = counts[0] + counts[1] + counts[2]
        if total == 0:
            continue
        
        rows.append({
            "Acción": action_key,
            "decrease_count": counts[0],
            "maintain_count": counts[1],
            "increase_count": counts[2],
            "decrease_%": round(counts[0] / total * 100, 2),
            "maintain_%": round(counts[1] / total * 100, 2),
            "increase_%": round(counts[2] / total * 100, 2),
        })
    
    return pd.DataFrame(rows)


def build_interval_signal_stats(history_window):
    """Hoja: Estadísticas interval-level desde history JSON."""
    rows = []
    
    for signal_name, values in sorted(history_window["interval_signal_values"].items()):
        if not values:
            continue
        
        arr = np.array(values, dtype=float)
        rows.append({
            "Señal": signal_name,
            "Min": round(float(np.min(arr)), 6),
            "Max": round(float(np.max(arr)), 6),
            "Mean": round(float(np.mean(arr)), 6),
            "Std": round(float(np.std(arr)), 6),
            "Median": round(float(np.median(arr)), 6),
            "p5": round(float(np.percentile(arr, 5)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
        })
    
    return pd.DataFrame(rows)


def build_saturation_window(history_window):
    """Hoja: Métricas por episodio útiles para revisar saturación y esfuerzo."""
    episode_df = pd.DataFrame(history_window["episode_rows"])
    return episode_df.sort_values("episode_id").reset_index(drop=True)


def build_last_episode_snapshot(history_window):
    """Hoja: Snapshot detallado del último episodio de la ventana."""
    last_episode = history_window["last_episode"]
    rows = []
    
    end_episode = last_episode["end_episode_data"]
    step_data = last_episode["step_data"]
    interval_data = last_episode["interval_data"]
    
    rows.append({"Aspecto": "episode_id", "Valor": last_episode["episode_id"]})
    rows.append({"Aspecto": "termination_reason", "Valor": end_episode["end_termination_reason"]})
    rows.append({"Aspecto": "total_reward", "Valor": round(end_episode["total_reward"], 6)})
    rows.append({"Aspecto": "n_step", "Valor": step_data["n_step"]})
    rows.append({"Aspecto": "n_interval", "Valor": interval_data["n_interval"]})
    rows.append({"Aspecto": "final_t_sec", "Valor": round(step_data["t_sec"][-1], 6)})
    rows.append({"Aspecto": "sat_ratio", "Valor": round(float(np.mean(np.array(step_data["is_saturated_global"], dtype=float))), 6)})
    
    action_keys = [k for k in interval_data.keys() if k.startswith("action_")]
    for action_key in action_keys:
        action_arr = np.array(interval_data[action_key], dtype=int)
        rows.append({"Aspecto": f"{action_key}_decrease_%", "Valor": round(float(np.mean(action_arr == 0) * 100), 2)})
        rows.append({"Aspecto": f"{action_key}_maintain_%", "Valor": round(float(np.mean(action_arr == 1) * 100), 2)})
        rows.append({"Aspecto": f"{action_key}_increase_%", "Valor": round(float(np.mean(action_arr == 2) * 100), 2)})
    
    signal_keys = [
        k for k in interval_data.keys()
        if k == "global_interval_reward"
        or k.startswith("reward_")
        or k.startswith("L_")
        or k.startswith("extra_")
    ]
    for signal_key in signal_keys:
        signal_arr = np.array(interval_data[signal_key], dtype=float)
        rows.append({"Aspecto": f"{signal_key}_mean", "Valor": round(float(np.mean(signal_arr)), 6)})
    
    return pd.DataFrame(rows)


def build_credit_assignment_audit(history_window):
    """Hoja: Auditoría de crédito usando rewards interval-level reales."""
    interval_signal_values = history_window["interval_signal_values"]
    
    global_reward_mean = float(np.mean(np.array(interval_signal_values["global_interval_reward"], dtype=float)))
    
    pendulum_reward_keys = [
        "reward_kp_pendulum_angle",
        "reward_ki_pendulum_angle",
        "reward_kd_pendulum_angle",
    ]
    cart_reward_keys = [
        "reward_kp_cart_position",
        "reward_ki_cart_position",
        "reward_kd_cart_position",
    ]
    
    pendulum_agent_reward_mean = float(np.mean(np.concatenate([
        np.array(interval_signal_values[key], dtype=float) for key in pendulum_reward_keys
    ])))
    cart_agent_reward_mean = float(np.mean(np.concatenate([
        np.array(interval_signal_values[key], dtype=float) for key in cart_reward_keys
    ])))
    
    extra_bonus_pendulum_mean = float(np.mean(np.array(interval_signal_values["extra_bonus_band_pendulum_angle"], dtype=float)))
    extra_bonus_cart_mean = float(np.mean(np.array(interval_signal_values["extra_bonus_band_cart_position"], dtype=float)))
    extra_penalty_pendulum_mean = float(np.mean(np.array(interval_signal_values["extra_conditional_dynamic_penalty_pendulum_angle"], dtype=float)))
    extra_penalty_cart_mean = float(np.mean(np.array(interval_signal_values["extra_conditional_dynamic_penalty_cart_position"], dtype=float)))
    
    rows = [
        {"Chequeo": "global_interval_reward_mean", "Valor": round(global_reward_mean, 6), "Lectura": "Reward global medio por decision", "Accion sugerida": ""},
        {"Chequeo": "pendulum_agent_reward_mean", "Valor": round(pendulum_agent_reward_mean, 6), "Lectura": "Reward medio recibido por agentes del lazo pendulo", "Accion sugerida": ""},
        {"Chequeo": "cart_agent_reward_mean", "Valor": round(cart_agent_reward_mean, 6), "Lectura": "Reward medio recibido por agentes del lazo carro", "Accion sugerida": ""},
        {"Chequeo": "extra_bonus_band_pendulum_mean", "Valor": round(extra_bonus_pendulum_mean, 6), "Lectura": "Bonus medio por decision del pendulo", "Accion sugerida": ""},
        {"Chequeo": "extra_bonus_band_cart_mean", "Valor": round(extra_bonus_cart_mean, 6), "Lectura": "Bonus medio por decision del carro", "Accion sugerida": ""},
        {"Chequeo": "extra_dynamic_penalty_pendulum_mean", "Valor": round(extra_penalty_pendulum_mean, 6), "Lectura": "Penalidad dinamica media del pendulo", "Accion sugerida": ""},
        {"Chequeo": "extra_dynamic_penalty_cart_mean", "Valor": round(extra_penalty_cart_mean, 6), "Lectura": "Penalidad dinamica media del carro", "Accion sugerida": ""},
        {"Chequeo": "global_minus_pendulum_reward", "Valor": round(global_reward_mean - pendulum_agent_reward_mean, 6), "Lectura": "Gap entre reward global y reward del lazo pendulo", "Accion sugerida": "Si el gap es grande, revisar credit assignment"},
        {"Chequeo": "global_minus_cart_reward", "Valor": round(global_reward_mean - cart_agent_reward_mean, 6), "Lectura": "Gap entre reward global y reward del lazo carro", "Accion sugerida": "Si el gap es grande, revisar credit assignment"},
    ]
    
    return pd.DataFrame(rows)


def build_diagnostic_summary(df, history_window, tr_col):
    """Hoja: Diagnóstico resumido y orientado a tuning."""
    rows = []
    last_df = df.tail(HISTORY_WINDOW_EPISODES)
    episode_df = pd.DataFrame(history_window["episode_rows"])
    interval_signal_values = history_window["interval_signal_values"]
    
    stabilization_count = 0
    if tr_col:
        stabilization_count = int(np.sum(last_df[tr_col] == "stabilization_success"))
    
    maintain_ratios = []
    for action_key, counts in history_window["action_counts"].items():
        total = counts[0] + counts[1] + counts[2]
        maintain_ratio = counts[1] / total * 100 if total > 0 else 0.0
        swing_bias = abs(counts[2] - counts[0]) / total * 100 if total > 0 else 0.0
        maintain_ratios.append((action_key, maintain_ratio, swing_bias))
    
    maintain_ratios.sort(key=lambda x: x[1])
    lowest_maintain_action, lowest_maintain_value, lowest_maintain_bias = maintain_ratios[0]
    mean_maintain = float(np.mean([item[1] for item in maintain_ratios]))
    
    sat_ratio_mean = float(episode_df["sat_ratio"].mean())
    sat_ratio_p95 = float(episode_df["sat_ratio"].quantile(0.95))
    
    extra_bonus_pendulum_mean = float(np.mean(np.array(interval_signal_values["extra_bonus_band_pendulum_angle"], dtype=float)))
    extra_bonus_cart_mean = float(np.mean(np.array(interval_signal_values["extra_bonus_band_cart_position"], dtype=float)))
    extra_penalty_pendulum_mean = float(np.mean(np.array(interval_signal_values["extra_conditional_dynamic_penalty_pendulum_angle"], dtype=float)))
    extra_penalty_cart_mean = float(np.mean(np.array(interval_signal_values["extra_conditional_dynamic_penalty_cart_position"], dtype=float)))
    
    rows.append({
        "Chequeo": "stabilization_success_last500",
        "Valor": stabilization_count,
        "Lectura": "Numero de episodios estabilizados en la ventana detallada",
        "Accion sugerida": "Si sigue en 0, ajustar reward shaping y granularidad de accion"
    })
    rows.append({
        "Chequeo": "mean_maintain_action_percent",
        "Valor": round(mean_maintain, 4),
        "Lectura": "Promedio de uso de maintain entre agentes",
        "Accion sugerida": "Si es bajo, reducir delta_gain para permitir asentamiento"
    })
    rows.append({
        "Chequeo": "lowest_maintain_action",
        "Valor": f"{lowest_maintain_action}: {round(lowest_maintain_value, 4)}%",
        "Lectura": f"Sesgo subir/bajar asociado = {round(lowest_maintain_bias, 4)}%",
        "Accion sugerida": "Si maintain es muy bajo y subir/bajar estan empatados, la politica aun oscila"
    })
    rows.append({
        "Chequeo": "global_saturation_last500_mean",
        "Valor": round(sat_ratio_mean, 6),
        "Lectura": f"p95 = {round(sat_ratio_p95, 6)}",
        "Accion sugerida": "Si es moderado, no tratar la saturacion como cuello principal"
    })
    rows.append({
        "Chequeo": "bonus_balance_pendulum_vs_cart",
        "Valor": round(extra_bonus_pendulum_mean - extra_bonus_cart_mean, 6),
        "Lectura": f"pendulum={round(extra_bonus_pendulum_mean, 6)}, cart={round(extra_bonus_cart_mean, 6)}",
        "Accion sugerida": "Si el pendulo recibe mucho menos bonus, rebalancear bandas"
    })
    rows.append({
        "Chequeo": "dynamic_penalty_balance_pendulum_vs_cart",
        "Valor": round(extra_penalty_pendulum_mean - extra_penalty_cart_mean, 6),
        "Lectura": f"pendulum={round(extra_penalty_pendulum_mean, 6)}, cart={round(extra_penalty_cart_mean, 6)}",
        "Accion sugerida": "Si el pendulo paga mucho mas, revisar weights y strength/scaled"
    })
    
    return pd.DataFrame(rows)




def build_decision_summary_tables(df, meta, history_window, tr_col):
    """Construye tablas resumen neutrales para la primera hoja."""
    config_main = meta["config_main"]
    sim_cfg = config_main["simulation"]
    agent_base_cfg = config_main["agent_base"]
    actions_cfg = agent_base_cfg["agent_config"]["actions"]["actions_values"]
    reward_cfg = config_main["reward_base"]["reward_calculation"]
    extra_reward_cfg = reward_cfg["extra_rewards"]
    bandwidth_cfg = extra_reward_cfg["bonus_approach"]["bandwidth_bonus"]
    dynamic_cfg = extra_reward_cfg["conditional_approach"]["dynamic_penalty"]["dynamic_penalty_params"]
    pendulum_penalty_cfg = dynamic_cfg["control_action_pendulum_angle"]
    cart_penalty_cfg = dynamic_cfg["control_action_cart_position"]
    pendulum_cond = pendulum_penalty_cfg["condition"]
    cart_cond = cart_penalty_cfg["condition"]
    lagrange_features = reward_cfg["principal_reward"]["lineal_combination_params"]["features"]
    boundary_cfg = config_main["dynamic_system"]["termination_conditions"]["boundary_constraint"]
    
    selected_episodes = history_window["selected_episodes"]
    interval_signal_values = history_window["interval_signal_values"]
    episode_df = pd.DataFrame(history_window["episode_rows"])
    action_df = build_action_usage(history_window).copy()
    
    stabilization_count = 0
    if tr_col:
        stabilization_count = int(np.sum(df.tail(HISTORY_WINDOW_EPISODES)[tr_col] == "stabilization_success"))
    
    pendulum_reward_keys = [
        "reward_kp_pendulum_angle",
        "reward_ki_pendulum_angle",
        "reward_kd_pendulum_angle",
    ]
    cart_reward_keys = [
        "reward_kp_cart_position",
        "reward_ki_cart_position",
        "reward_kd_cart_position",
    ]
    
    pendulum_agent_reward_mean = float(np.mean(np.concatenate([
        np.array(interval_signal_values[key], dtype=float) for key in pendulum_reward_keys
    ])))
    cart_agent_reward_mean = float(np.mean(np.concatenate([
        np.array(interval_signal_values[key], dtype=float) for key in cart_reward_keys
    ])))
    
    pendulum_bonus_mean = 0.0
    cart_bonus_mean = 0.0
    pendulum_penalty_mean = 0.0
    cart_penalty_mean = 0.0
    pendulum_extra_mean = 0.0
    cart_extra_mean = 0.0
    
    for signal_name in interval_signal_values:
        signal_mean = float(np.mean(np.array(interval_signal_values[signal_name], dtype=float)))
        if signal_name.startswith("extra_bonus_band_"):
            if signal_name.endswith("pendulum_angle"):
                pendulum_bonus_mean += signal_mean
            if signal_name.endswith("cart_position"):
                cart_bonus_mean += signal_mean
        if signal_name.startswith("extra_conditional_dynamic_penalty_"):
            if signal_name.endswith("pendulum_angle"):
                pendulum_penalty_mean += signal_mean
            if signal_name.endswith("cart_position"):
                cart_penalty_mean += signal_mean
        if signal_name.startswith("extra_"):
            if signal_name.endswith("pendulum_angle"):
                pendulum_extra_mean += signal_mean
            if signal_name.endswith("cart_position"):
                cart_extra_mean += signal_mean
    
    loop_cost_rows = []
    pendulum_principal_mean = 0.0
    cart_principal_mean = 0.0
    for loop_name in ["pendulum_angle", "cart_position"]:
        loop_row = {"loop": loop_name}
        principal_reward_mean = 0.0
        for feature_name in lagrange_features:
            signal_name = f"{feature_name}_{loop_name}"
            if signal_name in interval_signal_values:
                signal_mean = float(np.mean(np.array(interval_signal_values[signal_name], dtype=float)))
                loop_row[f"{feature_name}_mean"] = round(signal_mean, 6)
                principal_reward_mean -= lagrange_features[feature_name]["weight"] * signal_mean
        loop_row["principal_reward_mean"] = round(principal_reward_mean, 6)
        loop_cost_rows.append(loop_row)
        if loop_name == "pendulum_angle":
            pendulum_principal_mean = principal_reward_mean
        if loop_name == "cart_position":
            cart_principal_mean = principal_reward_mean
    
    angle_limit = abs(boundary_cfg["pendulum_angle"][1])
    cart_limit = abs(boundary_cfg["cart_position"][1])
    angle_only_fail = 0
    cart_only_fail = 0
    both_fail = 0
    
    pendulum_angle_abs_means = []
    pendulum_velocity_abs_means = []
    cart_position_abs_means = []
    cart_velocity_abs_means = []
    control_action_pendulum_abs_means = []
    control_action_cart_abs_means = []
    pendulum_band_bonus_episode = []
    cart_band_bonus_episode = []
    
    for episode in selected_episodes:
        step_data = episode["step_data"]
        end_episode = episode["end_episode_data"]
        
        pendulum_angle_abs_means.append(float(np.mean(np.abs(np.array(step_data["pendulum_angle_raw"], dtype=float)))))
        pendulum_velocity_abs_means.append(float(np.mean(np.abs(np.array(step_data["pendulum_velocity_raw"], dtype=float)))))
        cart_position_abs_means.append(float(np.mean(np.abs(np.array(step_data["cart_position_raw"], dtype=float)))))
        cart_velocity_abs_means.append(float(np.mean(np.abs(np.array(step_data["cart_velocity_raw"], dtype=float)))))
        control_action_pendulum_abs_means.append(float(np.mean(np.abs(np.array(step_data["control_action_pendulum_angle"], dtype=float)))))
        control_action_cart_abs_means.append(float(np.mean(np.abs(np.array(step_data["control_action_cart_position"], dtype=float)))))
        
        if "accumulated_band_bonus_pendulum_angle" in end_episode:
            pendulum_band_bonus_episode.append(float(end_episode["accumulated_band_bonus_pendulum_angle"]))
        if "accumulated_band_bonus_cart_position" in end_episode:
            cart_band_bonus_episode.append(float(end_episode["accumulated_band_bonus_cart_position"]))
        
        final_abs_angle = abs(float(step_data["pendulum_angle_raw"][-1]))
        final_abs_cart = abs(float(step_data["cart_position_raw"][-1]))
        angle_failed = final_abs_angle >= angle_limit
        cart_failed = final_abs_cart >= cart_limit
        
        if angle_failed and cart_failed:
            both_fail += 1
        elif angle_failed:
            angle_only_fail += 1
        elif cart_failed:
            cart_only_fail += 1
    
    total_window_episodes = len(selected_episodes)
    angle_only_ratio = angle_only_fail / total_window_episodes * 100
    cart_only_ratio = cart_only_fail / total_window_episodes * 100
    both_fail_ratio = both_fail / total_window_episodes * 100
    
    pendulum_cap = bandwidth_cfg["max_total_band_bonus"]["pendulum_angle"]
    cart_cap = bandwidth_cfg["max_total_band_bonus"]["cart_position"]
    pendulum_cap_hit_ratio = np.nan
    cart_cap_hit_ratio = np.nan
    if pendulum_band_bonus_episode:
        pendulum_cap_hits = np.sum(np.array(pendulum_band_bonus_episode, dtype=float) >= pendulum_cap - 1e-9)
        pendulum_cap_hit_ratio = float(pendulum_cap_hits) / len(pendulum_band_bonus_episode) * 100
    if cart_band_bonus_episode:
        cart_cap_hits = np.sum(np.array(cart_band_bonus_episode, dtype=float) >= cart_cap - 1e-9)
        cart_cap_hit_ratio = float(cart_cap_hits) / len(cart_band_bonus_episode) * 100
    
    config_table = pd.DataFrame([
        {"Parametro": "simulation.decision_interval_sec", "Valor": sim_cfg["decision_interval_sec"]},
        {"Parametro": "simulation.dt_sec", "Valor": sim_cfg["dt_sec"]},
        {"Parametro": "agent_base.agent_config.actions.actions_values.universal_params.delta_gain", "Valor": actions_cfg["universal_params"]["delta_gain"]},
        {"Parametro": "agent_base.params.discount_factor", "Valor": agent_base_cfg["params"]["discount_factor"]},
        {"Parametro": "reward_base.reward_config.reward_approach", "Valor": config_main["reward_base"]["reward_config"]["reward_approach"]},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.bonus_approach.bandwidth_bonus.per_step_band_bonus", "Valor": bandwidth_cfg["per_step_band_bonus"]},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.bonus_approach.bandwidth_bonus.max_total_band_bonus.pendulum_angle", "Valor": pendulum_cap},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.bonus_approach.bandwidth_bonus.max_total_band_bonus.cart_position", "Valor": cart_cap},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.bonus_approach.bandwidth_bonus.ranges.error_pendulum_angle", "Valor": str(bandwidth_cfg["ranges"]["error_pendulum_angle"])},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.bonus_approach.bandwidth_bonus.ranges.error_cart_position", "Valor": str(bandwidth_cfg["ranges"]["error_cart_position"])},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.conditional_approach.dynamic_penalty.dynamic_penalty_params.control_action_pendulum_angle.weight", "Valor": pendulum_penalty_cfg["weight"]},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.conditional_approach.dynamic_penalty.dynamic_penalty_params.control_action_pendulum_angle.condition.type", "Valor": pendulum_cond.get("type")},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.conditional_approach.dynamic_penalty.dynamic_penalty_params.control_action_pendulum_angle.condition.gain", "Valor": pendulum_cond.get("strength", pendulum_cond.get("scaled"))},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.conditional_approach.dynamic_penalty.dynamic_penalty_params.control_action_pendulum_angle.condition.setpoint", "Valor": pendulum_cond.get("setpoint", pendulum_cond.get("x_sp", 0.0))},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.conditional_approach.dynamic_penalty.dynamic_penalty_params.control_action_cart_position.weight", "Valor": cart_penalty_cfg["weight"]},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.conditional_approach.dynamic_penalty.dynamic_penalty_params.control_action_cart_position.condition.type", "Valor": cart_cond.get("type")},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.conditional_approach.dynamic_penalty.dynamic_penalty_params.control_action_cart_position.condition.gain", "Valor": cart_cond.get("strength", cart_cond.get("scaled"))},
        {"Parametro": "reward_base.reward_calculation.extra_rewards.conditional_approach.dynamic_penalty.dynamic_penalty_params.control_action_cart_position.condition.setpoint", "Valor": cart_cond.get("setpoint", cart_cond.get("x_sp", 0.0))},
    ])
    
    outcome_table = pd.DataFrame([
        {"Metrica": "window_episodes", "Valor": total_window_episodes},
        {"Metrica": "stabilization_success_last500", "Valor": stabilization_count},
        {"Metrica": "final_t_sec_mean_last500", "Valor": round(float(episode_df["final_t_sec"].mean()), 6)},
        {"Metrica": "final_t_sec_p95_last500", "Valor": round(float(episode_df["final_t_sec"].quantile(0.95)), 6)},
        {"Metrica": "total_reward_mean_last500", "Valor": round(float(episode_df["total_reward"].mean()), 6)},
        {"Metrica": "total_reward_std_last500", "Valor": round(float(episode_df["total_reward"].std()), 6)},
        {"Metrica": "sat_ratio_mean_last500", "Valor": round(float(episode_df["sat_ratio"].mean()), 6)},
        {"Metrica": "sat_ratio_p95_last500", "Valor": round(float(episode_df["sat_ratio"].quantile(0.95)), 6)},
        {"Metrica": "angle_only_fail_pct_last500", "Valor": round(angle_only_ratio, 6)},
        {"Metrica": "cart_only_fail_pct_last500", "Valor": round(cart_only_ratio, 6)},
        {"Metrica": "both_fail_pct_last500", "Valor": round(both_fail_ratio, 6)},
    ])
    
    loop_cost_table = pd.DataFrame(loop_cost_rows)
    
    loop_reward_table = pd.DataFrame([
        {
            "loop": "pendulum_angle",
            "principal_reward_mean": round(pendulum_principal_mean, 6),
            "extra_reward_mean": round(pendulum_extra_mean, 6),
            "agent_reward_mean": round(pendulum_agent_reward_mean, 6),
            "bonus_band_mean": round(pendulum_bonus_mean, 6),
            "dynamic_penalty_mean": round(pendulum_penalty_mean, 6),
            "extra_abs_over_principal_abs": round(abs(pendulum_extra_mean) / abs(pendulum_principal_mean), 6) if pendulum_principal_mean != 0 else np.nan,
        },
        {
            "loop": "cart_position",
            "principal_reward_mean": round(cart_principal_mean, 6),
            "extra_reward_mean": round(cart_extra_mean, 6),
            "agent_reward_mean": round(cart_agent_reward_mean, 6),
            "bonus_band_mean": round(cart_bonus_mean, 6),
            "dynamic_penalty_mean": round(cart_penalty_mean, 6),
            "extra_abs_over_principal_abs": round(abs(cart_extra_mean) / abs(cart_principal_mean), 6) if cart_principal_mean != 0 else np.nan,
        },
    ])
    
    bonus_rows = []
    if pendulum_band_bonus_episode:
        pend_arr = np.array(pendulum_band_bonus_episode, dtype=float)
        bonus_rows.append({
            "loop": "pendulum_angle",
            "accumulated_band_bonus_mean": round(float(np.mean(pend_arr)), 6),
            "accumulated_band_bonus_p95": round(float(np.percentile(pend_arr, 95)), 6),
            "accumulated_band_bonus_max": round(float(np.max(pend_arr)), 6),
            "cap_config": pendulum_cap,
            "cap_hit_pct": round(float(pendulum_cap_hit_ratio), 6),
        })
    if cart_band_bonus_episode:
        cart_arr = np.array(cart_band_bonus_episode, dtype=float)
        bonus_rows.append({
            "loop": "cart_position",
            "accumulated_band_bonus_mean": round(float(np.mean(cart_arr)), 6),
            "accumulated_band_bonus_p95": round(float(np.percentile(cart_arr, 95)), 6),
            "accumulated_band_bonus_max": round(float(np.max(cart_arr)), 6),
            "cap_config": cart_cap,
            "cap_hit_pct": round(float(cart_cap_hit_ratio), 6),
        })
    bonus_table = pd.DataFrame(bonus_rows)
    
    policy_table = action_df.copy()
    if not policy_table.empty:
        policy_table["decrease_%"] = policy_table["decrease_%"].round(4)
        policy_table["maintain_%"] = policy_table["maintain_%"].round(4)
        policy_table["increase_%"] = policy_table["increase_%"].round(4)
    
    physical_table = pd.DataFrame([
        {"Metrica": "abs_pendulum_angle_raw_mean", "Valor": round(float(np.mean(pendulum_angle_abs_means)), 6)},
        {"Metrica": "abs_pendulum_velocity_raw_mean", "Valor": round(float(np.mean(pendulum_velocity_abs_means)), 6)},
        {"Metrica": "abs_cart_position_raw_mean", "Valor": round(float(np.mean(cart_position_abs_means)), 6)},
        {"Metrica": "abs_cart_velocity_raw_mean", "Valor": round(float(np.mean(cart_velocity_abs_means)), 6)},
        {"Metrica": "abs_control_action_pendulum_mean", "Valor": round(float(np.mean(control_action_pendulum_abs_means)), 6)},
        {"Metrica": "abs_control_action_cart_mean", "Valor": round(float(np.mean(control_action_cart_abs_means)), 6)},
        {"Metrica": "mean_abs_u_total_raw_mean", "Valor": round(float(episode_df["mean_abs_u_total_raw"].mean()), 6)},
        {"Metrica": "max_abs_u_total_raw_mean", "Valor": round(float(episode_df["max_abs_u_total_raw"].mean()), 6)},
    ])
    
    gains_rows = []
    gain_cols = sorted([c for c in df.columns if c.startswith("final_k")])
    last_df = df.tail(HISTORY_WINDOW_EPISODES)
    for col in gain_cols:
        s = last_df[col].dropna()
        gains_rows.append({
            "Ganancia": col,
            "Min_last500": round(float(s.min()), 6),
            "Max_last500": round(float(s.max()), 6),
            "Mean_last500": round(float(s.mean()), 6),
            "Std_last500": round(float(s.std()), 6),
        })
    gains_table = pd.DataFrame(gains_rows)
    
    return [
        ("config_context", config_table),
        ("outcome_last500", outcome_table),
        ("loop_cost_components_last500", loop_cost_table),
        ("loop_reward_balance_last500", loop_reward_table),
        ("loop_band_bonus_episode_last500", bonus_table),
        ("policy_action_usage_last500", policy_table),
        ("physical_behavior_last500", physical_table),
        ("final_gains_last500", gains_table),
    ]


def write_decision_summary_sheet(writer, sheet_name, tables):
    """Escribe varias tablas una debajo de otra en una misma hoja."""
    startrow = 0
    for table_name, table_df in tables:
        pd.DataFrame([[table_name]]).to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
            header=False,
            startrow=startrow,
        )
        table_df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
            startrow=startrow + 1,
        )
        startrow += len(table_df) + 4


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
    history_window = load_history_window(paths)
    
    # Crear carpeta de salida
    os.makedirs(paths["output_dir"], exist_ok=True)
    
    # Construir hojas
    print("\n  Generando hojas de análisis...")
    
    sheets = {}
    decision_summary_tables = None
    
    if history_window:
        decision_summary_tables = build_decision_summary_tables(df, meta, history_window, tr_col)
    
    sheets["general_stats"] = build_general_stats(df, meta, tr_col)
    sheets["lagrange_components"] = build_component_stats(df, "L_", "Lagrange")
    sheets["agent_rewards"] = build_component_stats(df, "reward_", "Agent Rewards")
    sheets["extra_rewards"] = build_component_stats(df, "extra_", "Extra Rewards")
    sheets["normalization_audit"] = build_normalization_audit(df, meta)
    sheets["control_signals"] = build_control_signals(df)
    sheets["final_gains"] = build_final_gains(df, tr_col)
    sheets["reward_scale"] = build_reward_scale_analysis(df, meta)
    sheets["state_vars"] = build_state_vars(df)
    
    if history_window:
        sheets["diagnostic_summary"] = build_diagnostic_summary(df, history_window, tr_col)
        sheets["credit_assignment_last500"] = build_credit_assignment_audit(history_window)
        sheets["history_window"] = build_history_window_overview(history_window)
        sheets["action_usage_last500"] = build_action_usage(history_window)
        sheets["interval_signals_last500"] = build_interval_signal_stats(history_window)
        sheets["saturation_last500"] = build_saturation_window(history_window)
        sheets["last_episode_snapshot"] = build_last_episode_snapshot(history_window)
    
    # Escribir Excel
    print(f"  Escribiendo Excel en: {paths['output_xlsx']}")
    with pd.ExcelWriter(paths["output_xlsx"], engine="openpyxl") as writer:
        if decision_summary_tables:
            write_decision_summary_sheet(writer, "decision_summary", decision_summary_tables)
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
    if history_window:
        window_df = pd.DataFrame(history_window["episode_rows"])
        print(f"  Ventana history JSON: {history_window['window_episodes']} episodios detallados")
        print(f"  Saturación global (last500): mean={window_df['sat_ratio'].mean():.4f}, p95={window_df['sat_ratio'].quantile(0.95):.4f}")
    
    print(f"\n  Excel generado: {paths['output_xlsx']}")
    print(f"  Carpeta de análisis: {paths['output_dir']}")
    

def main():
    if len(sys.argv) < 2:
        print("Uso: python analysis/run_analysis.py <sim_id>")
        print("Ejemplo: python analysis/run_analysis.py 20260227_1310")
        sys.exit(1)
    
    sim_id = sys.argv[1]
    run_analysis(sim_id)


if __name__ == "__main__":
    main()
