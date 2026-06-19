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
        or k.startswith("local_")
        or k.startswith("extra_")
        or k.startswith("cooperative_")
        or k.startswith("credit_")
        or k.startswith("agent_credit_")
        or k.startswith("regime_")
        or k.startswith("internal_risk_")
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
    """Hoja: Auditoría de normalización - rangos config vs rangos reales."""
    reward_cfg = meta["config_main"]["reward_base"]["reward_calculation"]
    metric_processing_cfg = reward_cfg["metric_processing"]
    rows = []

    if "normalization" in metric_processing_cfg:
        norm_cfg = metric_processing_cfg["normalization"]
        if "signals_params" in norm_cfg:
            for variable_name, signals_cfg in sorted(norm_cfg["signals_params"].items()):
                for normalized_signal, signal_cfg in sorted(signals_cfg.items()):
                    cfg_range = signal_cfg["range"]
                    source_signal = signal_cfg["signal"]
                    max_abs = max(abs(float(cfg_range[0])), abs(float(cfg_range[1])))
                    actual_col = None
                    for candidate in [
                        f"{normalized_signal}_mean",
                        normalized_signal,
                        f"{source_signal}_mean",
                        source_signal,
                    ]:
                        if candidate in df.columns:
                            actual_col = candidate
                            break
                    if actual_col:
                        s = df[actual_col].dropna()
                        actual_min = round(float(s.min()), 6)
                        actual_max = round(float(s.max()), 6)
                        actual_mean = round(float(s.mean()), 6)
                        observed_abs = max(abs(float(s.min())), abs(float(s.max())))
                        coverage = round(observed_abs / max_abs * 100.0, 1) if max_abs > 0 else "N/A"
                    else:
                        actual_min = actual_max = actual_mean = coverage = "N/A"
                    rows.append({
                        "Familia": "normalization.signals_params",
                        "Variable": variable_name,
                        "Métrica": normalized_signal,
                        "Fuente": source_signal,
                        "Método": "linear_range",
                        "Rango Config": str(cfg_range),
                        "max_abs": round(max_abs, 6),
                        "Actual Col": actual_col if actual_col else "N/A",
                        "Actual Min": actual_min,
                        "Actual Max": actual_max,
                        "Actual Mean": actual_mean,
                        "Cobertura %": coverage,
                    })

    if "aggregation" in metric_processing_cfg:
        aggregation_cfg = metric_processing_cfg["aggregation"]
        if "features_params" in aggregation_cfg:
            for variable_name, features_cfg in sorted(aggregation_cfg["features_params"].items()):
                for feature_name, feature_cfg in sorted(features_cfg.items()):
                    method = feature_cfg["method"]
                    cfg_range = feature_cfg["range"]
                    if "record" in feature_cfg:
                        record_name = feature_cfg["record"].format(method=method)
                    else:
                        record_name = f"{feature_cfg['root_var']}_{method}"
                    max_abs = max(abs(float(cfg_range[0])), abs(float(cfg_range[1])))
                    scaled_max = max_abs ** 2 if method in ["mean_squared", "rms"] else max_abs
                    actual_col = None
                    for candidate in [f"{record_name}_mean", record_name]:
                        if candidate in df.columns:
                            actual_col = candidate
                            break
                    if actual_col:
                        s = df[actual_col].dropna()
                        actual_min = round(float(s.min()), 6)
                        actual_max = round(float(s.max()), 6)
                        actual_mean = round(float(s.mean()), 6)
                        observed_max = float(s.max())
                        coverage = round(observed_max / scaled_max * 100.0, 1) if scaled_max > 0 else "N/A"
                    else:
                        actual_min = actual_max = actual_mean = coverage = "N/A"
                    if variable_name == "global_vars":
                        l_col = f"L_{feature_name}_mean"
                    else:
                        l_col = f"L_{feature_name}_{variable_name}_mean"
                    if l_col in df.columns:
                        l_s = df[l_col].dropna()
                        l_mean = round(float(l_s.mean()), 6)
                        l_std = round(float(l_s.std()), 6)
                    else:
                        l_mean = l_std = "N/A"
                    rows.append({
                        "Familia": "aggregation.features_params",
                        "Variable": variable_name,
                        "Métrica": feature_name,
                        "Fuente": feature_cfg["root_var"],
                        "Método": method,
                        "Rango Config": str(cfg_range),
                        "max_abs": round(scaled_max, 6),
                        "Actual Col": actual_col if actual_col else "N/A",
                        "Actual Min": actual_min,
                        "Actual Max": actual_max,
                        "Actual Mean": actual_mean,
                        "Cobertura %": coverage,
                        "L_* Mean": l_mean,
                        "L_* Std": l_std,
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
    """Hoja: Analisis de escala de recompensas y familias persistidas."""
    reward_cfg = meta["config_main"]["reward_base"]["reward_calculation"]
    reward_config = meta["config_main"]["reward_base"]["reward_config"]
    rows = []

    rows.append({"Aspecto": "reward_approach", "Valor": reward_config["reward_approach"], "Detalle": ""})
    if "reward_composition" in reward_config:
        for name, value in sorted(reward_config["reward_composition"].items()):
            rows.append({"Aspecto": f"reward_composition.{name}", "Valor": value, "Detalle": ""})

    if "local_control_quality" in reward_cfg:
        local_cfg = reward_cfg["local_control_quality"]
        rows.append({"Aspecto": "local_control_quality.enabled", "Valor": local_cfg["enabled"], "Detalle": ""})
        for cost_name, weights_by_variable in sorted(local_cfg["cost_weights"].items()):
            for variable_name, weight in sorted(weights_by_variable.items()):
                rows.append({
                    "Aspecto": f"local_control_quality.cost_weights.{cost_name}.{variable_name}",
                    "Valor": weight,
                    "Detalle": "",
                })

    if "cooperative_transition_evaluator" in reward_cfg:
        global_potential = reward_cfg["cooperative_transition_evaluator"]["global_potential"]
        rows.append({
            "Aspecto": "cooperative_transition_evaluator.global_potential.aggregation_mode",
            "Valor": global_potential["aggregation_mode"],
            "Detalle": "",
        })
        if global_potential["aggregation_mode"] == "additive_sync_blend":
            sync_quality = global_potential["sync_quality"]
            rows.append({
                "Aspecto": "cooperative_transition_evaluator.global_potential.sync_quality.weight",
                "Valor": sync_quality["weight"],
                "Detalle": f"enabled={sync_quality['enabled']}, epsilon={sync_quality['epsilon']}",
            })

    if "total_reward" in df.columns:
        tr_mean = abs(df["total_reward"].mean())
        rows.append({"Aspecto": "|total_reward| mean", "Valor": round(tr_mean, 4), "Detalle": "Referencia"})

        component_prefixes = [
            "reward_",
            "local_",
            "cooperative_",
            "credit_",
            "agent_credit_",
            "internal_risk_",
            "regime_",
            "extra_",
            "L_",
        ]
        component_cols = sorted([
            col for col in df.columns
            if col.endswith("_mean") and any(col.startswith(prefix) for prefix in component_prefixes)
        ])
        for col in component_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            abs_mean = float(s.abs().mean())
            ratio = abs_mean / tr_mean * 100.0 if tr_mean > 0 else np.nan
            rows.append({
                "Aspecto": col,
                "Valor": round(abs_mean, 6),
                "Detalle": f"{round(ratio, 2)}% del |total_reward| mean" if tr_mean > 0 else "total_reward mean cero",
            })
    
    return pd.DataFrame(rows)


def build_state_vars(df):
    """Hoja: Variables de estado (raw y normalized)."""
    stat_suffixes = ["_mean", "_std", "_min", "_p25", "_p50", "_p75", "_max"]
    state_vars = []
    for col in df.columns:
        for suffix in ["_raw", "_norm"]:
            for stat in stat_suffixes:
                if col.endswith(f"{suffix}{stat}"):
                    variable_name = col[: -len(f"{suffix}{stat}")]
                    if variable_name not in state_vars:
                        state_vars.append(variable_name)
    rows = []
    
    for var in state_vars:
        for suffix in ["_raw", "_norm"]:
            for stat in stat_suffixes:
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
        or k.startswith("local_")
        or k.startswith("extra_")
        or k.startswith("cooperative_")
        or k.startswith("credit_")
        or k.startswith("agent_credit_")
        or k.startswith("regime_")
        or k.startswith("internal_risk_")
    ]
    for signal_key in signal_keys:
        signal_arr = np.array(interval_data[signal_key], dtype=float)
        rows.append({"Aspecto": f"{signal_key}_mean", "Valor": round(float(np.mean(signal_arr)), 6)})
    
    return pd.DataFrame(rows)


SIGNAL_SUMMARY_COLUMNS = ["Signal", "Min", "Max", "Mean", "Std", "Median", "p5", "p95"]


def interval_signal_mean(interval_signal_values, signal_name):
    """Media estricta de una señal interval-level ya seleccionada."""
    values = interval_signal_values[signal_name]
    arr = np.array(values, dtype=float)
    return float(np.mean(arr))


def summarize_interval_signal_prefixes(interval_signal_values, prefixes):
    """Resume señales interval-level por prefijos definidos por el análisis."""
    rows = []
    for signal_name in sorted(interval_signal_values):
        if not any(signal_name.startswith(prefix) for prefix in prefixes):
            continue
        values = interval_signal_values[signal_name]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        rows.append({
            "Signal": signal_name,
            "Min": round(float(np.min(arr)), 6),
            "Max": round(float(np.max(arr)), 6),
            "Mean": round(float(np.mean(arr)), 6),
            "Std": round(float(np.std(arr)), 6),
            "Median": round(float(np.median(arr)), 6),
            "p5": round(float(np.percentile(arr, 5)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
        })
    return pd.DataFrame(rows, columns=SIGNAL_SUMMARY_COLUMNS)


def collect_declared_reward_variables(reward_cfg):
    """Variables de control declaradas por los componentes configurados."""
    variables = []
    if "local_control_quality" in reward_cfg:
        for weights_by_variable in reward_cfg["local_control_quality"]["cost_weights"].values():
            for variable_name in weights_by_variable:
                if variable_name not in variables:
                    variables.append(variable_name)
    if "cooperative_transition_evaluator" in reward_cfg:
        variable_weights = reward_cfg["cooperative_transition_evaluator"]["global_potential"]["variable_weights"]
        for variable_name in variable_weights:
            if variable_name not in variables:
                variables.append(variable_name)
    if "credit_allocator" in reward_cfg:
        controller_assignment = reward_cfg["credit_allocator"]["controller_assignment"]
        if "correction_sign" in controller_assignment:
            for variable_name in controller_assignment["correction_sign"]:
                if variable_name not in variables:
                    variables.append(variable_name)
    return variables


def signal_belongs_to_variable(signal_name, variable_name):
    return signal_name.endswith(f"_{variable_name}")


def build_credit_assignment_audit(history_window):
    """Hoja: Auditoria de credito usando señales interval-level reales."""
    interval_signal_values = history_window["interval_signal_values"]
    
    global_reward_mean = float(np.mean(np.array(interval_signal_values["global_interval_reward"], dtype=float)))
    rows = [
        {
            "Chequeo": "global_interval_reward_mean",
            "Valor": round(global_reward_mean, 6),
            "Lectura": "Reward global medio por decision",
            "Accion sugerida": "",
        },
    ]

    reward_signed_keys = [
        signal_name for signal_name in interval_signal_values
        if signal_name.startswith("reward_") and "_signed_" in signal_name
    ]
    for signal_name in sorted(reward_signed_keys):
        signal_mean = interval_signal_mean(interval_signal_values, signal_name)
        rows.append({
            "Chequeo": f"{signal_name}_mean",
            "Valor": round(signal_mean, 6),
            "Lectura": "Componente signed de reward persistido por el sistema",
            "Accion sugerida": "",
        })
        rows.append({
            "Chequeo": f"global_minus_{signal_name}",
            "Valor": round(global_reward_mean - signal_mean, 6),
            "Lectura": "Gap entre reward global y componente signed",
            "Accion sugerida": "Si el gap domina, revisar composicion y asignacion de credito",
        })

    for signal_name in sorted(interval_signal_values):
        if not (
            signal_name.startswith("cooperative_global_")
            or signal_name.startswith("credit_")
            or signal_name.startswith("agent_credit_")
        ):
            continue
        rows.append({
            "Chequeo": f"{signal_name}_mean",
            "Valor": round(interval_signal_mean(interval_signal_values, signal_name), 6),
            "Lectura": "Señal cooperativa/crediticia media en ventana detallada",
            "Accion sugerida": "",
        })
    
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
    if maintain_ratios:
        lowest_maintain_action, lowest_maintain_value, lowest_maintain_bias = maintain_ratios[0]
        mean_maintain = float(np.mean([item[1] for item in maintain_ratios]))
    else:
        lowest_maintain_action = ""
        lowest_maintain_value = np.nan
        lowest_maintain_bias = np.nan
        mean_maintain = np.nan
    
    sat_ratio_mean = float(episode_df["sat_ratio"].mean())
    sat_ratio_p95 = float(episode_df["sat_ratio"].quantile(0.95))
    
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
    cooperative_global_quality = "cooperative_global_quality_signed"
    cooperative_linear_quality = "cooperative_global_linear_quality_signed"
    cooperative_sync_quality = "cooperative_global_sync_quality_signed"
    cooperative_sync_weight = "cooperative_global_sync_weight"

    if cooperative_global_quality in interval_signal_values:
        rows.append({
            "Chequeo": "cooperative_global_quality_signed_mean",
            "Valor": round(interval_signal_mean(interval_signal_values, cooperative_global_quality), 6),
            "Lectura": "Calidad cooperativa global firmada media",
            "Accion sugerida": "Si cae mientras sube la reward, revisar composicion o credito",
        })
    if cooperative_linear_quality in interval_signal_values and cooperative_sync_quality in interval_signal_values:
        linear_arr = np.array(interval_signal_values[cooperative_linear_quality], dtype=float)
        sync_arr = np.array(interval_signal_values[cooperative_sync_quality], dtype=float)
        gap_arr = linear_arr - sync_arr
        rows.append({
            "Chequeo": "cooperative_linear_minus_sync_quality_signed_mean",
            "Valor": round(float(np.mean(gap_arr)), 6),
            "Lectura": "Gap medio entre calidad lineal y calidad de sincronizacion",
            "Accion sugerida": "Si aumenta, auditar asincronia entre variables y beta_coop",
        })
        rows.append({
            "Chequeo": "cooperative_sync_gap_abs_mean",
            "Valor": round(float(np.mean(np.abs(gap_arr))), 6),
            "Lectura": "Desalineacion absoluta media entre calidad lineal y sync",
            "Accion sugerida": "Si domina la ventana, revisar peso sync_quality y credit_allocator",
        })
    if cooperative_sync_weight in interval_signal_values:
        rows.append({
            "Chequeo": "cooperative_global_sync_weight_mean",
            "Valor": round(interval_signal_mean(interval_signal_values, cooperative_sync_weight), 6),
            "Lectura": "Peso de sincronizacion aplicado por el evaluador cooperativo",
            "Accion sugerida": "",
        })

    credit_fraction_signals = [
        signal_name for signal_name in interval_signal_values
        if signal_name.startswith("credit_") and signal_name.endswith("_fraction")
    ]
    if not credit_fraction_signals:
        credit_fraction_signals = [
            signal_name for signal_name in interval_signal_values
            if signal_name.startswith("credit_") and "_fraction_" in signal_name
        ]
    for signal_name in sorted(credit_fraction_signals):
        rows.append({
            "Chequeo": f"{signal_name}_mean",
            "Valor": round(interval_signal_mean(interval_signal_values, signal_name), 6),
            "Lectura": "Fraccion de credito persistida por el allocator",
            "Accion sugerida": "",
        })
    
    return pd.DataFrame(rows)




def build_decision_summary_tables(df, meta, history_window, tr_col):
    """Construye tablas resumen desde la configuracion efectiva y señales persistidas."""
    config_main = meta["config_main"]
    sim_cfg = config_main["simulation"]
    agent_base_cfg = config_main["agent_base"]
    actions_cfg = agent_base_cfg["agent_config"]["actions"]["actions_values"]
    reward_base_cfg = config_main["reward_base"]
    reward_config = reward_base_cfg["reward_config"]
    reward_cfg = reward_base_cfg["reward_calculation"]
    boundary_cfg = config_main["dynamic_system"]["termination_conditions"]["boundary_constraint"]

    declared_variables = collect_declared_reward_variables(reward_cfg)
    if not declared_variables:
        raise KeyError("No se encontraron variables declaradas en reward_calculation")

    selected_episodes = history_window["selected_episodes"]
    interval_signal_values = history_window["interval_signal_values"]
    episode_df = pd.DataFrame(history_window["episode_rows"])
    action_df = build_action_usage(history_window).copy()

    stabilization_count = 0
    if tr_col:
        stabilization_count = int(np.sum(df.tail(HISTORY_WINDOW_EPISODES)[tr_col] == "stabilization_success"))

    config_rows = [
        {"Parametro": "simulation.decision_interval_sec", "Valor": sim_cfg["decision_interval_sec"]},
        {"Parametro": "simulation.dt_sec", "Valor": sim_cfg["dt_sec"]},
        {
            "Parametro": "agent_base.agent_config.actions.actions_values.universal_params.delta_gain",
            "Valor": actions_cfg["universal_params"]["delta_gain"],
        },
        {"Parametro": "agent_base.params.discount_factor", "Valor": agent_base_cfg["params"]["discount_factor"]},
        {"Parametro": "reward_base.reward_config.reward_approach", "Valor": reward_config["reward_approach"]},
    ]

    if "reward_composition" in reward_config:
        for name, value in sorted(reward_config["reward_composition"].items()):
            config_rows.append({
                "Parametro": f"reward_base.reward_config.reward_composition.{name}",
                "Valor": value,
            })

    if "local_control_quality" in reward_cfg:
        local_cfg = reward_cfg["local_control_quality"]
        config_rows.extend([
            {"Parametro": "reward_base.reward_calculation.local_control_quality.enabled", "Valor": local_cfg["enabled"]},
            {
                "Parametro": "reward_base.reward_calculation.local_control_quality.normalized_reward_mode",
                "Valor": local_cfg["normalized_reward_mode"],
            },
            {
                "Parametro": "reward_base.reward_calculation.local_control_quality.strict_weight_sum",
                "Valor": local_cfg["strict_weight_sum"],
            },
        ])
        for cost_name, weights_by_variable in sorted(local_cfg["cost_weights"].items()):
            for variable_name, weight in sorted(weights_by_variable.items()):
                config_rows.append({
                    "Parametro": f"reward_base.reward_calculation.local_control_quality.cost_weights.{cost_name}.{variable_name}",
                    "Valor": weight,
                })

    if "cooperative_transition_evaluator" in reward_cfg:
        cooperative_cfg = reward_cfg["cooperative_transition_evaluator"]
        global_potential = cooperative_cfg["global_potential"]
        config_rows.extend([
            {
                "Parametro": "reward_base.reward_calculation.cooperative_transition_evaluator.enabled",
                "Valor": cooperative_cfg["enabled"],
            },
            {
                "Parametro": "reward_base.reward_calculation.cooperative_transition_evaluator.global_potential.aggregation_mode",
                "Valor": global_potential["aggregation_mode"],
            },
        ])
        if global_potential["aggregation_mode"] == "additive_sync_blend":
            sync_quality = global_potential["sync_quality"]
            config_rows.extend([
                {
                    "Parametro": "reward_base.reward_calculation.cooperative_transition_evaluator.global_potential.sync_quality.enabled",
                    "Valor": sync_quality["enabled"],
                },
                {
                    "Parametro": "reward_base.reward_calculation.cooperative_transition_evaluator.global_potential.sync_quality.weight",
                    "Valor": sync_quality["weight"],
                },
                {
                    "Parametro": "reward_base.reward_calculation.cooperative_transition_evaluator.global_potential.sync_quality.epsilon",
                    "Valor": sync_quality["epsilon"],
                },
            ])
        for feature_name, weights_by_variable in sorted(global_potential["feature_weights"].items()):
            for variable_name, weight in sorted(weights_by_variable.items()):
                config_rows.append({
                    "Parametro": f"reward_base.reward_calculation.cooperative_transition_evaluator.global_potential.feature_weights.{feature_name}.{variable_name}",
                    "Valor": weight,
                })
        for variable_name, weight in sorted(global_potential["variable_weights"].items()):
            config_rows.append({
                "Parametro": f"reward_base.reward_calculation.cooperative_transition_evaluator.global_potential.variable_weights.{variable_name}",
                "Valor": weight,
            })

    if "credit_allocator" in reward_cfg:
        credit_cfg = reward_cfg["credit_allocator"]
        controller_assignment = credit_cfg["controller_assignment"]
        agent_assignment = credit_cfg["agent_assignment"]
        config_rows.extend([
            {"Parametro": "reward_base.reward_calculation.credit_allocator.enabled", "Valor": credit_cfg["enabled"]},
            {
                "Parametro": "reward_base.reward_calculation.credit_allocator.controller_assignment.mode",
                "Valor": controller_assignment["mode"],
            },
            {
                "Parametro": "reward_base.reward_calculation.credit_allocator.agent_assignment.mode",
                "Valor": agent_assignment["mode"],
            },
        ])
        for weight_family in ["credit_gain", "harmful_conflict_weight"]:
            if weight_family in controller_assignment:
                for variable_name, weight in sorted(controller_assignment[weight_family].items()):
                    config_rows.append({
                        "Parametro": f"reward_base.reward_calculation.credit_allocator.controller_assignment.{weight_family}.{variable_name}",
                        "Valor": weight,
                    })

    if "internal_risk_penalty" in reward_cfg:
        risk_cfg = reward_cfg["internal_risk_penalty"]
        config_rows.append({
            "Parametro": "reward_base.reward_calculation.internal_risk_penalty.enabled",
            "Valor": risk_cfg["enabled"],
        })
        for risk_name, weight in sorted(risk_cfg["risk_weights"].items()):
            config_rows.append({
                "Parametro": f"reward_base.reward_calculation.internal_risk_penalty.risk_weights.{risk_name}",
                "Valor": weight,
            })

    config_table = pd.DataFrame(config_rows)

    outcome_rows = [
        {"Metrica": "window_episodes", "Valor": len(selected_episodes)},
        {"Metrica": "stabilization_success_last500", "Valor": stabilization_count},
        {"Metrica": "final_t_sec_mean_last500", "Valor": round(float(episode_df["final_t_sec"].mean()), 6)},
        {"Metrica": "final_t_sec_p95_last500", "Valor": round(float(episode_df["final_t_sec"].quantile(0.95)), 6)},
        {"Metrica": "total_reward_mean_last500", "Valor": round(float(episode_df["total_reward"].mean()), 6)},
        {"Metrica": "total_reward_std_last500", "Valor": round(float(episode_df["total_reward"].std()), 6)},
        {"Metrica": "sat_ratio_mean_last500", "Valor": round(float(episode_df["sat_ratio"].mean()), 6)},
        {"Metrica": "sat_ratio_p95_last500", "Valor": round(float(episode_df["sat_ratio"].quantile(0.95)), 6)},
    ]
    for signal_name in [
        "cooperative_global_quality_signed",
        "cooperative_global_linear_quality_signed",
        "cooperative_global_sync_quality_signed",
        "cooperative_global_marginal_signed",
        "cooperative_global_sync_weight",
    ]:
        if signal_name in interval_signal_values:
            outcome_rows.append({
                "Metrica": f"{signal_name}_mean_last500",
                "Valor": round(interval_signal_mean(interval_signal_values, signal_name), 6),
            })
    if (
        "cooperative_global_linear_quality_signed" in interval_signal_values
        and "cooperative_global_sync_quality_signed" in interval_signal_values
    ):
        linear_arr = np.array(interval_signal_values["cooperative_global_linear_quality_signed"], dtype=float)
        sync_arr = np.array(interval_signal_values["cooperative_global_sync_quality_signed"], dtype=float)
        gap_arr = linear_arr - sync_arr
        outcome_rows.extend([
            {
                "Metrica": "cooperative_linear_minus_sync_quality_signed_mean_last500",
                "Valor": round(float(np.mean(gap_arr)), 6),
            },
            {
                "Metrica": "cooperative_sync_gap_abs_mean_last500",
                "Valor": round(float(np.mean(np.abs(gap_arr))), 6),
            },
        ])
    outcome_table = pd.DataFrame(outcome_rows)

    local_rows = []
    local_cost_weights = {}
    if "local_control_quality" in reward_cfg:
        local_cost_weights = reward_cfg["local_control_quality"]["cost_weights"]
    for variable_name in declared_variables:
        row = {"variable": variable_name}
        for cost_name in sorted(local_cost_weights):
            signal_name = f"local_{cost_name}_cost_01_{variable_name}"
            if signal_name in interval_signal_values:
                row[f"{cost_name}_cost_01_mean"] = round(interval_signal_mean(interval_signal_values, signal_name), 6)
        for signal_root in [
            "local_cost_01",
            "local_quality_01",
            "local_quality_signed",
            "local_marginal_quality_signed",
        ]:
            signal_name = f"{signal_root}_{variable_name}"
            if signal_name in interval_signal_values:
                row[f"{signal_root}_mean"] = round(interval_signal_mean(interval_signal_values, signal_name), 6)
        local_rows.append(row)
    local_quality_table = pd.DataFrame(local_rows)

    reward_rows = []
    for variable_name in declared_variables:
        for signal_name in sorted(interval_signal_values):
            if not signal_belongs_to_variable(signal_name, variable_name):
                continue
            if not (
                signal_name.startswith("reward_")
                or signal_name.startswith("cooperative_potential_")
                or signal_name.startswith("cooperative_marginal_")
                or signal_name.startswith("credit_")
                or signal_name.startswith("agent_credit_")
            ):
                continue
            reward_rows.append({
                "variable": variable_name,
                "signal": signal_name,
                "mean": round(interval_signal_mean(interval_signal_values, signal_name), 6),
            })
    reward_signal_table = pd.DataFrame(reward_rows, columns=["variable", "signal", "mean"])

    cooperative_quality_table = summarize_interval_signal_prefixes(interval_signal_values, ["cooperative_"])
    credit_table = summarize_interval_signal_prefixes(interval_signal_values, ["credit_", "agent_credit_"])
    internal_risk_table = summarize_interval_signal_prefixes(interval_signal_values, ["internal_risk_"])

    physical_rows = []
    for variable_name in declared_variables:
        raw_key = f"{variable_name}_raw"
        control_key = f"control_action_{variable_name}"
        raw_abs_means = []
        control_abs_means = []
        boundary_hits = 0
        boundary_samples = 0
        for episode in selected_episodes:
            step_data = episode["step_data"]
            if raw_key in step_data:
                raw_arr = np.array(step_data[raw_key], dtype=float)
                raw_abs_means.append(float(np.mean(np.abs(raw_arr))))
                if variable_name in boundary_cfg and len(raw_arr) > 0:
                    limit = max(abs(float(value)) for value in boundary_cfg[variable_name])
                    boundary_hits += int(abs(float(raw_arr[-1])) >= limit)
                    boundary_samples += 1
            if control_key in step_data:
                control_abs_means.append(float(np.mean(np.abs(np.array(step_data[control_key], dtype=float)))))
        row = {"variable": variable_name}
        if raw_abs_means:
            row["raw_abs_mean_last500"] = round(float(np.mean(raw_abs_means)), 6)
        if control_abs_means:
            row["control_abs_mean_last500"] = round(float(np.mean(control_abs_means)), 6)
        if boundary_samples > 0:
            row["final_boundary_hit_pct_last500"] = round(float(boundary_hits) / boundary_samples * 100.0, 6)
        physical_rows.append(row)
    physical_table = pd.DataFrame(physical_rows)

    policy_table = action_df.copy()
    if not policy_table.empty:
        policy_table["decrease_%"] = policy_table["decrease_%"].round(4)
        policy_table["maintain_%"] = policy_table["maintain_%"].round(4)
        policy_table["increase_%"] = policy_table["increase_%"].round(4)

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
        ("local_quality_components_last500", local_quality_table),
        ("reward_components_last500", reward_signal_table),
        ("cooperative_quality_last500", cooperative_quality_table),
        ("credit_components_last500", credit_table),
        ("internal_risk_last500", internal_risk_table),
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
    sheets["local_control_quality"] = build_component_stats(df, "local_", "Local Control Quality")
    sheets["agent_rewards"] = build_component_stats(df, "reward_", "Agent Rewards")
    sheets["extra_rewards"] = build_component_stats(df, "extra_", "Extra Rewards")
    sheets["cooperative_transition"] = build_component_stats(
        df,
        "cooperative_",
        "Cooperative Transition",
    )
    sheets["credit_components"] = build_component_stats(df, "credit_", "Credit Components")
    sheets["agent_credit_components"] = build_component_stats(
        df,
        "agent_credit_",
        "Agent Credit Components",
    )
    sheets["internal_risk_penalty"] = build_component_stats(df, "internal_risk_", "Internal Risk")
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
