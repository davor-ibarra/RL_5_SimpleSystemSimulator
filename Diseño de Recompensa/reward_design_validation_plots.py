import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("reward_validation_plots")
out_dir.mkdir(parents=True, exist_ok=True)

reward_design = {
    "weights": {"L_e": 0.82, "L_edot": 0.12, "L_delta_u": 0.06},
    "scaled": {"L_e": 5.0, "L_edot": 2.5, "L_delta_u": 4.0},
    "gates": {
        "pendulum_angle": {
            "L_e": {"type": "constant", "value": 1.0},
            "L_edot": {"type": "exp", "scaled": 0.12, "min_factor": 0.0, "max_factor": 1.0},
            "L_delta_u": {"type": "exp", "scaled": 0.10, "min_factor": 0.0, "max_factor": 1.0},
        },
        "cart_position": {
            "L_e": {"type": "exp", "scaled": 0.22, "min_factor": 0.05, "max_factor": 1.0},
            "L_edot": {"type": "exp", "scaled": 0.08, "min_factor": 0.0, "max_factor": 1.0},
            "L_delta_u": {"type": "exp", "scaled": 0.07, "min_factor": 0.0, "max_factor": 1.0},
        },
    },
    "band_bonus_per_step": {"pendulum_angle": 0.004, "cart_position": 0.004},
    "band_threshold_abs_error": {"pendulum_angle": 0.025, "cart_position": 0.020},
}

def exp_gate_from_abs_error(abs_error, scaled, min_factor, max_factor):
    factor = np.exp(-((abs_error / scaled) ** 2))
    return np.clip(factor, min_factor, max_factor)

def gate_value(var_obj, feature, abs_error):
    cfg = reward_design["gates"][var_obj][feature]
    if cfg["type"] == "constant":
        return np.ones_like(abs_error) * cfg["value"]
    return exp_gate_from_abs_error(abs_error, cfg["scaled"], cfg["min_factor"], cfg["max_factor"])

def feature_reward(weight, scaled, value_L, gate):
    return -gate * weight * (1.0 - np.exp(-scaled * (value_L ** 2)))

def total_local_reward(var_obj, abs_error_rms, abs_edot_rms_norm, abs_delta_u_rms_norm):
    L_e = np.clip(abs_error_rms ** 2, 0.0, 1.0)
    L_edot = np.clip(abs_edot_rms_norm ** 2, 0.0, 1.0)
    L_du = np.clip(abs_delta_u_rms_norm ** 2, 0.0, 1.0)

    g_e = gate_value(var_obj, "L_e", abs_error_rms)
    g_edot = gate_value(var_obj, "L_edot", abs_error_rms)
    g_du = gate_value(var_obj, "L_delta_u", abs_error_rms)

    r_e = feature_reward(reward_design["weights"]["L_e"], reward_design["scaled"]["L_e"], L_e, g_e)
    r_edot = feature_reward(reward_design["weights"]["L_edot"], reward_design["scaled"]["L_edot"], L_edot, g_edot)
    r_du = feature_reward(reward_design["weights"]["L_delta_u"], reward_design["scaled"]["L_delta_u"], L_du, g_du)

    band_bonus = np.where(
        abs_error_rms <= reward_design["band_threshold_abs_error"][var_obj],
        reward_design["band_bonus_per_step"][var_obj],
        0.0,
    )
    return r_e + r_edot + r_du + band_bonus

def save_reward_map(var_obj):
    e_vals = np.linspace(0.0, 1.0, 400)
    edot_vals = np.linspace(0.0, 1.0, 400)
    E, EDOT = np.meshgrid(e_vals, edot_vals)

    reward_map = total_local_reward(var_obj, E, EDOT, np.zeros_like(E))

    plt.figure(figsize=(9, 6))
    im = plt.imshow(
        reward_map,
        origin="lower",
        aspect="auto",
        extent=[e_vals.min(), e_vals.max(), edot_vals.min(), edot_vals.max()],
    )
    plt.colorbar(im, label="Recompensa local (principal + banda/step)")
    plt.xlabel(r"$|e|_{\mathrm{RMS}}$ normalizado")
    plt.ylabel(r"$|\dot{e}|_{\mathrm{RMS}}$ normalizado")
    plt.title(f"Mapa de recompensa local — {var_obj}  ($|\\Delta u|_{{RMS}}=0$)")
    plt.axvline(reward_design["band_threshold_abs_error"][var_obj], linestyle=":")
    plt.tight_layout()
    plt.savefig(out_dir / f"reward_map_{var_obj}.png", dpi=180, bbox_inches="tight")
    plt.close()

def save_reward_profile(var_obj):
    e_vals = np.linspace(0.0, 1.0, 600)

    scenarios = [
        {"label": r"ideal: $|\dot e|=0.00,\ |\Delta u|=0.00$", "edot": 0.00, "du": 0.00},
        {"label": r"suave: $|\dot e|=0.15,\ |\Delta u|=0.10$", "edot": 0.15, "du": 0.10},
        {"label": r"medio: $|\dot e|=0.30,\ |\Delta u|=0.20$", "edot": 0.30, "du": 0.20},
        {"label": r"agresivo: $|\dot e|=0.45,\ |\Delta u|=0.35$", "edot": 0.45, "du": 0.35},
    ]

    plt.figure(figsize=(9, 6))
    for s in scenarios:
        reward_curve = total_local_reward(
            var_obj=var_obj,
            abs_error_rms=e_vals,
            abs_edot_rms_norm=np.full_like(e_vals, s["edot"]),
            abs_delta_u_rms_norm=np.full_like(e_vals, s["du"]),
        )
        plt.plot(e_vals, reward_curve, label=s["label"])

    plt.axvline(reward_design["band_threshold_abs_error"][var_obj], linestyle=":")
    plt.xlabel(r"$|e|_{\mathrm{RMS}}$ normalizado")
    plt.ylabel("Recompensa local (principal + banda/step)")
    plt.title(f"Perfiles de recompensa local — {var_obj}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"reward_profile_{var_obj}.png", dpi=180, bbox_inches="tight")
    plt.close()

for var in ["pendulum_angle", "cart_position"]:
    save_reward_map(var)
    save_reward_profile(var)