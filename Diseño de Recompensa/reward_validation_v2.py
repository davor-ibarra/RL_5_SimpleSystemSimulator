import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("reward_validation_v2")
out_dir.mkdir(parents=True, exist_ok=True)

proposal = {
    "principal_reward": {
        "features": {
            "L_e": {"weight": 0.78, "scaled": 8.0, "setpoint": 0.0},
            "L_edot": {"weight": 0.10, "scaled": 3.0, "setpoint": 0.0},
            "L_u": {"weight": 0.12, "scaled": 2.5, "setpoint": 0.0},
        },
        "feature_gates": {
            "L_e": {
                "pendulum_angle": {"type": "constant", "value": 1.0},
                "cart_position": {"type": "exp", "scaled": 0.35, "min_factor": 0.20, "max_factor": 1.0},
            },
            "L_edot": {
                "pendulum_angle": {"type": "exp", "scaled": 0.45, "min_factor": 0.05, "max_factor": 1.0},
                "cart_position": {"type": "exp", "scaled": 0.18, "min_factor": 0.00, "max_factor": 1.0},
            },
            "L_u": {
                "pendulum_angle": {"type": "exp", "scaled": 0.35, "min_factor": 0.05, "max_factor": 1.0},
                "cart_position": {"type": "exp", "scaled": 0.16, "min_factor": 0.00, "max_factor": 1.0},
            },
        },
    },
    "extra_rewards": {
        "bandwidth_bonus": {
            "per_step_band_bonus": 0.003,
            "ranges": {"pendulum_angle": 0.020, "cart_position": 0.015},
        },
    },
}

def exp_gate(abs_error_rms, scaled, min_factor=0.0, max_factor=1.0):
    factor = np.exp(-((abs_error_rms / scaled) ** 2))
    return np.clip(factor, min_factor, max_factor)

def gate_value(var_obj, feature, abs_error_rms):
    cfg = proposal["principal_reward"]["feature_gates"][feature][var_obj]
    if cfg["type"] == "constant":
        return np.ones_like(abs_error_rms) * cfg["value"]
    return exp_gate(abs_error_rms, cfg["scaled"], cfg["min_factor"], cfg["max_factor"])

def feature_reward(weight, scaled, L_value, gate):
    return -gate * weight * (1.0 - np.exp(-scaled * (L_value ** 2)))

def total_local_reward(var_obj, e_rms, edot_rms, u_rms):
    L_e = np.clip(e_rms, 0.0, 1.0)
    L_edot = np.clip(edot_rms, 0.0, 1.0)
    L_u = np.clip(u_rms, 0.0, 1.0)

    w = proposal["principal_reward"]["features"]
    g_e = gate_value(var_obj, "L_e", e_rms)
    g_edot = gate_value(var_obj, "L_edot", e_rms)
    g_u = gate_value(var_obj, "L_u", e_rms)

    r_e = feature_reward(w["L_e"]["weight"], w["L_e"]["scaled"], L_e, g_e)
    r_edot = feature_reward(w["L_edot"]["weight"], w["L_edot"]["scaled"], L_edot, g_edot)
    r_u = feature_reward(w["L_u"]["weight"], w["L_u"]["scaled"], L_u, g_u)

    band_bonus = np.where(
        e_rms <= proposal["extra_rewards"]["bandwidth_bonus"]["ranges"][var_obj],
        proposal["extra_rewards"]["bandwidth_bonus"]["per_step_band_bonus"],
        0.0,
    )
    return r_e + r_edot + r_u + band_bonus

def save_reward_map_error_vs_u(var_obj, fixed_edot=0.20):
    e_vals = np.linspace(0.0, 1.0, 400)
    u_vals = np.linspace(0.0, 1.0, 400)
    E, U = np.meshgrid(e_vals, u_vals)
    reward_map = total_local_reward(var_obj, E, np.full_like(E, fixed_edot), U)
    plt.figure(figsize=(10, 7))
    im = plt.imshow(
        reward_map, origin="lower", aspect="auto",
        extent=[e_vals.min(), e_vals.max(), u_vals.min(), u_vals.max()]
    )
    plt.colorbar(im, label="Recompensa local (principal + banda/step)")
    plt.xlabel(r"$|e|_{\mathrm{RMS}}$ normalizado")
    plt.ylabel(r"$|u|_{\mathrm{RMS}}$ normalizado")
    plt.title(f"Mapa de recompensa — {var_obj}  (|edot|_RMS = {fixed_edot:.2f})")
    plt.axvline(proposal["extra_rewards"]["bandwidth_bonus"]["ranges"][var_obj], linestyle=":")
    plt.tight_layout()
    plt.savefig(out_dir / f"reward_map_error_vs_u_{var_obj}.png", dpi=180, bbox_inches="tight")
    plt.close()

def save_reward_map_error_vs_edot(var_obj, fixed_u=0.20):
    e_vals = np.linspace(0.0, 1.0, 400)
    edot_vals = np.linspace(0.0, 1.0, 400)
    E, EDOT = np.meshgrid(e_vals, edot_vals)
    reward_map = total_local_reward(var_obj, E, EDOT, np.full_like(E, fixed_u))
    plt.figure(figsize=(10, 7))
    im = plt.imshow(
        reward_map, origin="lower", aspect="auto",
        extent=[e_vals.min(), e_vals.max(), edot_vals.min(), edot_vals.max()]
    )
    plt.colorbar(im, label="Recompensa local (principal + banda/step)")
    plt.xlabel(r"$|e|_{\mathrm{RMS}}$ normalizado")
    plt.ylabel(r"$|\\dot e|_{\mathrm{RMS}}$ normalizado")
    plt.title(f"Mapa de recompensa — {var_obj}  (|u|_RMS = {fixed_u:.2f})")
    plt.axvline(proposal["extra_rewards"]["bandwidth_bonus"]["ranges"][var_obj], linestyle=":")
    plt.tight_layout()
    plt.savefig(out_dir / f"reward_map_error_vs_edot_{var_obj}.png", dpi=180, bbox_inches="tight")
    plt.close()

def save_reward_profile(var_obj):
    e_vals = np.linspace(0.0, 1.0, 700)
    scenarios = [
        {"label": r"ideal: $|edot|=0.00,\ |u|=0.00$", "edot": 0.00, "u": 0.00},
        {"label": r"suave: $|edot|=0.10,\ |u|=0.10$", "edot": 0.10, "u": 0.10},
        {"label": r"medio: $|edot|=0.25,\ |u|=0.25$", "edot": 0.25, "u": 0.25},
        {"label": r"agresivo: $|edot|=0.50,\ |u|=0.50$", "edot": 0.50, "u": 0.50},
    ]
    plt.figure(figsize=(10, 7))
    for s in scenarios:
        reward_curve = total_local_reward(
            var_obj=var_obj,
            e_rms=e_vals,
            edot_rms=np.full_like(e_vals, s["edot"]),
            u_rms=np.full_like(e_vals, s["u"]),
        )
        plt.plot(e_vals, reward_curve, label=s["label"])
    plt.axvline(proposal["extra_rewards"]["bandwidth_bonus"]["ranges"][var_obj], linestyle=":")
    plt.xlabel(r"$|e|_{\mathrm{RMS}}$ normalizado")
    plt.ylabel("Recompensa local (principal + banda/step)")
    plt.title(f"Perfil de recompensa — {var_obj}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"reward_profile_{var_obj}.png", dpi=180, bbox_inches="tight")
    plt.close()

def save_gate_profile(var_obj):
    e_vals = np.linspace(0.0, 1.0, 700)
    plt.figure(figsize=(10, 7))
    plt.plot(e_vals, gate_value(var_obj, "L_e", e_vals), label=r"$g_e$")
    plt.plot(e_vals, gate_value(var_obj, "L_edot", e_vals), label=r"$g_{edot}$")
    plt.plot(e_vals, gate_value(var_obj, "L_u", e_vals), label=r"$g_u$")
    plt.axvline(proposal["extra_rewards"]["bandwidth_bonus"]["ranges"][var_obj], linestyle=":")
    plt.xlabel(r"$|e|_{\mathrm{RMS}}$ normalizado")
    plt.ylabel("Valor de compuerta")
    plt.title(f"Perfil de compuertas — {var_obj}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"gate_profile_{var_obj}.png", dpi=180, bbox_inches="tight")
    plt.close()

for var in ["pendulum_angle", "cart_position"]:
    save_reward_map_error_vs_u(var, fixed_edot=0.20)
    save_reward_map_error_vs_edot(var, fixed_u=0.20)
    save_reward_profile(var)
    save_gate_profile(var)