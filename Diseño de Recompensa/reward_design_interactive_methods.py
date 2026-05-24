"""
reward_design_interactive_methods.py

Interactive reward-design sandbox.

This file mirrors the project reward architecture in a compact form:
- a config dictionary with the same reward_base / reward_calculation shape
- a MetricProcessingPreview module
- PrincipalRewardPreview for weighted_exponential and nonlinear_local_cost
- CoordinationRewardPreview for task_progress_credit_assignment components
- LocalRewardExtensionPreview for marginal, movement, and synergy terms
- RewardCompositionPreview
- RewardDesignOrchestrator
- RewardDesignPlotter for notebook figures

The goal is exploration only. This module does not mutate the production YAML
config and intentionally keeps all preview logic in one file.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


VAR_OBJS = ("pendulum_angle", "cart_position")


def build_initial_reward_design_config() -> Dict[str, Any]:
    """
    Build an exploration config preserving the same shape used by config_CartPole.

    Only the reward-related blocks needed by the current notebook are active.
    Disabled sections are still present so later modules can be added naturally.
    """
    return {
        "dynamic_system": {
            "system_name": "cart_pole",
            "termination_conditions": {
                "stabilization_criteria": {
                    "pendulum_angle": [-0.02, 0.02],
                    "pendulum_velocity": [-0.1, 0.1],
                    "cart_position": [-0.1, 0.1],
                    "cart_velocity": [-0.1, 0.1],
                }
            },
            "state_normalization": {
                "enabled": True,
                "output_limits": [-1.0, 1.0],
                "ranges_params": {
                    "pendulum_angle": [-1.0472, 1.0472],
                    "pendulum_velocity": [-4.0, 4.0],
                    "cart_position": [-5.0, 5.0],
                    "cart_velocity": [-8.0, 8.0],
                },
            },
        },
        "controller_base": {
            "mixing_policy": "sum",
            "global_actuator_enabled": True,
            "global_actuator_limits": [-1.0, 1.0],
            "controllers": {
                "controller_pendulum_angle": {
                    "params": {
                        "name_objective_var": "pendulum_angle",
                        "setpoint": 0.0,
                        "error_is_setpoint_minus_pv": False,
                    }
                },
                "controller_cart_position": {
                    "params": {
                        "name_objective_var": "cart_position",
                        "setpoint": 0.0,
                        "error_is_setpoint_minus_pv": True,
                    }
                },
            },
        },
        "agent_base": {
            "agent_base_name": "pid_qlearning",
            "params": {
                "algorithm": "q-learning",
                "discount_factor": 0.95,
            },
            "agent_config": {
                "actions": {
                    "actions_space": {0: "decrease", 1: "maintain", 2: "increase"},
                    "actions_values": {
                        "mode": "universal",
                        "universal_params": {"delta_gain": 0.25},
                        "per_agent_params": {"delta_gain": {}},
                    },
                },
                "agents": {
                    "kp_pendulum_angle": {"enabled_agent": True, "min": 0.0, "max": 10.0, "state_vars": []},
                    "ki_pendulum_angle": {"enabled_agent": True, "min": 0.0, "max": 10.0, "state_vars": []},
                    "kd_pendulum_angle": {"enabled_agent": True, "min": 0.0, "max": 10.0, "state_vars": []},
                    "kp_cart_position": {"enabled_agent": True, "min": 0.0, "max": 10.0, "state_vars": []},
                    "ki_cart_position": {"enabled_agent": True, "min": 0.0, "max": 10.0, "state_vars": []},
                    "kd_cart_position": {"enabled_agent": True, "min": 0.0, "max": 10.0, "state_vars": []},
                },
            },
        },
        "reward_base": {
            "reward_base_name": "reward_base",
            "module_path": "rewards.reward_calculator_base",
            "class_name": "RewardCalculatorBase",
            "reward_config": {
                "reward_approach": "controller_reward",
                "reward_composition": {
                    "enabled": True,
                    "normalized_composition_mode": True,
                    "strict_weight_sum": True,
                    "block_weights": {
                        "principal_reward": 0.75,
                        "coordination_reward": 0.25,
                        "local_reward_extension": 0.0,
                        "extra_rewards": 0.0,
                    },
                },
                "individual_reward_params": {
                    "method": "lineal_combination",
                    "lineal_combination_params": {
                        "pendulum_angle": {},
                        "cart_position": {},
                    },
                },
            },
            "reward_calculation": {
                "metric_processing": {
                    "normalization": {
                        "enabled": True,
                        "output_limits": [0.0, 1.0],
                        "params": {
                            "pendulum_angle": {
                                "e": {"method": "rms", "range": [-0.35, 0.35]},
                                "edot": {"method": "rms", "range": [-1.20, 1.20]},
                                "I": {"method": "rms", "range": [-2.0, 2.0]},
                                "u": {"method": "rms", "range": [-1.0, 1.0]},
                            },
                            "cart_position": {
                                "e": {"method": "rms", "range": [-0.30, 0.30]},
                                "edot": {"method": "rms", "range": [-0.25, 0.25]},
                                "I": {"method": "rms", "range": [-0.7, 0.7]},
                                "u": {"method": "rms", "range": [-1.0, 1.0]},
                            },
                            "global_vars": {
                                "u_total": {"method": "mean_squared", "range": [-1.0, 1.0]},
                                "delta_u_total": {"method": "mean_squared", "range": [-0.05, 0.05]},
                            },
                        },
                    }
                },
                "principal_reward": {
                    "enabled": True,
                    "normalized_reward_mode": True,
                    "reward_principal_name": "lagrange_reward",
                    "module_path": "rewards.lagrange_reward_calculator",
                    "class_name": "LagrangeRewardCalculator",
                    "method": "nonlinear_local_cost",
                    "lineal_combination_params": {
                        "features": {
                            "L_e": {"weight": 0.35},
                            "L_edot": {"weight": 0.50},
                            "L_I": {"weight": 0.0},
                            "L_delta_u": {"weight": 0.15},
                        }
                    },
                    "weighted_exponential_params": {
                        "features": {
                            "L_e": {"weight": 0.60, "scaled": 5.0, "setpoint": 0.0},
                            "L_edot": {"weight": 0.40, "scaled": 2.5, "setpoint": 0.0},
                            "L_I": {"weight": 0.0, "scaled": 1.0, "setpoint": 0.0},
                            "L_u": {"weight": 0.05, "scaled": 1.0, "setpoint": 0.0},
                        },
                        "feature_gates": {
                            "L_e": {
                                "enabled": False,
                                "type": "exp",
                                "source": "L_e_{var_obj}",
                                "use_sqrt": False,
                                "setpoint": 0.0,
                                "per_var": {
                                    "pendulum_angle": {"scaled": 1.0, "min_factor": 1.0, "max_factor": 1.0},
                                    "cart_position": {"scaled": 1.0, "min_factor": 1.0, "max_factor": 1.0},
                                },
                            },
                            "L_edot": {
                                "enabled": True,
                                "type": "exp",
                                "source": "L_e_{var_obj}",
                                "use_sqrt": False,
                                "setpoint": 0.0,
                                "per_var": {
                                    "pendulum_angle": {"scaled": 0.50, "min_factor": 0.20, "max_factor": 1.0},
                                    "cart_position": {"scaled": 0.35, "min_factor": 0.20, "max_factor": 1.0},
                                },
                            },
                            "L_u": {
                                "enabled": True,
                                "type": "exp",
                                "source": "L_e_{var_obj}",
                                "use_sqrt": False,
                                "setpoint": 0.0,
                                "per_var": {
                                    "pendulum_angle": {"scaled": 0.30, "min_factor": 0.00, "max_factor": 1.0},
                                    "cart_position": {"scaled": 0.25, "min_factor": 0.00, "max_factor": 1.0},
                                },
                            },
                        },
                    },
                    "nonlinear_local_cost_params": {
                        "mode": "nonlinear_cost_with_progress",
                        "reward_form": "cost",
                        "component_weights": {
                            "base": {
                                "pendulum_angle": 0.80,
                                "cart_position": 0.80,
                            },
                            "progress": {
                                "pendulum_angle": 0.20,
                                "cart_position": 0.20,
                            },
                        },
                        "alpha_e": {
                            "pendulum_angle": 0.65,
                            "cart_position": 0.65,
                        },
                        "effort_weight": {
                            "pendulum_angle": 0.10,
                            "cart_position": 0.10,
                        },
                        "effort_gate_kappa": {
                            "pendulum_angle": 4.0,
                            "cart_position": 4.0,
                        },
                    },
                    "feature_budget_conservation": {
                        "enabled": False,
                        "strict_weight_sum": False,
                        "redistribution": {
                            "L_edot": {"L_e": 1.0},
                            "L_u": {"L_e": 0.5, "L_edot": 0.5},
                        },
                    },
                },
                "coordination_reward": {
                    "enabled": True,
                    "normalized_reward_mode": True,
                    "reward_name": "coordination_global_progress",
                    "module_path": "rewards.coordination_reward_handler",
                    "class_name": "CoordinationRewardHandler",
                    "reward_mode": "task_progress_credit_assignment",
                    "assign_mode": "per_var",
                    "global_weight": 12.0,
                    "potential_features": ["L_e", "L_edot"],
                    "potential_weights": {
                        "L_e": {
                            "pendulum_angle": 0.65,
                            "cart_position": 0.65,
                        },
                        "L_edot": {
                            "pendulum_angle": 0.35,
                            "cart_position": 0.35,
                        },
                    },
                    "potential_transform": {
                        "value_power": 1.0,
                        "square_terms": False,
                    },
                    "progress_positive_only": True,
                    "progress_clip_max": 0.02,
                    "contribution_error_signal": "error_{var_obj}",
                    "contribution_effort_signal": "u_alloc_{var_obj}",
                    "correction_signs": {
                        "pendulum_angle": 1.0,
                        "cart_position": 1.0,
                    },
                    "epsilon_credit": 1.0e-6,
                    "credit_reduction_mode": "average",
                    "component_weights": {
                        "global_progress": 1.0,
                        "local_progress": 0.0,
                        "local_debt": 0.0,
                    },
                    "potential_weight_normalization": {
                        "enabled": True,
                        "strict_weight_sum": False,
                        "tolerance": 1.0e-9,
                    },
                    "coordination_bonus_clip": {
                        "enabled": True,
                        "progress_max": 0.02,
                    },
                    "local_error_decay": {
                        "enabled": False,
                        "method": "hard",
                        "error_signal": "error_{var_obj}",
                        "source_reduction": "rms",
                        "hard_zero_above_threshold": False,
                        "thresholds": {
                            "pendulum_angle": 0.4,
                            "cart_position": 0.3,
                        },
                        "lambdas": {
                            "pendulum_angle": 1.0,
                            "cart_position": 1.0,
                        },
                    },
                    "local_potential_weights": {
                        "L_e": 0.6,
                        "L_edot": 0.4,
                    },
                    "local_progress": {
                        "enabled": False,
                        "weight": {
                            "pendulum_angle": 1.0,
                            "cart_position": 1.0,
                        },
                        "positive_only": True,
                        "condition_mode": "global_progress_positive",
                        "clip_enabled": True,
                        "clip_max": 0.02,
                    },
                    "local_debt": {
                        "enabled": False,
                        "persistence": 0.95,
                        "penalty_weight": {
                            "pendulum_angle": 0.25,
                            "cart_position": 0.25,
                        },
                        "clip_enabled": True,
                        "clip_max": 1.0,
                    },
                },
                "local_reward_extension": {
                    "enabled": False,
                    "normalized_reward_mode": True,
                    "reward_name": "local_reward_extension",
                    "module_path": "rewards.local_reward_extension_handler",
                    "class_name": "LocalRewardExtensionHandler",
                    "component_weights": {
                        "marginal_baseline": {
                            "pendulum_angle": 0.50,
                            "cart_position": 0.50,
                        },
                        "movement_penalty": {
                            "pendulum_angle": 0.25,
                            "cart_position": 0.25,
                        },
                        "synergy": {
                            "pendulum_angle": 0.25,
                            "cart_position": 0.25,
                        },
                    },
                    "cost": {
                        "enabled": False,
                        "alpha_e": {
                            "pendulum_angle": 0.65,
                            "cart_position": 0.65,
                        },
                        "effort_weight": {
                            "pendulum_angle": 0.25,
                            "cart_position": 0.25,
                        },
                        "effort_gate_kappa": {
                            "pendulum_angle": 4.0,
                            "cart_position": 4.0,
                        },
                    },
                    "marginal_baseline": {
                        "enabled": False,
                        "mode": "immediate_delta",
                        "beta_immediate": {
                            "pendulum_angle": 0.05,
                            "cart_position": 0.05,
                        },
                        "deterioration_weight_immediate": {
                            "pendulum_angle": 0.05,
                            "cart_position": 0.05,
                        },
                        "beta_smooth": {
                            "pendulum_angle": 0.05,
                            "cart_position": 0.05,
                        },
                        "deterioration_weight_smooth": {
                            "pendulum_angle": 0.05,
                            "cart_position": 0.05,
                        },
                        "baseline_tau": {
                            "pendulum_angle": 0.10,
                            "cart_position": 0.10,
                        },
                    },
                    "movement_penalty": {
                        "enabled": False,
                        "penalty_weight": {
                            "pendulum_angle": 0.01,
                            "cart_position": 0.01,
                        },
                    },
                    "synergy": {
                        "enabled": False,
                        "weight": {
                            "pendulum_angle": 0.05,
                            "cart_position": 0.05,
                        },
                        "error_signal": "error_{var_obj}",
                        "derivative_error_signal": "derivative_error_{var_obj}",
                        "effort_signal": "u_alloc_{var_obj}",
                        "correction_signs": {
                            "pendulum_angle": 1.0,
                            "cart_position": 1.0,
                        },
                        "gamma_e": {
                            "pendulum_angle": 1.0,
                            "cart_position": 1.0,
                        },
                        "gamma_edot": {
                            "pendulum_angle": 1.0,
                            "cart_position": 1.0,
                        },
                        "gamma_u": {
                            "pendulum_angle": 1.0,
                            "cart_position": 1.0,
                        },
                    },
                },
                "extra_rewards": {
                    "enabled": False,
                    "normalized_reward_mode": False,
                },
            },
        },
    }


def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively update a copy of base."""
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def clip01(value: Any) -> Any:
    return np.clip(value, 0.0, 1.0)


def resolve_var_value(value_config: Any, var_obj: str) -> float:
    if isinstance(value_config, Mapping):
        return float(value_config[var_obj])
    return float(value_config)


class MetricProcessingPreview:
    """Preview of metrics.metric_processing normalization behavior."""

    FEATURE_TO_SIGNAL = {
        "e": "error",
        "edot": "derivative_error",
        "I": "integral_error",
        "u": "u_alloc",
        "delta_u": "delta_u_alloc",
    }

    FEATURE_TO_REWARD = {
        "e": "L_e",
        "edot": "L_edot",
        "I": "L_I",
        "u": "L_u",
        "delta_u": "L_delta_u",
    }

    def __init__(self, config_main: Mapping[str, Any]):
        self.config_main = config_main
        self.config = config_main["reward_base"]["reward_calculation"]["metric_processing"]
        self.normalization = self.config["normalization"]

    def normalization_param(self, var_obj: str, feature_key: str) -> Dict[str, Any]:
        return self.normalization["params"][var_obj][feature_key]

    def declared_span(self, var_obj: str, feature_key: str) -> float:
        cfg = self.normalization_param(var_obj, feature_key)
        return max(abs(float(cfg["range"][0])), abs(float(cfg["range"][1])))

    def stability_threshold_normalized(self, var_obj: str, feature_key: str) -> float:
        criteria = self.config_main["dynamic_system"]["termination_conditions"]["stabilization_criteria"]
        ranges = self.config_main["dynamic_system"]["state_normalization"]["ranges_params"]
        if feature_key == "e":
            physical_key = var_obj
        elif feature_key == "edot" and var_obj == "pendulum_angle":
            physical_key = "pendulum_velocity"
        elif feature_key == "edot" and var_obj == "cart_position":
            physical_key = "cart_velocity"
        else:
            return 0.0
        threshold_raw = max(abs(criteria[physical_key][0]), abs(criteria[physical_key][1]))
        state_span = max(abs(ranges[physical_key][0]), abs(ranges[physical_key][1]))
        return threshold_raw / state_span if state_span > 0.0 else 0.0

    def aggregate_with_method(self, values: Iterable[float], method: str) -> float:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return 0.0
        if method == "mean":
            return float(np.mean(arr))
        if method == "mean_squared":
            return float(np.mean(arr ** 2))
        if method == "rms":
            return float(np.sqrt(np.mean(arr ** 2)))
        if method == "abs":
            return float(np.mean(np.abs(arr)))
        if method == "accum":
            return float(np.sum(arr))
        if method == "proportion":
            return float(np.mean(arr > 0))
        if method == "keep_last":
            return float(arr[-1])
        return float(np.mean(arr ** 2))

    def normalize_value(self, value: Any, value_range: Iterable[float], method: str) -> Any:
        range_min, range_max = [float(x) for x in value_range]
        out_min, out_max = [float(x) for x in self.normalization["output_limits"]]
        if method == "mean_squared":
            range_span = max(abs(range_min), abs(range_max)) ** 2
        elif method in {"rms", "abs"}:
            range_span = max(abs(range_min), abs(range_max))
        elif method == "proportion":
            range_span = 1.0
        else:
            range_span = range_max - range_min if range_max != range_min else 1.0
            value = np.abs(value - range_min)
        normalized_01 = clip01(np.abs(value) / range_span) if range_span > 0.0 else 0.0
        return out_min + normalized_01 * (out_max - out_min)

    def normalize_feature(self, value: Any, var_obj: str, feature_key: str) -> Any:
        cfg = self.normalization_param(var_obj, feature_key)
        return self.normalize_value(value, cfg["range"], cfg["method"])


class PrincipalRewardPreview:
    """Preview of LagrangeRewardCalculator nonlinear_local_cost."""

    def __init__(self, config_main: Mapping[str, Any]):
        self.config_main = config_main
        self.config = config_main["reward_base"]["reward_calculation"]["principal_reward"]
        self.normalized_reward_mode = bool(self.config.get("normalized_reward_mode", False))
        self.params = self.config["nonlinear_local_cost_params"]
        self.weighted_params = self.config["weighted_exponential_params"]

    def local_cost_input(self, value: Any) -> Any:
        if self.normalized_reward_mode:
            return clip01(np.abs(value))
        return np.abs(value)

    def feature_weight_sum(self, features: Optional[Mapping[str, Any]] = None) -> float:
        features = features or self.weighted_params["features"]
        total = 0.0
        for cfg in features.values():
            total += max(0.0, float(cfg.get("weight", 0.0)))
        return total if total > 0.0 else 1.0

    def effective_feature_weight(self, raw_weight: float, feature_sum: Optional[float] = None) -> float:
        if not self.normalized_reward_mode:
            return raw_weight
        normalizer = self.feature_weight_sum() if feature_sum is None else feature_sum
        return max(0.0, raw_weight) / normalizer

    def exponential_cost_01(self, value: Any, scaled: float, setpoint: float = 0.0) -> Any:
        if scaled < 0.0:
            raise ValueError(f"Exponential scale must be non-negative, got {scaled}")
        if scaled == 0.0:
            return np.zeros_like(value, dtype=float)
        value_01 = clip01(np.abs(value))
        raw_cost = 1.0 - np.exp(-scaled * ((value_01 - setpoint) ** 2))
        bound_0 = 1.0 - math.exp(-scaled * ((0.0 - setpoint) ** 2))
        bound_1 = 1.0 - math.exp(-scaled * ((1.0 - setpoint) ** 2))
        raw_bound = max(bound_0, bound_1, 1.0e-12)
        return clip01(raw_cost / raw_bound)

    def gate_factor(self, source_value: Any, gate_cfg: Optional[Mapping[str, Any]]) -> Any:
        if not gate_cfg or not gate_cfg.get("enabled", False):
            return np.ones_like(source_value, dtype=float)
        value = np.abs(source_value) if gate_cfg.get("use_abs", True) else source_value
        if gate_cfg.get("use_sqrt", False):
            value = np.sqrt(np.maximum(value, 0.0))
        gain = float(gate_cfg.get("strength", gate_cfg.get("scaled", 1.0)))
        setpoint = float(gate_cfg.get("x_sp", gate_cfg.get("setpoint", 0.0)))
        if gain < 0.0:
            raise ValueError(f"Gate gain must be non-negative, got {gain}")
        gate_type = gate_cfg.get("type", "exp")
        if gate_type == "exp":
            factor = np.exp(-(((value - setpoint) / gain) ** 2)) if gain > 0.0 else np.zeros_like(value, dtype=float)
        elif gate_type == "tanh":
            factor = np.tanh(gain * np.abs(value - setpoint))
        else:
            raise ValueError(f"Unsupported principal gate type: {gate_type}")
        if gate_cfg.get("invert", False):
            factor = 1.0 - factor
        min_factor = float(gate_cfg.get("min_factor", 0.0))
        max_factor = float(gate_cfg.get("max_factor", 1.0))
        return np.clip(factor, min_factor, max_factor)

    def resolve_weighted_gate_config(
        self,
        feature_name: str,
        var_obj: str,
        *,
        enabled: Optional[bool] = None,
        scaled: Optional[float] = None,
        min_factor: Optional[float] = None,
        max_factor: Optional[float] = None,
        invert: Optional[bool] = None,
        gate_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        gates = self.weighted_params.get("feature_gates", {})
        gate_cfg = deepcopy(gates.get(feature_name, {}))
        if not gate_cfg:
            return None
        per_var = gate_cfg.pop("per_var", {})
        merged = dict(gate_cfg)
        merged.update(per_var.get(var_obj, {}) or {})
        if enabled is not None:
            merged["enabled"] = bool(enabled)
        if scaled is not None:
            merged["scaled"] = float(scaled)
        if min_factor is not None:
            merged["min_factor"] = float(min_factor)
        if max_factor is not None:
            merged["max_factor"] = float(max_factor)
        if invert is not None:
            merged["invert"] = bool(invert)
        if gate_type is not None:
            merged["type"] = gate_type
        merged.setdefault("source", "L_e_{var_obj}")
        merged["source"] = merged["source"].format(var_obj=var_obj)
        merged.setdefault("type", "exp")
        merged.setdefault("scaled", merged.get("strength", 1.0))
        merged.setdefault("setpoint", merged.get("x_sp", 0.0))
        merged.setdefault("use_sqrt", False)
        merged.setdefault("use_abs", True)
        merged.setdefault("invert", False)
        merged.setdefault("min_factor", 0.0)
        merged.setdefault("max_factor", 1.0)
        return merged

    def weighted_exponential_terms(
        self,
        L_e: Any,
        L_edot: Any,
        L_u: Any,
        var_obj: str,
        *,
        w_e: Optional[float] = None,
        w_edot: Optional[float] = None,
        w_u: Optional[float] = None,
        scale_e: Optional[float] = None,
        scale_edot: Optional[float] = None,
        scale_u: Optional[float] = None,
        gate_edot_enabled: Optional[bool] = None,
        gate_edot_scaled: Optional[float] = None,
        gate_edot_min: Optional[float] = None,
        gate_u_enabled: Optional[bool] = None,
        gate_u_scaled: Optional[float] = None,
        gate_u_min: Optional[float] = None,
        budget_conservation: bool = False,
    ) -> Dict[str, Any]:
        features = deepcopy(self.weighted_params["features"])
        if w_e is not None:
            features["L_e"]["weight"] = float(w_e)
        if w_edot is not None:
            features["L_edot"]["weight"] = float(w_edot)
        if w_u is not None:
            features["L_u"]["weight"] = float(w_u)
        if scale_e is not None:
            features["L_e"]["scaled"] = float(scale_e)
        if scale_edot is not None:
            features["L_edot"]["scaled"] = float(scale_edot)
        if scale_u is not None:
            features["L_u"]["scaled"] = float(scale_u)

        values = {"L_e": L_e, "L_edot": L_edot, "L_u": L_u}
        feature_sum = self.feature_weight_sum(features)
        base_weights = {}
        costs = {}
        gates = {}
        for feature_name in ("L_e", "L_edot", "L_u"):
            cfg = features[feature_name]
            base_weights[feature_name] = self.effective_feature_weight(float(cfg["weight"]), feature_sum)
            costs[feature_name] = self.exponential_cost_01(
                values[feature_name],
                float(cfg.get("scaled", 1.0)),
                float(cfg.get("setpoint", 0.0)),
            )

        gates["L_e"] = np.ones_like(L_e, dtype=float)
        gates["L_edot"] = self.gate_factor(
            L_e,
            self.resolve_weighted_gate_config(
                "L_edot",
                var_obj,
                enabled=gate_edot_enabled,
                scaled=gate_edot_scaled,
                min_factor=gate_edot_min,
            ),
        )
        gates["L_u"] = self.gate_factor(
            L_e,
            self.resolve_weighted_gate_config(
                "L_u",
                var_obj,
                enabled=gate_u_enabled,
                scaled=gate_u_scaled,
                min_factor=gate_u_min,
            ),
        )

        effective_weights = {
            feature_name: base_weights[feature_name] * gates[feature_name]
            for feature_name in ("L_e", "L_edot", "L_u")
        }
        residual_weights = {
            feature_name: base_weights[feature_name] - effective_weights[feature_name]
            for feature_name in ("L_e", "L_edot", "L_u")
        }

        if budget_conservation:
            effective_weights["L_e"] = (
                effective_weights["L_e"]
                + residual_weights["L_edot"]
                + 0.5 * residual_weights["L_u"]
            )
            effective_weights["L_edot"] = effective_weights["L_edot"] + 0.5 * residual_weights["L_u"]

        loop_cost = (
            effective_weights["L_e"] * costs["L_e"]
            + effective_weights["L_edot"] * costs["L_edot"]
            + effective_weights["L_u"] * costs["L_u"]
        )
        return {
            "cost": loop_cost,
            "reward": -loop_cost,
            "base_weights": base_weights,
            "effective_weights": effective_weights,
            "residual_weights": residual_weights,
            "costs": costs,
            "gates": gates,
        }

    def weighted_exponential_surface(
        self,
        var_obj: str,
        L_u_fixed: float = 0.10,
        n: int = 160,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        L_e = np.linspace(0.0, 1.0, n)
        L_edot = np.linspace(0.0, 1.0, n)
        E_grid, Edot_grid = np.meshgrid(L_e, L_edot)
        terms = self.weighted_exponential_terms(
            E_grid,
            Edot_grid,
            np.full_like(E_grid, L_u_fixed),
            var_obj,
            **kwargs,
        )
        return E_grid, Edot_grid, terms

    def local_error_potential(self, L_e: Any, L_edot: Any, var_obj: str) -> Any:
        alpha_e = resolve_var_value(self.params["alpha_e"], var_obj)
        return alpha_e * L_e + (1.0 - alpha_e) * L_edot

    def effort_gate(self, E: Any, var_obj: str) -> Any:
        kappa = resolve_var_value(self.params["effort_gate_kappa"], var_obj)
        return np.exp(-kappa * E)

    def nonlinear_terms(self, L_e: Any, L_edot: Any, L_u: Any, var_obj: str) -> Dict[str, Any]:
        alpha_e = resolve_var_value(self.params["alpha_e"], var_obj)
        effort_weight = resolve_var_value(self.params["effort_weight"], var_obj)
        L_e = self.local_cost_input(L_e)
        L_edot = self.local_cost_input(L_edot)
        L_u = self.local_cost_input(L_u)
        E = alpha_e * L_e + (1.0 - alpha_e) * L_edot
        g_u = self.effort_gate(E, var_obj)
        denominator = 1.0 + effort_weight
        J = (E + effort_weight * g_u * L_u) / denominator
        return {
            "L_e": L_e,
            "L_edot": L_edot,
            "L_u": L_u,
            "E": E,
            "g_u": g_u,
            "J": J,
            "effective_weight_L_e": alpha_e / denominator,
            "effective_weight_L_edot": (1.0 - alpha_e) / denominator,
            "effective_weight_L_u": effort_weight * g_u / denominator,
            "effort_residual_weight": effort_weight * (1.0 - g_u) / denominator,
        }

    def component_weights(self, var_obj: str) -> Tuple[float, float]:
        mode = self.params["mode"]
        base_weight = resolve_var_value(self.params["component_weights"]["base"], var_obj)
        progress_weight = resolve_var_value(self.params["component_weights"]["progress"], var_obj)
        if mode != "nonlinear_cost_with_progress":
            progress_weight = 0.0
        if self.normalized_reward_mode:
            weight_sum = base_weight + progress_weight
            if weight_sum <= 0.0:
                raise ValueError(f"Component weights must be positive for {var_obj}")
            base_weight /= weight_sum
            progress_weight /= weight_sum
        return base_weight, progress_weight

    def controller_reward(
        self,
        L_e: float,
        L_edot: float,
        L_u: float,
        var_obj: str,
        previous_J: Optional[float] = None,
    ) -> Dict[str, float]:
        terms = self.nonlinear_terms(L_e, L_edot, L_u, var_obj)
        base_weight, progress_weight = self.component_weights(var_obj)
        cost_delta = 0.0 if previous_J is None else previous_J - float(terms["J"])
        progress_signal = max(0.0, cost_delta)
        deterioration_signal = max(0.0, -cost_delta)
        if self.normalized_reward_mode:
            progress_signal = float(clip01(progress_signal))
            deterioration_signal = float(clip01(deterioration_signal))
        score_reward = 1.0 - float(terms["J"])
        cost_reward = -float(terms["J"])
        base_reward = score_reward if self.params["reward_form"] == "score" else cost_reward
        if self.params["reward_form"] == "score":
            progress_component = progress_weight * progress_signal
        else:
            progress_component = -progress_weight * deterioration_signal
        reward = base_weight * base_reward + progress_component
        return {
            "reward": reward,
            "J": float(terms["J"]),
            "E": float(terms["E"]),
            "g_u": float(terms["g_u"]),
            "progress_signal": progress_signal,
            "deterioration_signal": deterioration_signal,
            "base_weight": base_weight,
            "progress_weight": progress_weight,
        }

    def surface(
        self,
        var_obj: str,
        L_u_fixed: float = 0.10,
        n: int = 160,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        L_e = np.linspace(0.0, 1.0, n)
        L_edot = np.linspace(0.0, 1.0, n)
        E_grid, Edot_grid = np.meshgrid(L_e, L_edot)
        J_grid = self.nonlinear_terms(E_grid, Edot_grid, L_u_fixed, var_obj)["J"]
        return E_grid, Edot_grid, J_grid


class CoordinationRewardPreview:
    """Preview of CoordinationRewardHandler task progress and credit logic."""

    def __init__(self, config_main: Mapping[str, Any]):
        self.config_main = config_main
        self.config = config_main["reward_base"]["reward_calculation"]["coordination_reward"]
        self.normalized_reward_mode = bool(self.config["normalized_reward_mode"])

    def potential_weight_sum_effective(self) -> float:
        total = 0.0
        for var_obj in VAR_OBJS:
            for feature_name in self.config["potential_features"]:
                total += self.effective_potential_weight(feature_name, var_obj)
        return total if total > 0.0 else 1.0

    def task_progress_bound(self) -> float:
        return self.potential_weight_sum_effective()

    @staticmethod
    def normalize_by_bound(value: Any, bound: float) -> Any:
        if bound <= 0.0:
            raise ValueError("Normalization bound must be positive")
        return clip01(value / bound)

    def transform_potential_value(self, value: Any) -> Any:
        transform = self.config["potential_transform"]
        power = 2.0 if transform.get("square_terms", False) else float(transform["value_power"])
        if power == 1.0:
            return value
        return np.abs(value) ** power

    def potential_weight_sum_by_var(self, var_obj: str) -> float:
        total = 0.0
        for feature_name in self.config["potential_features"]:
            total += float(self.config["potential_weights"][feature_name][var_obj])
        return total

    def effective_potential_weight(self, feature_name: str, var_obj: str) -> float:
        weight = float(self.config["potential_weights"][feature_name][var_obj])
        norm_cfg = self.config["potential_weight_normalization"]
        if norm_cfg["enabled"]:
            total = self.potential_weight_sum_by_var(var_obj)
            return weight / total if total > 0.0 else 0.0
        return weight

    def local_potential(self, L_e: Any, L_edot: Any, var_obj: str) -> Any:
        values = {"L_e": L_e, "L_edot": L_edot}
        potential = 0.0
        for feature_name in self.config["potential_features"]:
            weight = self.effective_potential_weight(feature_name, var_obj)
            potential += weight * self.transform_potential_value(values[feature_name])
        return potential

    def task_potential(
        self,
        L_e_pendulum: Any,
        L_edot_pendulum: Any,
        L_e_cart: Any,
        L_edot_cart: Any,
    ) -> Any:
        return (
            self.local_potential(L_e_pendulum, L_edot_pendulum, "pendulum_angle")
            + self.local_potential(L_e_cart, L_edot_cart, "cart_position")
        )

    def rewardable_progress(self, task_progress_delta: Any) -> Any:
        progress = np.maximum(0.0, task_progress_delta) if self.config["progress_positive_only"] else task_progress_delta
        if self.normalized_reward_mode:
            return np.minimum(progress, self.task_progress_bound())
        clip_cfg = self.config["coordination_bonus_clip"]
        if clip_cfg.get("enabled", False):
            progress = np.minimum(progress, float(clip_cfg["progress_max"]))
        elif float(self.config.get("progress_clip_max", 0.0)) > 0.0:
            progress = np.minimum(progress, float(self.config["progress_clip_max"]))
        return progress

    def task_progress_01(self, task_progress_rewardable: Any) -> Any:
        return self.normalize_by_bound(task_progress_rewardable, self.task_progress_bound())

    def coordination_bonus_total(self, task_progress_delta: Any) -> Any:
        rewardable = self.rewardable_progress(task_progress_delta)
        progress_01 = self.task_progress_01(rewardable)
        if self.normalized_reward_mode:
            return progress_01
        return float(self.config["global_weight"]) * rewardable

    def credit_share(self, credit_self: Any, credit_other: Any) -> Any:
        eps = float(self.config["epsilon_credit"])
        return credit_self / (eps + credit_self + credit_other)

    def normalized_component_weights(
        self,
        global_progress: float,
        local_progress: float,
        local_debt: float,
    ) -> Dict[str, float]:
        raw = {
            "global_progress": max(0.0, float(global_progress)),
            "local_progress": max(0.0, float(local_progress)),
            "local_debt": max(0.0, float(local_debt)),
        }
        total = sum(raw.values())
        if total <= 0.0:
            return {"global_progress": 1.0, "local_progress": 0.0, "local_debt": 0.0}
        return {name: value / total for name, value in raw.items()}

    def normalized_var_share(self, pendulum_weight: float, cart_weight: float) -> Dict[str, float]:
        p = max(0.0, float(pendulum_weight))
        c = max(0.0, float(cart_weight))
        total = p + c
        if total <= 0.0:
            return {"pendulum_angle": 0.5, "cart_position": 0.5}
        return {"pendulum_angle": p / total, "cart_position": c / total}

    def error_decay(self, error_abs: Any, threshold: float, lambda_v: float, method: str, hard_zero: bool = False) -> Any:
        threshold = max(float(threshold), 1.0e-12)
        ratio = np.abs(error_abs) / threshold
        if method == "linear":
            factor = np.maximum(0.0, 1.0 - ratio)
        elif method == "exponential":
            factor = np.exp(-float(lambda_v) * (ratio ** 2))
        elif method == "hard":
            factor = np.where(ratio <= 1.0, 1.0, 0.0)
        else:
            raise ValueError(f"Unsupported coordination error decay method: {method}")
        if hard_zero:
            factor = np.where(ratio > 1.0, 0.0, factor)
        return clip01(factor)

    def local_progress_condition(self, delta_local_p: Any, delta_local_c: Any, delta_task: Any, mode: str) -> Any:
        if mode == "always":
            return np.ones_like(delta_task, dtype=float)
        if mode == "global_progress_positive":
            return np.where(delta_task > 0.0, 1.0, 0.0)
        if mode == "all_local_progress_positive":
            return np.where((delta_local_p > 0.0) & (delta_local_c > 0.0), 1.0, 0.0)
        if mode == "global_and_all_local_positive":
            return np.where((delta_task > 0.0) & (delta_local_p > 0.0) & (delta_local_c > 0.0), 1.0, 0.0)
        raise ValueError(f"Unsupported local progress condition mode: {mode}")

    def local_progress_component_01(
        self,
        delta_local: Any,
        local_potential_bound: float,
        var_share: float,
        condition_factor: Any,
        positive_only: bool = True,
    ) -> Any:
        progress = np.maximum(0.0, delta_local) if positive_only else delta_local
        progress = np.minimum(progress, local_potential_bound)
        progress_01 = self.normalize_by_bound(progress, local_potential_bound)
        return var_share * progress_01 * condition_factor

    def debt_component_01(
        self,
        local_potential: Any,
        persistence: float,
        clip_max: float,
        var_share: float,
        previous_debt: float = 0.0,
    ) -> Any:
        debt = float(persistence) * previous_debt + np.maximum(0.0, local_potential)
        if clip_max > 0.0:
            debt = np.minimum(debt, clip_max)
            bound = clip_max
        elif persistence < 1.0:
            bound = 1.0 / (1.0 - persistence)
        else:
            bound = 1.0
        return var_share * self.normalize_by_bound(debt, bound)

    def coordination_module_reward(
        self,
        global_component_01: Any,
        local_progress_component_01: Any,
        debt_component_01: Any,
        component_weights: Mapping[str, float],
    ) -> Any:
        return (
            component_weights["global_progress"] * global_component_01
            + component_weights["local_progress"] * local_progress_component_01
            - component_weights["local_debt"] * debt_component_01
        )


class LocalRewardExtensionPreview:
    """Preview of LocalRewardExtensionHandler local components."""

    def __init__(self, config_main: Mapping[str, Any]):
        self.config_main = config_main
        self.config = config_main["reward_base"]["reward_calculation"]["local_reward_extension"]
        self.normalized_reward_mode = bool(self.config["normalized_reward_mode"])
        self.cost_config = self.config["cost"]
        self.marginal_config = self.config["marginal_baseline"]
        self.movement_config = self.config["movement_penalty"]
        self.synergy_config = self.config["synergy"]

    def local_cost_input(self, value: Any) -> Any:
        if self.normalized_reward_mode:
            return clip01(np.abs(value))
        return np.abs(value)

    def cost_terms(
        self,
        L_e: Any,
        L_edot: Any,
        L_u: Any,
        var_obj: str,
        *,
        alpha_e: Optional[float] = None,
        effort_weight: Optional[float] = None,
        effort_gate_kappa: Optional[float] = None,
    ) -> Dict[str, Any]:
        alpha = resolve_var_value(self.cost_config["alpha_e"], var_obj) if alpha_e is None else float(alpha_e)
        beta = resolve_var_value(self.cost_config["effort_weight"], var_obj) if effort_weight is None else float(effort_weight)
        kappa = resolve_var_value(self.cost_config["effort_gate_kappa"], var_obj) if effort_gate_kappa is None else float(effort_gate_kappa)
        L_e = self.local_cost_input(L_e)
        L_edot = self.local_cost_input(L_edot)
        L_u = self.local_cost_input(L_u)
        E = alpha * L_e + (1.0 - alpha) * L_edot
        g_u = np.exp(-kappa * E)
        J = (E + beta * g_u * L_u) / (1.0 + beta)
        return {"L_e": L_e, "L_edot": L_edot, "L_u": L_u, "E": E, "g_u": g_u, "J": J}

    @staticmethod
    def clip_signed_01(value: Any) -> Any:
        return np.clip(value, -1.0, 1.0)

    def marginal_signal(
        self,
        current_cost: Any,
        previous_cost: Any,
        baseline_cost: Any,
        *,
        mode: str,
        tau: float,
        normalized: bool = True,
        beta: float = 0.05,
        deterioration_weight: float = 0.05,
    ) -> Dict[str, Any]:
        delta_j = previous_cost - current_cost
        advantage = baseline_cost - current_cost
        next_baseline = (1.0 - tau) * baseline_cost + tau * current_cost
        move_reference = advantage if mode == "smoothed_baseline" else delta_j
        raw_signal = advantage if mode == "smoothed_baseline" else delta_j
        if normalized:
            reward = self.clip_signed_01(raw_signal)
        else:
            reward = beta * np.maximum(0.0, raw_signal) - deterioration_weight * np.maximum(0.0, -raw_signal)
        return {
            "delta_j": delta_j,
            "advantage": advantage,
            "next_baseline": next_baseline,
            "move_reference": move_reference,
            "reward": reward,
        }

    @staticmethod
    def movement_penalty(move_reference: Any, moved: Any) -> Any:
        return np.where((moved > 0.0) & (move_reference <= 0.0), 1.0, 0.0)

    def synergy_terms(
        self,
        error: Any,
        derivative_error: Any,
        effort: Any,
        var_obj: str,
        *,
        gamma_e: Optional[float] = None,
        gamma_edot: Optional[float] = None,
        gamma_u: Optional[float] = None,
        correction_sign: Optional[float] = None,
    ) -> Dict[str, Any]:
        ge = resolve_var_value(self.synergy_config["gamma_e"], var_obj) if gamma_e is None else float(gamma_e)
        gd = resolve_var_value(self.synergy_config["gamma_edot"], var_obj) if gamma_edot is None else float(gamma_edot)
        gu = resolve_var_value(self.synergy_config["gamma_u"], var_obj) if gamma_u is None else float(gamma_u)
        sign = resolve_var_value(self.synergy_config["correction_signs"], var_obj) if correction_sign is None else float(correction_sign)
        error_gate = np.tanh(ge * np.abs(error))
        sedot = np.tanh(gd * (-error * derivative_error))
        seu = np.tanh(gu * sign * error * effort)
        synergy = error_gate * sedot * seu
        return {"error_gate": error_gate, "sedot": sedot, "seu": seu, "synergy": synergy, "reward": synergy}

    def normalized_component_weights(
        self,
        marginal_baseline: float,
        movement_penalty: float,
        synergy: float,
        *,
        marginal_enabled: bool = True,
        movement_enabled: bool = True,
        synergy_enabled: bool = True,
    ) -> Dict[str, float]:
        raw = {
            "marginal_baseline": max(0.0, float(marginal_baseline)) if marginal_enabled else 0.0,
            "movement_penalty": max(0.0, float(movement_penalty)) if movement_enabled else 0.0,
            "synergy": max(0.0, float(synergy)) if synergy_enabled else 0.0,
        }
        total = sum(raw.values())
        if total <= 0.0:
            return {"marginal_baseline": 1.0, "movement_penalty": 0.0, "synergy": 0.0}
        return {name: value / total for name, value in raw.items()}


class RewardCompositionPreview:
    """Preview of RewardCalculatorBase normalized block composition."""

    def __init__(self, config_main: Mapping[str, Any]):
        self.config_main = config_main
        self.config = config_main["reward_base"]["reward_config"]["reward_composition"]

    def block_weights(self) -> Dict[str, float]:
        raw = {
            name: float(value)
            for name, value in self.config["block_weights"].items()
        }
        if self.config["strict_weight_sum"]:
            total = sum(raw.values())
            if abs(total - 1.0) > 1.0e-9:
                raise ValueError(f"Reward block weights must sum 1.0, got {total}")
            return raw
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()} if total > 0.0 else raw

    def compose_loop_reward(
        self,
        principal_reward: Any,
        coordination_reward: Any,
        extra_reward: Any = 0.0,
        local_extension_reward: Any = 0.0,
    ) -> Any:
        weights = self.block_weights()
        return (
            weights["principal_reward"] * principal_reward
            + weights["coordination_reward"] * coordination_reward
            + weights["extra_rewards"] * extra_reward
            + weights["local_reward_extension"] * local_extension_reward
        )


class RewardDesignOrchestrator:
    """
    Single-file orchestrator for interactive parameter exploration.

    Mirrors RewardCalculatorBase at exploration scale: metric processing,
    principal reward, coordination reward, then composition.
    """

    def __init__(self, config_main: Optional[Mapping[str, Any]] = None):
        self.config_main = deepcopy(config_main) if config_main is not None else build_initial_reward_design_config()
        self.metric_processing = MetricProcessingPreview(self.config_main)
        self.principal_reward = PrincipalRewardPreview(self.config_main)
        self.coordination_reward = CoordinationRewardPreview(self.config_main)
        self.local_reward_extension = LocalRewardExtensionPreview(self.config_main)
        self.reward_composition = RewardCompositionPreview(self.config_main)

    def clone_with_overrides(self, overrides: Mapping[str, Any]) -> "RewardDesignOrchestrator":
        return RewardDesignOrchestrator(deep_update(self.config_main, overrides))

    def with_slider_values(
        self,
        *,
        alpha_e: Optional[float] = None,
        effort_weight: Optional[float] = None,
        effort_gate_kappa: Optional[float] = None,
        principal_weight: Optional[float] = None,
        coordination_weight: Optional[float] = None,
        progress_clip_max: Optional[float] = None,
        potential_weight_e: Optional[float] = None,
        potential_weight_edot: Optional[float] = None,
        value_power: Optional[float] = None,
        square_terms: Optional[bool] = None,
        pendulum_e_span: Optional[float] = None,
        pendulum_edot_span: Optional[float] = None,
        cart_e_span: Optional[float] = None,
        cart_edot_span: Optional[float] = None,
    ) -> "RewardDesignOrchestrator":
        config = deepcopy(self.config_main)
        nonlinear = config["reward_base"]["reward_calculation"]["principal_reward"]["nonlinear_local_cost_params"]
        coordination = config["reward_base"]["reward_calculation"]["coordination_reward"]
        composition = config["reward_base"]["reward_config"]["reward_composition"]["block_weights"]
        norm = config["reward_base"]["reward_calculation"]["metric_processing"]["normalization"]["params"]

        if alpha_e is not None:
            nonlinear["alpha_e"] = {var_obj: float(alpha_e) for var_obj in VAR_OBJS}
        if effort_weight is not None:
            nonlinear["effort_weight"] = {var_obj: float(effort_weight) for var_obj in VAR_OBJS}
        if effort_gate_kappa is not None:
            nonlinear["effort_gate_kappa"] = {var_obj: float(effort_gate_kappa) for var_obj in VAR_OBJS}
        if principal_weight is not None or coordination_weight is not None:
            p = composition["principal_reward"] if principal_weight is None else float(principal_weight)
            c = composition["coordination_reward"] if coordination_weight is None else float(coordination_weight)
            p = max(0.0, p)
            c = max(0.0, c)
            total = p + c
            if total <= 0.0:
                p, c = 1.0, 0.0
            else:
                p, c = p / total, c / total
            composition["principal_reward"] = p
            composition["coordination_reward"] = c
            composition["extra_rewards"] = 0.0
            composition["local_reward_extension"] = 0.0
        if progress_clip_max is not None:
            coordination["progress_clip_max"] = float(progress_clip_max)
            coordination["coordination_bonus_clip"]["progress_max"] = float(progress_clip_max)
        if potential_weight_e is not None:
            coordination["potential_weights"]["L_e"] = {var_obj: float(potential_weight_e) for var_obj in VAR_OBJS}
        if potential_weight_edot is not None:
            coordination["potential_weights"]["L_edot"] = {var_obj: float(potential_weight_edot) for var_obj in VAR_OBJS}
        if value_power is not None:
            coordination["potential_transform"]["value_power"] = float(value_power)
        if square_terms is not None:
            coordination["potential_transform"]["square_terms"] = bool(square_terms)
        if pendulum_e_span is not None:
            norm["pendulum_angle"]["e"]["range"] = [-float(pendulum_e_span), float(pendulum_e_span)]
        if pendulum_edot_span is not None:
            norm["pendulum_angle"]["edot"]["range"] = [-float(pendulum_edot_span), float(pendulum_edot_span)]
        if cart_e_span is not None:
            norm["cart_position"]["e"]["range"] = [-float(cart_e_span), float(cart_e_span)]
        if cart_edot_span is not None:
            norm["cart_position"]["edot"]["range"] = [-float(cart_edot_span), float(cart_edot_span)]

        return RewardDesignOrchestrator(config)


class RewardDesignPlotter:
    """Matplotlib-based views used by the notebook widgets."""

    def __init__(self, orchestrator: Optional[RewardDesignOrchestrator] = None):
        self.orchestrator = orchestrator or RewardDesignOrchestrator()

    @staticmethod
    def _plt():
        import matplotlib.pyplot as plt

        return plt

    def plot_normalization_ranges(
        self,
        pendulum_e_span: float = 0.35,
        pendulum_edot_span: float = 1.20,
        cart_e_span: float = 0.30,
        cart_edot_span: float = 0.25,
    ):
        orch = self.orchestrator.with_slider_values(
            pendulum_e_span=pendulum_e_span,
            pendulum_edot_span=pendulum_edot_span,
            cart_e_span=cart_e_span,
            cart_edot_span=cart_edot_span,
        )
        mp = orch.metric_processing
        plt = self._plt()
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), constrained_layout=True)
        specs = [
            ("pendulum_angle", "e", axes[0, 0], "Pendulo: L_e"),
            ("cart_position", "e", axes[0, 1], "Carro: L_e"),
            ("pendulum_angle", "edot", axes[1, 0], "Pendulo: L_edot"),
            ("cart_position", "edot", axes[1, 1], "Carro: L_edot"),
        ]
        for var_obj, feature_key, ax, title in specs:
            cfg = mp.normalization_param(var_obj, feature_key)
            span = mp.declared_span(var_obj, feature_key)
            stable = mp.stability_threshold_normalized(var_obj, feature_key)
            xs = np.linspace(0.0, span * 1.18, 400)
            ys = mp.normalize_value(xs, cfg["range"], cfg["method"])
            ax.plot(xs, ys, lw=2.8)
            ax.axvspan(0.0, stable, color="#dcfce7", alpha=0.75, label="banda estable")
            ax.axvline(span, color="#dc2626", ls="--", lw=2, label="saturacion")
            ax.axvline(stable, color="#16a34a", ls="--", lw=2)
            ax.set_title(title)
            ax.set_xlabel(f"|{feature_key}| RMS normalizado")
            ax.set_ylabel(f"L_{feature_key}")
            ax.set_ylim(0.0, 1.08)
            ax.grid(alpha=0.25)
            ax.text(0.02, 0.94, f"span={span:.3g}\\nestable={stable:.3g}", transform=ax.transAxes, va="top")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle("Normalizacion de senales para interpretar costos L_e y L_edot", y=1.04)
        return fig

    def plot_local_cost_surface(
        self,
        alpha_e: float = 0.65,
        effort_weight: float = 0.10,
        effort_gate_kappa: float = 4.0,
        L_u_fixed: float = 0.10,
        var_obj: str = "pendulum_angle",
    ):
        orch = self.orchestrator.with_slider_values(
            alpha_e=alpha_e,
            effort_weight=effort_weight,
            effort_gate_kappa=effort_gate_kappa,
        )
        pr = orch.principal_reward
        X, Y, Z = pr.surface(var_obj, L_u_fixed=L_u_fixed)
        plt = self._plt()
        fig, ax = plt.subplots(figsize=(8.8, 6.3), constrained_layout=True)
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        contours = ax.contour(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8], colors="white", linewidths=1.8)
        ax.clabel(contours, inline=True, fontsize=9, fmt="J=%.1f")
        ax.set_xlabel("L_e")
        ax.set_ylabel("L_edot")
        ax.set_title(f"Costo local J_v - {var_obj}")
        ax.scatter([0.18, 0.65, 0.12], [0.18, 0.15, 0.70], c=["white", "black", "black"], edgecolors="white")
        ax.text(0.20, 0.19, "zona deseable", color="white", weight="bold")
        ax.text(0.67, 0.16, "lejos lento", color="black")
        ax.text(0.14, 0.71, "rapido cerca", color="black")
        fig.colorbar(im, ax=ax, label="J_v")
        return fig

    def plot_weighted_gate_surface(
        self,
        w_e: float = 0.60,
        w_edot: float = 0.35,
        w_u: float = 0.05,
        scale_e: float = 5.0,
        scale_edot: float = 2.5,
        scale_u: float = 1.0,
        gate_edot_scaled: float = 0.50,
        gate_u_scaled: float = 0.30,
        gate_min: float = 0.0,
        L_u_fixed: float = 0.10,
        budget_conservation: bool = False,
        var_obj: str = "pendulum_angle",
    ):
        pr = self.orchestrator.principal_reward
        X, Y, terms = pr.weighted_exponential_surface(
            var_obj,
            L_u_fixed=L_u_fixed,
            w_e=w_e,
            w_edot=w_edot,
            w_u=w_u,
            scale_e=scale_e,
            scale_edot=scale_edot,
            scale_u=scale_u,
            gate_edot_enabled=True,
            gate_edot_scaled=gate_edot_scaled,
            gate_edot_min=gate_min,
            gate_u_enabled=True,
            gate_u_scaled=gate_u_scaled,
            gate_u_min=gate_min,
            budget_conservation=budget_conservation,
        )
        plt = self._plt()
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), constrained_layout=True)
        im = axes[0].imshow(
            terms["cost"],
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="magma_r",
        )
        contours = axes[0].contour(X, Y, terms["cost"], levels=[0.15, 0.30, 0.50, 0.70], colors="white", linewidths=1.5)
        axes[0].clabel(contours, inline=True, fontsize=8, fmt="c=%.2f")
        axes[0].set_title(f"weighted_exponential con gates - {var_obj}")
        axes[0].set_xlabel("L_e")
        axes[0].set_ylabel("L_edot")
        fig.colorbar(im, ax=axes[0], label="costo local 01")

        e_axis = np.linspace(0.0, 1.0, 400)
        terms_line = pr.weighted_exponential_terms(
            e_axis,
            np.full_like(e_axis, 0.25),
            np.full_like(e_axis, L_u_fixed),
            var_obj,
            w_e=w_e,
            w_edot=w_edot,
            w_u=w_u,
            scale_e=scale_e,
            scale_edot=scale_edot,
            scale_u=scale_u,
            gate_edot_enabled=True,
            gate_edot_scaled=gate_edot_scaled,
            gate_edot_min=gate_min,
            gate_u_enabled=True,
            gate_u_scaled=gate_u_scaled,
            gate_u_min=gate_min,
            budget_conservation=budget_conservation,
        )
        axes[1].plot(e_axis, terms_line["gates"]["L_edot"], lw=2.8, label="gate L_edot")
        axes[1].plot(e_axis, terms_line["gates"]["L_u"], lw=2.8, label="gate L_u")
        axes[1].plot(e_axis, terms_line["effective_weights"]["L_e"], lw=2.2, ls="--", label="peso ef. L_e")
        axes[1].plot(e_axis, terms_line["effective_weights"]["L_edot"], lw=2.2, ls="--", label="peso ef. L_edot")
        axes[1].plot(e_axis, terms_line["effective_weights"]["L_u"], lw=2.2, ls="--", label="peso ef. L_u")
        axes[1].set_title("Gates y pesos efectivos vs L_e")
        axes[1].set_xlabel("L_e usado como fuente de compuerta")
        axes[1].set_ylabel("factor / peso")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].grid(alpha=0.25)
        axes[1].legend(ncol=2, fontsize=8)
        return fig

    def plot_effort_gate(
        self,
        alpha_e: float = 0.65,
        effort_weight: float = 0.10,
        effort_gate_kappa: float = 4.0,
        var_obj: str = "pendulum_angle",
    ):
        orch = self.orchestrator.with_slider_values(
            alpha_e=alpha_e,
            effort_weight=effort_weight,
            effort_gate_kappa=effort_gate_kappa,
        )
        pr = orch.principal_reward
        xs = np.linspace(0.0, 1.0, 400)
        gates = pr.effort_gate(xs, var_obj)
        max_weight = effort_weight / (1.0 + effort_weight) if effort_weight > 0.0 else 0.0
        effective_weight = effort_weight * gates / (1.0 + effort_weight)
        plt = self._plt()
        fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
        ax.plot(xs, gates, lw=3, label="g_u(E_v)")
        ax.set_xlabel("E_v = alpha_e L_e + (1-alpha_e) L_edot")
        ax.set_ylabel("compuerta")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(xs, effective_weight, lw=2.5, color="#ea580c", label="peso efectivo L_u")
        ax2.set_ylabel("peso efectivo L_u")
        ax2.set_ylim(0.0, max(0.01, max_weight * 1.08))
        ax.axvline(0.25, color="#2563eb", ls="--")
        ax.axvline(0.50, color="#dc2626", ls="--")
        ax.set_title("Compuerta de esfuerzo y peso efectivo de L_u")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
        return fig

    def plot_progress_and_composition(
        self,
        principal_weight: float = 0.75,
        coordination_weight: float = 0.25,
        progress_clip_max: float = 0.02,
    ):
        orch = self.orchestrator.with_slider_values(
            principal_weight=principal_weight,
            coordination_weight=coordination_weight,
            progress_clip_max=progress_clip_max,
        )
        coord = orch.coordination_reward
        composition = orch.reward_composition
        effective_weights = composition.block_weights()
        principal_weight_eff = effective_weights["principal_reward"]
        coordination_weight_eff = effective_weights["coordination_reward"]
        xs = np.linspace(0.0, max(0.001, progress_clip_max * 3.0), 400)
        progress = coord.coordination_bonus_total(xs)
        J = np.linspace(0.0, 1.0, 400)
        plt = self._plt()
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
        axes[0].plot(xs, progress, lw=3)
        axes[0].axvline(progress_clip_max, color="#dc2626", ls="--", lw=2)
        axes[0].set_title("Normalizacion del progreso global")
        axes[0].set_xlabel("Delta Phi")
        axes[0].set_ylabel("bonus coordinativo")
        axes[0].set_ylim(0.0, 1.08)
        axes[0].grid(alpha=0.25)
        for C, color in [(0.0, "#dc2626"), (0.5, "#ea580c"), (1.0, "#16a34a")]:
            reward = composition.compose_loop_reward(-J, C)
            axes[1].plot(J, reward, color=color, lw=2.8, label=f"C={C:.1f}")
        if principal_weight_eff > 0.0:
            break_even = coordination_weight_eff / principal_weight_eff
            axes[1].axvline(break_even, color="#dc2626", ls="--", lw=2, label=f"J={break_even:.2f}")
        axes[1].axhline(0.0, color="black", lw=1)
        axes[1].set_title(
            f"Dominancia local/global (pesos efectivos {principal_weight_eff:.2f}/{coordination_weight_eff:.2f})"
        )
        axes[1].set_xlabel("J_v")
        axes[1].set_ylabel("reward compuesto")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        return fig

    def plot_coordination_potential_and_progress(
        self,
        potential_weight_e: float = 0.65,
        potential_weight_edot: float = 0.35,
        value_power: float = 1.0,
        square_terms: bool = False,
        progress_clip_max: float = 0.02,
    ):
        orch = self.orchestrator.with_slider_values(
            potential_weight_e=potential_weight_e,
            potential_weight_edot=potential_weight_edot,
            value_power=value_power,
            square_terms=square_terms,
            progress_clip_max=progress_clip_max,
        )
        coord = orch.coordination_reward
        L = np.linspace(0.0, 1.0, 400)
        potential = coord.local_potential(L, np.full_like(L, 0.25), "pendulum_angle")
        progress_delta = np.linspace(0.0, coord.task_progress_bound(), 400)
        progress_rewardable = coord.rewardable_progress(progress_delta)
        progress_01 = coord.task_progress_01(progress_rewardable)
        plt = self._plt()
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)
        axes[0].plot(L, potential, lw=3)
        axes[0].set_title("Potencial local al variar L_e (L_edot=0.25)")
        axes[0].set_xlabel("L_e")
        axes[0].set_ylabel("Phi_local")
        axes[0].grid(alpha=0.25)
        axes[0].text(0.02, 0.94, f"power efectivo={2.0 if square_terms else value_power:.2f}", transform=axes[0].transAxes, va="top")
        axes[1].plot(progress_delta, progress_01, lw=3, label="logica normalizada actual")
        axes[1].axvline(progress_clip_max, color="#dc2626", ls="--", lw=2, label="progress_clip_max (referencia config)")
        axes[1].set_title(f"Progreso global normalizado por bound={coord.task_progress_bound():.2f}")
        axes[1].set_xlabel("Delta Phi")
        axes[1].set_ylabel("task_progress_01")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        return fig

    def plot_coordination_component_mixer(
        self,
        global_progress_weight: float = 0.70,
        local_progress_weight: float = 0.20,
        local_debt_weight: float = 0.10,
        global_component_fixed: float = 0.40,
        progress_share_pendulum: float = 1.0,
        debt_share_pendulum: float = 1.0,
        condition_mode: str = "global_progress_positive",
    ):
        coord = self.orchestrator.coordination_reward
        component_weights = coord.normalized_component_weights(
            global_progress_weight,
            local_progress_weight,
            local_debt_weight,
        )
        progress_share = coord.normalized_var_share(progress_share_pendulum, 1.0)["pendulum_angle"]
        debt_share = coord.normalized_var_share(debt_share_pendulum, 1.0)["pendulum_angle"]
        progress = np.linspace(0.0, 1.0, 180)
        debt = np.linspace(0.0, 1.0, 180)
        P, D = np.meshgrid(progress, debt)
        condition = 1.0 if condition_mode in {"always", "global_progress_positive"} else np.where(P > 0, 1.0, 0.0)
        local_progress_component = progress_share * P * condition
        debt_component = debt_share * D
        reward = coord.coordination_module_reward(
            global_component_fixed,
            local_progress_component,
            debt_component,
            component_weights,
        )
        plt = self._plt()
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True, gridspec_kw={"width_ratios": [0.9, 1.4, 1.2]})
        axes[0].bar(component_weights.keys(), component_weights.values(), color=["#2563eb", "#16a34a", "#dc2626"])
        axes[0].set_title("Pesos internos normalizados")
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis="x", rotation=25)
        im = axes[1].imshow(reward, origin="lower", extent=[0, 1, 0, 1], aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        axes[1].contour(P, D, reward, levels=[0.0], colors="black", linewidths=2)
        axes[1].set_title("Reward coordinativo neto")
        axes[1].set_xlabel("progreso local 01")
        axes[1].set_ylabel("deuda local 01")
        fig.colorbar(im, ax=axes[1], label="reward modulo")

        e = np.linspace(0.0, 1.0, 400)
        for method, color in [("linear", "#2563eb"), ("exponential", "#16a34a"), ("hard", "#dc2626")]:
            axes[2].plot(e, coord.error_decay(e, threshold=0.35, lambda_v=1.0, method=method), lw=2.5, label=method, color=color)
        axes[2].set_title("Decaimiento por error local")
        axes[2].set_xlabel("|error| reducido")
        axes[2].set_ylabel("factor decay")
        axes[2].grid(alpha=0.25)
        axes[2].legend()
        return fig

    def plot_credit_share(self, epsilon_credit: float = 1.0e-6):
        orch = self.orchestrator.clone_with_overrides(
            {
                "reward_base": {
                    "reward_calculation": {
                        "coordination_reward": {"epsilon_credit": float(epsilon_credit)}
                    }
                }
            }
        )
        coord = orch.coordination_reward
        x = np.linspace(0.0, 1.0, 220)
        y = np.linspace(0.0, 1.0, 220)
        X, Y = np.meshgrid(x, y)
        Z = coord.credit_share(X, Y)
        plt = self._plt()
        fig, ax = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="coolwarm",
        )
        ax.plot([0, 1], [0, 1], color="black", lw=2, label="share=0.5")
        ax.scatter([0.82, 0.20, 0.55], [0.20, 0.80, 0.55], c="#dc2626", edgecolors="white")
        ax.text(0.84, 0.20, "pendulo mas")
        ax.text(0.22, 0.80, "carro mas")
        ax.text(0.57, 0.55, "parejo")
        ax.set_title("Reparto de credito coordinativo")
        ax.set_xlabel("credito correctivo pendulo")
        ax.set_ylabel("credito correctivo carro")
        ax.legend(loc="upper left")
        fig.colorbar(im, ax=ax, label="share pendulo")
        return fig

    def plot_local_extension_marginal(
        self,
        previous_cost: float = 0.45,
        baseline_cost: float = 0.50,
        tau: float = 0.10,
        mode: str = "immediate_delta",
        moved: bool = True,
    ):
        local = self.orchestrator.local_reward_extension
        current = np.linspace(0.0, 1.0, 500)
        terms = local.marginal_signal(
            current,
            previous_cost,
            baseline_cost,
            mode=mode,
            tau=tau,
            normalized=True,
        )
        penalty = local.movement_penalty(terms["move_reference"], np.ones_like(current) if moved else np.zeros_like(current))
        plt = self._plt()
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
        axes[0].plot(current, terms["reward"], lw=3, label="marginal reward")
        axes[0].axvline(previous_cost, color="#2563eb", ls="--", label="J previo")
        axes[0].axvline(baseline_cost, color="#16a34a", ls="--", label="baseline")
        axes[0].axhline(0, color="black", lw=1)
        axes[0].set_title(f"Marginal baseline ({mode})")
        axes[0].set_xlabel("J actual")
        axes[0].set_ylabel("senal marginal")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        axes[1].fill_between(current, 0, penalty, color="#fecaca")
        axes[1].plot(current, penalty, color="#dc2626", lw=2.5)
        axes[1].set_title("Penalizacion por mover sin mejorar")
        axes[1].set_xlabel("J actual")
        axes[1].set_ylabel("penalty signal")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(alpha=0.25)
        return fig

    def plot_local_extension_synergy(
        self,
        gamma_e: float = 1.0,
        gamma_edot: float = 1.0,
        gamma_u: float = 1.0,
        effort_fixed: float = 0.50,
        var_obj: str = "pendulum_angle",
    ):
        local = self.orchestrator.local_reward_extension
        error = np.linspace(-1.0, 1.0, 220)
        edot = np.linspace(-1.0, 1.0, 220)
        E, ED = np.meshgrid(error, edot)
        terms = local.synergy_terms(
            E,
            ED,
            np.full_like(E, effort_fixed),
            var_obj,
            gamma_e=gamma_e,
            gamma_edot=gamma_edot,
            gamma_u=gamma_u,
        )
        plt = self._plt()
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
        im = axes[0].imshow(terms["synergy"], origin="lower", extent=[-1, 1, -1, 1], aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        axes[0].contour(E, ED, terms["synergy"], levels=[0.0], colors="black", linewidths=1.8)
        axes[0].set_title("Synergy: error, derivada y esfuerzo")
        axes[0].set_xlabel("error")
        axes[0].set_ylabel("derivative_error")
        fig.colorbar(im, ax=axes[0], label="synergy")

        e_axis = np.linspace(0.0, 1.0, 400)
        axes[1].plot(e_axis, np.tanh(gamma_e * e_axis), lw=2.8, label="gate error")
        axes[1].plot(e_axis, np.tanh(gamma_edot * e_axis), lw=2.8, label="tanh edot")
        axes[1].plot(e_axis, np.tanh(gamma_u * e_axis), lw=2.8, label="tanh esfuerzo")
        axes[1].set_title("Sensibilidad de gammas")
        axes[1].set_xlabel("producto positivo")
        axes[1].set_ylabel("factor")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        return fig

    def plot_local_extension_composition(
        self,
        marginal_weight: float = 0.50,
        movement_weight: float = 0.25,
        synergy_weight: float = 0.25,
        moved: bool = True,
    ):
        local = self.orchestrator.local_reward_extension
        weights = local.normalized_component_weights(marginal_weight, movement_weight, synergy_weight)
        marginal = np.linspace(-1.0, 1.0, 200)
        synergy = np.linspace(-1.0, 1.0, 200)
        M, S = np.meshgrid(marginal, synergy)
        penalty = local.movement_penalty(M, np.ones_like(M) if moved else np.zeros_like(M))
        reward = weights["marginal_baseline"] * M + weights["synergy"] * S - weights["movement_penalty"] * penalty
        plt = self._plt()
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
        axes[0].bar(weights.keys(), weights.values(), color=["#2563eb", "#dc2626", "#16a34a"])
        axes[0].set_title("Pesos internos local_reward_extension")
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis="x", rotation=25)
        im = axes[1].imshow(reward, origin="lower", extent=[-1, 1, -1, 1], aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        axes[1].contour(M, S, reward, levels=[0.0], colors="black", linewidths=2)
        axes[1].set_title("Reward local por marginal/synergy")
        axes[1].set_xlabel("senal marginal")
        axes[1].set_ylabel("senal synergy")
        fig.colorbar(im, ax=axes[1], label="local extension reward")
        return fig


def build_plotter(config_main: Optional[Mapping[str, Any]] = None) -> RewardDesignPlotter:
    return RewardDesignPlotter(RewardDesignOrchestrator(config_main))
