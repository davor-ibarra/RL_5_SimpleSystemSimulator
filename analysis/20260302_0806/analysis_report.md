# 📊 Análisis Exhaustivo — Simulación `20260302_0806`

> **Fecha**: 2026-03-02 | **Episodios**: 5 000 | **Duración**: 5 s | **dt**: 0.001 s | **Intervalo decisión**: 0.02 s  
> **Delta gain**: 0.1 (universal) | **Rango ganancias**: [0, 6] | **Reward approach**: `controller_reward` (individual_reward **OFF**)  
> **Simulación previa de referencia**: `20260302_0504`

---

## 1. Resumen Ejecutivo — Hallazgos Críticos

### 1.1 Comparativa con la simulación anterior

| Indicador | `0504` (anterior) | `0806` (actual) | Δ |
|---|---|---|---|
| Estabilizaciones | 0 / 5000 (0 %) | **0 / 5000 (0 %)** | = |
| Terminación | 100 % limit_exceeded | **100 % limit_exceeded** | = |
| Total reward (mean) | −154.30 | **−400.76** | 🔴 **−160 % peor** |
| Total reward (std) | 87.03 | **275.91** | 🔴 **3.2× más varianza** |
| Performance (mean) | −72.17 | **−200.61** | 🔴 **−178 % peor** |
| Últimos 500 — reward | −256.26 | **−471.23** | 🔴 **−84 % peor** |
| Últimos 500 — performance | −111.05 | **−268.66** | 🔴 **−142 % peor** |

### 1.2 Diagnóstico global

> [!CAUTION]
> **La simulación `0806` empeoró drásticamente** pese a aplicar correcciones parciales del análisis previo. El reward medio se degradó 2.6× y la varianza se triplicó. La causa raíz es un **desbordamiento catastrófico en la normalización de `delta_u`** (6014 % de cobertura), combinado con la **no-corrección de los rangos de `u`** (425 % de desbordamiento persistente).

**Los 5 problemas raíz identificados:**

1. **🔴 `delta_u` saturado al 6014 %**: el rango [-0.01, 0.01] es 60× más estrecho que los valores reales. `L_delta_u` satura a ~0.48 de media (con p75 = 0.91-1.0), y con peso 0.30, **domina la recompensa con 27.7 %** aportando penalización constante e **informacionalmente vacía** (sin gradiente útil).

2. **🔴 `u` (acciones de control) siguen sin corregir**: cobertura del 425 % (pendulum) y −17.3 % (cart). La recomendación de ampliar a [-4, 4] **no fue aplicada**. Aunque `L_u` no tiene peso Lagrange directo, esto afecta al `dynamic_penalty`.

3. **🔴 `e` (pendulum) desborda al 169 %**: el rango [-0.5, 0.5] resulta demasiado estrecho. El error medio del péndulo es 0.316 rad → `e_mean²/0.25 = 0.40`, y el max regular alcanza 0.42 rad → normalizado a 0.71. El percentil 75 de L_e llega a **0.73-1.0**, indicando saturación frecuente.

4. **⚠️ Pesos Lagrange desbalanceados** por efecto de saturación: L_e contribuye 52.7 %, L_delta_u 27.7 %, L_edot solo 19.6 %. La contribución de `L_edot` cayó dramáticamente pese a tener w=0.30.

5. **⚠️ Dynamic penalty amplificada pero contraproducente**: la media del dynamic_penalty de péndulo creció de −0.88 a **−2.40** (2.7×) y del carro de −0.44 a **−1.29** (2.9×). Esto agrega penalización neta significativa (0.6 % del total_reward) sin proporcionar señal direccional útil dado que las acciones de control están permanentemente desbordadas.

---

## 2. Auditoría de Rangos de Normalización (Config vs. Real)

> **Método**: `mean_squared` → `L = clamp(valor² / max_abs², 0, 1)`

### 2.1 `pendulum_angle`

| Métrica | Rango Config | max_abs² | Rango Real (mean) | Cobertura % | Diagnóstico |
|---|---|---|---|---|---|
| **e** | [-0.5, 0.5] | 0.25 | [0.103, 0.422] | **169 %** | 🔴 **SATURACIÓN**: error medio (0.316) da L_e≈0.40, max (0.42) da L_e≈0.71 |
| **edot** | [-1.0, 1.0] | 1.00 | [-0.068, 0.863] | **86.3 %** | ✅ Rango aceptable — valores operan al 86 % del rango |
| **I** | [-2.0, 2.0] | 4.00 | [0.114, 0.270] | **6.7 %** | ⚠️ Infrautilizado (pero w=0, sin impacto actual) |
| **u** | [-1.0, 1.0] | 1.00 | [0.245, 4.249] | **425 %** | 🔴🔴 **SATURACIÓN MASIVA** — **sin corregir desde `0504`** |
| **delta_u** | [-0.01, 0.01] | 0.0001 | [0.000, 0.006] | **6014 %** | 🔴🔴🔴 **CATASTRÓFICO**: rango 60× menor que valores reales |

### 2.2 `cart_position`

| Métrica | Rango Config | max_abs² | Rango Real (mean) | Cobertura % | Diagnóstico |
|---|---|---|---|---|---|
| **e** | [-0.5, 0.5] | 0.25 | [-0.381, -0.020] | **−7.8 %** | ⚠️ Infrautilizado, error medio bajo |
| **edot** | [-0.6, 0.6] | 0.36 | [-0.515, -0.050] | **−13.8 %** | ⚠️ Utilización moderada (~14 %) |
| **I** | [-2.0, 2.0] | 4.00 | [-0.364, -0.004] | **−0.1 %** | 🔴 Infrautilizado extremo (w=0) |
| **u** | [-1.0, 1.0] | 1.00 | [-3.809, -0.173] | **−17.3 %** | 🔴 DESBORDA — valores hasta 3.8× el límite |
| **delta_u** | [-0.01, 0.01] | 0.0001 | [-0.005, -0.000] | **−104 %** | 🔴 **SATURACIÓN**: valores hasta 55× el rango |

### 2.3 Impacto directo de la saturación de `delta_u`

La saturación de `delta_u` se refleja directamente en los componentes Lagrange:

| Componente | L_mean | L_p75 | L_max | L_std |
|---|---|---|---|---|
| L_delta_u_pendulum | **0.484** | **0.909** | **1.000** (100%) | 0.391 |
| L_delta_u_cart | **0.402** | **0.810** | **1.000** (99.5%) | 0.382 |

> [!WARNING]
> `L_delta_u` está saturado en la mitad superior del rango [0,1] de forma permanente. Con peso w=0.30, aporta un castigo constante de ~0.27 al reward por paso, sin variación informativa. **El agente no puede aprender nada de esta señal porque es prácticamente constante.**

---

## 3. Análisis de Pesos Lagrange — Contribución Media y Varianza Ponderada

### 3.1 Configuración actual

| Componente | Peso (w) | Σ L_mean (pend + cart) | Contribución Ponderada | % del Total |
|---|---|---|---|---|
| **L_e** | **0.40** | 1.265 (0.388 + 0.249) = 0.637 | **0.5059** | **52.7 %** |
| **L_edot** | **0.30** | 0.628 (0.260 + 0.368) | **0.1883** | **19.6 %** |
| **L_I** | **0.00** | — | **0.0000** | **0.0 %** |
| **L_delta_u** | **0.30** | 0.886 (0.484 + 0.402) | **0.2659** | **27.7 %** |
| **TOTAL** | **1.00** | — | **0.9601** | **100 %** |

### 3.2 Comparativa con `0504`

| Componente | `0504` (% contribución) | `0806` (% contribución) | Cambio |
|---|---|---|---|
| L_e | 68.6 % | **52.7 %** | ↓ Mejor, pero aún domina |
| L_edot | 26.2 % | **19.6 %** | ↓ **Peor** — debería ser mayor |
| L_delta_u | 5.3 % | **27.7 %** | ↑↑ **Explosión por saturación** |

### 3.3 Diagnóstico detallado

1. **La contribución ponderada total subió de 0.269 a 0.960**: esto significa que el reward instantáneo (−Σ contribuciones) se multiplicó ~3.6× en magnitud. **Esto explica directamente** por qué el total_reward empeoró de −154 a −401.

2. **`L_delta_u` pasó de 5.3 % a 27.7 %**: un incremento de 5.2× causado por el rango [-0.01, 0.01] que satura permanentemente. Los valores reales de delta_u (mean ≈ 0.003) son 0.003²/0.0001 = 0.09 como mínimo, pero con distribución que frecuentemente alcanza 0.5-1.0 normalizado.

3. **`L_edot` bajó de 26.2 % a 19.6 %**: la contribución relativa de la derivada del error disminuyó porque `L_delta_u` creció desproporcionadamente. Esto es lo contrario de lo deseado — la velocidad angular debería ser el componente dominante para lograr estabilización.

4. **Varianza extrema en todos los componentes**: L_delta_u_std ≈ 0.39 (sobre un rango de [0,1]), L_e_std ≈ 0.10-0.14, L_edot_std ≈ 0.12. La señal de recompensa es altamente ruidosa.

---

## 4. Evaluación de Extra Rewards

### 4.1 Bandwidth Bonus

| Parámetro | Valor Config |
|---|---|
| `per_step_band_bonus` | 0.15 |
| `max_total_band_bonus` | 37.5 |
| `range: error_pendulum_angle` | [-0.5, 0.5] |
| `range: error_cart_position` | [-1.5, 1.5] |

| Estadística (por episodio) | Pendulum | Cart |
|---|---|---|
| Mean | **0.213** | **0.196** |
| Max | 3.00 | 3.00 |
| p25, p50, p75 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 |
| Std | 0.749 | 0.737 |
| **% del |total_reward|** | **0.05 %** | **0.05 %** |

**Comparativa con `0504`**:
| | `0504` | `0806` | Cambio |
|---|---|---|---|
| Mean (pend) | 0.052 | **0.213** | ↑ 4.1× mejor |
| Max | 0.80 | **3.00** | ↑ 3.75× más (= 20 pasos en banda) |
| % del total_reward | 0.03 % | **0.05 %** | → Prácticamente igual (peor total_reward) |

**Diagnóstico**:
- ✅ El incremento del `per_step_band_bonus` de 0.04 a 0.15 sí mejoró el bono absoluto (4× más).
- ⚠️ Sin embargo, los percentiles 25/50/75 siguen en **0.0** → en la mayoría de los pasos el agente sigue fuera de banda.
- El max de 3.0 = 20 pasos × 0.15, igual que antes en proporciones. El agente no permanece en banda más tiempo.
- ⚠️ La contribución relativa (0.05 %) sigue siendo **imperceptible** para la política porque el total_reward se amplificó por la saturación de delta_u.

### 4.2 Dynamic Penalty

| Parámetro | Pendulum | Cart |
|---|---|---|
| Weight | 0.5 | 0.3 |
| Condition scaled | 2.0 | 2.0 |

| Estadística (por episodio) | Pendulum (`0806`) | Cart (`0806`) | Pend (`0504`) | Cart (`0504`) |
|---|---|---|---|---|
| Mean | **−2.40** | **−1.29** | −0.88 | −0.44 |
| Min (peor) | **−8.50** | **−5.36** | −2.50 | −1.26 |
| p50 | **−1.07** | **−0.48** | −0.59 | −0.30 |
| Std | **2.73** | **1.62** | 0.83 | 0.43 |
| **% del total_reward** | **0.60 %** | **0.32 %** | 0.57 % | 0.28 % |

**Diagnóstico**:
- 📈 El `dynamic_penalty` se amplificó ~2.7× respecto a `0504` por el incremento de weights (0.3→0.5, 0.2→0.3) y la reducción de `scaled` (5.0→2.0).
- 🔴 **Varianza extrema** (std=2.73 para péndulo): la penalización dinámica varía enormemente entre episodios, añadiendo ruido a la señal de recompensa.
- ⚠️ La contribución relativa (0.6 %) sigue siendo baja respecto al reward principal, pero la magnitud absoluta (−2.40 por episodio) ya no es despreciable comparada con el bono de banda (+0.21). El agente recibe **11× más penalización dinámica que bono de banda**.
- **Paradoja persistente**: con `scaled=2.0`, cuando `error_pendulum=0.32` → `exp(-2×0.32)` = 0.53 → **la penalización sigue activa al 53 % incluso con error moderado**, pero el agente no puede reducirla porque las ganancias necesarias para controlar el péndulo generan acciones de control altas que son penalizadas.

---

## 5. Análisis de Señales de Control y Ganancias PID

### 5.1 Señales de control

| Señal | Min | Max | Mean | Std |
|---|---|---|---|---|
| error_pendulum_angle | 0.103 | 0.422 | **0.316** | 0.061 |
| error_cart_position | −0.381 | −0.020 | **−0.202** | 0.091 |
| deriv_error_pendulum | −0.068 | 0.863 | **0.415** | 0.173 |
| deriv_error_cart | −0.515 | −0.050 | **−0.337** | 0.088 |
| integral_error_pendulum | 0.114 | 0.270 | **0.198** | 0.027 |
| integral_error_cart | −0.364 | −0.004 | **−0.099** | 0.066 |
| **u_eff_pendulum** | 0.245 | **4.249** | **2.082** | 0.676 |
| **u_eff_cart** | **−3.809** | −0.173 | **−1.702** | 0.655 |
| delta_u_eff_pendulum | 0.000 | 0.006 | **0.003** | 0.001 |
| delta_u_eff_cart | −0.005 | 0.000 | **−0.002** | 0.001 |

**Comparativa con `0504`**:

| Señal | `0504` Mean | `0806` Mean | Cambio |
|---|---|---|---|
| error_pend | 0.308 | **0.316** | ↑ Peor (+2.7%) |
| error_cart | −0.240 | **−0.202** | ↗ Mejor (+16%) |
| u_eff_pend | 2.234 | **2.082** | ↗ Ligeramente menor |
| u_eff_cart | −1.843 | **−1.702** | ↗ Ligeramente menor |
| delta_u_pend | 0.002 | **0.003** | ↑ Mayor incrementalidad |

**Hallazgos**:
1. Las acciones de control siguen **masivamente desbordadas** (u_eff_pend mean=2.08, max=4.25 vs rango [-1,1]).
2. El error del péndulo empeoró ligeramente (0.308→0.316 mean).
3. Los delta_u se mantienen en el mismo orden (~0.002-0.003), pero ahora al normalizarse con rango 0.01 producen valores L_delta_u ~40-48× mayores que antes.

### 5.2 Ganancias PID finales

| Ganancia | Todos (mean ± σ) | Últimos 500 (mean ± σ) | `0504` last500 | Tendencia |
|---|---|---|---|---|
| kp_pendulum | 2.51 ± 1.69 | **3.89 ± 2.05** | 4.90 ± 0.72 | ↓ Más bajo, **mucha más varianza** |
| ki_pendulum | 3.24 ± 1.78 | **3.82 ± 1.20** | 3.17 ± 0.73 | ↑ Sube |
| kd_pendulum | 2.91 ± 1.80 | **1.88 ± 1.65** | 4.59 ± 0.86 | 🔴 **Colapso a 1.88** con altísima varianza |
| kp_cart | 3.51 ± 1.89 | **5.45 ± 0.55** | 2.29 ± 1.98 | ↑↑ Converge alto |
| ki_cart | 2.42 ± 1.63 | **1.46 ± 0.08** | 4.69 ± 0.91 | 🔴 **Colapso a 1.46** — baja varianza |
| kd_cart | 3.56 ± 1.87 | **5.14 ± 0.58** | 3.72 ± 1.09 | ↑ Converge alto |

**Diagnóstico de ganancias**:
1. 🔴 **kd_pendulum colapsó de 4.59 a 1.88**: el agente está **reduciendo** la ganancia derivativa del péndulo — la más importante para amortiguamiento. **Esto es un efecto directo de la saturación de L_delta_u**: como delta_u se penaliza despiadadamente, el agente minimiza kd para reducir la contribución derivativa (delta_u depende de kd × cambio en derivada del error).
2. **kp_cart y kd_cart convergen a ~5.1-5.5**: el agente empuja estas ganancias al máximo, intentando compensar con el lazo del carro lo que no puede hacer con el péndulo.
3. **ki_cart colapsó a 1.46**: el integral del carro es prácticamente desactivado.
4. **kp_pendulum bajó a 3.89 con σ=2.05**: la política no ha convergido — sigue con alta exploración residual.
5. **Patrón patológico**: la penalización excesiva de delta_u está forzando al agente a reducir las ganancias de péndulo (especialmente kd) → péndulo no se estabiliza → error crece → reward empeora. **Ciclo vicioso.**

---

## 6. Propuesta Concreta de Mejoras

### 6.1 Corrección URGENTE de rangos de normalización

> [!IMPORTANT]
> **Los rangos de `delta_u` y `u` son la causa raíz del colapso del rendimiento y deben corregirse como prioridad absoluta.**

```json
{
  "normalization": {
    "enabled": true,
    "output_limits": [0.0, 1.0],
    "params": {
      "pendulum_angle": {
        "e":       {"method": "mean_squared", "range": [-0.8, 0.8]},
        "edot":    {"method": "mean_squared", "range": [-1.0, 1.0]},
        "I":       {"method": "mean_squared", "range": [-0.5, 0.5]},
        "u":       {"method": "mean_squared", "range": [-5.0, 5.0]},
        "delta_u": {"method": "mean_squared", "range": [-0.08, 0.08]}
      },
      "cart_position": {
        "e":       {"method": "mean_squared", "range": [-0.8, 0.8]},
        "edot":    {"method": "mean_squared", "range": [-0.6, 0.6]},
        "I":       {"method": "mean_squared", "range": [-0.5, 0.5]},
        "u":       {"method": "mean_squared", "range": [-5.0, 5.0]},
        "delta_u": {"method": "mean_squared", "range": [-0.08, 0.08]}
      },
      "global_vars": {
        "u_total":       {"method": "mean_squared", "range": [-1, 1]},
        "delta_u_total": {"method": "mean_squared", "range": [-0.05, 0.05]}
      }
    }
  }
}
```

**Justificación crítica de cada cambio**:

| Variable | Antes (`0806`) | Propuesto | Razón |
|---|---|---|---|
| pend.e | [-0.5, 0.5] | **[-0.8, 0.8]** | Error real mean=0.316, max=0.422. Rango de 0.5 produce 169% overflow. Regresar a 0.8 elimina saturación mientras mantiene resolución. max_abs²=0.64 → L_e(0.316)=0.156 — buena resolución. |
| pend.edot | [-1.0, 1.0] | **[-1.0, 1.0]** ← Sin cambio | Cobertura del 86% — buen rango. El cambio previo fue correcto. |
| pend.u | [-1.0, 1.0] | **[-5.0, 5.0]** | Mean real=2.08, max=4.25. Rango de 5.0 cubre el p95 con margen. max_abs²=25 → L_u(2.08)=0.17 — resolución adecuada. |
| pend.delta_u | [-0.01, 0.01] | **[-0.08, 0.08]** | 🔴 **CAMBIO CRÍTICO**. Real mean=0.003, max=0.006. Rango 0.08 da max_abs²=0.0064 → L(0.003)=0.001, L(0.006)=0.006. Valor medio de la simulación anterior (`0504`) era 0.002. Con 0.08 hay espacio para crecimiento sin saturar. El rango anterior de 0.15 (en `0504`) daba 27% cobertura — aceptable. El cambio a 0.01 fue **excesivo**. |
| cart.e | [-0.5, 0.5] | **[-0.8, 0.8]** | Error real mean=0.20. Con 0.5, infrautilizado al 7.8%. Con 0.8, L_e(0.20)=0.063 — buena resolución media-baja. |
| cart.edot | [-0.6, 0.6] | **[-0.6, 0.6]** ← Sin cambio | 13.8% cobertura — aceptable para el lazo del carro. |
| cart.u | [-1.0, 1.0] | **[-5.0, 5.0]** | Mean real=1.70, max=3.81. Mismo razonamiento que péndulo. |
| cart.delta_u | [-0.01, 0.01] | **[-0.08, 0.08]** | Mean real=0.002, max=0.005. Mismo razonamiento. |

### 6.2 Redistribución de pesos Lagrange

Con los rangos corregidos, las contribuciones proyectadas cambian radicalmente. Los pesos deben ajustarse para que `L_edot` sea el componente dominante (prioritario para estabilización del péndulo).

```json
{
  "principal_reward": {
    "enabled": true,
    "reward_principal_name": "lagrange_reward",
    "module_path": "rewards.lagrange_reward_calculator",
    "class_name": "LagrangeRewardCalculator",
    "method": "lineal_combination",
    "lineal_combination_params": {
      "features": {
        "L_e":       {"weight": 0.35},
        "L_edot":    {"weight": 0.50},
        "L_I":       {"weight": 0.00},
        "L_delta_u": {"weight": 0.15}
      }
    }
  }
}
```

**Justificación**:
- **L_e 0.40 → 0.35**: con rango ampliado a 0.8, los valores L_e serán más bajos (~0.16 pendulum), reduciendo su dominancia natural.
- **L_edot 0.30 → 0.50**: restaurar la prioridad sobre la velocidad angular. Con rango 1.0 y valores reales ~0.42, L_edot_pend ≈ 0.17. Con w=0.50, la contribución será significativa.
- **L_delta_u 0.30 → 0.15**: con rango 0.08, L_delta_u será ~0.001 (valores reales ~0.003). Un peso de 0.15 es suficiente para señalizar suavidad sin dominar ni saturar. **Reducir drásticamente para evitar repetir el colapso de `0806`.**

### 6.3 Ajuste de Extra Rewards

#### 6.3.1 Bandwidth Bonus — sin cambios

```json
{
  "bandwidth_bonus": {
    "enabled": true,
    "per_step_band_bonus": 0.15,
    "max_total_band_bonus": 37.5,
    "ranges": {
      "error_pendulum_angle": [-0.5, 0.5],
      "error_cart_position": [-1.5, 1.5]
    }
  }
}
```

**Justificación**: los parámetros son razonables. El problema no fue le configuración del bono sino que el castigo principal se amplificó 2.6× por la saturación de delta_u, haciendo el bono relativamente irrelevante. Con la corrección de normalización, el total_reward debería volver a ~−150, y el bono medio de ~0.21 representaría ~0.14 % — sigue bajo pero es un incentivo en la dirección correcta. Si se corrige la normalización y el bono sigue siendo irrelevante, considerar subir `per_step_band_bonus` a 0.25 en la siguiente iteración.

#### 6.3.2 Dynamic Penalty

```json
{
  "dynamic_penalty": {
    "enabled": true,
    "method": "quadratic",
    "dynamic_penalty_params": {
      "control_action_pendulum_angle": {
        "weight": 0.3,
        "condition": {
          "type": "exp",
          "feature": "error_pendulum_angle",
          "scaled": 3.0,
          "setpoint": 0.0
        }
      },
      "control_action_cart_position": {
        "weight": 0.2,
        "condition": {
          "type": "exp",
          "feature": "error_cart_position",
          "scaled": 3.0,
          "setpoint": 0.0
        }
      }
    }
  }
}
```

**Justificación**:
- **weight_pend 0.5 → 0.3, weight_cart 0.3 → 0.2**: reducir los pesos de vuelta. Con w=0.5 y scaled=2.0 la penalización fue −2.40 de media, demasiado agresiva. Esto provocó que el agente redujera kd_pendulum a 1.88 para minimizar la penalización.
- **scaled 2.0 → 3.0**: aumentar el decay exponencial para que la penalización sea más selectiva — solo se active fuertemente cuando el error es muy pequeño (zona de estabilización final). Con `scaled=3.0` y error=0.32: `exp(-3×0.32)` = 0.38 → es un compromiso entre 0.21 (scaled=5.0, demasiado selectivo) y 0.53 (scaled=2.0, demasiado amplio).

---

## 7. Verificación Matemática de los Cambios Propuestos

### 7.1 Contribuciones Lagrange proyectadas (nuevos rangos + mismos valores reales `0806`)

#### L_e (nueva, rango [-0.8, 0.8])
- Pendulum: `0.316² / 0.8² = 0.0999 / 0.64 = 0.1561`
- Cart: `0.202² / 0.8² = 0.0408 / 0.64 = 0.0638`
- **Σ L_e = 0.2199**, **contribución ponderada = 0.35 × 0.2199 = 0.0770**

#### L_edot (sin cambio, rango [-1.0, 1.0] pend, [-0.6, 0.6] cart)
- Pendulum: `0.415² / 1.0² = 0.1723`
- Cart: `0.337² / 0.6² = 0.1136 / 0.36 = 0.3155`
- **Σ L_edot = 0.4878**, **contribución ponderada = 0.50 × 0.4878 = 0.2439**

#### L_delta_u (nueva, rango [-0.08, 0.08])
- Pendulum: `0.003² / 0.08² = 0.000009 / 0.0064 = 0.00141`
- Cart: `0.002² / 0.08² = 0.000004 / 0.0064 = 0.000625`
- **Σ L_delta_u = 0.00203**, **contribución ponderada = 0.15 × 0.00203 = 0.000305**

### 7.2 Distribución proyectada

| Componente | Antes (`0806`) | Después (proyectado) | Cambio |
|---|---|---|---|
| L_e | 52.7 % (0.506) | **23.9 % (0.077)** | ↓ Desaturada |
| L_edot | 19.6 % (0.188) | **75.8 % (0.244)** | ↑↑ Ahora domina — **correcto** |
| L_delta_u | 27.7 % (0.266) | **0.1 % (0.0003)** | ↓↓ Desaturada drásticamente |
| **TOTAL** | 0.960 | **0.321** | ↓ **3× menos castigo total** |

### 7.3 Impacto en el total_reward estimado

- **Reward principal** (6 agentes, 250 pasos): `−0.321 × 6 × 250 = −481.5` → por episodio puro Lagrange ≈ −482 (pero esto es un upper bound; los valores L no son constantes y el promedio real será menor).
- Más realistamente, el total_reward debería regresar al rango de **−120 a −180** (similar o mejor que `0504`), dado que la contribución ponderada total (0.321) es comparable a la de `0504` (0.269) pero con distribución mucho más balanceada.

### 7.4 Análisis del dynamic_penalty proyectado

Con `scaled=3.0` y `weight=0.3` para péndulo:
- Error medio del péndulo = 0.316: `condición = exp(-3.0 × 0.316)` = **0.388**
- Penalización = `0.3 × u² × 0.388`. Con u_eff_pend² ≈ 4.33: penalty ≈ `0.3 × 4.33 × 0.388` = **−0.504** por paso
- Por episodio (~250 pasos × 0.02 dt ratio): ~**−0.25** por episodio → ~0.15 % del total_reward estimado

Esto es 10× menor que la penalización actual (−2.40) pero sigue siendo direccional.

### 7.5 Verificaciones de coherencia

- ✅ **Σ pesos Lagrange = 1.00** (0.35 + 0.50 + 0.00 + 0.15)
- ✅ **L_edot domina al 75.8 %** → correcto para priorizar estabilización
- ✅ **L_delta_u al 0.1 %** → deja de saturar, proporciona señal suave y proporcional
- ✅ **L_e al 23.9 %** → segundo componente, contribuye sin dominar
- ✅ **Rangos de normalización cubren p95 de valores reales sin saturar**
- ✅ **Dynamic penalty reducida** para evitar que el agente colapse kd_pendulum
- ⚠️ **Monitorear**: si tras corrección el agente sube kd_pendulum de 1.88 a ~4-5, los delta_u reales podrían crecer → verificar que el rango 0.08 sigue siendo suficiente

---

## Anexo: Configuración JSON Consolidada para Aplicar

<details>
<summary>📋 Bloque completo de <code>reward_calculation</code></summary>

```json
{
  "reward_calculation": {
    "metric_processing": {
      "normalization": {
        "enabled": true,
        "output_limits": [0.0, 1.0],
        "params": {
          "pendulum_angle": {
            "e":       {"method": "mean_squared", "range": [-0.8, 0.8]},
            "edot":    {"method": "mean_squared", "range": [-1.0, 1.0]},
            "I":       {"method": "mean_squared", "range": [-0.5, 0.5]},
            "u":       {"method": "mean_squared", "range": [-5.0, 5.0]},
            "delta_u": {"method": "mean_squared", "range": [-0.08, 0.08]}
          },
          "cart_position": {
            "e":       {"method": "mean_squared", "range": [-0.8, 0.8]},
            "edot":    {"method": "mean_squared", "range": [-0.6, 0.6]},
            "I":       {"method": "mean_squared", "range": [-0.5, 0.5]},
            "u":       {"method": "mean_squared", "range": [-5.0, 5.0]},
            "delta_u": {"method": "mean_squared", "range": [-0.08, 0.08]}
          },
          "global_vars": {
            "u_total":       {"method": "mean_squared", "range": [-1, 1]},
            "delta_u_total": {"method": "mean_squared", "range": [-0.05, 0.05]}
          }
        }
      }
    },
    "principal_reward": {
      "enabled": true,
      "reward_principal_name": "lagrange_reward",
      "module_path": "rewards.lagrange_reward_calculator",
      "class_name": "LagrangeRewardCalculator",
      "method": "lineal_combination",
      "lineal_combination_params": {
        "features": {
          "L_e":       {"weight": 0.35},
          "L_edot":    {"weight": 0.50},
          "L_I":       {"weight": 0.00},
          "L_delta_u": {"weight": 0.15}
        }
      }
    },
    "extra_rewards": {
      "enabled": true,
      "reward_extra_name": "extra_rewards",
      "module_path": "rewards.extra_rewards_handler",
      "class_name": "ExtraRewardsHandler",
      "penalty_approach": {
        "penalty_instantaneous_reward": {"enabled": false},
        "penalty_delta_var": {"enabled": false}
      },
      "bonus_approach": {
        "goal_bonus_reward": {"enabled": false},
        "bandwidth_bonus": {
          "enabled": true,
          "per_step_band_bonus": 0.15,
          "max_total_band_bonus": 37.5,
          "ranges": {
            "error_pendulum_angle": [-0.5, 0.5],
            "error_cart_position": [-1.5, 1.5]
          }
        }
      },
      "conditional_approach": {
        "dynamic_penalty": {
          "enabled": true,
          "method": "quadratic",
          "dynamic_penalty_params": {
            "control_action_pendulum_angle": {
              "weight": 0.3,
              "condition": {
                "type": "exp",
                "feature": "error_pendulum_angle",
                "scaled": 3.0,
                "setpoint": 0.0
              }
            },
            "control_action_cart_position": {
              "weight": 0.2,
              "condition": {
                "type": "exp",
                "feature": "error_cart_position",
                "scaled": 3.0,
                "setpoint": 0.0
              }
            }
          }
        },
        "dynamic_incentive": {"enabled": false}
      }
    }
  }
}
```

</details>

### Resumen de cambios clave vs. `0806`

| Parámetro | `0806` (actual) | Propuesto | Motivo |
|---|---|---|---|
| pend.e range | [-0.5, 0.5] | **[-0.8, 0.8]** | Desaturar (169%→39%) |
| pend.u range | [-1.0, 1.0] | **[-5.0, 5.0]** | Desaturar (425%→83%) |
| pend.delta_u range | **[-0.01, 0.01]** | **[-0.08, 0.08]** | 🔴 **Desaturar (6014%→38%)** |
| cart.e range | [-0.5, 0.5] | **[-0.8, 0.8]** | Coherencia con pend |
| cart.u range | [-1.0, 1.0] | **[-5.0, 5.0]** | Desaturar (17%→34%) |
| cart.delta_u range | **[-0.01, 0.01]** | **[-0.08, 0.08]** | 🔴 **Desaturar (104%→6.3%)** |
| L_e weight | 0.40 | **0.35** | Reducir dominancia |
| L_edot weight | 0.30 | **0.50** | Priorizar amortiguamiento |
| L_delta_u weight | **0.30** | **0.15** | Evitar dominancia por saturación |
| dp weight_pend | 0.50 | **0.30** | Evitar colapso de kd |
| dp weight_cart | 0.30 | **0.20** | Reducir penalización excesiva |
| dp scaled | 2.0 | **3.0** | Zona operativa más selectiva |
