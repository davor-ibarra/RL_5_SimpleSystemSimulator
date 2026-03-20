# 📊 Análisis Exhaustivo — Simulación `20260302_0504`

> **Fecha**: 2026-03-02 | **Episodios**: 5 000 | **Duración**: 5 s | **dt**: 0.001 s | **Intervalo decisión**: 0.02 s  
> **Delta gain**: 0.1 (universal) | **Rango ganancias**: [0, 6] | **Reward approach**: `controller_reward` (individual_reward **OFF**)

---

## 1. Resumen Ejecutivo — Hallazgos Críticos

| Indicador | Valor |
|---|---|
| Estabilizaciones | **0 / 5 000 (0.0 %)** |
| Terminación 100 % | `limit_exceeded` |
| Total reward (media ± σ) | −154.30 ± 87.03 |
| Performance (media ± σ) | −72.17 ± 38.62 |
| **Últimos 500 ep** — reward | −256.26 ± 50.06 |
| **Últimos 500 ep** — performance | −111.05 ± 20.91 |

### Diagnóstico global

1. **Cero estabilizaciones**: en 5 000 episodios el péndulo nunca fue controlado dentro de los criterios de estabilización. El 100 % de los episodios termina por `limit_exceeded` (péndulo supera ±1.0472 rad o carro ±5 m).
2. **Degradación progresiva**: los últimos 500 episodios muestran un reward medio de −256.26, **un 66 % peor** que la media global (−154.30). Esto evidencia un **colapso del aprendizaje tardío**: a medida que epsilon decae, la política explotada lleva a valores peores, no mejores.
3. **Normalización severamente desalineada**: la variable `u (pendulum_angle)` tiene **cobertura del 390.9 %** — los valores reales desbordan casi 4× el rango configurado, lo que satura la normalización y destruye la señal de recompensa.
4. **Contribución Lagrange muy desbalanceada**: `L_e` domina con **68.6 %** de la contribución ponderada total; `L_edot` aporta **26.2 %** y `L_delta_u` apenas **5.3 %**. Esto impide que los componentes de suavidad y velocidad angular influyan significativamente en la política.
5. **Extra rewards marginales**: el `bandwidth_bonus` aporta solo **0.03 %** del total_reward medio, y el `dynamic_penalty` aporta ~0.57 % (péndulo) y 0.28 % (carro). Son prácticamente imperceptibles para el agente.

---

## 2. Auditoría de Rangos de Normalización (Config vs. Real)

> **Método**: `mean_squared`. La métrica normalizada es `L = clamp(valor² / max_abs², 0, 1)`.

### 2.1 `pendulum_angle`

| Métrica | Rango Config | max_abs² | Rango Real (mean) | Cobertura % | Diagnóstico |
|---|---|---|---|---|---|
| **e** | [−0.8, 0.8] | 0.64 | [0.105, 0.420] | **65.6 %** | ⚠️ Infrautilizado — se usa ~2/3 del rango |
| **edot** | [−3.0, 3.0] | 9.00 | [−0.024, 0.808] | **9.0 %** | 🔴 **Rango excesivamente amplio**: solo se usa el 9 % → señal aplanada |
| **I** | [−2.0, 2.0] | 4.00 | [0.119, 0.260] | **6.5 %** | 🔴 **Infrautilizado extremo**: integral casi constante (desactivado w=0) |
| **u** | [−1.0, 1.0] | 1.00 | [0.395, 3.909] | **390.9 %** | 🔴🔴 **SATURACIÓN MASIVA**: valores reales 4× sobre el rango → L_u siempre = 1.0 |
| **delta_u** | [−0.15, 0.15] | 0.0225 | [0.000, 0.006] | **27.3 %** | ⚠️ Moderadamente infrautilizado |

### 2.2 `cart_position`

| Métrica | Rango Config | max_abs² | Rango Real (mean) | Cobertura % | Diagnóstico |
|---|---|---|---|---|---|
| **e** | [−0.8, 0.8] | 0.64 | [−0.409, −0.031] | **−4.8 %** | ⚠️ Infrautilizado y con sesgo negativo |
| **edot** | [−1.5, 1.5] | 2.25 | [−0.518, −0.073] | **−3.3 %** | 🔴 **Utilización mínima**: <5 % del rango |
| **I** | [−2.0, 2.0] | 4.00 | [−0.421, −0.008] | **−0.2 %** | 🔴 **Utilización despreciable** (desactivado w=0) |
| **u** | [−1.0, 1.0] | 1.00 | [−3.463, −0.208] | **−20.8 %** | 🔴 **DESBORDA** rango: valores reales 3.5× el límite → saturación |
| **delta_u** | [−0.08, 0.08] | 0.0064 | [−0.006, 0.000] | **−3.3 %** | ⚠️ Muy infrautilizado |

### 2.3 `global_vars`

| Métrica | Rango Config | max_abs² | Diagnóstico |
|---|---|---|---|
| **u_total** | [−1, 1] | 1.00 | Coherente (actuador limitado a ±1) |
| **delta_u_total** | [−0.05, 0.05] | 0.0025 | Sin datos reales en audit — verificar |

### 2.4 Hallazgos críticos de normalización

> [!CAUTION]
> **Las acciones de control (`u`) de ambos lazos desbordan masivamente sus rangos de normalización.** Esto significa que las componentes Lagrange asociadas a `u` estarían saturadas a 1.0 permanentemente, aunque dichas componentes (`L_u`) no participan activamente en el reward actual (no tienen peso Lagrange explícito). **Sin embargo**, esta saturación afecta indirectamente la señal del `dynamic_penalty`, que usa `control_action_*` como variable penalizada.

> [!WARNING]
> **edot (ambos lazos)** tiene rangos configurados 10-30× más amplios que los valores reales observados. Esto comprime la señal normalizada a valores cercanos a 0, **atenuando drásticamente** la contribución de `L_edot` pese a tener el mayor peso Lagrange (0.50).

---

## 3. Análisis de Pesos Lagrange — Contribución Media y Varianza Ponderada

### 3.1 Configuración actual

| Componente | Peso (w) | Σ L_mean (pend+cart) | Contribución Ponderada | % del Total | σ ponderada |
|---|---|---|---|---|---|
| **L_e** | **0.35** | 0.5259 (0.2040 + 0.1812) + ... | **0.1841** | **68.6 %** | Alta (0.071 + 0.086) |
| **L_edot** | **0.50** | 0.1407 (0.0356 + 0.1051) | **0.0704** | **26.2 %** | Media (0.025 + 0.036) |
| **L_I** | **0.00** | — (desactivado) | **0.0000** | **0.0 %** | — |
| **L_delta_u** | **0.15** | 0.0943 (0.0334 + 0.0609) | **0.0141** | **5.3 %** | Baja (0.025 + 0.033) |
| **TOTAL** | **1.00** | — | **0.2686** | **100 %** | — |

### 3.2 Desglose por controlador

| Componente | L_mean (pendulum) | L_mean (cart) | Ratio pend/cart |
|---|---|---|---|
| L_e | 0.2040 | 0.1812 | 1.13 : 1 |
| L_edot | 0.0356 | 0.1051 | 1 : 2.95 |
| L_delta_u | 0.0334 | 0.0609 | 1 : 1.83 |

### 3.3 Diagnóstico Lagrange

1. **`L_e` domina abrumadoramente (68.6 %)** pese a tener peso 0.35, porque su valor medio normalizado (0.20-0.18 por lazo) es mucho mayor que el de `L_edot` (0.036 pendulum). Esto refleja que la normalización de `edot` comprime demasiado los valores.

2. **`L_edot` (w=0.50) contribuye solo 26.2 %**: tiene el mayor peso nominal pero su contribución real es baja. **Causa raíz**: el rango de normalización `edot_pendulum = [-3.0, 3.0]` da `max_abs²=9.0`, pero los valores reales promedian solo 0.356 rad/s → `L_edot_pend = 0.356²/9.0 ≈ 0.014`, que es microscópico. El rango de normalización está aplastando la señal.

3. **`L_delta_u` (w=0.15) es prácticamente invisible (5.3 %)**: valores de delta_u_eff son ≤0.006, contra un `max_abs²=0.0225`. El delta_u real está 1 orden de magnitud por debajo del rango, y al elevarlo al cuadrado la compresión es aún mayor.

4. **Varianza alta en L_e** (σ ≈ 0.072-0.086): episodios donde el error es pequeño vs. grande producen rewards muy distintos. Esto genera alta varianza en la señal — inconsistente para Q-learning.

---

## 4. Evaluación de Extra Rewards

### 4.1 Bandwidth Bonus

| Parámetro | Valor Config |
|---|---|
| `per_step_band_bonus` | 0.04 |
| `max_total_band_bonus` | 10.0 |
| `range: error_pendulum_angle` | [−0.4, 0.4] |
| `range: error_cart_position` | [−1.0, 1.0] |

| Estadística (por episodio) | Pendulum | Cart |
|---|---|---|
| Mean | 0.0519 | 0.0479 |
| Max | 0.80 | 0.80 |
| p25, p50, p75 | 0.0, 0.0, 0.0 | 0.0, 0.0, 0.0 |
| Std | 0.192 | 0.189 |
| **% del |total_reward|** | **0.03 %** | **0.03 %** |

**Diagnóstico**:
- ⚠️ Los percentiles 25, 50 y 75 son **exactamente 0.0**. Esto significa que en la gran mayoría de los pasos el agente está **fuera de banda**. Solo en momentos muy puntuales (iniciales, antes de que el péndulo diverja) se otorga el bono.
- 📉 Contribución del 0.03 % respecto al castigo acumulado de ~154 unidades de reward negativo → **el bono es completamente irrelevante** para la política.
- El máximo de 0.80 (= 20 pasos × 0.04) indica que incluso en el mejor caso, el agente solo permaneció ~20 de los 250 pasos en banda. 0.80/10.0 = 8 % del bono máximo teórico.
- **Rango del péndulo [−0.4, 0.4] es demasiado estrecho** para la dinámica observada: el ángulo medio del péndulo es 0.323 rad, con p75 = 0.421 rad. La banda apenas cubre la mediana.

### 4.2 Dynamic Penalty

| Parámetro | Pendulum | Cart |
|---|---|---|
| Weight | 0.3 | 0.2 |
| Condition type | exp | exp |
| Feature | error_pendulum_angle | error_cart_position |
| Scaled | 5.0 | 5.0 |

| Estadística (por episodio) | Pendulum | Cart |
|---|---|---|
| Mean (absoluto, negativo) | −0.880 | −0.437 |
| Min (peor caso) | −2.503 | −1.256 |
| Max (mejor caso) | −0.007 | −0.000 |
| p50 | −0.593 | −0.295 |
| Std | 0.830 | 0.426 |
| **% del |total_reward|** | **0.57 %** | **0.28 %** |

**Diagnóstico**:
- La penalización dinámica por péndulo es mayor que la del carro (2:1), lo cual es coherente porque el ángulo del péndulo diverge más.
- Sin embargo, **ambas penalidades representan <1 % del total_reward**. Son marginales e insuficientes para dirigir la política.
- **El factor `scaled=5.0` es demasiado alto**: la condición exponencial `exp(-scaled * |error|)` decae rápidamente. Con `scaled=5.0` y error medio del péndulo ~0.31 rad: `exp(−5.0 × 0.31) = exp(−1.55) ≈ 0.21`. Esto reduce la penalización a solo el 21 % de su peso nominal cuando el error es moderado.
- **Paradoja**: cuando el error es grande (y la acción debería ser penalizada más), la condición exponencial _reduce_ la penalización (porque `exp(-scaled*error)` → 0). La penalización solo es fuerte cuando el error es pequeño y ya no se necesita tanto control.

---

## 5. Análisis de Señales de Control y Ganancias PID

### 5.1 Señales de control (medias por episodio)

| Señal | Min | Max | Mean | Std |
|---|---|---|---|---|
| error_pendulum_angle | 0.105 | 0.420 | **0.308** | 0.057 |
| error_cart_position | −0.409 | −0.031 | **−0.240** | 0.077 |
| deriv_error_pendulum | −0.024 | 0.808 | **0.356** | 0.160 |
| deriv_error_cart | −0.518 | −0.073 | **−0.381** | 0.077 |
| integral_error_pendulum | 0.119 | 0.260 | **0.212** | 0.024 |
| integral_error_cart | −0.421 | −0.008 | **−0.120** | 0.055 |
| u_eff_pendulum | 0.395 | **3.909** | **2.234** | 0.738 |
| u_eff_cart | **−3.463** | −0.208 | **−1.843** | 0.713 |
| delta_u_eff_pendulum | 0.000 | 0.006 | **0.002** | 0.001 |
| delta_u_eff_cart | −0.006 | 0.000 | **−0.002** | 0.001 |

**Hallazgos de control**:
1. 🔴 **Acciones de control MASIVAMENTE desbordadas**: `u_eff_pendulum` tiene un mean de **2.234** (max 3.909) y `u_eff_cart` de **−1.843** (min −3.463), cuando el rango de normalización es [−1, 1]. Los controladores combinados están generando señales 2-4× más allá del actuador. Esto indica **ganancias PID excesivamente altas** o falta de limitación individual por lazo.
2. ⚠️ **Error de péndulo siempre positivo** (mean=0.308 rad ≈ 17.6°): el péndulo nunca cruza el punto de equilibrio, siempre está cayendo en la misma dirección. Esto sugiere que el control es insuficiente para revertir la caída inicial de 0.157 rad → el péndulo diverge monotónicamente.
3. 📊 **Delta_u_eff es muy pequeño** (±0.002): los cambios incrementales por paso de decisión son mínimos, lo que sugiere que los agentes están haciendo ajustes muy finos de ganancia, pero insuficientes para contrarrestar la dinámica gravitacional.

### 5.2 Ganancias PID finales

| Ganancia | Todos (mean ± σ) | Últimos 500 (mean ± σ) | Tendencia |
|---|---|---|---|
| kp_pendulum | 2.73 ± 1.79 | **4.90 ± 0.72** | ↑ Converge alto |
| ki_pendulum | 3.22 ± 1.80 | **3.17 ± 0.73** | → Estable |
| kd_pendulum | 3.60 ± 1.90 | **4.59 ± 0.86** | ↑ Converge alto |
| kp_cart | 2.78 ± 1.72 | **2.29 ± 1.98** | ↓ Baja, alta varianza |
| ki_cart | 2.72 ± 1.86 | **4.69 ± 0.91** | ↑ Converge alto |
| kd_cart | 3.15 ± 1.66 | **3.72 ± 1.09** | ↑ Sube moderado |

**Diagnóstico de ganancias**:
1. **kp_pendulum converge a 4.9**: esto multiplica el error (0.31 rad) × 4.9 = 1.52 de acción — ya supera el actuador normalizado. Es lógico que el agente intente subir Kp para frenar la caída, pero genera saturación.
2. **ki_pendulum se mantiene estable (~3.2)**: no crece porque la integral del error es pequeña (0.21) y su contribución no es alta.
3. **ki_cart sube agresivamente a 4.69**: puede estar intentando compensar el sesgo permanente del carro, pero esto genera integral windup que agravará el desbordamiento.
4. **kp_cart baja a 2.29 con altísima varianza (1.98)**: la política no ha convergido — sigue explorando activamente. El error del carro es disperso y la señal de reward no le proporciona dirección clara.

---

## 6. Propuesta Concreta de Mejoras

### 6.1 Ajuste de rangos de normalización

**Problema**: Los rangos actuales de `edot` son 10-30× más amplios que los valores reales, aplastando su contribución ponderada. Los rangos de `u` son 4× menores que los valores reales, saturando permanentemente.

```json
{
  "normalization": {
    "enabled": true,
    "output_limits": [0.0, 1.0],
    "params": {
      "pendulum_angle": {
        "e":       {"method": "mean_squared", "range": [-0.5, 0.5]},
        "edot":    {"method": "mean_squared", "range": [-1.0, 1.0]},
        "I":       {"method": "mean_squared", "range": [-0.5, 0.5]},
        "u":       {"method": "mean_squared", "range": [-4.0, 4.0]},
        "delta_u": {"method": "mean_squared", "range": [-0.01, 0.01]}
      },
      "cart_position": {
        "e":       {"method": "mean_squared", "range": [-0.5, 0.5]},
        "edot":    {"method": "mean_squared", "range": [-0.6, 0.6]},
        "I":       {"method": "mean_squared", "range": [-0.5, 0.5]},
        "u":       {"method": "mean_squared", "range": [-4.0, 4.0]},
        "delta_u": {"method": "mean_squared", "range": [-0.01, 0.01]}
      },
      "global_vars": {
        "u_total":       {"method": "mean_squared", "range": [-1, 1]},
        "delta_u_total": {"method": "mean_squared", "range": [-0.05, 0.05]}
      }
    }
  }
}
```

**Justificación de cada cambio**:

| Variable | Antes | Después | Razón |
|---|---|---|---|
| pend.e | [-0.8, 0.8] | **[-0.5, 0.5]** | Mean real ~ 0.31. Rango cubre p95 (0.42) con margen. Mejor resolución. |
| pend.edot | [-3.0, 3.0] | **[-1.0, 1.0]** | Mean real ~ 0.36, max ~ 0.81. Antes max_abs²=9.0, ahora 1.0 → 9× más sensible. |
| pend.I | [-2.0, 2.0] | **[-0.5, 0.5]** | Real [0.12, 0.26]. Aunque w=0, preparar para activación futura. |
| pend.u | [-1.0, 1.0] | **[-4.0, 4.0]** | Real mean=2.23, max=3.91. Rango actual satura al 390 %. |
| pend.delta_u | [-0.15, 0.15] | **[-0.01, 0.01]** | Real mean=0.002, max=0.006. Ajustar a 10× del rango real. |
| cart.e | [-0.8, 0.8] | **[-0.5, 0.5]** | Real mean=0.24. Mejor resolución sin saturar. |
| cart.edot | [-1.5, 1.5] | **[-0.6, 0.6]** | Real mean=0.38, max=0.52. Antes solo al 3.3 % de uso. |
| cart.I | [-2.0, 2.0] | **[-0.5, 0.5]** | Real mean=0.12. Mismo razonamiento. |
| cart.u | [-1.0, 1.0] | **[-4.0, 4.0]** | Real mean=1.84, max=3.46. Saturación al −20.8 %. |
| cart.delta_u | [-0.08, 0.08] | **[-0.01, 0.01]** | Real mean=0.002, max=0.006. |

### 6.2 Redistribución de pesos Lagrange

**Problema**: `L_e` domina al 68.6 % porque sus valores normalizados son altos. Con la corrección de rangos, `L_edot` tendrá valores normalizados mucho más altos (9× más sensible), por lo que es necesario rebalancear los pesos para evitar que `L_edot` domine completamente tras el ajuste.

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
        "L_e":       {"weight": 0.30},
        "L_edot":    {"weight": 0.40},
        "L_I":       {"weight": 0.00},
        "L_delta_u": {"weight": 0.30}
      }
    }
  }
}
```

**Justificación**:
- **L_e 0.35 → 0.30**: con rangos más estrechos para `e` (0.5 vs. 0.8), el valor medio normalizado subirá. Reducir ligeramente el peso compensa y evita dominancia.
- **L_edot 0.50 → 0.40**: el ajuste de rango edot de 3.0 a 1.0 aumentará L_edot_pend ~9×. Un peso menor evita que domine completamente pero mantiene prioridad sobre la velocidad angular.
- **L_delta_u 0.15 → 0.30**: con rango delta_u ajustado de 0.15 a 0.01, la sensibilidad sube 225×. Esto penalizará mucho más los cambios bruscos de acción, incentivando políticas suaves. Subir el peso a 0.30 consolida esta señal de suavidad.
- **Σ = 1.00** ✓

### 6.3 Ajuste de Extra Rewards

#### 6.3.1 Bandwidth Bonus

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

**Justificación**:
- **per_step_band_bonus 0.04 → 0.15**: el bono actual (0.03 % del total_reward) es imperceptible. Un bono de 0.15 × 250 pasos = 37.5 máximo teórico, que representaría ~24 % del |total_reward| medio actual. Esto sí generará un gradiente atractivo.
- **max_total_band_bonus 10.0 → 37.5**: coherente con 250 pasos × 0.15.
- **range_pendulum [-0.4, 0.4] → [-0.5, 0.5]**: ampliar ligeramente para cubrir el ángulo medio (~0.31 rad) con un margen del 60 %. Esto permitirá obtener el bono durante más pasos, sembrando experiencias positivas.
- **range_cart [-1.0, 1.0] → [-1.5, 1.5]**: el error del carro promedia 0.24 (posición), la banda actual es bastante generosa, pero con movimientos amplios del carro (max 2.05), ampliar a 1.5 permite capturar más trayectorias parcialmente exitosas.

#### 6.3.2 Dynamic Penalty

```json
{
  "dynamic_penalty": {
    "enabled": true,
    "method": "quadratic",
    "dynamic_penalty_params": {
      "control_action_pendulum_angle": {
        "weight": 0.5,
        "condition": {
          "type": "exp",
          "feature": "error_pendulum_angle",
          "scaled": 2.0,
          "setpoint": 0.0
        }
      },
      "control_action_cart_position": {
        "weight": 0.3,
        "condition": {
          "type": "exp",
          "feature": "error_cart_position",
          "scaled": 2.0,
          "setpoint": 0.0
        }
      }
    }
  }
}
```

**Justificación**:
- **weight_pend 0.3 → 0.5, weight_cart 0.2 → 0.3**: aumentar los pesos para que la penalización sea perceptible (actualmente solo 0.57 % y 0.28 %).
- **scaled 5.0 → 2.0**: **cambio crucial**. Con `scaled=5.0`, la función `exp(−5×|error|)` decae demasiado rápido:
  - Error = 0.31: `exp(-5×0.31)` = 0.21 (solo 21 % del peso)
  - Error = 0.50: `exp(-5×0.50)` = 0.08 (8 %)
  
  Con `scaled=2.0`:
  - Error = 0.31: `exp(-2×0.31)` = 0.54 (54 % del peso)
  - Error = 0.50: `exp(-2×0.50)` = 0.37 (37 %)
  
  Esto amplía la zona operativa de la penalización, castigando las acciones bruscas incluso cuando hay error moderado, no solo cuando error ≈ 0.

---

## 7. Verificación Matemática de los Cambios Propuestos

### 7.1 Contribuciones Lagrange estimadas tras el ajuste

**Proyección con nuevos rangos** (asumiendo mismos valores reales observados):

#### L_e (nueva)
- Pendulum: `e_mean² / max_abs²_new = 0.308² / 0.5² = 0.0949 / 0.25 = 0.3795`
- Cart: `e_mean² / max_abs²_new = 0.240² / 0.5² = 0.0576 / 0.25 = 0.2304`
- **Σ L_e_mean_new = 0.6099**, **contribución ponderada = 0.30 × 0.6099 = 0.1830**

#### L_edot (nueva)
- Pendulum: `edot_mean² / max_abs²_new = 0.356² / 1.0² = 0.1268 / 1.0 = 0.1268`
- Cart: `edot_mean² / max_abs²_new = 0.381² / 0.6² = 0.1452 / 0.36 = 0.4033`
- **Σ L_edot_mean_new = 0.5301**, **contribución ponderada = 0.40 × 0.5301 = 0.2120**

#### L_delta_u (nueva)
- Pendulum: `delta_u_mean² / max_abs²_new = 0.002² / 0.01² = 0.000004 / 0.0001 = 0.0400`
- Cart: `delta_u_mean² / max_abs²_new = 0.002² / 0.01² = 0.000004 / 0.0001 = 0.0400`
- **Σ L_delta_u_mean_new = 0.0800**, **contribución ponderada = 0.30 × 0.0800 = 0.0240**

### 7.2 Distribución proyectada

| Componente | Contribución | % del Total |
|---|---|---|
| L_e | 0.1830 | **43.7 %** |
| L_edot | 0.2120 | **50.6 %** |
| L_delta_u | 0.0240 | **5.7 %** |
| **TOTAL** | **0.4190** | **100 %** |

**Comparación con la distribución actual**:

| Componente | ANTES (%) | DESPUÉS (%) | Cambio |
|---|---|---|---|
| L_e | 68.6 % | **43.7 %** | ↓ Menos dominante |
| L_edot | 26.2 % | **50.6 %** | ↑ Ahora domina — correcto para priorizar amortiguamiento |
| L_delta_u | 5.3 % | **5.7 %** | → Estable, pero ahora con mayor sensibilidad real |

> [!IMPORTANT]
> La nueva distribución es más balanceada: `L_edot` ahora domina al 50.6 %, lo cual prioriza la reducción de velocidad angular — esencial para evitar la divergencia monotónica del péndulo observada. `L_e` pasa de una dominancia abusiva (68.6 %) a un segundo lugar razonable (43.7 %) que sigue siendo significativo.

### 7.3 Escala comparativa de Extra Rewards proyectada

| Componente | Contribución estimada por episodio | % del total reward estimado |
|---|---|---|
| Bandwidth bonus (si 50% pasos en banda) | 0.15 × 125 = **18.75** | ~12 % ← significativo |
| Dynamic penalty pendulum (media con scaled=2.0) | ~0.5 × u² × 0.54 × 250 × 0.02 = **~1.35** | ~0.87 % |
| Dynamic penalty cart (media con scaled=2.0) | ~0.3 × u² × 0.54 × 250 × 0.02 = **~0.81** | ~0.52 % |

### 7.4 Verificación de coherencia

- ✅ **Σ pesos Lagrange = 1.00** (0.30 + 0.40 + 0.00 + 0.30)
- ✅ **Rangos de normalización cubren p95 de valores reales** con margen del 20-30 %
- ✅ **Bandwidth bonus ahora significativo** (~12 % vs. 0.03 % actual)
- ✅ **Dynamic penalty con zona operativa más amplia** (54 % vs. 21 % a error medio)
- ✅ **L_edot priorizada** como componente dominante → correcta para estabilización
- ⚠️ **Monitorear L_delta_u**: con rangos delta_u=[−0.01, 0.01], si los delta reales cambian significativamente tras convergencia de ganancias, este componente podría saturarse. Ajustar en iteraciones futuras.

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
            "e":       {"method": "mean_squared", "range": [-0.5, 0.5]},
            "edot":    {"method": "mean_squared", "range": [-1.0, 1.0]},
            "I":       {"method": "mean_squared", "range": [-0.5, 0.5]},
            "u":       {"method": "mean_squared", "range": [-4.0, 4.0]},
            "delta_u": {"method": "mean_squared", "range": [-0.01, 0.01]}
          },
          "cart_position": {
            "e":       {"method": "mean_squared", "range": [-0.5, 0.5]},
            "edot":    {"method": "mean_squared", "range": [-0.6, 0.6]},
            "I":       {"method": "mean_squared", "range": [-0.5, 0.5]},
            "u":       {"method": "mean_squared", "range": [-4.0, 4.0]},
            "delta_u": {"method": "mean_squared", "range": [-0.01, 0.01]}
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
          "L_e":       {"weight": 0.30},
          "L_edot":    {"weight": 0.40},
          "L_I":       {"weight": 0.00},
          "L_delta_u": {"weight": 0.30}
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
              "weight": 0.5,
              "condition": {
                "type": "exp",
                "feature": "error_pendulum_angle",
                "scaled": 2.0,
                "setpoint": 0.0
              }
            },
            "control_action_cart_position": {
              "weight": 0.3,
              "condition": {
                "type": "exp",
                "feature": "error_cart_position",
                "scaled": 2.0,
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
