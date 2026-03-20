# 📊 Análisis de Incongruencias: Simulación 20260223_1024

Tras analizar los resultados del archivo `summary.xlsx` de la simulación `20260223_1024` y el archivo de configuración actual `config_CartPole.yaml`, se han detectado **incongruencias críticas** que están limitando severamente la capacidad de los agentes de reinforcement learning (Q-Learning) para converger u optimizar correctamente. 

A continuación detallo las áreas de mejora y los ajustes paramétricos sugeridos:

---

### 1. 🚨 Agentes P, I y D recibiendo exactamente la misma recompensa
* **Incongruencia:** El análisis estadístico muestra que los valores (Mín, Máx, Media, y Percentiles) de `reward_kp_pendulum_angle`, `reward_ki_pendulum_angle` y `reward_kd_pendulum_angle` **son exactamente idénticos** en todo el historial, y lo mismo ocurre para `cart_position`.
* **Causa (en `config_CartPole.yaml`):** En la sección `individual_reward_params > lineal_combination_params`, los pesos para la asignación de crédito están configurados todos en `1.0`:
  ```yaml
  kp_pendulum_angle: {w_e: 1.0, w_edot: 1.0, w_I: 1.0, w_u: 1.0, w_s: 1.0}
  ki_pendulum_angle: {w_e: 1.0, w_edot: 1.0, w_I: 1.0, w_u: 1.0, w_s: 1.0}
  kd_pendulum_angle: {w_e: 1.0, w_edot: 1.0, w_I: 1.0, w_u: 1.0, w_s: 1.0}
  ```
  Al ser iguales, el RL no puede hacer asignación de crédito (Credit Assignment). El agente Kp no sabe si lo hizo bien respecto al error (e), ni Kd respecto a la derivada (edot). 
* **Acción recomendada:** Diferenciar los pesos por responsabilidad. Por ejemplo, seguir los comentarios que tienes a la derecha en el archivo YAML:
  * Kp -> Priorizar `w_e`
  * Kd -> Priorizar `w_edot`
  * Ki -> Priorizar `w_I`

---

### 2. 📉 Insensibilidad de `L_delta_u` (Penalización por esfuerzo de control)
* **Incongruencia:** Las métricas de recompensa relacionadas con la variación del control (`L_delta_u_pendulum_angle` y `L_delta_u_cart_position`) se estancan permanentemente en el valor de **`~0.5000`**, con una desviación estándar minúscula (`0.0004`). Es decir, el agente no percibe ningún castigo o recompensa relevante al hacer variaciones bruscas en su acción.
* **Causa:** En la configuración de normalización de métricas, el rango para el cálculo está establecido de forma exagerada:
  ```yaml
  delta_u: {method: mean, range: [-2.0, 2.0]} # Variación acción máximo
  ```
  La variación real paso a paso (`delta_u_eff`) oscila típicamente en `0.003`. Un cambio de `0.003` evaluado en un rango teórico de `[-2.0, 2.0]` al ser normalizado al espacio de `[0.0, 1.0]`, se convierte en `0.5`, anulando la señal de aprendizaje y desperdiciando el peso de `0.1` que tiene en el reward principal.
* **Acción recomendada:** Reducir drásticamente los rangos en `metric_processing`. De `range: [-2.0, 2.0]` a `range: [-0.05, 0.05]` o algo cercano al comportamiento real (`std ~ 0.01`).

---

### 3. 🔥 Picos Extremos en la Acción de Control (Over-actuation / Windup)
* **Incongruencia:** El PID interno está demandando acciones de control desproporcionadas (`control_action_pendulum_angle` llega a **10.86** y `control_action_cart_position` cae a **-9.19**). 
* **Causa:** Si tienes los límites del actuador global en `[-1.0, 1.0]`, significan que las ganancias calculadas por los agentes, combinadas con los errores, están generando señales de control 10 veces mayores a lo que el sistema puede aplicar (`u_eff` las recorta adecuadamente). Sin embargo, calcular internamente un factor de `10.8` suele significar:
  * Saturación prematura porque las ganancias de exploración inician muy alto.
  * El sistema de **Anti-Windup condicional** recientemente ajustado podría estar dejando acumular error rápido porque P y D por sí solos ya rebasan el `±1.0` provocando picos enormes que I no alcanza a mitigar.
* **Acción recomendada:** Revisar/Reducir los rangos máximos de las acciones dinámicas de Kp, Ki y Kd en la inicialización (el `max` en `agent_config.agents` está en `5.0`). Bajar la cota máxima (`max: 3.0` o menor) ayudará a no crear picos instantáneos inestables por control P o D.

---

### 4. 🗜️ Capping en el error (Cart Position)
* **Incongruencia:** Las métricas de penalización de cart_position en ciertos percentiles o máximos se anclan artificialmente. `error_cart_position` muestra un valor mínimo constante de `-1.0` y máximo de `1.0`, pero la posición real cruda (`cart_position_raw`) en algunas ocasiones alcanza posiciones de `5.0` metros (los límites de los bordes del sim).
* **Causa:** En `state_normalization.ranges_params`, el rango declarado para `cart_position` es `[-2.0, 2.0]`. Cuando el carrito cruza el umbral de `2.0` y escala a `5.0`, la lógica normaliza cortando el error a `-1.0` o `1.0`. 
* **Acción recomendada:** Ajustar los `ranges_params` de las posiciones / velocidades crudas para que coincidan con o incluso cubran los verdaderos espacios antes de la falla (por ejemplo, escalar `cart_position` de `[-5.0, 5.0]` si esa es la barrera final de penalización física del entorno).

### Conclusión
Para darle al sistema RL una oportunidad real de aprender, tu prioridad **#1 debe ser implementar los pesos diferenciados (Credit Assignment) en el reward base de `config_CartPole.yaml`** y ajustar el rango para normalizar `delta_u`, el cual actualmente anula toda la información transitoria.
