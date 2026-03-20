# 📊 Análisis Exhaustivo y Minucioso: Simulación 20260224_1211

El análisis de la simulación `20260224_1211` revela el avance conseguido tras modificar los parámetros de inicialización del PID ($K_p=2.0$, $K_i=2.0$, $K_d=2.0$) y potenciar el incremento del agente (`delta_gain: 0.5`). Sin embargo, persisten trabas fundamentales en la estructura de asignación de crédito.

## 1. 🔍 Hallazgos y Limitaciones de Rendimiento Actuales

Aunque la inyección inicial de fuerza mitiga el problema de "Too little, too late" del reporte anterior, exponemos las verdaderas limitaciones de optimización del agente Q-Learning:

### A. Mejor Cota de Supervivencia Inicial pero Aprendizaje Ciego
* **Diagnóstico:** Los controladores arrancan con ganancias de $2.0$ y pueden dar saltos de $0.5$. Esto detiene la caída libre gravitacional en los primeros milisegundos. Sin embargo, no hay optimización fina.
* **Consecuencia:** El agente "sobrevive" más tiempo empíricamente que cuando arrancaba en 0. Pero sin saber "por qué" sobrevive, los episodios degeneran en ruido. El agente explora, satura el péndulo y falla irremediablemente en los siguientes segundos sin descubrir una política direccional de convergencia hacia $0.0$.

### B. El Bloqueo Principal Sigue Intacto: Credit Assignment Placa (Idéntico)
* **Diagnóstico (Crítico):** Al revisar el subárbol `reward_base` en `metadata.json`, los parámetros `individual_reward_params/lineal_combination` **siguen configurados exactamente igual** para Kp, Ki y Kd en todos los ejes de estado:
  * `w_e: 1.0`, `w_edot: 1.0`, `w_I: 1.0`, `w_u: 1.0`, `w_s: 1.0`
* **Consecuencia:** Como se predijo en retrospectiva del lote `1024`, los tres agentes independientes bajo una variable objetivo (como `pendulum_angle`) están compitiendo a ciegas. 
  * Si el error de posición es enorme, P, I y D son penalizados por igual, cuando D no tiene la culpa de la posición.
  * Si el comportamiento es oscilatorio inestable, P, I y D son castigados, cuando el verdadero responsable de amortiguar es D, y el causante del lag es I o P. 
  Esta interferencia cruzada impide construir una Q-table coherente para cada sub-agente.

### C. Sistema de "Extra Rewards" Sigue en el Banquillo (Desactivado)
* **Diagnóstico:** A pesar del gran desafío de mantener un péndulo invertido sin caerse (donde evitar cruzar los $0.4 rads$ exige maestría), todos los Extra Rewards están globalmente en `"enabled": false`. 
* **Consecuencia:** La señal dominante es un goteo punitivo (castigo continuo basado en distancia al objetivo). Sin un `goal_bonus_reward` (premio por alcanzar la meta) o un `bandwidth_bonus` (premio por sostenerse dentro de ciertos límites aunque no sea perfecto), el agente asume que cualquier trayectoria a largo plazo es una causa perdida. 

## 2. 📝 Propuestas Fundamentadas de Ajuste de Parámetros

Para destrabar irrevocablemente el proceso de aprendizaje direccional en la próxima iteración, es menester la aplicación de las siguientes modificaciones directamente sobre los archivos `yaml` que orquestan el simulador (típicamente de la carpeta `./config`):

### Propuesta 1 (MANDATORIA): Inyectar la Diferenciación Numérica en los Recompensas
Modificar drásticamente `lineal_combination_params` en la base del Reward. Esto es no-negociable para que los agentes operen matemáticamente.
*   **Agente Kp (`kp_pendulum_angle` y `kp_cart_position`):** Responsable principal del error angular instantáneo.
    *   `w_e: 1.0` *(Enfoque total aquí)*
    *   `w_edot: 0.1` 
    *   `w_I: 0.0`
    *   `w_u: 0.5`, `w_s: 0.5`
*   **Agente Kd (`kd_pendulum_angle` y `kd_cart_position`):** Responsable principal de amortiguar, ergo, reducir la derivada del error.
    *   `w_e: 0.1`
    *   `w_edot: 1.0` *(Enfoque total aquí)*
    *   `w_I: 0.0`
    *   `w_u: 0.5`, `w_s: 0.5`
*   **Agente Ki (`ki_pendulum_angle` y `ki_cart_position`):** Responsable del sesgo sostenido en régimen permanente.
    *   `w_e: 0.1`
    *   `w_edot: 0.0`
    *   `w_I: 1.0` *(Enfoque total aquí)*
    *   `w_u: 0.5`, `w_s: 0.5`

### Propuesta 2: Activar y Sintonizar los "Extra Rewards"
Convertir `"enabled": true` en el nodo secundario y configurar:
*   **`bandwidth_bonus`:** Configurar un bono de `0.05` por paso cuando el error del péndulo se mantenga dentro de `[-0.1, 0.1]`. Esto sembrará un parche verde de "éxito parcial" en el mapa de recompensas que atraerá la política codiciosa (greedy) de exploración hacia el centro.
*   **`dynamic_penalty` para Acciones Bruscas:** Evitará que el delta_gain alto que programamos ($0.5$) genere espasmos de control en el carro inyectando un leve castigo condicionado a usar altas acciones cerca del `setpoint`.

### Propuesta 3: Desaceleración Gradual del Aprendizaje y Ajuste de Initial Bounds 
*   **Diagnóstico de Cota:** Aunque iniciar con `Kp/Ki/Kd` en 2.0 y `delta=0.5` salva al sistema de morir instantáneamente, su comportamiento a largo de 5000 episodios satura muy fácilmente si crece libremente.
*   **Ajuste:** Bajar el parámetro `max: 10.0` a `max: 6.0` en la cota de los agentes. Matemáticamente, una ganancia proporcional de 6.0 sobre un error de 0.2 rad arroja una demanda de torque de $1.2$ que ya supera la zona lineal nominal. Permitirle saltar hasta 10.0 incentiva la saturación irremediable rápida. Permutemos a `delta_gain=0.1` para que aprenda a balancear fino alrededor del 2.0.
