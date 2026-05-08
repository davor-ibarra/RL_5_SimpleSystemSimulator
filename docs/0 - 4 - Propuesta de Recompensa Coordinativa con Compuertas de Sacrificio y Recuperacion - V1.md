# Propuesta de Recompensa Coordinativa con Compuertas de Sacrificio y Recuperacion - V1

## 1. Proposito del documento

Este documento desarrolla una propuesta matematica para una recompensa coordinativa con compuertas dinamicas. La propuesta esta pensada para sistemas fisicos controlados por varios lazos locales, donde cada lazo posee un controlador parametrico y donde agentes de aprendizaje por refuerzo autosintonizan las ganancias de dichos controladores.

La propuesta se formula de manera general. Por ello, no depende de una planta fisica particular, ni de nombres concretos de variables como posicion, angulo, presion, nivel, temperatura o velocidad. La formulacion se apoya en magnitudes abstractas de desempeno, tales como costos normalizados por lazo, progreso local, progreso global y credito coordinativo.

La intencion central es resolver un problema tipico de sistemas coordinados: durante ciertas fases de una maniobra, un lazo puede necesitar empeorar temporalmente su propio objetivo local para permitir que el sistema completo avance hacia una condicion global mejor. Sin embargo, una vez que el objetivo global ha mejorado lo suficiente, el mismo lazo debe volver a su propio setpoint. La recompensa debe reconocer ambas fases sin mezclar directamente las observaciones de los agentes ni introducir reglas especificas de una planta.

La propuesta recibe el nombre:

**Recompensa Coordinativa con Compuertas de Sacrificio y Recuperacion.**

El nombre resume las dos ideas principales:

- **Sacrificio:** un lazo puede recibir credito cuando su actuacion contribuye al progreso global aunque su costo local aumente temporalmente.
- **Recuperacion:** cuando el sistema global ya se encuentra suficientemente encaminado, cada lazo debe volver a ser incentivado a reducir su propio costo local.

## 2. Principio arquitectonico que se desea preservar

La arquitectura considerada posee varios agentes autosintonizantes. Cada agente modifica una ganancia de un controlador local. Por ejemplo, en un controlador PID podrian existir agentes distintos para \(K_p\), \(K_i\) y \(K_d\). Sin embargo, la propuesta no depende de que el controlador sea PID; solamente requiere que existan parametros ajustables.

El principio de diseno es el siguiente:

> Cada agente debe aprender principalmente a partir de su propio lazo y de su propia ganancia. La coordinacion entre lazos debe aparecer solamente mediante la recompensa coordinativa, no mediante una mezcla directa de observaciones de distintos lazos.

Esto significa que el agente no necesita observar el estado completo del sistema fisico. Tampoco necesita recibir como estado la variable fisica de otro controlador. El mecanismo de coordinacion se introduce como una senal escalar de aprendizaje calculada externamente a partir de metricas del sistema.

El diseno conserva entonces tres niveles separados:

1. **Nivel local:** cada controlador posee su propio error, esfuerzo y costo.
2. **Nivel global:** el sistema completo posee un potencial de tarea que resume el estado global de desempeno.
3. **Nivel coordinativo:** la recompensa coordinativa reparte credito global entre lazos sin convertir las observaciones de los agentes en observaciones acopladas.

Esta separacion es importante porque permite que la propuesta sea reutilizable en distintos sistemas fisicos.

## 3. Descripcion intuitiva del problema

En un sistema con multiples lazos, no siempre es optimo que cada controlador minimice su error local de manera inmediata y aislada. En algunas maniobras, un lazo de soporte debe realizar una accion que parece localmente mala, pero que globalmente es necesaria.

La situacion puede describirse asi:

- un lazo principal necesita alcanzar una region de operacion favorable;
- otro lazo actua como soporte para permitir ese avance;
- el lazo de soporte puede alejarse de su propio setpoint durante esa ayuda;
- si la recompensa local lo castiga demasiado, el aprendizaje inhibe la maniobra util;
- si se premia solamente el avance global, el lazo de soporte puede aprender a ayudar pero no necesariamente a volver.

El problema no es solamente que falte una recompensa global. El problema mas fino es que la recompensa debe distinguir dos fases:

1. **Fase de asistencia global:** se permite que un lazo sacrifique temporalmente su objetivo local si eso produce progreso global.
2. **Fase de recuperacion local:** cuando el sistema global ya mejoro, cada lazo debe volver a su setpoint.

Una recompensa coordinativa sin compuertas puede reconocer el progreso global, pero puede no generar suficiente presion para el retorno local. Por eso se propone agregar compuertas suaves que modulen el tipo de coordinacion segun el estado abstracto de desempeno del sistema.

## 4. Notacion general

Se considera un sistema con \(m\) lazos de control. El conjunto de lazos se denota como:

$$
\mathcal{V} = \{1,2,\dots,m\}.
$$

Cada indice \(i \in \mathcal{V}\) representa un lazo local. El lazo \(i\) posee:

- una variable objetivo fisica, que no necesita aparecer explicitamente en la recompensa propuesta;
- un setpoint local;
- un controlador local;
- uno o mas parametros ajustables por agentes;
- un costo local normalizado que resume que tan lejos esta el lazo de su objetivo.

El tiempo se organiza en dos escalas:

- \(t\): indice de simulacion o tiempo fisico fino;
- \(k\): indice de intervalo de decision del agente.

La recompensa se calcula a nivel de intervalo \(k\), porque los agentes modifican ganancias cada cierto intervalo y no necesariamente en cada paso fino de simulacion.

## 5. Costo local abstracto por lazo

Para cada lazo \(i\), se define un costo local normalizado:

$$
J_i(k) \ge 0.
$$

Este costo representa que tan malo fue el comportamiento del lazo \(i\) durante el intervalo \(k\). Un valor pequeno indica buen desempeno local; un valor grande indica mal desempeno local.

El costo \(J_i(k)\) puede construirse a partir de metricas locales normalizadas. Una forma general es:

$$
J_i(k)
=
\sum_{q \in \mathcal{Q}}
a_{i,q} L_{i,q}(k),
$$

donde:

- \(\mathcal{Q}\) es el conjunto de caracteristicas locales usadas para evaluar el lazo;
- \(L_{i,q}(k)\) es una metrica no negativa del lazo \(i\) para la caracteristica \(q\);
- \(a_{i,q} \ge 0\) es el peso de la caracteristica \(q\) para el lazo \(i\).

Ejemplos de caracteristicas locales son:

- costo de error;
- costo de derivada del error;
- costo de integral del error;
- costo de esfuerzo de control;
- costo de variacion del esfuerzo.

La propuesta no exige usar todas estas caracteristicas. Para una version inicial, suele ser suficiente usar error y derivada del error:

$$
J_i(k)
=
a_{i,e}L_{i,e}(k)
+
a_{i,\dot e}L_{i,\dot e}(k).
$$

Lo importante es que \(J_i(k)\) sea una cantidad abstracta, normalizada, comparable entre intervalos y definida por lazo.

## 6. Potencial global de tarea

Se define un potencial global de tarea:

$$
\Phi(k)
=
\sum_{i \in \mathcal{V}}
\beta_i J_i(k),
$$

donde:

- \(\Phi(k)\) mide el desempeno global del sistema durante el intervalo \(k\);
- \(\beta_i \ge 0\) es la importancia global asignada al lazo \(i\);
- \(J_i(k)\) es el costo local del lazo \(i\).

La interpretacion es directa:

- si \(\Phi(k)\) es grande, el sistema global esta lejos de una condicion deseada;
- si \(\Phi(k)\) es pequeno, el sistema global esta cerca de una condicion deseada.

El potencial global no es una observacion entregada al agente. Es una magnitud usada por el calculador de recompensa para decidir como repartir credito.

La forma anterior es lineal por simplicidad. Tambien se podria usar una forma no lineal, pero la version lineal tiene tres ventajas:

1. Es interpretable.
2. Es facil de normalizar.
3. Permite analizar matematicamente el progreso global.

## 7. Progreso global y progreso local

El progreso global entre dos intervalos consecutivos se define como:

$$
\Delta \Phi(k)
=
\Phi(k-1)-\Phi(k).
$$

Si \(\Delta \Phi(k)>0\), el potencial global disminuyo. Por lo tanto, el sistema mejoro globalmente.

Si \(\Delta \Phi(k)<0\), el potencial global aumento. Por lo tanto, el sistema empeoro globalmente.

La parte positiva del progreso global se define como:

$$
\Delta \Phi^+(k)
=
\max(0,\Delta \Phi(k)).
$$

Esta cantidad premia solamente mejora global. No castiga empeoramiento global, porque el castigo ya puede estar contenido en la recompensa local principal. Esta decision reduce interferencias y hace que el termino coordinativo sea un complemento positivo.

De manera analoga, para cada lazo se define el progreso local:

$$
\Delta J_i(k)
=
J_i(k-1)-J_i(k).
$$

Si \(\Delta J_i(k)>0\), el lazo \(i\) mejoro localmente, porque su costo disminuyo.

Si \(\Delta J_i(k)<0\), el lazo \(i\) empeoro localmente, porque su costo aumento.

Se definen dos partes no negativas:

$$
\Delta J_i^+(k)
=
\max(0,\Delta J_i(k)),
$$

$$
\Delta J_i^-(k)
=
\max(0,-\Delta J_i(k)).
$$

Estas cantidades tienen interpretaciones distintas:

- \(\Delta J_i^+(k)\) mide recuperacion o mejora local.
- \(\Delta J_i^-(k)\) mide sacrificio o deterioro local.

La propuesta no asume que todo deterioro local sea bueno. El deterioro local solo puede ser considerado sacrificio util si ocurre junto con progreso global y credito coordinativo.

## 8. Credito coordinativo por lazo

Para repartir el progreso global entre lazos se define una medida de contribucion no negativa:

$$
c_i(k) \ge 0.
$$

Esta cantidad debe medir cuanto contribuyo el lazo \(i\) al avance coordinado durante el intervalo \(k\). La definicion exacta puede depender de las senales locales disponibles, pero debe cumplir una condicion importante:

> La contribucion \(c_i(k)\) debe construirse a partir de informacion atribuible al lazo \(i\), no a partir de una accion global mezclada que impida distinguir responsables.

Una forma general es:

$$
c_i(k)
=
\frac{1}{N_k}
\sum_{t \in k}
\chi_i(t),
$$

donde:

- \(N_k\) es el numero de pasos finos dentro del intervalo \(k\);
- \(\chi_i(t)\ge 0\) es una contribucion instantanea atribuible al lazo \(i\).

Por ejemplo, \(\chi_i(t)\) puede medir esfuerzo correctivo local, alineamiento entre error local y accion local, reduccion local de costo o cualquier magnitud local no negativa que sea consistente con la planta.

Luego, el credito relativo se define como:

$$
\rho_i(k)
=
\frac{c_i(k)}
{\varepsilon + \sum_{j \in \mathcal{V}} c_j(k)},
$$

donde \(\varepsilon>0\) evita division por cero.

La interpretacion de \(\rho_i(k)\) es:

- si el lazo \(i\) no contribuye, entonces \(\rho_i(k)\) se aproxima a cero;
- si el lazo \(i\) concentra la contribucion, entonces \(\rho_i(k)\) se aproxima a uno;
- la suma de los creditos relativos es menor o igual a uno:

$$
\sum_{i \in \mathcal{V}}\rho_i(k)
=
\frac{\sum_i c_i(k)}
{\varepsilon+\sum_i c_i(k)}
\le 1.
$$

Esta propiedad es util porque impide que el reparto coordinativo cree mas credito relativo que el progreso disponible.

## 9. Necesidad de compuertas

Una recompensa coordinativa basica podria escribirse como:

$$
R_i^{coord}(k)
=
\lambda \rho_i(k)\Delta\Phi^+(k),
$$

donde \(\lambda\ge 0\) regula la intensidad del bonus coordinativo.

Esta forma premia a los lazos que contribuyen cuando el sistema global mejora. Sin embargo, esta recompensa no distingue fases. En particular, no diferencia entre:

- la fase en que conviene permitir movimiento de soporte;
- la fase en que conviene volver a minimizar el costo local.

Por ello se introducen compuertas. Una compuerta es una funcion escalar, normalmente acotada entre cero y uno, que modula una recompensa segun una condicion abstracta.

La compuerta no debe entenderse como una regla discreta del tipo "si ocurre A, entonces hacer B". Se propone como una funcion suave que cambia gradualmente el peso de cada objetivo.

## 10. Funcion logistica de compuerta

Se define la funcion logistica:

$$
\sigma_\tau(x)
=
\frac{1}{1+\exp(-x/\tau)},
$$

donde:

- \(\tau>0\) controla la suavidad de la transicion;
- si \(\tau\) es pequeno, la transicion es mas abrupta;
- si \(\tau\) es grande, la transicion es mas gradual.

La funcion cumple:

$$
0 < \sigma_\tau(x) < 1.
$$

Ademas:

$$
\sigma_\tau(-x)
=
1-\sigma_\tau(x).
$$

Esta propiedad permite definir compuertas complementarias.

## 11. Umbral abstracto de suficiencia global

Se introduce un parametro:

$$
\Phi^\star \ge 0.
$$

Este parametro representa un nivel de potencial global considerado suficientemente bueno para comenzar a priorizar la recuperacion local.

No debe interpretarse como un estado fisico especifico. Es un umbral en el espacio de costos normalizados. Por ejemplo:

- si \(\Phi(k)>\Phi^\star\), el sistema global todavia requiere asistencia;
- si \(\Phi(k)<\Phi^\star\), el sistema global esta suficientemente encaminado y puede abrirse la recuperacion local.

La ventaja de usar \(\Phi^\star\) es que la condicion de fase se expresa con una magnitud general y reusable.

## 12. Compuerta de asistencia

La compuerta de asistencia se define como:

$$
g_A(k)
=
\sigma_{\tau_A}\big(\Phi(k)-\Phi^\star\big).
$$

Esta compuerta es grande cuando el potencial global esta por encima del umbral de suficiencia. En ese caso, el sistema aun necesita avanzar globalmente.

Interpretacion:

- \(g_A(k)\approx 1\): el sistema esta lejos del objetivo global; se permite coordinacion de asistencia.
- \(g_A(k)\approx 0\): el sistema ya esta cerca del objetivo global; la asistencia pierde prioridad.

La compuerta de asistencia no dice que lazo debe actuar. Solo abre o cierra el canal de recompensa coordinativa asociado al progreso global. El reparto entre lazos sigue estando determinado por \(\rho_i(k)\).

## 13. Compuerta de recuperacion

La compuerta de recuperacion se define como:

$$
g_R(k)
=
\sigma_{\tau_R}\big(\Phi^\star-\Phi(k)\big).
$$

Esta compuerta es grande cuando el potencial global esta por debajo del umbral de suficiencia.

Interpretacion:

- \(g_R(k)\approx 0\): el sistema aun no esta suficientemente encaminado; la recuperacion local no debe dominar.
- \(g_R(k)\approx 1\): el sistema ya alcanzo una condicion global razonable; los lazos deben volver a reducir sus costos locales.

Si se elige \(\tau_A=\tau_R=\tau\), entonces:

$$
g_R(k)
=
1-g_A(k).
$$

Esta complementariedad produce una transicion suave entre asistencia y recuperacion.

## 14. Recompensa coordinativa propuesta

La recompensa coordinativa con compuertas se define como:

$$
R_i^{SR}(k)
=
\lambda_A g_A(k)\rho_i(k)\Delta\Phi^+(k)
+
\lambda_R g_R(k)\Delta J_i^+(k),
$$

donde:

- \(R_i^{SR}(k)\) es la recompensa coordinativa de sacrificio-recuperacion para el lazo \(i\);
- \(\lambda_A\ge 0\) regula el peso de asistencia global;
- \(\lambda_R\ge 0\) regula el peso de recuperacion local;
- \(g_A(k)\) abre la fase de asistencia;
- \(g_R(k)\) abre la fase de recuperacion;
- \(\rho_i(k)\) asigna credito coordinativo por contribucion;
- \(\Delta\Phi^+(k)\) mide progreso global positivo;
- \(\Delta J_i^+(k)\) mide mejora local del lazo \(i\).

La recompensa final de cada lazo queda:

$$
R_i^{final}(k)
=
R_i^{local}(k)
+
R_i^{SR}(k).
$$

El termino \(R_i^{local}(k)\) representa la recompensa local principal del lazo. Puede ser una recompensa basada en el costo local, por ejemplo:

$$
R_i^{local}(k)=-J_i(k),
$$

o una transformacion acotada equivalente. La propuesta no exige una forma especifica para la recompensa local.

## 15. Interpretacion del primer termino

El primer termino de \(R_i^{SR}(k)\) es:

$$
\lambda_A g_A(k)\rho_i(k)\Delta\Phi^+(k).
$$

Este termino premia asistencia global. Solo puede ser positivo cuando:

1. el potencial global mejora, es decir, \(\Delta\Phi^+(k)>0\);
2. el canal de asistencia esta abierto, es decir, \(g_A(k)>0\);
3. el lazo recibe credito relativo, es decir, \(\rho_i(k)>0\).

Este termino permite que un lazo reciba credito aunque su costo local no mejore, siempre que haya contribuido a una mejora global.

La recompensa local puede seguir penalizando el deterioro local, pero el termino coordinativo puede compensarlo cuando el deterioro forma parte de una maniobra global util.

## 16. Interpretacion del segundo termino

El segundo termino de \(R_i^{SR}(k)\) es:

$$
\lambda_R g_R(k)\Delta J_i^+(k).
$$

Este termino premia recuperacion local. Solo puede ser positivo cuando:

1. el lazo mejora localmente, es decir, \(\Delta J_i^+(k)>0\);
2. el canal de recuperacion esta abierto, es decir, \(g_R(k)>0\).

La compuerta \(g_R(k)\) evita que la recuperacion local domine demasiado pronto. Si el sistema global aun esta lejos del objetivo, la prioridad sigue siendo el progreso global. Cuando el potencial global disminuye lo suficiente, la recompensa empieza a favorecer el retorno de cada lazo a su setpoint.

Esta es la parte que corrige el problema de "capturar pero no volver". La recompensa no solo premia que el sistema global avance, sino que tambien abre una fase donde el avance local vuelve a ser explicitamente valioso.

## 17. Version con refuerzo explicito del sacrificio util

La formulacion base es deliberadamente simple. Sin embargo, puede agregarse un factor opcional que haga explicito el concepto de sacrificio local.

Se define:

$$
s_i(k)
=
1+
\alpha_S
\frac{\Delta J_i^-(k)}
{\varepsilon_S+\Delta J_i^+(k)+\Delta J_i^-(k)},
$$

donde:

- \(\alpha_S\ge 0\) controla cuanto se amplifica el credito cuando hay deterioro local;
- \(\varepsilon_S>0\) evita division por cero;
- \(\Delta J_i^-(k)\) mide deterioro local.

Como el cociente esta entre cero y uno, se cumple:

$$
1 \le s_i(k) \le 1+\alpha_S.
$$

La recompensa extendida seria:

$$
R_i^{SR+}(k)
=
\lambda_A g_A(k)s_i(k)\rho_i(k)\Delta\Phi^+(k)
+
\lambda_R g_R(k)\Delta J_i^+(k).
$$

Esta version enfatiza la idea de que un lazo que se deteriora localmente mientras contribuye al progreso global puede recibir un reconocimiento adicional. Sin embargo, para una primera implementacion se recomienda comenzar con la version base, porque es mas interpretable y posee menos parametros.

## 18. Propiedad de no negatividad

Se demostrara que \(R_i^{SR}(k)\) es no negativa si sus parametros son no negativos.

Por definicion:

$$
\lambda_A \ge 0,
\quad
\lambda_R \ge 0,
\quad
g_A(k)>0,
\quad
g_R(k)>0,
\quad
\rho_i(k)\ge 0,
\quad
\Delta\Phi^+(k)\ge 0,
\quad
\Delta J_i^+(k)\ge 0.
$$

Por lo tanto:

$$
\lambda_A g_A(k)\rho_i(k)\Delta\Phi^+(k)\ge 0,
$$

y tambien:

$$
\lambda_R g_R(k)\Delta J_i^+(k)\ge 0.
$$

Como \(R_i^{SR}(k)\) es suma de dos terminos no negativos, entonces:

$$
R_i^{SR}(k)\ge 0.
$$

Esta propiedad es importante porque el termino coordinativo actua como bonus. No introduce castigos adicionales que puedan interferir con la recompensa local principal.

## 19. Propiedad de acotamiento

Se supone que los costos locales estan normalizados:

$$
0 \le J_i(k) \le J_{\max}.
$$

Si se usa \(J_{\max}=1\), el analisis se simplifica, pero no es obligatorio.

El potencial global satisface:

$$
0 \le \Phi(k)
=
\sum_i \beta_i J_i(k)
\le
J_{\max}\sum_i\beta_i.
$$

Se define:

$$
\Phi_{\max}
=
J_{\max}\sum_i\beta_i.
$$

Como \(\Delta\Phi^+(k)\) es una disminucion positiva del potencial, se cumple:

$$
0 \le \Delta\Phi^+(k) \le \Phi_{\max}.
$$

De manera analoga:

$$
0 \le \Delta J_i^+(k) \le J_{\max}.
$$

Ademas:

$$
0<g_A(k)<1,
\quad
0<g_R(k)<1,
\quad
0\le \rho_i(k)\le 1.
$$

Por lo tanto:

$$
R_i^{SR}(k)
\le
\lambda_A \Phi_{\max}
+
\lambda_R J_{\max}.
$$

Esto demuestra que la recompensa coordinativa esta acotada si los costos locales estan normalizados. El acotamiento es fundamental para aprendizaje tabular o discretizado, porque evita que la senal de recompensa tenga picos desproporcionados.

## 20. Propiedad de equilibrio en el objetivo

Se considera una condicion ideal en la que todos los lazos estan en su objetivo:

$$
J_i(k)=0
\quad
\forall i\in\mathcal{V}.
$$

Entonces:

$$
\Phi(k)=0.
$$

Si el sistema permanece en el objetivo, tambien:

$$
J_i(k-1)=0,
\quad
\Phi(k-1)=0.
$$

Por lo tanto:

$$
\Delta\Phi^+(k)=\max(0,0-0)=0,
$$

y:

$$
\Delta J_i^+(k)=\max(0,0-0)=0.
$$

Luego:

$$
R_i^{SR}(k)=0.
$$

Esto demuestra que la recompensa coordinativa no crea un incentivo artificial para abandonar el objetivo. Cuando no hay progreso que realizar ni recuperacion local pendiente, el termino coordinativo desaparece.

## 21. Condicion matematica para compensar un sacrificio local util

Se analiza ahora una situacion de asistencia. Supongase que, durante un intervalo \(k\), el lazo \(i\) empeora localmente:

$$
J_i(k)>J_i(k-1).
$$

Entonces:

$$
\Delta J_i(k)<0.
$$

Una recompensa local basada en costo podria disminuir. Si se usa:

$$
R_i^{local}(k)=-J_i(k),
$$

entonces el aumento de \(J_i(k)\) reduce la recompensa local.

Pero si el sistema global mejora:

$$
\Delta\Phi^+(k)>0,
$$

y el lazo recibe credito:

$$
\rho_i(k)>0,
$$

entonces el primer termino coordinativo es positivo:

$$
\lambda_A g_A(k)\rho_i(k)\Delta\Phi^+(k)>0.
$$

La recompensa total mejora respecto a la recompensa puramente local si el bonus coordinativo supera el deterioro local efectivo. Si se denota por \(D_i^{local}(k)\) la perdida local atribuible al sacrificio, la condicion suficiente es:

$$
\lambda_A g_A(k)\rho_i(k)\Delta\Phi^+(k)
>
D_i^{local}(k).
$$

Despejando \(\lambda_A\), se obtiene:

$$
\lambda_A
>
\frac{D_i^{local}(k)}
{g_A(k)\rho_i(k)\Delta\Phi^+(k)}.
$$

Siempre que el denominador sea positivo, existe un valor de \(\lambda_A\) que permite compensar el sacrificio local util.

Esta demostracion no dice que \(\lambda_A\) deba ser grande. Solo demuestra que la arquitectura posee la capacidad matematica de reconocer un sacrificio cuando hay progreso global atribuible.

## 22. Condicion matematica para favorecer el retorno

Se analiza ahora una situacion de recuperacion. Supongase que el sistema global ya esta suficientemente cerca del objetivo:

$$
\Phi(k)<\Phi^\star.
$$

Entonces:

$$
g_R(k)
=
\sigma_{\tau_R}(\Phi^\star-\Phi(k))
>
\frac{1}{2}.
$$

Si el lazo \(i\) reduce su costo local:

$$
J_i(k)<J_i(k-1),
$$

entonces:

$$
\Delta J_i^+(k)>0.
$$

El termino de recuperacion es:

$$
\lambda_R g_R(k)\Delta J_i^+(k)>0.
$$

Si se comparan dos decisiones que producen igual recompensa local inmediata y similar progreso global, pero una reduce \(J_i\) y la otra no, la decision que reduce \(J_i\) recibe mayor recompensa total por una cantidad:

$$
\lambda_R g_R(k)\Delta J_i^+(k).
$$

Por lo tanto, en la fase de recuperacion, la recompensa propuesta ordena preferentemente las decisiones que hacen volver el lazo a su setpoint.

## 23. Transicion suave entre asistencia y recuperacion

Si se usa la misma temperatura \(\tau\) en ambas compuertas:

$$
g_A(k)
=
\sigma_{\tau}(\Phi(k)-\Phi^\star),
$$

$$
g_R(k)
=
\sigma_{\tau}(\Phi^\star-\Phi(k)),
$$

entonces:

$$
g_R(k)=1-g_A(k).
$$

La recompensa puede escribirse como:

$$
R_i^{SR}(k)
=
\lambda_A g_A(k)\rho_i(k)\Delta\Phi^+(k)
+
\lambda_R (1-g_A(k))\Delta J_i^+(k).
$$

Esta expresion muestra que el peso se desplaza gradualmente desde asistencia global hacia recuperacion local.

No existe un cambio discontinuo de objetivo. Esto es deseable porque los agentes con estados discretos y acciones discretas pueden ser sensibles a recompensas abruptas. Una transicion suave reduce la posibilidad de oscilaciones causadas por cambios repentinos de senal.

## 24. Relacion con agentes de observabilidad limitada

La propuesta es especialmente adecuada cuando los agentes tienen observabilidad limitada. Supongase que cada agente observa solamente:

$$
s_a(k)=\text{bin}(\theta_a(k)),
$$

donde:

- \(a\) es un agente;
- \(\theta_a(k)\) es la ganancia controlada por ese agente;
- \(\text{bin}(\cdot)\) es una discretizacion.

Las acciones posibles son:

$$
\mathcal{A}=\{-1,0,+1\},
$$

que representan bajar, mantener o subir una unidad discreta de ganancia.

En este caso, el agente no conoce explicitamente la fase fisica del sistema. Sin embargo, la recompensa coordinativa puede hacer que ciertas acciones de ganancia sean reforzadas en contextos donde producen progreso global o recuperacion local.

El agente no necesita observar \(\Phi(k)\), \(J_i(k)\) ni \(\rho_i(k)\) como estado. Estas magnitudes se usan para calcular la recompensa escalar:

$$
r_a(k)=R_i^{final}(k),
$$

donde el agente \(a\) pertenece al lazo \(i\).

Por lo tanto, la coordinacion ocurre a traves de la evaluacion del resultado, no a traves de una mezcla de observaciones.

## 25. Por que la propuesta no mezcla lazos de forma indebida

La propuesta usa informacion global para calcular recompensa, pero no entrega informacion global como observacion local al agente. Esta diferencia es fundamental.

Mezclar lazos en observacion significaria que un agente del lazo \(i\) recibe directamente variables del lazo \(j\) como parte de su estado:

$$
s_i(k)=\big(\theta_i(k),x_j(k)\big).
$$

Esto aumenta la dimensionalidad, reduce la generalidad y acopla explicitamente la politica del agente con una planta concreta.

La propuesta, en cambio, mantiene:

$$
s_i(k)=\theta_i(k)
\quad \text{o una extension local equivalente}.
$$

La informacion de otros lazos solo aparece comprimida en:

$$
\Phi(k)
=
\sum_j \beta_j J_j(k),
$$

y se usa dentro de una recompensa escalar. El agente no aprende una politica condicionada directamente por el estado fisico de otro lazo. Aprende, en cambio, que ciertos cambios de ganancia tienden a producir mejores consecuencias evaluadas globalmente.

## 26. Discusion de parametros

La propuesta introduce pocos parametros nuevos:

### 26.1. \(\Phi^\star\)

Define el umbral abstracto de suficiencia global. Debe elegirse en el espacio de costos normalizados. Si los costos \(J_i\) estan en \([0,1]\), entonces \(\Phi^\star\) puede interpretarse como una tolerancia global.

Un valor alto abre recuperacion demasiado pronto. Un valor bajo exige demasiado progreso global antes de premiar retorno local.

### 26.2. \(\tau_A\) y \(\tau_R\)

Controlan la suavidad de las compuertas. Valores pequenos crean transiciones mas parecidas a un interruptor. Valores grandes crean transiciones graduales.

Para una primera version, puede usarse:

$$
\tau_A=\tau_R=\tau.
$$

### 26.3. \(\lambda_A\)

Controla la fuerza del credito por asistencia global. Si es demasiado pequeno, el sistema puede no aprender sacrificios utiles. Si es demasiado grande, puede ignorar los costos locales.

### 26.4. \(\lambda_R\)

Controla la fuerza de la recuperacion local. Si es demasiado pequeno, el sistema puede capturar pero no volver. Si es demasiado grande, puede volver demasiado pronto y dificultar la asistencia.

### 26.5. \(\beta_i\)

Define la importancia de cada lazo dentro del potencial global. No necesariamente todos los lazos deben pesar igual. Sin embargo, los pesos deben elegirse en funcion de la tarea global y no como una regla cruzada ad hoc.

## 27. Forma recomendada para una primera version

La version inicial recomendada es:

$$
R_i^{final}(k)
=
R_i^{local}(k)
+
\lambda_A g_A(k)\rho_i(k)\Delta\Phi^+(k)
+
\lambda_R g_R(k)\Delta J_i^+(k).
$$

con:

$$
g_A(k)
=
\sigma_{\tau}\big(\Phi(k)-\Phi^\star\big),
$$

$$
g_R(k)
=
\sigma_{\tau}\big(\Phi^\star-\Phi(k)\big),
$$

$$
\Phi(k)
=
\sum_i\beta_iJ_i(k),
$$

$$
\Delta\Phi^+(k)
=
\max(0,\Phi(k-1)-\Phi(k)),
$$

$$
\Delta J_i^+(k)
=
\max(0,J_i(k-1)-J_i(k)).
$$

Esta forma es suficientemente simple para implementarse y suficientemente expresiva para distinguir asistencia y recuperacion.

## 28. Interpretacion operacional

La recompensa propuesta actua asi:

1. Si el sistema global esta lejos del objetivo, \(g_A(k)\) es alto. En esa fase, el progreso global se premia y se reparte segun credito coordinativo.
2. Si un lazo ayuda al progreso global, recibe parte del bonus aunque localmente no haya mejorado.
3. Cuando el potencial global baja lo suficiente, \(g_R(k)\) aumenta.
4. En esa segunda fase, cada lazo recibe recompensa adicional por reducir su propio costo local.
5. Si el sistema ya esta en equilibrio, ambos progresos son nulos y el bonus desaparece.

Este comportamiento es coherente con la intuicion de una maniobra coordinada:

- primero se permite ayudar;
- luego se exige volver.

## 29. Criterios de consistencia

La propuesta es consistente con el objetivo de agentes autosintonizantes porque:

- usa costos normalizados, no variables fisicas especificas;
- no exige observabilidad global en los agentes;
- conserva recompensas locales por lazo;
- introduce coordinacion solo por recompensa;
- distingue fases sin reglas discretas especificas de planta;
- produce una senal acotada si los costos estan acotados;
- desaparece en equilibrio;
- permite trazabilidad completa por intervalo.

## 30. Variables minimas que deben registrarse

Para validar empiricamente la propuesta, conviene registrar por intervalo:

- \(J_i(k)\): costo local por lazo;
- \(\Phi(k)\): potencial global;
- \(\Delta\Phi(k)\): progreso global firmado;
- \(\Delta\Phi^+(k)\): progreso global premiable;
- \(\Delta J_i(k)\): progreso local firmado;
- \(\Delta J_i^+(k)\): recuperacion local;
- \(g_A(k)\): compuerta de asistencia;
- \(g_R(k)\): compuerta de recuperacion;
- \(c_i(k)\): contribucion local bruta;
- \(\rho_i(k)\): credito coordinativo relativo;
- \(R_i^{SR}(k)\): bonus coordinativo total del lazo;
- \(R_i^{final}(k)\): recompensa final entregada a agentes del lazo.

Estas magnitudes permiten responder si el sistema:

- premia sacrificios utiles;
- abre recuperacion en el momento correcto;
- reparte credito entre lazos de forma razonable;
- evita recompensar acciones sin progreso global;
- incentiva retorno al setpoint cuando la tarea global ya esta encaminada.

## 31. Conclusion

La Recompensa Coordinativa con Compuertas de Sacrificio y Recuperacion propone una forma simple de hacer que una recompensa coordinativa sea dinamica sin abandonar la generalidad.

La idea central es reemplazar una coordinacion estatica por una coordinacion con fases suaves. Mientras el sistema global esta lejos del objetivo, la recompensa favorece asistencia global. Cuando el potencial global cae por debajo de un umbral abstracto de suficiencia, la recompensa favorece recuperacion local.

Matematicamente, la propuesta se apoya en costos locales normalizados, un potencial global, progreso global positivo, progreso local positivo y compuertas logisticas. La demostracion muestra que el bonus es no negativo, acotado, nulo en equilibrio y capaz de compensar sacrificios locales utiles bajo una condicion explicita de ganancia.

La propuesta permite declarar de manera precisa el comportamiento deseado:

> Un lazo puede ser premiado por ayudar al sistema global aun si temporalmente empeora su costo local, pero una vez que el sistema global esta suficientemente encaminado, el mismo lazo debe ser premiado por volver a reducir su propio costo.

Esta frase resume la innovacion algoritimica de la recompensa.
