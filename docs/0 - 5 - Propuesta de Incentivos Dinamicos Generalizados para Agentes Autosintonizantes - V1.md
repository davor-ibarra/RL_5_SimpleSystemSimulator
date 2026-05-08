# Propuesta de Incentivos Dinamicos Generalizados para Agentes Autosintonizantes - V1

## 1. Proposito del documento

Este documento desarrolla una propuesta matematica para integrar incentivos dinamicos en un sistema de agentes autosintonizantes. La propuesta se formula de manera general y no depende de una planta fisica particular.

La motivacion principal es ampliar la idea de incentivo dinamico. Un incentivo dinamico no tiene por que depender solamente de otra variable fisica del sistema. Tambien puede depender de:

- el desempeno del propio lazo;
- la recompensa local del propio lazo;
- el progreso local;
- la recompensa coordinativa recibida;
- el progreso global;
- la confianza, saturacion, estabilidad o cualquier metrica abstracta de desempeno.

Por lo tanto, el incentivo dinamico se entiende como una pieza de recompensa adicional que depende de una variable moduladora. Dicha variable moduladora puede ser fisica, local, global, coordinativa o incluso una recompensa previamente calculada.

El documento tambien distingue cuidadosamente entre:

- **compuertas**, que modulan recompensas ya existentes;
- **incentivos dinamicos**, que agregan terminos nuevos de recompensa asociados a una relacion deseada entre variables.

Esta distincion es importante porque ambos mecanismos pueden parecer similares al escribirse como productos matematicos, pero cumplen funciones conceptuales distintas.

## 2. Contexto de agentes autosintonizantes

Se considera un conjunto de lazos de control:

$$
\mathcal{V}=\{1,2,\dots,m\}.
$$

Cada lazo \(i\in\mathcal{V}\) posee un controlador local. Los parametros del controlador se ajustan mediante agentes de aprendizaje por refuerzo. Un agente puede controlar una ganancia, un coeficiente o cualquier parametro ajustable.

En el caso de agentes de observabilidad limitada, cada agente puede observar solamente su propia ganancia discretizada:

$$
s_a(k)=\text{bin}(\theta_a(k)),
$$

donde:

- \(a\) identifica al agente;
- \(\theta_a(k)\) es el parametro controlado por el agente;
- \(\text{bin}(\cdot)\) es una discretizacion.

Las acciones pueden ser:

$$
\mathcal{A}=\{-1,0,+1\},
$$

que representan bajar, mantener o subir un paso de ganancia.

La limitada observabilidad no impide calcular recompensas ricas. La recompensa puede usar informacion del desempeno del sistema, siempre que esa informacion no se introduzca como estado del agente si el diseno busca evitar mezcla de lazos en observacion.

## 3. Recompensa base y necesidad de incentivos dinamicos

Se supone que cada lazo posee una recompensa local:

$$
R_i^{local}(k).
$$

Esta recompensa mide el desempeno del lazo \(i\) durante el intervalo \(k\). En general, puede depender de un costo local:

$$
J_i(k)\ge 0.
$$

Una forma simple es:

$$
R_i^{local}(k)=-J_i(k).
$$

Tambien puede existir una recompensa coordinativa:

$$
R_i^{coord}(k),
$$

que representa credito recibido por contribuir al progreso global del sistema.

La recompensa total basica podria ser:

$$
R_i^{base}(k)
=
R_i^{local}(k)
+
R_i^{coord}(k).
$$

Sin embargo, esta forma puede no ser suficiente cuando se desea reforzar patrones mas especificos de conducta, por ejemplo:

- mejorar el propio lazo despues de haber recibido credito coordinativo;
- reducir esfuerzo cuando el lazo ya esta cerca del objetivo;
- aumentar respuesta cuando el costo local es alto;
- premiar alineamiento entre una accion local y un objetivo abstracto;
- reforzar consistencia entre progreso global y progreso local;
- transformar una recompensa coordinativa en una senal de aprendizaje mas informativa para el retorno al setpoint.

Para esto se introducen incentivos dinamicos.

## 4. Definicion general de incentivo dinamico

Un incentivo dinamico para el lazo \(i\) se define como:

$$
I_i(k)
=
\omega_i
\Gamma_i(z_i(k))
\Omega_i(y_i(k),y_i^\star(z_i(k))),
$$

donde:

- \(I_i(k)\) es el incentivo dinamico agregado al lazo \(i\);
- \(\omega_i\ge 0\) es el peso del incentivo;
- \(z_i(k)\) es la variable moduladora;
- \(\Gamma_i(\cdot)\) es una funcion de activacion o intensidad;
- \(y_i(k)\) es la respuesta observada que se desea evaluar;
- \(y_i^\star(z_i(k))\) es una respuesta deseada que puede depender de la variable moduladora;
- \(\Omega_i(\cdot,\cdot)\) mide compatibilidad entre la respuesta observada y la respuesta deseada.

La recompensa final queda:

$$
R_i^{final}(k)
=
R_i^{base}(k)
+
I_i(k).
$$

Esta formulacion es general. Permite expresar distintos incentivos cambiando \(z_i\), \(\Gamma_i\), \(y_i\), \(y_i^\star\) y \(\Omega_i\).

## 5. Interpretacion de cada componente

### 5.1. Variable moduladora \(z_i(k)\)

La variable moduladora es la magnitud que decide en que contexto se activa o modifica el incentivo.

Puede ser:

- una variable fisica;
- un costo local;
- una recompensa local;
- una recompensa coordinativa;
- una medida de progreso global;
- una medida de progreso local;
- una medida de saturacion;
- una medida de incertidumbre o aprendizaje.

El punto central es que \(z_i(k)\) no tiene que ser una variable fisica de otro lazo.

### 5.2. Funcion de activacion \(\Gamma_i(z_i(k))\)

La funcion \(\Gamma_i\) transforma la variable moduladora en una intensidad. Usualmente se elige acotada:

$$
0\le \Gamma_i(z)\le 1.
$$

Ejemplos comunes:

$$
\Gamma_i(z)=\sigma_\tau(z-z^\star),
$$

o:

$$
\Gamma_i(z)=\exp\left(-\left(\frac{z-z^\star}{\sigma_z}\right)^2\right).
$$

La primera forma aumenta cuando \(z\) supera un umbral. La segunda forma premia cercania a un valor deseado.

### 5.3. Respuesta observada \(y_i(k)\)

La respuesta observada es aquello que se desea reforzar. Puede ser:

- progreso local \(\Delta J_i^+(k)\);
- reduccion de esfuerzo;
- alineamiento entre error y accion;
- recompensa local positiva;
- cambio positivo en la recompensa local;
- accion efectiva local;
- suavidad de control.

### 5.4. Respuesta deseada \(y_i^\star(z_i(k))\)

La respuesta deseada puede ser constante o depender de la variable moduladora. Si depende de \(z_i\), el incentivo se vuelve adaptativo.

Por ejemplo, podria definirse:

$$
y_i^\star(z)
=
y_{\max}\tanh(\alpha z),
$$

lo que significa que la respuesta deseada aumenta suavemente con la magnitud del modulador.

### 5.5. Funcion de compatibilidad \(\Omega_i\)

La funcion \(\Omega_i\) mide si la respuesta observada se parece a la respuesta deseada. Puede tomar distintas formas.

Una forma tipo seguimiento es:

$$
\Omega_i(y,y^\star)
=
\exp\left(
-\left(\frac{y-y^\star}{\sigma_y}\right)^2
\right).
$$

Una forma direccional es:

$$
\Omega_i(y,y^\star)
=
\max(0, yy^\star).
$$

Una forma basada en mejora positiva es:

$$
\Omega_i(y,y^\star)=y,
\quad
y=\Delta J_i^+(k).
$$

La eleccion depende de la conducta que se desea reforzar.

## 6. Familias de variables moduladoras

La propuesta distingue cuatro familias principales de moduladores.

## 6.1. Modulador fisico

Un modulador fisico usa una variable del sistema:

$$
z_i(k)=x_j(k),
$$

donde \(x_j(k)\) puede pertenecer al mismo lazo o a otro lazo.

Esta familia es intuitiva porque permite expresar relaciones fisicas directas. Por ejemplo, un lazo puede ser incentivado a actuar con mayor fuerza cuando cierta variable fisica esta lejos de su region deseada.

Sin embargo, esta opcion tiene un costo de generalidad. Si el incentivo depende de una variable fisica especifica, la recompensa queda mas ligada a una planta concreta.

### Ventajas

- Puede ser muy efectiva cuando se conoce bien la fisica del sistema.
- Permite capturar relaciones direccionales claras.
- Puede acelerar el aprendizaje en una planta particular.

### Riesgos

- Puede convertirse en una regla ad hoc.
- Puede mezclar lazos de manera dificil de justificar.
- Puede no transferirse a otros sistemas fisicos.
- Puede contradecir el objetivo de alta generalidad.

Por ello, los moduladores fisicos deben usarse con cuidado.

## 6.2. Modulador de desempeno local

Un modulador de desempeno local usa el costo o recompensa del propio lazo:

$$
z_i(k)=J_i(k),
$$

o:

$$
z_i(k)=R_i^{local}(k).
$$

Esta familia no mezcla lazos. El incentivo se adapta al estado abstracto del propio lazo.

Un ejemplo es:

$$
I_i^{self}(k)
=
\omega_i
\sigma_\tau(J_i(k)-J_i^\star)
\Delta J_i^+(k).
$$

Interpretacion:

- si el costo local \(J_i(k)\) es alto, la funcion \(\sigma_\tau(J_i(k)-J_i^\star)\) se activa;
- si el lazo reduce su costo, \(\Delta J_i^+(k)>0\);
- el incentivo premia mejorar cuando el propio lazo esta mal.

Este incentivo no requiere informacion de otro lazo ni de una variable fisica particular. Es general para cualquier lazo con costo local normalizado.

## 6.3. Modulador de recompensa local

Otra opcion es usar la recompensa local como modulador:

$$
z_i(k)=R_i^{local}(k).
$$

Como muchas recompensas locales son negativas cuando el costo es alto, puede convenir usar:

$$
z_i(k)=-R_i^{local}(k).
$$

Si:

$$
R_i^{local}(k)=-J_i(k),
$$

entonces:

$$
-R_i^{local}(k)=J_i(k).
$$

Esto muestra que usar recompensa local o costo local puede ser equivalente bajo ciertas definiciones.

Un incentivo posible es:

$$
I_i^{localR}(k)
=
\omega_i
\sigma_\tau(-R_i^{local}(k)-\eta_i)
\Delta R_i^{local,+}(k),
$$

donde:

$$
\Delta R_i^{local,+}(k)
=
\max(0,R_i^{local}(k)-R_i^{local}(k-1)).
$$

Este incentivo premia aumentos de recompensa local cuando la recompensa local aun es mala.

## 6.4. Modulador de recompensa coordinativa

Una opcion especialmente relevante es usar la recompensa coordinativa como modulador:

$$
z_i(k)=R_i^{coord}(k).
$$

Esto permite construir incentivos del tipo:

$$
I_i^{coordR}(k)
=
\omega_i
\sigma_\tau(R_i^{coord}(k)-\eta_C)
\Delta J_i^+(k).
$$

Interpretacion:

- si el lazo recibio recompensa coordinativa, significa que participo en una mejora global;
- si despues o durante esa condicion reduce su costo local, recibe un incentivo adicional;
- la recompensa coordinativa se transforma en una senal que activa retorno local.

Esta idea es muy importante porque no requiere usar directamente otra variable fisica. La variable moduladora es una recompensa ya construida por el sistema de coordinacion.

Conceptualmente, el incentivo dice:

> Cuando un lazo ya fue reconocido por ayudar al sistema global, se le puede incentivar a consolidar esa ayuda recuperando su propio desempeno local.

Esta formulacion es general y compatible con el principio de no mezclar observaciones entre lazos.

## 6.5. Modulador de progreso global

Tambien se puede usar el progreso global:

$$
z_i(k)=\Delta\Phi^+(k).
$$

Un incentivo posible es:

$$
I_i^{globalP}(k)
=
\omega_i
\sigma_\tau(\Delta\Phi^+(k)-\eta_\Phi)
\Delta J_i^+(k).
$$

Este incentivo premia recuperacion local cuando existe progreso global suficiente.

La diferencia con usar \(R_i^{coord}(k)\) es que \(\Delta\Phi^+(k)\) mide progreso global sin asignacion de credito, mientras que \(R_i^{coord}(k)\) ya contiene informacion de cuanto credito recibio el lazo \(i\).

Por lo tanto:

- usar \(\Delta\Phi^+(k)\) es mas global y menos personalizado;
- usar \(R_i^{coord}(k)\) es mas especifico al lazo.

## 7. Diferencia matematica entre compuerta e incentivo dinamico

Una compuerta tiene la forma:

$$
\widetilde R_i(k)
=
g_i(z_i(k))R_i(k).
$$

Aqui:

- \(R_i(k)\) ya existe;
- \(g_i(z_i(k))\) solamente cambia su intensidad;
- si \(R_i(k)=0\), la compuerta no puede crear recompensa nueva.

Un incentivo dinamico tiene la forma:

$$
R_i^{final}(k)
=
R_i(k)+I_i(k),
$$

con:

$$
I_i(k)
=
\omega_i\Gamma_i(z_i(k))\Omega_i(y_i(k),y_i^\star(z_i(k))).
$$

Aqui:

- el incentivo agrega un termino nuevo;
- puede crear recompensa aunque la recompensa base sea cero;
- puede reforzar una relacion especifica entre contexto y respuesta.

La diferencia esencial es:

> La compuerta decide cuando pesa una recompensa existente. El incentivo dinamico decide que relacion adicional merece recompensa.

Aunque ambos mecanismos pueden usar funciones como \(\sigma(\cdot)\), no son equivalentes en significado.

## 8. Diferencia conceptual entre usar otra variable fisica, el propio lazo o la coordinacion

### 8.1. Incentivo respecto a otra variable fisica

La forma general es:

$$
I_i^{phys}(k)
=
\omega_i
\Gamma(x_j(k))
\Omega(y_i(k),y_i^\star(x_j(k))).
$$

Esto significa que el comportamiento deseado del lazo \(i\) depende directamente de una variable fisica \(x_j\).

Esta forma puede ser potente, pero tambien es la menos general. Introduce conocimiento de planta y puede mezclar lazos en la recompensa.

No necesariamente es incorrecta. Puede ser apropiada si la tesis desea declarar un mecanismo especializado para una familia de sistemas. Pero si la propuesta busca generalidad fuerte, esta opcion debe quedar como una variante secundaria o experimental.

### 8.2. Incentivo respecto al propio lazo

La forma general es:

$$
I_i^{self}(k)
=
\omega_i
\Gamma(J_i(k))
\Omega(\Delta J_i^+(k),0).
$$

En este caso, el incentivo no depende de otro lazo. Solo dice que el lazo debe mejorar cuando su propio desempeno es malo.

Esta opcion es muy general y robusta. Sin embargo, por si sola no resuelve coordinacion entre lazos, porque no reconoce que un sacrificio local pueda ayudar al sistema global.

### 8.3. Incentivo respecto a recompensa coordinativa

La forma general es:

$$
I_i^{coordR}(k)
=
\omega_i
\Gamma(R_i^{coord}(k))
\Omega(\Delta J_i^+(k),0).
$$

Este incentivo usa la coordinacion como variable moduladora, no como observacion fisica.

Su interpretacion es:

- el lazo recibio credito coordinativo;
- ese credito indica que su accion tuvo valor global;
- luego se refuerza que el lazo consolide ese valor reduciendo su propio costo.

Esta opcion es una de las mas interesantes para agentes autosintonizantes de observabilidad limitada, porque crea un puente entre ayuda global y retorno local sin entregar estados fisicos cruzados al agente.

## 9. Propuesta A: Incentivo dinamico de recuperacion post-coordinativa

Se propone el incentivo:

$$
I_i^{postC}(k)
=
\omega_C
\sigma_\tau(R_i^{coord}(k)-\eta_C)
\Delta J_i^+(k).
$$

donde:

- \(\omega_C\ge 0\) es el peso del incentivo;
- \(R_i^{coord}(k)\) es la recompensa coordinativa recibida por el lazo \(i\);
- \(\eta_C\ge 0\) es un umbral minimo de coordinacion;
- \(\Delta J_i^+(k)\) es la mejora local del lazo \(i\).

### Interpretacion

Este incentivo premia al lazo cuando mejora localmente en una situacion donde tambien recibio credito coordinativo.

No premia simplemente moverse. No premia simplemente recibir coordinacion. Premia la combinacion:

$$
\text{credito coordinativo} + \text{recuperacion local}.
$$

Esto es distinto a una compuerta de recuperacion, porque el modulador no es directamente el potencial global \(\Phi(k)\), sino la recompensa coordinativa atribuida al lazo.

### Demostracion de no negatividad

Como:

$$
\omega_C\ge 0,
$$

$$
0<\sigma_\tau(R_i^{coord}(k)-\eta_C)<1,
$$

y:

$$
\Delta J_i^+(k)\ge 0,
$$

se cumple:

$$
I_i^{postC}(k)\ge 0.
$$

### Demostracion de acotamiento

Si:

$$
0\le J_i(k)\le J_{\max},
$$

entonces:

$$
0\le \Delta J_i^+(k)\le J_{\max}.
$$

Como la sigmoide es menor que uno:

$$
I_i^{postC}(k)
\le
\omega_CJ_{\max}.
$$

Por lo tanto, el incentivo esta acotado.

### Utilidad

Este incentivo es apropiado cuando el problema observado es:

> El sistema aprende a producir progreso global, pero no consolida ese progreso haciendo que los lazos vuelvan a su objetivo local.

## 10. Propuesta B: Incentivo dinamico de mejora local activada por costo propio

Se propone:

$$
I_i^{selfJ}(k)
=
\omega_J
\sigma_\tau(J_i(k)-J_i^\star)
\Delta J_i^+(k).
$$

donde:

- \(J_i(k)\) es el costo local;
- \(J_i^\star\) es el nivel de costo a partir del cual se desea incentivar mejora;
- \(\Delta J_i^+(k)\) es la mejora local.

### Interpretacion

Cuando el lazo esta mal, la compuerta interna del incentivo se activa. Si el lazo mejora, recibe recompensa adicional.

Este incentivo es autorreferencial y local. No usa coordinacion ni variables fisicas de otros lazos.

### Diferencia con la recompensa local

Si ya existe:

$$
R_i^{local}(k)=-J_i(k),
$$

podria parecer que \(I_i^{selfJ}\) es redundante. No lo es necesariamente.

La recompensa local castiga el estado actual del costo. El incentivo \(I_i^{selfJ}\) premia la reduccion del costo:

$$
\Delta J_i^+(k)=\max(0,J_i(k-1)-J_i(k)).
$$

Por lo tanto, introduce una senal de progreso, no solo una senal de estado.

Esta diferencia puede ser importante para agentes con observabilidad limitada, porque el agente puede aprender que ciertos cambios de ganancia producen mejoras, aunque el costo absoluto siga siendo alto.

## 11. Propuesta C: Incentivo dinamico de alineamiento entre recompensa local y coordinativa

Se propone:

$$
I_i^{align}(k)
=
\omega_A
[R_i^{coord}(k)]_+
[\Delta R_i^{local}(k)]_+,
$$

donde:

$$
[x]_+=\max(0,x),
$$

y:

$$
\Delta R_i^{local}(k)
=
R_i^{local}(k)-R_i^{local}(k-1).
$$

### Interpretacion

El incentivo premia cuando ocurren simultaneamente dos cosas:

1. el lazo recibe credito coordinativo positivo;
2. la recompensa local del lazo mejora.

El producto actua como una medida de alineamiento:

- si hay coordinacion sin mejora local, el incentivo es cero;
- si hay mejora local sin coordinacion, el incentivo es cero;
- si ambas ocurren, el incentivo es positivo.

Este incentivo es util para consolidar comportamientos que son buenos local y globalmente.

### Demostracion de acotamiento

Supongase:

$$
0\le R_i^{coord}(k)\le C_{\max},
$$

y:

$$
0\le [\Delta R_i^{local}(k)]_+\le L_{\max}.
$$

Entonces:

$$
0\le I_i^{align}(k)
\le
\omega_A C_{\max}L_{\max}.
$$

Por lo tanto, el incentivo esta acotado si sus componentes estan acotadas.

## 12. Propuesta D: Incentivo dinamico de seguimiento adaptativo

Algunos incentivos no buscan simplemente premiar mejora, sino que una respuesta local siga una referencia dinamica. La forma general es:

$$
I_i^{track}(k)
=
\omega_T
\Gamma_i(z_i(k))
\exp\left(
-\left(
\frac{y_i(k)-f_i(z_i(k))}
{\sigma_y}
\right)^2
\right).
$$

donde:

- \(z_i(k)\) es el modulador;
- \(y_i(k)\) es la respuesta local;
- \(f_i(z_i(k))\) es la respuesta deseada;
- \(\sigma_y>0\) controla tolerancia.

Una eleccion simple para la referencia es:

$$
f_i(z)
=
y_{\max}\tanh(\alpha z).
$$

### Interpretacion

Si el modulador crece, la respuesta deseada aumenta suavemente. El incentivo premia que la respuesta real se acerque a esa referencia.

Esta forma es mas expresiva, pero tambien mas delicada. Requiere definir que respuesta se desea seguir y por que.

### Comparacion con las propuestas anteriores

Las propuestas \(I_i^{postC}\), \(I_i^{selfJ}\) e \(I_i^{align}\) son mas simples porque usan progreso o recompensa directamente. La propuesta de seguimiento adaptativo es mas flexible, pero puede ser menos interpretable si no se justifica bien la funcion \(f_i\).

## 13. Propuesta E: Incentivo dinamico basado en recompensa coordinativa acumulada

Puede ser util recordar si un lazo recibio coordinacion recientemente. Se define una traza:

$$
M_i(k)
=
\gamma_M M_i(k-1)
+
R_i^{coord,+}(k),
$$

donde:

- \(0\le \gamma_M<1\) es un factor de memoria;
- \(R_i^{coord,+}(k)=\max(0,R_i^{coord}(k))\).

Luego:

$$
I_i^{memC}(k)
=
\omega_M
\sigma_\tau(M_i(k)-\eta_M)
\Delta J_i^+(k).
$$

### Interpretacion

Este incentivo no exige que la recuperacion local ocurra exactamente en el mismo intervalo en que se recibio coordinacion. Permite que la coordinacion deje una memoria breve.

Esto puede ser importante si el sistema fisico tiene retardos o si la recuperacion natural toma varios intervalos.

### Propiedad de acotamiento de la traza

Si:

$$
0\le R_i^{coord,+}(k)\le C_{\max},
$$

entonces:

$$
M_i(k)
\le
\gamma_M M_i(k-1)+C_{\max}.
$$

En regimen acotado:

$$
M_i(k)
\le
\frac{C_{\max}}{1-\gamma_M}.
$$

Por lo tanto, la memoria coordinativa permanece acotada si \(0\le\gamma_M<1\).

## 14. Como difieren estos incentivos de las compuertas de sacrificio y recuperacion

La recompensa con compuertas de sacrificio y recuperacion usa el potencial global para distribuir fases:

$$
R_i^{SR}(k)
=
\lambda_A g_A(k)\rho_i(k)\Delta\Phi^+(k)
+
\lambda_R g_R(k)\Delta J_i^+(k).
$$

El rol principal de \(g_A\) y \(g_R\) es decidir que tipo de objetivo pesa mas en cada momento:

- asistencia global;
- recuperacion local.

En cambio, un incentivo dinamico como:

$$
I_i^{postC}(k)
=
\omega_C
\sigma_\tau(R_i^{coord}(k)-\eta_C)
\Delta J_i^+(k)
$$

no solamente abre una fase. Agrega una recompensa nueva que expresa una relacion:

> si el lazo recibio coordinacion, entonces se premia que mejore localmente.

La diferencia es sutil pero importante:

- la compuerta organiza fases de la recompensa;
- el incentivo dinamico define una conducta adicional deseada.

En terminos practicos, las compuertas son mas adecuadas cuando se desea modular una recompensa general. Los incentivos dinamicos son mas adecuados cuando se desea reforzar una asociacion particular entre una condicion y una respuesta.

## 15. Clasificacion segun generalidad

Las variantes pueden ordenarse desde mas general a mas especifica:

1. Incentivos basados en \(J_i(k)\), \(\Delta J_i(k)\) o \(R_i^{local}(k)\). Son puramente locales y generales.
2. Incentivos basados en \(R_i^{coord}(k)\) o \(\Delta\Phi(k)\). Son coordinativos y generales, porque usan senales abstractas.
3. Incentivos basados en costos de otros lazos \(J_j(k)\). Son cruzados, pero aun abstractos.
4. Incentivos basados en variables fisicas \(x_j(k)\). Son los mas especificos y menos transferibles.

Esta clasificacion no significa que las opciones especificas sean incorrectas. Significa que deben justificarse con mas cuidado.

## 16. Criterio para elegir la variable moduladora

La variable moduladora debe elegirse segun la pregunta que se quiere responder.

### Si la pregunta es:

> Cuando el propio lazo esta mal, como se premia que mejore?

Entonces conviene usar:

$$
z_i(k)=J_i(k)
$$

o:

$$
z_i(k)=R_i^{local}(k).
$$

### Si la pregunta es:

> Cuando el lazo ayudo globalmente, como se premia que vuelva?

Entonces conviene usar:

$$
z_i(k)=R_i^{coord}(k)
$$

o una memoria de coordinacion:

$$
z_i(k)=M_i(k).
$$

### Si la pregunta es:

> Cuando el sistema global avanza, como se premia que los lazos consoliden?

Entonces conviene usar:

$$
z_i(k)=\Delta\Phi^+(k).
$$

### Si la pregunta es:

> Cuando una variable fisica especifica tiene cierto valor, como debe responder otro lazo?

Entonces puede usarse:

$$
z_i(k)=x_j(k).
$$

Pero esta ultima opcion debe declararse como una especializacion fisica, no como el nucleo general de la propuesta.

## 17. Recomendacion para una propuesta robusta

La version mas robusta y general de incentivos dinamicos para agentes autosintonizantes deberia comenzar con moduladores abstractos:

1. costo local \(J_i(k)\);
2. progreso local \(\Delta J_i^+(k)\);
3. recompensa coordinativa \(R_i^{coord}(k)\);
4. progreso global \(\Delta\Phi^+(k)\).

Una propuesta compacta podria ser:

$$
R_i^{final}(k)
=
R_i^{local}(k)
+
R_i^{coord}(k)
+
I_i^{selfJ}(k)
+
I_i^{postC}(k).
$$

con:

$$
I_i^{selfJ}(k)
=
\omega_J
\sigma_\tau(J_i(k)-J_i^\star)
\Delta J_i^+(k),
$$

y:

$$
I_i^{postC}(k)
=
\omega_C
\sigma_\tau(R_i^{coord}(k)-\eta_C)
\Delta J_i^+(k).
$$

Esta combinacion dice:

- si el propio lazo esta mal, se premia que mejore;
- si el lazo recibio coordinacion, se premia que consolide esa coordinacion mejorando localmente.

Ambos incentivos usan magnitudes abstractas y no requieren variables fisicas cruzadas.

## 18. Demostracion de acotamiento de la recompensa final

Supongase:

$$
|R_i^{local}(k)|\le L_{\max},
$$

$$
0\le R_i^{coord}(k)\le C_{\max},
$$

$$
0\le J_i(k)\le J_{\max},
$$

$$
0\le \Delta J_i^+(k)\le J_{\max}.
$$

Para los incentivos recomendados:

$$
0\le I_i^{selfJ}(k)\le \omega_JJ_{\max},
$$

y:

$$
0\le I_i^{postC}(k)\le \omega_CJ_{\max}.
$$

Por lo tanto:

$$
|R_i^{final}(k)|
\le
L_{\max}
+
C_{\max}
+
\omega_JJ_{\max}
+
\omega_CJ_{\max}.
$$

Esto demuestra que la recompensa final permanece acotada si sus componentes base estan acotadas.

## 19. Demostracion de consistencia en equilibrio

Se considera una condicion ideal donde el lazo esta en su objetivo y no hay progreso pendiente:

$$
J_i(k)=J_i(k-1)=0.
$$

Entonces:

$$
\Delta J_i^+(k)=0.
$$

Por lo tanto:

$$
I_i^{selfJ}(k)=0,
$$

y:

$$
I_i^{postC}(k)=0.
$$

Si ademas no hay progreso global pendiente, entonces:

$$
R_i^{coord}(k)=0.
$$

La recompensa final se reduce a la recompensa local en equilibrio:

$$
R_i^{final}(k)=R_i^{local}(k).
$$

Si la recompensa local tambien es cero o maxima en el objetivo, no aparece ningun incentivo dinamico para abandonar el setpoint.

## 20. Demostracion de preferencia por recuperacion tras coordinacion

Se comparan dos acciones \(a\) y \(b\) de un agente del lazo \(i\). Supongase que ambas producen la misma recompensa local y la misma recompensa coordinativa:

$$
R_i^{local,a}(k)=R_i^{local,b}(k),
$$

$$
R_i^{coord,a}(k)=R_i^{coord,b}(k).
$$

Pero la accion \(a\) produce mayor mejora local:

$$
\Delta J_i^{+,a}(k)
>
\Delta J_i^{+,b}(k).
$$

Para el incentivo post-coordinativo:

$$
I_i^{postC,a}(k)-I_i^{postC,b}(k)
=
\omega_C
\sigma_\tau(R_i^{coord}(k)-\eta_C)
\left(
\Delta J_i^{+,a}(k)
-
\Delta J_i^{+,b}(k)
\right).
$$

Si:

$$
\omega_C>0,
$$

y:

$$
\sigma_\tau(R_i^{coord}(k)-\eta_C)>0,
$$

entonces:

$$
I_i^{postC,a}(k)>I_i^{postC,b}(k).
$$

Por lo tanto:

$$
R_i^{final,a}(k)>R_i^{final,b}(k).
$$

Esto demuestra que, bajo igualdad de recompensa local y coordinativa, el incentivo dinamico prefiere la accion que recupera mejor el lazo.

## 21. Riesgos de diseno

Los incentivos dinamicos deben ser simples y acotados. Si se agregan demasiados incentivos, la recompensa puede volverse dificil de interpretar.

Los riesgos principales son:

- duplicar una senal que ya existe en la recompensa local;
- hacer que el incentivo domine la recompensa principal;
- usar variables fisicas demasiado especificas;
- introducir reglas cruzadas que reduzcan generalidad;
- premiar correlaciones espurias;
- provocar oscilaciones si la activacion es abrupta;
- generar recompensas positivas permanentes aun cuando no hay progreso.

Para evitar estos riesgos, cada incentivo debe cumplir:

1. tener una interpretacion verbal clara;
2. depender de variables normalizadas;
3. estar acotado;
4. desaparecer cuando no hay progreso relevante;
5. registrar sus componentes para trazabilidad;
6. probarse primero de manera aislada.

## 22. Diferencia practica entre tres alternativas

Se comparan tres formas de resolver el problema de "ayudar y luego volver".

### 22.1. Compuertas de sacrificio y recuperacion

Forma:

$$
R_i^{SR}(k)
=
\lambda_A g_A(k)\rho_i(k)\Delta\Phi^+(k)
+
\lambda_R g_R(k)\Delta J_i^+(k).
$$

Caracter:

- organiza fases;
- usa potencial global;
- es general;
- modula asistencia y recuperacion.

### 22.2. Incentivo post-coordinativo

Forma:

$$
I_i^{postC}(k)
=
\omega_C
\sigma_\tau(R_i^{coord}(k)-\eta_C)
\Delta J_i^+(k).
$$

Caracter:

- no organiza fases globales directamente;
- usa la recompensa coordinativa recibida como senal;
- premia que quien ayudo tambien se recupere;
- es muy compatible con coordinacion por credito.

### 22.3. Incentivo fisico cruzado

Forma:

$$
I_i^{phys}(k)
=
\omega_i
\Gamma(x_j(k))
\Omega(y_i(k),y_i^\star(x_j(k))).
$$

Caracter:

- puede ser muy efectivo;
- depende de una relacion fisica concreta;
- es menos general;
- debe justificarse como especializacion.

## 23. Recomendacion conceptual final

La propuesta mas consistente para agentes autosintonizantes con observabilidad limitada es desarrollar los incentivos dinamicos primero sobre recompensas y costos abstractos, no sobre variables fisicas cruzadas.

La forma recomendada es:

$$
R_i^{final}(k)
=
R_i^{local}(k)
+
R_i^{coord}(k)
+
\omega_J
\sigma_\tau(J_i(k)-J_i^\star)
\Delta J_i^+(k)
+
\omega_C
\sigma_\tau(R_i^{coord}(k)-\eta_C)
\Delta J_i^+(k).
$$

Esta expresion puede leerse asi:

1. El lazo recibe su recompensa local.
2. El lazo recibe credito coordinativo si contribuyo al progreso global.
3. Si su propio costo es alto y mejora, recibe un incentivo local de progreso.
4. Si recibio credito coordinativo y mejora localmente, recibe un incentivo de consolidacion.

La propuesta no requiere que el agente observe variables fisicas de otros lazos. La coordinacion sigue encapsulada en la recompensa.

## 24. Conclusion

Los incentivos dinamicos son una familia mas amplia que las compuertas. Una compuerta decide cuando una recompensa existente debe pesar mas o menos. Un incentivo dinamico agrega una recompensa nueva para reforzar una relacion entre una variable moduladora y una respuesta deseada.

La variable moduladora puede ser una variable fisica, pero no tiene por que serlo. Tambien puede ser el costo local, la recompensa local, el progreso local, la recompensa coordinativa o el progreso global.

Para una propuesta general de agentes autosintonizantes, los moduladores mas robustos son aquellos construidos desde magnitudes abstractas de desempeno. Entre ellos, el uso de \(R_i^{coord}(k)\) como modulador es especialmente prometedor, porque permite que el sistema convierta el credito global recibido por un lazo en un incentivo posterior de recuperacion local.

La idea central puede resumirse asi:

> Un incentivo dinamico no solamente cambia el peso de una recompensa; declara que, bajo cierto contexto medido por una variable moduladora, una respuesta especifica merece credito adicional.

Esta formulacion permite disenar recompensas flexibles, interpretables y matematicamente consistentes sin abandonar la separacion entre lazos ni depender obligatoriamente de variables fisicas particulares.
