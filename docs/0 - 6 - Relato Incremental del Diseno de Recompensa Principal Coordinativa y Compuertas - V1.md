# Relato Incremental del Diseno de Recompensa Principal, Coordinativa y Compuertas - V1

## 1. Proposito del documento

Este documento presenta un relato incremental del diseno de recompensa para agentes autosintonizantes de controladores locales. La exposicion parte desde los metodos que ya se encuentran implementados en la arquitectura actual y, sobre esa base, desarrolla las extensiones conceptuales que conviene formular de manera matematica.

La intencion no es comenzar desde una propuesta aislada. La intencion es ordenar el camino completo:

1. Primero se presenta la recompensa principal por lazo.
2. Luego se presentan las compuertas de la recompensa principal.
3. Despues se introduce una pieza conceptual que debio aparecer antes de la coordinacion: un potencial independiente por cada lazo.
4. Luego se presenta la recompensa de coordinacion por progreso global.
5. Despues se proponen compuertas para la recompensa de coordinacion.
6. Finalmente se demuestra matematicamente la recompensa principal con compuertas y la recompensa coordinativa con compuertas actuando juntas.
7. Al final se discute como la logica interna de funciones tipo `tanh`, ya presente en incentivos dinamicos, puede integrarse como herramienta funcional dentro de la recompensa principal y coordinativa, sin declarar necesariamente incentivos dinamicos como terminos separados.

El documento se escribe de forma general. No depende de una planta fisica especifica. La planta puede ser mecanica, electrica, termica, hidraulica, quimica o de cualquier otro dominio fisico. Lo unico necesario es que existan varios lazos de control locales y que cada lazo pueda evaluarse mediante metricas de desempeno.

## 2. Contexto general de la arquitectura

Se considera un sistema fisico controlado por varios lazos locales. El conjunto de lazos se denota como:

$$
\mathcal{V}=\{1,2,\dots,m\}.
$$

Cada elemento \(i\in\mathcal{V}\) representa un lazo de control. Cada lazo posee:

- una variable objetivo;
- un setpoint local;
- un controlador local;
- una accion efectiva local;
- metricas de desempeno calculadas durante un intervalo;
- agentes que ajustan parametros o ganancias del controlador.

La arquitectura de aprendizaje opera en intervalos de decision. El indice de intervalo se denota por:

$$
k=0,1,2,\dots
$$

Dentro de cada intervalo \(k\), el sistema fisico evoluciona durante varios pasos de simulacion o de muestreo. Al final del intervalo, se agregan metricas y se calcula una recompensa para los agentes.

El objetivo general es que cada agente ajuste su parametro local para mejorar el desempeno del lazo al que pertenece, pero sin impedir que el sistema completo pueda realizar maniobras coordinadas. Esta tension entre desempeno local y desempeno global es el centro del problema.

## 3. Principio de separacion entre lazos

El diseno busca preservar un principio fuerte:

> Cada lazo debe conservar su evaluacion local, y la coordinacion entre lazos debe aparecer por medio de la recompensa coordinativa, no mediante la mezcla directa de estados fisicos entre agentes.

Esto significa que el agente de una ganancia no necesita observar directamente variables fisicas de otros lazos. El agente puede tener observabilidad limitada. Por ejemplo, puede observar solamente la discretizacion de su propia ganancia.

La recompensa, en cambio, si puede ser calculada con informacion agregada del sistema, porque la recompensa representa una evaluacion externa del resultado. La distincion es importante:

- **mezclar observaciones** cambia el estado del agente y acopla politicas;
- **usar recompensa coordinativa** mantiene estados locales y acopla solo la evaluacion del resultado.

El documento mantiene esta separacion durante toda la formulacion.

## 4. Metricas locales por lazo

Durante cada intervalo \(k\), cada lazo \(i\) produce un conjunto de metricas locales. Se denota por:

$$
L_{q,i}(k)\ge 0
$$

la metrica local asociada a la caracteristica \(q\) del lazo \(i\).

El subindice \(q\) puede representar distintas caracteristicas de desempeno:

- \(e\): error local;
- \(\dot e\): derivada del error local;
- \(I\): integral del error;
- \(u\): esfuerzo de control o accion efectiva;
- \(\Delta u\): variacion del esfuerzo.

Asi, por ejemplo:

$$
L_{e,i}(k)
$$

representa el costo de error del lazo \(i\) durante el intervalo \(k\), y:

$$
L_{\dot e,i}(k)
$$

representa el costo asociado a la velocidad de cambio del error.

Estas metricas deben entenderse como magnitudes normalizadas. La normalizacion es esencial porque permite comparar costos con unidades fisicas distintas. Sin normalizacion, no seria consistente sumar un error angular, un error de posicion, una velocidad, un esfuerzo de actuacion o una integral acumulada.

## 5. Recompensa principal: metodo implementado

La primera capa de recompensa es la recompensa principal. Esta recompensa evalua cada lazo de forma local. Su objetivo es premiar configuraciones de ganancias que reduzcan los costos del lazo.

La arquitectura actual contempla una recompensa principal basada en una combinacion de caracteristicas locales. Una de las formas implementadas es la version exponencial ponderada.

Para cada caracteristica \(q\) y lazo \(i\), se define una contribucion:

$$
C_{q,i}^{P}(k)
=
-w_q
\left(
1-\exp\left[-s_q\left(L_{q,i}(k)-b_q\right)^2\right]
\right),
$$

donde:

- \(C_{q,i}^{P}(k)\) es la contribucion principal de la caracteristica \(q\) para el lazo \(i\);
- \(w_q\ge 0\) es el peso de la caracteristica;
- \(s_q>0\) es un factor de escala o sensibilidad;
- \(b_q\) es el setpoint deseado para la metrica;
- \(L_{q,i}(k)\) es la metrica local normalizada.

La recompensa principal del lazo \(i\) se obtiene sumando contribuciones:

$$
R_i^{P}(k)
=
\sum_{q\in\mathcal{Q}}
C_{q,i}^{P}(k).
$$

Como cada contribucion tiene signo negativo o nulo, la recompensa principal se interpreta como una penalizacion acotada:

- vale cero cuando las metricas estan exactamente en su setpoint;
- se vuelve negativa cuando las metricas se alejan del setpoint;
- no decrece sin limite, porque la funcion exponencial esta acotada.

## 6. Interpretacion de la forma exponencial

La expresion:

$$
1-\exp\left[-s_q\left(L_{q,i}(k)-b_q\right)^2\right]
$$

tiene varias propiedades utiles.

Primero, siempre esta entre cero y uno:

$$
0
\le
1-\exp\left[-s_q\left(L_{q,i}(k)-b_q\right)^2\right]
<
1.
$$

Segundo, toma valor cero si:

$$
L_{q,i}(k)=b_q.
$$

Tercero, aumenta suavemente cuando la metrica se aleja de su setpoint.

Por lo tanto, la contribucion:

$$
C_{q,i}^{P}(k)
$$

esta acotada por:

$$
-w_q < C_{q,i}^{P}(k)\le 0.
$$

Esto evita recompensas de magnitud explosiva y permite que el aprendizaje reciba una senal estable.

## 7. Recompensa principal con compuertas: metodo implementado

Sobre la recompensa principal existe una segunda pieza ya implementada: las compuertas de caracteristicas, o `feature_gates`.

Una compuerta de la recompensa principal se denota como:

$$
G_{q,i}^{P}(k).
$$

Esta compuerta multiplica la contribucion de una caracteristica \(q\) en un lazo \(i\). La recompensa principal con compuertas queda:

$$
R_i^{P,G}(k)
=
\sum_{q\in\mathcal{Q}}
-G_{q,i}^{P}(k)
w_q
\left(
1-\exp\left[-s_q\left(L_{q,i}(k)-b_q\right)^2\right]
\right).
$$

La compuerta no crea una recompensa nueva. Solo cambia la intensidad con la que una caracteristica pesa en la recompensa principal.

Si:

$$
G_{q,i}^{P}(k)=1,
$$

la caracteristica se evalua con su peso original.

Si:

$$
0<G_{q,i}^{P}(k)<1,
$$

la penalizacion asociada a esa caracteristica se atenual.

Si la compuerta tiene un limite superior menor que uno, entonces esa caracteristica nunca puede dominar completamente la recompensa.

## 8. Forma general de una compuerta principal

Una compuerta principal se construye a partir de una senal fuente:

$$
z_{q,i}^{P}(k).
$$

Esta senal fuente puede ser una metrica local, como:

$$
z_{q,i}^{P}(k)=L_{e,i}(k),
$$

o cualquier otra metrica disponible en la recompensa principal.

La compuerta se define como:

$$
G_{q,i}^{P}(k)
=
\text{clip}
\left(
\gamma_{q,i}(z_{q,i}^{P}(k)),
G_{q,i}^{\min},
G_{q,i}^{\max}
\right),
$$

donde:

- \(\gamma_{q,i}\) es una funcion suave;
- \(G_{q,i}^{\min}\) es el factor minimo permitido;
- \(G_{q,i}^{\max}\) es el factor maximo permitido;
- `clip` limita el valor de la compuerta al rango permitido.

La arquitectura actual soporta funciones suaves como exponencial y `tanh`. Una forma exponencial es:

$$
\gamma_{q,i}^{exp}(z)
=
\exp\left[-\left(\frac{|z-z^\star|}{\sigma}\right)^2\right].
$$

Una forma tipo `tanh` puede escribirse como:

$$
\gamma_{q,i}^{tanh}(z)
=
\tanh(\alpha |z-z^\star|).
$$

En ambos casos, la compuerta permite que la recompensa principal no trate todas las caracteristicas con peso constante durante todo el episodio.

## 9. Interpretacion de las compuertas principales

Las compuertas principales tienen una funcion conceptual clara:

> Permiten que el peso efectivo de cada termino de la recompensa principal dependa del estado abstracto de desempeno del propio lazo.

Por ejemplo, puede no convenir penalizar con la misma intensidad el esfuerzo de control cuando el error es muy grande que cuando el error ya es pequeno. Si el error es grande, el sistema puede necesitar actuar; si el error ya es pequeno, conviene exigir suavidad y menor esfuerzo.

La compuerta permite expresar esa idea sin agregar una recompensa adicional. Simplemente modifica la relevancia de una caracteristica ya existente.

Esto es distinto a un incentivo dinamico separado. La compuerta no dice "se entrega un nuevo premio". La compuerta dice "este termino de la recompensa principal pesa mas o menos bajo cierta condicion".

## 10. Necesidad de un potencial independiente por lazo

Antes de introducir coordinacion global, conviene declarar una pieza intermedia: el potencial independiente por lazo.

La recompensa principal calcula contribuciones por caracteristica, pero la coordinacion necesita una medida compacta del estado de cada lazo. Esa medida no debe ser todavia global. Debe ser primero local e independiente.

Se define entonces:

$$
\Psi_i(k)
=
\sum_{q\in\mathcal{Q}_{\Psi}}
\alpha_{q,i}L_{q,i}(k),
$$

donde:

- \(\Psi_i(k)\) es el potencial local independiente del lazo \(i\);
- \(\mathcal{Q}_{\Psi}\) es el conjunto de caracteristicas usadas para construir el potencial;
- \(\alpha_{q,i}\ge 0\) es el peso de la caracteristica \(q\) en el potencial del lazo \(i\);
- \(L_{q,i}(k)\) es la metrica local normalizada.

La interpretacion es:

- \(\Psi_i(k)\) grande indica que el lazo \(i\) esta lejos de su objetivo local;
- \(\Psi_i(k)\) pequeno indica que el lazo \(i\) esta cerca de su objetivo local.

Este potencial no reemplaza la recompensa principal. Es una variable de organizacion matematica. Sirve para medir progreso local, construir potencial global y definir compuertas de coordinacion.

## 11. Diferencia entre recompensa principal y potencial local

La recompensa principal y el potencial local no son lo mismo.

La recompensa principal:

$$
R_i^{P,G}(k)
$$

es una senal de aprendizaje. Esta disenada para ser entregada a los agentes.

El potencial local:

$$
\Psi_i(k)
$$

es una medida de estado abstracto del lazo. Esta disenado para comparar intervalos, medir progreso y construir coordinacion.

La recompensa principal puede ser no lineal, acotada, exponencial y con compuertas. El potencial local puede ser mas simple y directamente interpretable.

Esta separacion es util porque evita que la coordinacion dependa de detalles internos de la forma exacta de recompensa. La coordinacion puede usar potenciales, mientras que la recompensa principal puede conservar su forma mas rica.

## 12. Progreso local por lazo

Una vez definido el potencial local, se define el progreso local:

$$
\Delta \Psi_i(k)
=
\Psi_i(k-1)-\Psi_i(k).
$$

Si:

$$
\Delta \Psi_i(k)>0,
$$

el lazo \(i\) mejoro localmente, porque su potencial disminuyo.

Si:

$$
\Delta \Psi_i(k)<0,
$$

el lazo \(i\) empeoro localmente, porque su potencial aumento.

La parte positiva del progreso local es:

$$
\Delta \Psi_i^+(k)
=
\max(0,\Delta \Psi_i(k)).
$$

La parte negativa, entendida como deterioro local, es:

$$
\Delta \Psi_i^-(k)
=
\max(0,-\Delta \Psi_i(k)).
$$

Estas dos cantidades son relevantes para la coordinacion:

- \(\Delta \Psi_i^+(k)\) mide recuperacion local;
- \(\Delta \Psi_i^-(k)\) mide sacrificio o deterioro local.

El sacrificio no es bueno por si mismo. Solo puede ser interpretado como util si ocurre junto con progreso global.

## 13. Recompensa de coordinacion: metodo implementado

La recompensa de coordinacion ya implementada se basa en progreso global de tarea y reparto de credito.

Primero se define un potencial global:

$$
\Phi(k)
=
\sum_{i\in\mathcal{V}}
\beta_i \Psi_i(k).
$$

En la implementacion actual, este potencial puede construirse directamente desde caracteristicas como \(L_e\) y \(L_{\dot e}\) por lazo. Matematicamente, ambas formas son equivalentes si \(\Psi_i(k)\) se define como la combinacion local de esas caracteristicas.

La interpretacion de \(\Phi(k)\) es:

- \(\Phi(k)\) grande indica mal desempeno global;
- \(\Phi(k)\) pequeno indica buen desempeno global.

Luego se define el progreso global:

$$
\Delta \Phi(k)
=
\Phi(k-1)-\Phi(k).
$$

La parte premiable es:

$$
\Delta \Phi^+(k)
=
\max(0,\Delta \Phi(k)).
$$

Esta parte positiva asegura que la coordinacion premie solamente mejoras globales.

## 14. Credito coordinativo implementado

El progreso global no debe repartirse por igual entre todos los lazos. Debe repartirse segun una medida de contribucion.

Se define una contribucion bruta del lazo \(i\):

$$
c_i(k)\ge 0.
$$

En la forma implementada, esta contribucion se calcula a partir de una senal de error local y una senal de esfuerzo local efectivo. Una forma abstracta es:

$$
c_i(k)
=
\frac{1}{N_k}
\sum_{t\in k}
\max\left(0,s_i e_i(t)u_i^{eff}(t)\right),
$$

donde:

- \(N_k\) es el numero de pasos dentro del intervalo;
- \(e_i(t)\) es el error local del lazo \(i\);
- \(u_i^{eff}(t)\) es el esfuerzo efectivo atribuible al lazo \(i\);
- \(s_i\in\{-1,1\}\) representa el signo correctivo esperado.

Esta expresion mide esfuerzo correctivo local. Si el producto tiene signo correctivo, se cuenta como contribucion positiva. Si no, se descarta.

Luego se define el credito relativo:

$$
\rho_i(k)
=
\frac{c_i(k)}
{\varepsilon+\sum_{j\in\mathcal{V}}c_j(k)},
$$

donde \(\varepsilon>0\) evita division por cero.

La recompensa coordinativa basica queda:

$$
R_i^{C}(k)
=
\lambda_C
\rho_i(k)
\Delta \Phi^+(k),
$$

donde \(\lambda_C\ge 0\) es el peso global de coordinacion.

## 15. Interpretacion de la recompensa de coordinacion

La recompensa coordinativa implementada expresa la siguiente idea:

> Si el sistema completo mejora y un lazo contribuyo correctivamente durante el intervalo, ese lazo recibe una fraccion del progreso global.

Este diseno es general porque no dice que un lazo dependa de una variable fisica especifica de otro lazo. Solamente usa:

- potencial global;
- progreso global;
- contribucion local;
- reparto relativo de credito.

La recompensa coordinativa se suma a la recompensa local. Por lo tanto, un lazo puede recibir:

- castigo local por estar lejos de su setpoint;
- credito coordinativo por ayudar a que el sistema global mejore.

Esta es la tension necesaria para permitir maniobras coordinadas.

## 16. Limitacion de la coordinacion sin compuertas

La coordinacion basica premia progreso global, pero no distingue suficientemente entre fases.

El sistema puede encontrarse en dos situaciones distintas:

1. Una fase donde el sistema global esta lejos de su objetivo y algunos lazos deben ayudar aunque localmente empeoren.
2. Una fase donde el sistema global ya mejoro y los lazos deben volver a minimizar sus potenciales locales.

La recompensa coordinativa basica:

$$
R_i^{C}(k)
=
\lambda_C\rho_i(k)\Delta\Phi^+(k)
$$

solo premia progreso global. No declara explicitamente cuando debe aumentar la presion de retorno local.

Por eso se propone incorporar compuertas de coordinacion.

## 17. Compuertas para la recompensa de coordinacion

Las compuertas de coordinacion deben cumplir una condicion fundamental:

> Deben depender de potenciales y progresos abstractos, no de reglas fisicas especificas de una planta.

Se proponen dos compuertas suaves:

- una compuerta de asistencia global;
- una compuerta de recuperacion local coordinada.

La compuerta de asistencia se define como:

$$
G_A(k)
=
H_{\tau_A}\left(\Phi(k)-\Phi^\star\right),
$$

donde:

- \(\Phi^\star\) es un umbral abstracto de suficiencia global;
- \(H_{\tau_A}\) es una funcion suave creciente entre cero y uno.

La compuerta de recuperacion se define como:

$$
G_R(k)
=
H_{\tau_R}\left(\Phi^\star-\Phi(k)\right).
$$

Cuando el potencial global es alto, \(G_A(k)\) es grande y \(G_R(k)\) es pequeno. Cuando el potencial global baja, \(G_R(k)\) aumenta.

## 18. Funcion suave para compuertas

Una eleccion natural es la sigmoide:

$$
H_{\tau}(x)
=
\frac{1}{1+\exp(-x/\tau)}.
$$

Otra eleccion equivalente basada en `tanh` es:

$$
H_{\kappa}(x)
=
\frac{1}{2}\left(1+\tanh(\kappa x)\right).
$$

Ambas funciones toman valores entre cero y uno. La forma con `tanh` es especialmente interesante porque la logica interna de los incentivos dinamicos ya implementados utiliza funciones `tanh` para crear transiciones suaves, saturadas e interpretables.

La funcion \(H_{\kappa}\) tiene estas propiedades:

$$
0 < H_{\kappa}(x) < 1,
$$

$$
H_{\kappa}(0)=\frac{1}{2},
$$

y:

$$
H_{\kappa}(-x)=1-H_{\kappa}(x).
$$

Por lo tanto, si se usa la misma pendiente para asistencia y recuperacion:

$$
G_R(k)=1-G_A(k).
$$

Esto genera una transicion suave entre fases.

## 19. Recompensa coordinativa con compuertas

Se propone extender la recompensa coordinativa asi:

$$
R_i^{C,G}(k)
=
\lambda_A
G_A(k)
\rho_i(k)
\Delta\Phi^+(k)
+
\lambda_R
G_R(k)
\Delta\Psi_i^+(k).
$$

donde:

- \(R_i^{C,G}(k)\) es la recompensa coordinativa con compuertas;
- \(\lambda_A\ge 0\) regula asistencia global;
- \(\lambda_R\ge 0\) regula recuperacion local coordinada;
- \(G_A(k)\) abre la fase de asistencia;
- \(G_R(k)\) abre la fase de recuperacion;
- \(\rho_i(k)\) reparte credito global entre lazos;
- \(\Delta\Phi^+(k)\) mide progreso global positivo;
- \(\Delta\Psi_i^+(k)\) mide progreso local positivo del lazo \(i\).

El primer termino corresponde al metodo coordinativo ya implementado, pero modulado por una compuerta de fase. El segundo termino agrega la pieza que faltaba para consolidar retorno local cuando el sistema global ya esta suficientemente encaminado.

## 20. Por que el segundo termino sigue siendo coordinativo

El termino:

$$
\lambda_R G_R(k)\Delta\Psi_i^+(k)
$$

puede parecer local, porque usa el progreso del lazo \(i\). Sin embargo, su activacion depende del estado global mediante:

$$
G_R(k)
=
H_{\tau_R}\left(\Phi^\star-\Phi(k)\right).
$$

Por eso se interpreta como recuperacion local coordinada. No es simplemente una recompensa local adicional. Es una recompensa local de recuperacion que solo se abre cuando el potencial global indica que el sistema ya puede transitar desde asistencia hacia retorno.

Esta distincion es importante. El lazo no recibe este bonus por mejorar localmente en cualquier circunstancia, sino por mejorar localmente cuando la fase global lo permite.

## 21. Recompensa total propuesta

La recompensa total para el lazo \(i\) queda:

$$
R_i^{total}(k)
=
R_i^{P,G}(k)
+
R_i^{C,G}(k).
$$

Sustituyendo las definiciones:

$$
R_i^{total}(k)
=
\sum_{q\in\mathcal{Q}}
-G_{q,i}^{P}(k)
w_q
\left(
1-\exp\left[-s_q\left(L_{q,i}(k)-b_q\right)^2\right]
\right)
$$

$$
+
\lambda_A
G_A(k)
\rho_i(k)
\Delta\Phi^+(k)
+
\lambda_R
G_R(k)
\Delta\Psi_i^+(k).
$$

Esta ecuacion reune todas las piezas:

- recompensa principal local;
- compuertas de recompensa principal;
- potencial local;
- potencial global;
- progreso global;
- credito coordinativo;
- compuertas de coordinacion;
- recuperacion local coordinada.

## 22. Demostracion de acotamiento de la recompensa principal

Se asume:

$$
0\le G_{q,i}^{P}(k)\le G_{q,i}^{\max},
$$

con:

$$
G_{q,i}^{\max}<\infty.
$$

Tambien:

$$
0
\le
1-\exp\left[-s_q\left(L_{q,i}(k)-b_q\right)^2\right]
<
1.
$$

Entonces, cada contribucion principal cumple:

$$
-G_{q,i}^{\max}w_q
<
C_{q,i}^{P,G}(k)
\le
0.
$$

Sumando sobre \(q\):

$$
-\sum_{q\in\mathcal{Q}}G_{q,i}^{\max}w_q
<
R_i^{P,G}(k)
\le
0.
$$

Por lo tanto, la recompensa principal con compuertas esta acotada inferior y superiormente.

## 23. Demostracion de acotamiento del potencial local y global

Se asume que las metricas locales estan normalizadas:

$$
0\le L_{q,i}(k)\le L_{q}^{\max}.
$$

Entonces:

$$
0
\le
\Psi_i(k)
=
\sum_{q\in\mathcal{Q}_{\Psi}}\alpha_{q,i}L_{q,i}(k)
\le
\sum_{q\in\mathcal{Q}_{\Psi}}\alpha_{q,i}L_q^{\max}.
$$

Se define:

$$
\Psi_i^{\max}
=
\sum_{q\in\mathcal{Q}_{\Psi}}\alpha_{q,i}L_q^{\max}.
$$

Por lo tanto:

$$
0\le \Psi_i(k)\le \Psi_i^{\max}.
$$

El potencial global cumple:

$$
\Phi(k)=\sum_{i\in\mathcal{V}}\beta_i\Psi_i(k).
$$

Como \(\beta_i\ge 0\), entonces:

$$
0
\le
\Phi(k)
\le
\sum_{i\in\mathcal{V}}\beta_i\Psi_i^{\max}.
$$

Se define:

$$
\Phi^{\max}
=
\sum_{i\in\mathcal{V}}\beta_i\Psi_i^{\max}.
$$

Por lo tanto:

$$
0\le \Phi(k)\le \Phi^{\max}.
$$

## 24. Demostracion de acotamiento de los progresos

Como:

$$
0\le \Phi(k)\le \Phi^{\max},
$$

el progreso global positivo satisface:

$$
0\le \Delta\Phi^+(k)\le \Phi^{\max}.
$$

De manera analoga:

$$
0\le \Delta\Psi_i^+(k)\le \Psi_i^{\max}.
$$

Estas desigualdades son importantes porque la recompensa coordinativa usa directamente progresos.

## 25. Demostracion de acotamiento del credito coordinativo

Por definicion:

$$
c_i(k)\ge 0.
$$

El credito relativo es:

$$
\rho_i(k)
=
\frac{c_i(k)}
{\varepsilon+\sum_{j\in\mathcal{V}}c_j(k)}.
$$

Como el denominador es mayor que el numerador para \(\varepsilon>0\), se cumple:

$$
0\le \rho_i(k)<1.
$$

Ademas:

$$
\sum_{i\in\mathcal{V}}\rho_i(k)
=
\frac{\sum_i c_i(k)}
{\varepsilon+\sum_i c_i(k)}
<1.
$$

El reparto coordinativo no crea mas credito relativo que el disponible.

## 26. Demostracion de acotamiento de la coordinacion con compuertas

Se tiene:

$$
0<G_A(k)<1,
$$

$$
0<G_R(k)<1,
$$

$$
0\le \rho_i(k)<1,
$$

$$
0\le \Delta\Phi^+(k)\le \Phi^{\max},
$$

y:

$$
0\le \Delta\Psi_i^+(k)\le \Psi_i^{\max}.
$$

Entonces:

$$
0
\le
\lambda_A
G_A(k)
\rho_i(k)
\Delta\Phi^+(k)
\le
\lambda_A\Phi^{\max}.
$$

Tambien:

$$
0
\le
\lambda_R
G_R(k)
\Delta\Psi_i^+(k)
\le
\lambda_R\Psi_i^{\max}.
$$

Por lo tanto:

$$
0
\le
R_i^{C,G}(k)
\le
\lambda_A\Phi^{\max}
+
\lambda_R\Psi_i^{\max}.
$$

La recompensa coordinativa con compuertas es no negativa y acotada.

## 27. Demostracion de acotamiento de la recompensa total

La recompensa total es:

$$
R_i^{total}(k)
=
R_i^{P,G}(k)
+
R_i^{C,G}(k).
$$

Ya se demostro:

$$
-\sum_qG_{q,i}^{\max}w_q
<
R_i^{P,G}(k)
\le
0,
$$

y:

$$
0
\le
R_i^{C,G}(k)
\le
\lambda_A\Phi^{\max}
+
\lambda_R\Psi_i^{\max}.
$$

Por lo tanto:

$$
-\sum_qG_{q,i}^{\max}w_q
<
R_i^{total}(k)
<
\lambda_A\Phi^{\max}
+
\lambda_R\Psi_i^{\max}.
$$

Esto demuestra que la recompensa total permanece acotada.

El acotamiento es una propiedad critica para agentes tabulares o discretizados, porque evita que una componente de recompensa destruya la escala de aprendizaje de las demas.

## 28. Demostracion de equilibrio

Se considera una condicion ideal donde cada metrica local esta en su setpoint:

$$
L_{q,i}(k)=b_q
\quad
\forall q,i.
$$

Entonces:

$$
1-\exp\left[-s_q(L_{q,i}(k)-b_q)^2\right]=0.
$$

Por lo tanto:

$$
R_i^{P,G}(k)=0.
$$

Si ademas el sistema permanece en esa condicion, entonces:

$$
\Psi_i(k)=\Psi_i(k-1),
$$

y:

$$
\Phi(k)=\Phi(k-1).
$$

Por lo tanto:

$$
\Delta\Psi_i^+(k)=0,
$$

y:

$$
\Delta\Phi^+(k)=0.
$$

Luego:

$$
R_i^{C,G}(k)=0.
$$

Finalmente:

$$
R_i^{total}(k)=0.
$$

Esto demuestra que la recompensa propuesta no genera bonus artificial cuando no hay error ni progreso pendiente. En equilibrio, la recompensa coordinativa desaparece y la recompensa principal no penaliza.

## 29. Demostracion de mejora local bajo compuertas principales

Se analiza una caracteristica \(q\) de un lazo \(i\). Si la compuerta \(G_{q,i}^{P}(k)\) se mantiene fija durante la comparacion, la contribucion principal depende de:

$$
d_{q,i}(k)
=
\left(L_{q,i}(k)-b_q\right)^2.
$$

La contribucion es:

$$
C_{q,i}^{P,G}(k)
=
-G_{q,i}^{P}w_q
\left(1-\exp[-s_qd_{q,i}(k)]\right).
$$

Si la metrica se acerca al setpoint, entonces \(d_{q,i}(k)\) disminuye. Como:

$$
1-\exp[-s_qd]
$$

es una funcion creciente de \(d\), al disminuir \(d\) disminuye la penalizacion. Por lo tanto, la contribucion se vuelve menos negativa.

En consecuencia, la recompensa principal con compuertas sigue premiando la mejora local, siempre que la compuerta no invierta el signo de la contribucion. Como la compuerta es no negativa, esta propiedad se preserva.

## 30. Demostracion de asistencia global

Supongase que durante un intervalo \(k\):

$$
\Delta\Phi^+(k)>0.
$$

Esto significa que el sistema global mejoro.

Supongase ademas que:

$$
\rho_i(k)>0.
$$

Esto significa que el lazo \(i\) recibio credito relativo por su contribucion.

Si la compuerta de asistencia esta abierta:

$$
G_A(k)>0,
$$

entonces:

$$
\lambda_A
G_A(k)
\rho_i(k)
\Delta\Phi^+(k)
>0
$$

si \(\lambda_A>0\).

Por lo tanto, el lazo recibe recompensa coordinativa positiva por asistencia global. Esta propiedad permite que un lazo sea reconocido por contribuir a una mejora global, incluso si su recompensa principal local no mejora en ese mismo intervalo.

## 31. Demostracion de recuperacion local coordinada

Supongase que el potencial global ya se encuentra por debajo del umbral:

$$
\Phi(k)<\Phi^\star.
$$

Entonces:

$$
G_R(k)
=
H_{\tau_R}(\Phi^\star-\Phi(k))
>
\frac{1}{2}
$$

si \(H\) es una sigmoide centrada.

Si el lazo \(i\) mejora localmente:

$$
\Delta\Psi_i^+(k)>0,
$$

entonces:

$$
\lambda_R
G_R(k)
\Delta\Psi_i^+(k)
>0
$$

si \(\lambda_R>0\).

Por lo tanto, cuando el sistema global esta suficientemente encaminado, la recompensa coordinativa con compuertas incentiva que cada lazo reduzca su propio potencial local.

Esta propiedad es la que completa el relato de coordinacion:

- primero se premia asistir al progreso global;
- despues se premia volver al objetivo local.

## 32. Relacion entre sacrificio local y progreso global

Puede ocurrir que un lazo empeore localmente:

$$
\Delta\Psi_i(k)<0.
$$

Esto implica:

$$
\Delta\Psi_i^-(k)>0.
$$

Ese deterioro no se premia directamente en la formulacion base. Sin embargo, si el deterioro local coincide con progreso global y credito del lazo:

$$
\Delta\Phi^+(k)>0,
$$

$$
\rho_i(k)>0,
$$

entonces el lazo puede recibir el termino de asistencia:

$$
\lambda_A G_A(k)\rho_i(k)\Delta\Phi^+(k).
$$

La arquitectura no recompensa el sacrificio como tal. Recompensa el progreso global atribuible. Esta distincion evita premiar deterioros locales inutiles.

El sacrificio local queda justificado solo si aparece asociado a mejora global.

## 33. Papel de la logica `tanh` en el diseno

La arquitectura contiene logicas de incentivo dinamico donde aparecen funciones tipo `tanh`. En este documento no se propone incorporar esos incentivos como terminos separados de recompensa. En cambio, se propone reutilizar la logica matematica de `tanh` como herramienta interna para las compuertas y modulaciones de la recompensa principal y coordinativa.

La funcion `tanh` tiene forma:

$$
\tanh(x)
=
\frac{e^x-e^{-x}}{e^x+e^{-x}}.
$$

Sus propiedades principales son:

$$
-1<\tanh(x)<1,
$$

$$
\tanh(0)=0,
$$

y:

$$
\lim_{x\to\infty}\tanh(x)=1,
\quad
\lim_{x\to-\infty}\tanh(x)=-1.
$$

Esto la convierte en una funcion util para:

- saturar magnitudes;
- evitar crecimientos no acotados;
- crear transiciones suaves;
- representar alineamiento direccional;
- transformar variables firmadas en factores acotados.

## 34. Compuertas con `tanh`

Una compuerta entre cero y uno puede definirse como:

$$
H_{\kappa}(x)
=
\frac{1}{2}(1+\tanh(\kappa x)).
$$

Esta forma puede reemplazar o complementar la sigmoide exponencial.

Para la recompensa principal:

$$
G_{q,i}^{P}(k)
=
G_{q,i}^{\min}
+
(G_{q,i}^{\max}-G_{q,i}^{\min})
H_{\kappa}\left(z_{q,i}^{P}(k)-z_{q,i}^{\star}\right).
$$

Para la coordinacion:

$$
G_A(k)
=
H_{\kappa_A}\left(\Phi(k)-\Phi^\star\right),
$$

y:

$$
G_R(k)
=
H_{\kappa_R}\left(\Phi^\star-\Phi(k)\right).
$$

La ventaja de esta forma es que todas las compuertas quedan acotadas y suavemente diferenciables.

## 35. Saturacion de contribucion coordinativa con `tanh`

La contribucion bruta:

$$
c_i(k)
$$

puede estar dominada por un lazo si su esfuerzo correctivo es muy grande. Para evitar dominancias excesivas, puede transformarse mediante:

$$
\tilde c_i(k)
=
\tanh(\kappa_c c_i(k)).
$$

Luego el credito relativo se calcula como:

$$
\tilde\rho_i(k)
=
\frac{\tilde c_i(k)}
{\varepsilon+\sum_j\tilde c_j(k)}.
$$

Como:

$$
0\le \tilde c_i(k)<1,
$$

ningun lazo puede aumentar indefinidamente su credito bruto solo por magnitud. La funcion `tanh` conserva el orden para valores positivos, pero satura los extremos.

Esta idea no introduce un incentivo nuevo. Solo hace mas robusto el reparto de credito coordinativo.

## 36. Modulacion de progreso con `tanh`

El progreso global positivo tambien puede saturarse:

$$
\widetilde{\Delta\Phi^+}(k)
=
\tanh(\kappa_\Phi \Delta\Phi^+(k)).
$$

La recompensa de asistencia seria:

$$
R_{i,A}^{C,G}(k)
=
\lambda_A
G_A(k)
\rho_i(k)
\widetilde{\Delta\Phi^+}(k).
$$

Esto evita que un intervalo con progreso excepcionalmente grande genere una actualizacion de recompensa desproporcionada.

De manera analoga:

$$
\widetilde{\Delta\Psi_i^+}(k)
=
\tanh(\kappa_\Psi \Delta\Psi_i^+(k)).
$$

La recuperacion local coordinada seria:

$$
R_{i,R}^{C,G}(k)
=
\lambda_R
G_R(k)
\widetilde{\Delta\Psi_i^+}(k).
$$

Esta integracion usa la logica interna de los incentivos dinamicos, pero no crea una familia separada de incentivos. Solo modifica la forma funcional de los terminos ya definidos.

## 37. Alineamiento direccional con `tanh`

La logica de los incentivos dinamicos tambien usa productos direccionales y saturacion. Esta idea puede mejorar la contribucion coordinativa.

En vez de:

$$
c_i(t)=\max(0,s_i e_i(t)u_i^{eff}(t)),
$$

se puede definir:

$$
c_i(t)
=
\tanh\left(
\kappa_a
\max(0,s_i e_i(t)u_i^{eff}(t))
\right).
$$

Esto conserva la idea de alineamiento correctivo:

- si la accion no corrige, la contribucion es cero;
- si la accion corrige poco, la contribucion crece casi linealmente;
- si la accion corrige mucho, la contribucion se satura.

La saturacion evita que esfuerzos grandes obtengan credito ilimitado.

## 38. Version compacta final con `tanh` integrado

Una version compacta y robusta de la recompensa total puede escribirse como:

$$
R_i^{total}(k)
=
R_i^{P,G}(k)
+
\lambda_A
G_A(k)
\tilde\rho_i(k)
\tanh(\kappa_\Phi\Delta\Phi^+(k))
+
\lambda_R
G_R(k)
\tanh(\kappa_\Psi\Delta\Psi_i^+(k)).
$$

donde:

$$
G_A(k)
=
\frac{1}{2}
\left(
1+\tanh(\kappa_A(\Phi(k)-\Phi^\star))
\right),
$$

$$
G_R(k)
=
\frac{1}{2}
\left(
1+\tanh(\kappa_R(\Phi^\star-\Phi(k)))
\right),
$$

y:

$$
\tilde\rho_i(k)
=
\frac{\tanh(\kappa_c c_i(k))}
{\varepsilon+\sum_j\tanh(\kappa_c c_j(k))}.
$$

Esta formulacion integra la logica `tanh` en:

- compuertas de fase;
- saturacion de contribucion;
- saturacion de progreso global;
- saturacion de recuperacion local.

No se agregan incentivos dinamicos como bloques independientes. Se reutiliza su forma funcional para hacer mas suaves y robustas las recompensas ya existentes.

## 39. Criterios de implementacion conceptual

La secuencia conceptual recomendada es:

1. Mantener la recompensa principal local como base.
2. Mantener las compuertas principales por caracteristica.
3. Declarar explicitamente el potencial independiente por lazo \(\Psi_i(k)\).
4. Construir el potencial global \(\Phi(k)\) desde los potenciales locales.
5. Mantener la recompensa coordinativa por progreso global y credito relativo.
6. Agregar compuertas de coordinacion para asistencia y recuperacion.
7. Usar funciones `tanh` como herramientas internas para transiciones y saturaciones.

Esta secuencia conserva el relato incremental:

- primero se evalua cada lazo;
- luego se modula la evaluacion local;
- luego se resume cada lazo en un potencial;
- luego se resume el sistema en un potencial global;
- luego se reparte progreso global;
- luego se controla la fase de coordinacion;
- finalmente se suavizan los mecanismos con funciones saturadas.

## 40. Variables minimas de la propuesta

Para que la propuesta sea trazable, deberian registrarse las siguientes magnitudes por intervalo:

- \(L_{q,i}(k)\): metricas locales;
- \(G_{q,i}^{P}(k)\): compuertas principales;
- \(R_i^{P,G}(k)\): recompensa principal por lazo;
- \(\Psi_i(k)\): potencial independiente por lazo;
- \(\Delta\Psi_i(k)\): progreso local firmado;
- \(\Delta\Psi_i^+(k)\): progreso local positivo;
- \(\Phi(k)\): potencial global;
- \(\Delta\Phi(k)\): progreso global firmado;
- \(\Delta\Phi^+(k)\): progreso global positivo;
- \(c_i(k)\): contribucion coordinativa bruta;
- \(\rho_i(k)\): credito relativo;
- \(G_A(k)\): compuerta de asistencia;
- \(G_R(k)\): compuerta de recuperacion;
- \(R_i^{C,G}(k)\): recompensa coordinativa con compuertas;
- \(R_i^{total}(k)\): recompensa total por lazo.

Si se integra `tanh`, tambien conviene registrar:

- \(\tilde c_i(k)\): contribucion saturada;
- \(\tilde\rho_i(k)\): credito relativo saturado;
- \(\tanh(\kappa_\Phi\Delta\Phi^+(k))\): progreso global saturado;
- \(\tanh(\kappa_\Psi\Delta\Psi_i^+(k))\): recuperacion local saturada.

## 41. Conclusion

El diseno de recompensa debe entenderse como una construccion incremental. La recompensa principal ya implementada evalua cada lazo mediante costos locales y una forma exponencial acotada. Las compuertas principales ya implementadas permiten modular la importancia de cada caracteristica local segun senales abstractas.

Antes de la coordinacion, conviene declarar explicitamente un potencial independiente por lazo. Este potencial resume el estado local de cada lazo y permite definir progreso local de forma clara. Sobre esos potenciales locales se construye el potencial global, que permite medir progreso del sistema completo.

La recompensa de coordinacion ya implementada premia progreso global y reparte credito segun contribucion local. La extension propuesta agrega compuertas de coordinacion para distinguir asistencia global y recuperacion local coordinada. Con ello, la arquitectura puede reconocer cuando un lazo debe ayudar al sistema global y cuando debe volver a reducir su propio potencial.

Finalmente, la logica interna de los incentivos dinamicos, especialmente el uso de `tanh`, no necesita incorporarse como incentivos separados. Puede utilizarse como herramienta funcional dentro de la recompensa principal y coordinativa para construir compuertas suaves, saturar contribuciones, limitar progresos y mejorar la robustez de la senal de aprendizaje.

La formulacion total mantiene el principio de no mezclar lazos en observacion, conserva la generalidad para sistemas fisicos diversos y entrega una recompensa matematicamente acotada, interpretable y trazable.
