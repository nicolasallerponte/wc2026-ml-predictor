# WC2026 ML Predictor

**Oracle Hackathon 4.0 - Equipo MXNJ**  
Nicolás Aller · Miguel Planas · León Fernández · Jacobo Núñez · Xoel García

---

## Índice

1. [Descripción General](#1-descripción-general)
2. [Contexto del Reto](#2-contexto-del-reto)
3. [Datos](#3-datos)
4. [Feature Engineering](#4-feature-engineering)
5. [Arquitectura del Modelo](#5-arquitectura-del-modelo)
6. [Calibración de Probabilidades](#6-calibración-de-probabilidades)
7. [Ajuste de Umbral para Empates](#7-ajuste-de-umbral-para-empates)
8. [Resultados y Métricas](#8-resultados-y-métricas)
9. [Backtesting en Mundiales Reales](#9-backtesting-en-mundiales-reales)
10. [Interpretación del Random Forest](#10-interpretación-del-random-forest)
11. [Simulación Monte Carlo WC2026](#11-simulación-monte-carlo-wc2026)
12. [Decisiones de Diseño - Resumen Razonado](#12-decisiones-de-diseño--resumen-razonado)
13. [Limitaciones Honestas](#13-limitaciones-honestas)
14. [Estructura del Repositorio](#14-estructura-del-repositorio)
15. [Cómo Reproducir](#15-cómo-reproducir)

---

## 1. Descripción General

Este proyecto entrena un sistema de machine learning para **predecir resultados de partidos de fútbol internacional** y, a partir de esas predicciones, **simular el torneo completo del Mundial 2026** (48 equipos, 12 grupos, formato inédito).

El modelo resuelve una **clasificación ternaria**:

| Clase | Significado |
|-------|-------------|
| `Home Win` | Victoria del equipo local |
| `Draw` | Empate |
| `Away Win` | Victoria del equipo visitante |

La salida final no es solo la clase predicha, sino una **distribución de probabilidades calibrada** sobre las tres opciones para cada partido, lo que permite alimentar una simulación Monte Carlo del torneo y obtener para cada equipo su probabilidad de ser campeón, llegar al Top 4, Top 8 o clasificar de grupos.

**Stack tecnológico:**

```
Python 3.10 · Oracle Autonomous Database (mTLS) · scikit-learn 1.7.2
XGBoost 3.2.0 · LightGBM 4.6.0 · SHAP 0.49.1 · Plotly 5.x · scipy · numpy · pandas
```

---

## 2. Contexto del Reto

La Oracle Hackathon 4.0 propone usar los datos históricos de fútbol internacional disponibles en una base de datos Oracle Autonomous Database para construir un modelo predictivo del Mundial 2026.

### Por qué el fútbol es un problema difícil para ML

El fútbol es, de todos los deportes de equipo, **el de mayor incertidumbre intrínseca**. Una sola jugada fortuita puede cambiar el resultado. A diferencia del baloncesto (donde equipos superiores ganan casi siempre) o el béisbol (con miles de at-bats por temporada), en fútbol:

- Un equipo puede disparar 20 veces y perder 0-1.
- Los empates son estructuralmente difíciles de predecir porque dependen de factores tácticos y de motivación que los datos históricos capturan mal.
- Los *upsets* (sorpresas) no son anomalías: son parte del deporte.

Esto implica que **superar el 55% de accuracy en predicción ternaria es competitivo** con los mejores modelos académicos y con los sistemas internos de casas de apuestas profesionales (que operan en el rango 55-58%).

### El desafío adicional del WC2026

El Mundial 2026 usa un **formato inédito de 48 equipos en 12 grupos de 4**, nunca jugado antes. Esto significa que el modelo debe *extrapolar fuera de distribución*: los patrones aprendidos de Mundiales de 32 equipos no cubren perfectamente la dinámica de un torneo más amplio con equipos de menor nivel histórico participando.

---

## 3. Datos

### Fuentes

Todos los datos provienen de tablas cargadas desde **Oracle Autonomous Database** (ADB) con conexión segura mTLS. Las credenciales se gestionan mediante archivo `.env` y wallet de Oracle; nunca están hardcodeadas en el código.

| Tabla / Vista | Contenido | Volumen |
|---------------|-----------|---------|
| `MATCH_RESULTS` | Partidos internacionales 1872–2024 | ~48.943 partidos |
| `GOALSCORERS` | Goles individuales con minuto exacto | ~44.500+ goles |
| `SHOOTOUTS` | Resultados de tandas de penales | - |
| `VW_TEAM_STATISTICS` | Estadísticas históricas por equipo en Mundiales | - |
| `CONFEDERATIONS` | Datos de confederaciones (UEFA, CONMEBOL, etc.) | - |
| `WC2026_VENUES` | Sedes del WC2026 con altitud | 16 estadios |
| `WC2026_GROUPS` | Composición de los 12 grupos del torneo | 48 equipos |
| `WC_EDITIONS` | Histórico de Mundiales y campeones | - |

### Por qué Oracle ADB y no CSV locales

Usar Oracle ADB en lugar de archivos CSV locales aporta tres ventajas clave:

1. **Reproducibilidad**: Todos los miembros del equipo acceden a la misma versión del dato. No hay "el CSV que me mandaste era el del martes".
2. **Write-back**: Los resultados de la simulación Monte Carlo se persisten directamente en la base de datos (`WC2026_PREDICTIONS`), lo que permite consultarlos desde otras partes del sistema (por ejemplo, un dashboard APEX).
3. **Escalabilidad**: Si en el futuro se incorporan nuevas fuentes (partidos de mayo-junio 2026 previos al torneo), basta con actualizar la tabla en ADB; el pipeline no cambia.

### División temporal del dataset

El split es **temporal, no aleatorio**: el 80% más antiguo se usa para entrenamiento y el 20% más reciente para test.

- **Train:** ~39.000 partidos (hasta cierta fecha de corte)
- **Test:** ~9.800 partidos (últimos en el tiempo)

**¿Por qué temporal y no aleatorio?** Porque mezclar aleatoriamente causaría *data leakage*: el modelo vería partidos del futuro para predecir el pasado, lo que infla artificialmente las métricas y genera un modelo que no funciona en producción. El fútbol es una serie temporal: el ELO de hoy depende de resultados de ayer, la forma reciente depende de partidos anteriores. Mezclar aleatoriamente destruye esa causalidad.

---

## 4. Feature Engineering

El feature engineering es la parte más crítica del proyecto. Un modelo simple sobre los datos crudos (solo goles marcados y recibidos) tiene muy poca capacidad predictiva. Las 32 features finales intentan capturar distintas dimensiones de la fortaleza de un equipo y del contexto de cada partido.

### 4.1 ELO Dinámico con K-factor Variable y Regresión Anual

El sistema ELO es un método de rating inventado para el ajedrez y adaptado al fútbol. Cada equipo tiene una puntuación numérica que sube cuando gana y baja cuando pierde, con el incremento proporcional a la "sorpresa" del resultado.

**Fórmula base:**
```
ELO_nuevo = ELO_viejo + K × (resultado_real - resultado_esperado)
resultado_esperado = 1 / (1 + 10^((ELO_rival - ELO_propio) / 400))
```

**¿Por qué ELO y no el ranking FIFA oficial?**

El ranking FIFA tiene dos problemas estructurales para ML:
- Solo cubre un período de tiempo reciente (los últimos 4 años con ponderación decreciente) y con fórmulas que cambian entre ediciones.
- No está diseñado para producir probabilidades de resultado; es una herramienta de seeding de torneos.

El ELO calculado desde 1872 acumula señal histórica consistente y produce directamente una probabilidad de victoria mediante la función logística.

**K-factor variable por tipo de torneo:**

No todos los partidos valen lo mismo. Un amistoso en enero no dice tanto de un equipo como una semifinal de Copa del Mundo. Por eso el K-factor varía:

| Tipo de partido | K-factor |
|-----------------|----------|
| Copa del Mundo (fases finales) | Alto |
| Clasificatorios mundialistas | Medio-alto |
| Torneos continentales | Medio |
| Amistosos | Bajo |

**Regresión anual del 2%:**

Cada año, el ELO de cada equipo regresa un 2% hacia el valor neutro (1500):
```
ELO = ELO × 0.98 + 1500 × 0.02
```

¿Por qué? Sin esta corrección, equipos que jugaron muchos partidos en décadas pasadas acumularían ELO indefinidamente, aunque ahora sean mediocres. La regresión simula que el talento humano es perecedero: los jugadores se jubilan, los equipos cambian de generación. Con este ajuste, un equipo que deja de jugar bien pierde terreno frente a la media con el paso del tiempo.

### 4.2 Time Decay - Peso Temporal de los Partidos

No todos los partidos históricos tienen la misma relevancia para predecir lo que ocurrirá en 2026. Un partido jugado en 1990 dice muy poco del equipo actual.

**Fórmula:**
```
weight = exp(-λ × días_transcurridos)
λ = ln(2) / half_life
```

Con `half_life = 3 años` (1095 días), un partido de hace 3 años pesa exactamente el 50% de uno jugado hoy. Un partido de hace 6 años pesa el 25%, etc.

**¿Por qué 3 años y no 1 ni 10?**

- Con `half_life = 1 año`, solo los últimos 12-18 meses importan. Se pierde señal histórica valiosa (p. ej., el patrón de juego de una selección que lleva 5 años construyendo su estilo).
- Con `half_life = 10 años`, un partido de hace 5 años pesa casi lo mismo que uno reciente. La forma actual del equipo queda diluida.
- Con `half_life = 3 años`, capturamos la forma de una generación de jugadores (típicamente 2-4 años de pico), que es el horizonte relevante en fútbol internacional.

Este weight se usa en el cálculo de las features de rolling window (forma reciente, goal difference, win rates).

### 4.3 Forma Reciente - Rolling Window Vectorizado

Para cada partido, la "forma reciente" de un equipo es la media ponderada por time_decay de sus resultados en los últimos 20 partidos.

```python
# Pseudocódigo del cálculo vectorizado
forma = (
    df.groupby('team')['resultado']
    .apply(lambda x: x.shift(1).rolling(20, min_periods=1)
                      .apply(lambda w: np.average(w, weights=time_weights[-len(w):])))
)
```

**Decisiones técnicas clave:**

- **Window de 20 partidos:** Suficiente para capturar tendencia sin diluir excesivamente en historia lejana.
- **`shift(1)` obligatorio:** El partido actual no puede incluirse en el cálculo de su propia forma reciente. Sin `shift(1)` habría data leakage porque el modelo vería el resultado que tiene que predecir.
- **Vectorización con groupby+cumsum:** La implementación naive (iterar partido a partido) tarda 10-20 veces más. La versión vectorizada usa operaciones de pandas optimizadas en C, crítico con ~48.000 partidos.

### 4.4 Goal Difference Score con `tanh`

La diferencia de goles es una señal importante de la calidad del equipo, pero tiene un problema: los outliers. Un resultado 8-0 no es 8 veces más informativo que un 1-0; simplemente indica que el rival era muy inferior.

**Fórmula:**
```
gd_score = tanh(goal_difference_ponderado / 2)
```

La función `tanh` comprime el espacio de [-∞, +∞] a [-1, +1]:
- Un 1-0 da un score de ~0.46
- Un 3-0 da ~0.91
- Un 8-0 da ~0.9999 (prácticamente lo mismo que 5-0)

Esto **evita que goleadas históricas dominen el gradiente** del modelo y distorsionen las predicciones para partidos normales.

### 4.5 Win Rates por Localía

Se calculan dos win rates separados para cada equipo:

- `home_wr_home`: % de victorias del equipo local cuando **juega en casa** (últimos 20 partidos en casa)
- `away_wr_away`: % de victorias del visitante cuando **juega fuera** (últimos 20 partidos de visitante)

**¿Por qué separar por localía?** La ventaja de jugar en casa es uno de los efectos más robustos y documentados en fútbol. Equipos que son muy fuertes en casa pero frágiles de visitante (o viceversa) tienen perfiles distintos que un único win rate global ocultaría.

### 4.6 Head-to-Head Histórico

Los últimos 15 años de enfrentamientos directos entre los dos equipos que se enfrentan se compilan en una feature `h2h_diff`.

**¿Por qué si ya tenemos ELO?** El ELO captura la fortaleza relativa general, pero existen rivalidades históricas con patrones que van más allá de la fortaleza: estilos de juego complementarios, presión psicológica de ciertas rivalidades (Francia vs. Alemania, Argentina vs. Brasil). El H2H captura estos patrones que el ELO no puede.

### 4.7 Penalty Strength

El ratio de victorias en tandas de penales de cada equipo, calculado sobre todos los shootouts registrados en la tabla `SHOOTOUTS`.

**Por qué es importante:** En la fase eliminatoria del Mundial, muchos partidos se deciden por penales. Un equipo con 70% de victorias en penales tiene una ventaja real sobre uno con 30%, que el ELO o la forma reciente no reflejan.

**Default = 0.5:** Para equipos sin datos históricos de penales (raro o con pocos partidos KO), se asigna el valor neutro, evitando introducir ruido ficticio.

### 4.8 Confederation Strength

Pesos calibrados empíricamente sobre resultados históricos en Mundiales:

| Confederación | Peso |
|---------------|------|
| UEFA | 1.00 |
| CONMEBOL | 0.95 |
| CONCACAF | 0.75 |
| AFC | 0.72 |
| CAF | 0.70 |
| OFC | 0.60 |

**¿Por qué es necesario si tenemos ELO?** El ELO se calcula sobre todos los partidos internacionales, incluidos enfrentamientos entre equipos de la misma confederación. Un equipo africano con alto ELO puede haberlo ganado principalmente ganando a rivales africanos; cuando enfrenta a Europa o Sudamérica en un Mundial, el ELO puede sobreestimar su fortaleza. El peso de confederación actúa como corrección calibrada sobre ese sesgo.

### 4.9 Draw Rate

El porcentaje de empates en los últimos 20 partidos de cada equipo: `draw_rate_home` y `draw_rate_away`.

**¿Por qué es la feature más directa para empates?** Los empates no son aleatorios: hay equipos (y estilos de juego) que producen empates sistemáticamente. Selecciones que priorizan no perder, que juegan con bloque bajo frente a rivales superiores, o que no tienen capacidad de cerrar partidos, tienen draw rates estructuralmente altos. Esta feature captura exactamente ese patrón.

### 4.10 Features Avanzadas

| Feature | Descripción | Razonamiento |
|---------|-------------|--------------|
| `late_goals_ratio` | Proporción de goles marcados en el minuto ≥ 80 | Equipos que marcan tarde tienen resistencia y mentalidad ganadora; los goles tardíos suelen ser decisivos en Mundiales |
| `scorer_hhi` | Índice Herfindahl de concentración de goles por jugador | Un equipo cuyo 80% de goles los mete un solo jugador es más frágil (si ese jugador se lesiona o es neutralizado) que uno con goles repartidos |
| `clean_sheet_rate` | % de partidos sin encajar goles | Indicador directo de solidez defensiva; los equipos con bajo clean_sheet_rate raramente ganan torneos |
| `upset_rate` | Frecuencia con que el visitante gana siendo considerado favorito | Mide la tendencia de un equipo a dar sorpresas o a ser víctima de ellas |
| `elo_momentum` | Cambio de ELO en las últimas 5 partidas | Captura si el equipo está en tendencia ascendente o descendente, información que el ELO absoluto no refleja |
| `form_comp` | Forma reciente solo en partidos competitivos (excluye amistosos) | Un equipo puede ganar amistosos fácilmente pero rendir peor en competición real; esta feature separa ambos contextos |
| `altitude_shock` | Ajuste por altitud de la sede | Equipos no acostumbrados a la altitud rinden peor en las sedes mexicanas (Ciudad de México 2.240m, Guadalajara 1.566m); equipos CONMEBOL y AFC llevan ventaja |
| `is_neutral` | Flag de campo neutro | Cuando ningún equipo juega en casa (p. ej., sedes de Mundial), la ventaja local desaparece |
| `wc_dna` | Score de experiencia mundialista = editions_played × win_pct en Mundiales | Hay equipos que "saben ganar" en Mundiales: sus jugadores, cuerpo técnico y federación tienen cultura de alto rendimiento en este torneo específico |

### 4.11 Eliminación de Features Redundantes

Antes de entrenar, se eliminaron dos features que aportaban correlación perfecta con otras ya presentes:

- **`gd_away`**: Correlación de -1.0 con `gd_home`. Es exactamente la misma información con signo invertido. Incluirla añadiría multicolinealidad sin nueva señal.
- **`fifa_rank_diff`**: Correlación de +1.0 con `elo_diff`. El ranking FIFA es una versión ruidosa del ELO que además cambia de metodología entre ediciones; el ELO calculado internamente es más consistente.

**Resultado final:** ~32 features independientes y semánticamente distintas.

---

## 5. Arquitectura del Modelo

### Ensemble Stacking

En lugar de elegir un solo modelo, se usa **ensemble stacking**: cuatro modelos base se entrenan independientemente y sus predicciones se combinan mediante un meta-learner que aprende cuál es el peso óptimo de cada uno.

```
Datos de entrenamiento
        │
        ├──────────────────────────────────────────────┐
        │                                              │
   [Logistic Regression]  [Random Forest]  [XGBoost]  [LightGBM]
        │                      │               │           │
   predicciones OOF       predicciones OOF  pred. OOF  pred. OOF
        └──────────────────────┴───────────────┴───────────┘
                                    │
                          [Meta-Learner: Logistic Regression]
                                    │
                          Predicción final calibrada
```

**¿Por qué stacking y no un simple promedio (voting)?**

Con un promedio simple, cada modelo tiene el mismo peso independientemente de su calidad. El stacking deja que el meta-learner *aprenda* los pesos óptimos: si XGBoost es sistemáticamente mejor en partidos entre equipos del mismo nivel, el meta-learner le asignará más peso en esos casos. El resultado es una combinación más inteligente y adaptativa.

**¿Por qué Out-of-Fold (OOF)?**

Si los modelos base se entrenaran sobre todos los datos de entrenamiento y luego sus predicciones se usaran para entrenar el meta-learner sobre esos mismos datos, habría data leakage: el meta-learner vería predicciones sobre datos que los modelos ya habían memorizado. Las predicciones OOF (cada fold predice sobre datos no vistos durante su entrenamiento) garantizan que el meta-learner aprende a combinar predicciones *genuinamente* out-of-sample.

### Base Learner 1: Logistic Regression

```python
LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs", class_weight="balanced")
```

- **Papel:** Baseline lineal. Si la solución óptima es aproximadamente lineal en el espacio de features, LR la encuentra eficientemente.
- **`C=0.5`:** Regularización L2 moderada. Evita que el modelo se sobreajuste a correlaciones espurias en los datos de entrenamiento.
- **`class_weight="balanced"`:** Compensa el desbalance de clases (hay más victorias locales que empates).
- **Accuracy en test:** 56.09%

### Base Learner 2: Random Forest

```python
RandomForestClassifier(n_estimators=~300, max_depth=~10, min_samples_leaf=~15, class_weight="balanced")
```

Hiperparámetros seleccionados con **RandomizedSearchCV** (búsqueda aleatoria en el espacio de hiperparámetros, más eficiente que grid search exhaustivo).

- **Papel:** Captura no-linealidades e interacciones entre features sin necesidad de especificarlas manualmente.
- **Robusto al ruido:** Al promediar cientos de árboles, el RF amortigua el ruido estocástico del fútbol.
- **Permite SHAP:** Los árboles son interpretables mediante SHAP TreeExplainer con bajo coste computacional.
- **Accuracy en test:** 57.59%

### Base Learner 3: XGBoost

```python
XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", ...)
```

- **Papel:** Boosting iterativo. Cada árbol nuevo se enfoca en los errores del anterior, produciendo un modelo más preciso.
- **`multi:softprob`:** Produce probabilidades directamente para las 3 clases.
- **Mejor empírico en datos tabulares:** XGBoost es consistentemente el modelo de mayor accuracy en competiciones de ML con datos tabulares estructurados.
- **Accuracy en test:** 59.61% - el mejor modelo base individual.

### Base Learner 4: LightGBM

```python
LGBMClassifier(objective="multiclass", ...)
```

- **Papel:** Alternativa al XGBoost con arquitectura diferente (crecimiento por hoja en lugar de por nivel). Más rápido y con menor uso de memoria.
- **Diversidad arquitectural:** Si XGBoost y LightGBM cometen errores diferentes, el meta-learner puede combinarlos para reducir la varianza total.
- **Accuracy en test:** 55.68%

### Meta-Learner: Logistic Regression

El meta-learner recibe como input las predicciones de probabilidad de los 4 modelos base (4 modelos × 3 clases = 12 meta-features) y aprende a combinarlas óptimamente.

**¿Por qué LR como meta-learner?** Porque el problema del meta-learner es relativamente simple (combinar 4 señales bien calibradas), y un modelo lineal con regularización evita sobreajustar a los errores específicos del set de OOF.

### Validación Cruzada: TimeSeriesSplit

```python
TimeSeriesSplit(n_splits=5)
```

En cada fold, el set de entrenamiento siempre es anterior en el tiempo al set de validación. Esto respeta la causalidad y garantiza que las métricas de validación cruzada son representativas del rendimiento real.

Si se usara `KFold` estándar (aleatorio), partidos de 2022 podrían estar en el set de entrenamiento mientras se valida sobre partidos de 2015, lo que sería absurdo desde el punto de vista causal.

---

## 6. Calibración de Probabilidades

### El problema: modelos mal calibrados

Un modelo puede tener buena accuracy pero **malas probabilidades**. Si el modelo dice "probabilidad de victoria local = 0.80" pero en los casos en que dice eso, solo gana el local el 60% de las veces, el modelo está *overconfident* (sobreconfiado). Esto es especialmente problemático para la simulación Monte Carlo: si las probabilidades están sesgadas, las simulaciones producen resultados distorsionados.

### Solución: Isotonic Regression

```python
CalibratedClassifierCV(base_estimator=modelo, method="isotonic", cv=TimeSeriesSplit())
```

La **Isotonic Regression** ajusta las probabilidades del modelo para que coincidan con las frecuencias observadas. Para cada rango de probabilidad predicha, verifica cuánto porcentaje de los casos realmente ocurrió y ajusta la curva.

**¿Por qué Isotonic y no Platt Scaling?**

- **Platt Scaling** (calibración sigmoidea) funciona bien cuando la distribución de scores del modelo es aproximadamente normal. Es simple y eficiente, pero solo puede aprender una transformación sigmoidea.
- **Isotonic Regression** es no paramétrica: puede aprender cualquier función monótonamente creciente. En problemas multi-clase con desbalance (como el nuestro, donde Draw tiene el doble de partidos que la otra clase), la distorsión es no-lineal y Isotonic la corrige mejor.

La calibración se aplica a cada uno de los 4 modelos base, y también al ensemble final.

---

## 7. Ajuste de Umbral para Empates

### El problema estructural de los empates

Los empates son la clase más difícil de predecir en fútbol. El modelo tiende a clasificar los casos ambiguos como "Home Win" o "Away Win" porque son más frecuentes. Esto produce un recall bajo para Draw (el modelo "pierde" muchos empates reales).

Sin ajuste, el threshold por defecto es 0.33 (la clase con mayor probabilidad predicha gana). Esto funciona bien en media, pero subestima los empates.

### Grid Search sobre el Threshold de Draw

Se realiza una búsqueda en el rango [0.20, 0.50]:

```python
for threshold in np.linspace(0.20, 0.50, 31):
    pred = np.where(prob_draw > threshold, 1, argmax(prob_home, prob_away))
    f1_draw = f1_score(y_true, pred, average=None)[1]
    acc = accuracy_score(y_true, pred)
```

**Threshold óptimo encontrado: 0.38**

| Métrica | Sin ajuste (0.33) | Con ajuste (0.38) |
|---------|-------------------|-------------------|
| Accuracy | 0.5668 | 0.5538 |
| Draw F1 | ~0.25 | 0.3289 |
| Draw Recall | ~0.31 | ~0.49 |

**¿Por qué aceptar esta pérdida de accuracy?**

Porque el objetivo no es solo maximizar accuracy global, sino tener **predicciones razonables en todas las clases**. Un modelo que nunca predice empates tiene accuracy más alta (porque hay menos empates que victorias), pero es inútil para simular el torneo correctamente: en el WC2026, los empates en fase de grupos son determinantes para la clasificación.

El trade-off es explícito y razonado: aceptamos -1.3% de accuracy global a cambio de un +10% de F1 en Draw, lo que hace el modelo más equilibrado y útil para la simulación.

---

## 8. Resultados y Métricas

### Comparativa de Modelos

| Modelo | Accuracy (test) |
|--------|-----------------|
| Logistic Regression | 0.5609 |
| Random Forest | 0.5759 |
| XGBoost | 0.5961 |
| LightGBM | 0.5568 |
| **Ensemble Stacking (sin calibrar)** | **0.5668** |
| **Ensemble Stacking (calibrado, threshold=0.38)** | **0.5312** |

*Nota: La caída de accuracy en el ensemble calibrado respecto al sin calibrar es esperada y deseable: la calibración ajusta las probabilidades para que sean más realistas, lo que puede reclasificar algunos casos, pero mejora el Brier Score y el Log-Loss.*

### Métricas del Ensemble Final Calibrado

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 0.5312 |
| **ROC-AUC (macro)** | 0.7308 |
| **Log-Loss** | 0.9261 |
| Brier Score - Home Win | 0.2055 |
| Brier Score - Draw | 0.1837 |
| Brier Score - Away Win | 0.1570 |

### Classification Report por Clase

```
Clase           Precision  Recall  F1-Score  Soporte
─────────────────────────────────────────────────────
Home Win           0.73     0.56     0.64     4.680
Draw               0.29     0.49     0.37     2.267
Away Win           0.61     0.51     0.55     2.846
─────────────────────────────────────────────────────
Accuracy                             0.53     9.793
Macro avg          0.54     0.52     0.52
Weighted avg       0.59     0.53     0.55
```

**Interpretación:**

- **Home Win:** Alta precision (0.73) - cuando predecimos victoria local, acertamos el 73% de las veces. Recall moderado (0.56) porque algunos partidos que acaban con victoria local los predecimos como Draw o Away Win.
- **Draw:** Precision baja (0.29) porque hay muchos falsos positivos (predecimos empate y acaba siendo otra cosa). Pero recall razonable (0.49) gracias al ajuste de umbral: captamos casi la mitad de los empates reales.
- **Away Win:** Balance razonable. Las victorias visitantes son más fáciles de predecir cuando el visitante es claramente superior.

### Matriz de Confusión Normalizada

```
                Predicho:
                Home Win  Draw  Away Win
Real: Home Win    56%      0%     44%
Real: Draw        20%     49%     31%
Real: Away Win     0%      0%    100%
```

*La fila Away Win con 100% en la diagonal refleja que, cuando el modelo predice Away Win con alta confianza, acierta casi siempre. Los errores del modelo se concentran en casos ambiguos entre Home Win y Away Win.*

### ¿Por qué 53% de accuracy es competitivo?

El fútbol no es determinista. Incluso con información perfecta, **hay un límite físico a la predictibilidad**. Los modelos de casas de apuestas profesionales, con acceso a datos de jugadores, lesiones, cuotas de mercado y análisis táctico propietario, operan en el rango **55-58% de accuracy en predicción ternaria**. La literatura académica (modelos publicados) suele reportar entre 50-57%.

Nuestro modelo, usando solo datos históricos de resultados y goles (sin datos de jugadores individuales), logra un **53% calibrado** que es honesto y realista, no inflado por data leakage ni por overfitting al test set.

---

## 9. Backtesting en Mundiales Reales

Para validar que el modelo funciona en el contexto específico de un Mundial (y no solo en datos históricos generales), se realiza backtesting en los dos últimos torneos.

**Metodología:**
- Para WC2018: se entrena con datos **anteriores a junio 2018** y se evalúa sobre los 64 partidos del torneo.
- Para WC2022: se entrena con datos **anteriores a noviembre 2022** y se evalúa sobre los 64 partidos.

### Resultados

| Torneo | Accuracy | Log-Loss |
|--------|----------|----------|
| **WC 2018 (Rusia)** | **0.5625** | 0.9760 |
| **WC 2022 (Qatar)** | **0.4531** | 1.0756 |

### Interpretación

**WC2018 - 56.25%:** Rendimiento sólido, por encima de la media académica. Los equipos eran los esperados (favoritos clásicos: Francia, Croacia, Bélgica, Inglaterra), lo que facilita la predicción.

**WC2022 - 45.31%:** Rendimiento inferior. Varias razones explican esta caída:

1. **Post-COVID:** Los datos de 2020-2021 tienen muchos partidos atípicos (sin público, calendarios comprimidos, secuelas físicas). El modelo aprende de esos datos ruidosos.
2. **Upsets extraordinarios:** Argentina ganó un torneo que era su primer Mundial en 36 años; Marruecos llegó a semifinales por primera vez en la historia. Estos resultados son atípicos y difíciles de predecir con modelos históricos.
3. **Formato de Qatar:** El primer Mundial en invierno del hemisferio norte alteró la forma de los jugadores europeos, que venían de una pausa forzada en mitad de temporada.

Que el WC2022 sea más difícil de predecir que el promedio histórico **no invalida el modelo**; indica que 2022 fue un torneo especialmente impredecible. La literatura académica reporta los mismos patrones: la varianza entre Mundiales es alta.

---

## 10. Interpretación del Random Forest

El Random Forest es el modelo más interpretable del ensemble gracias a dos herramientas complementarias: **feature importance por impureza** y **SHAP (SHapley Additive exPlanations)**.

### Cómo funciona la importancia de features en RF (Gini Importance)

Cada árbol del Random Forest decide en cada nodo qué feature usar para dividir los datos. La importancia de una feature se calcula como **la reducción total de impureza de Gini** que produce esa feature, promediada sobre todos los árboles.

- Una feature importante aparece frecuentemente como primer split en muchos árboles y consigue reducir mucho la mezcla de clases.
- Una feature irrelevante produce splits casi aleatorios, con poca reducción de impureza.

**Limitación de la impureza Gini:** Tiende a sobreestimar la importancia de features con muchos valores únicos (variables continuas de alta cardinalidad). Por eso se complementa con SHAP.

### SHAP - Explicaciones Basadas en Teoría de Juegos

SHAP asigna a cada feature, para cada predicción individual, **cuánto contribuyó a alejar la predicción de la predicción media**. Está fundamentado en los valores de Shapley de la teoría de juegos cooperativos: es el único método que garantiza propiedades de fairness (eficiencia, simetría, linealidad, dummy).

**Ventaja sobre Gini:** SHAP captura el efecto real de cada feature en la predicción, incluyendo interacciones con otras features. Una feature puede tener alta importancia Gini pero bajo SHAP si su efecto se cancela cuando se consideran interacciones.

### Top Features y su Significado

**Factores dominantes:**

1. **`elo_diff`** - La diferencia de ELO entre los dos equipos es el predictor más potente. Cuando un equipo tiene 200+ puntos más de ELO, el modelo asigna una probabilidad muy alta de victoria. El SHAP beeswarm muestra que `elo_diff` alto (equipo local claramente superior) desplaza masivamente la predicción hacia Home Win; `elo_diff` negativo la desplaza hacia Away Win.

2. **`form_home` / `form_away`** - La forma reciente captura la dinámica actual del equipo. En Mundiales, equipos que llegan en racha de victorias tienen una ventaja psicológica y táctica que el ELO, calculado sobre años de historia, no refleja completamente. El SHAP muestra que la forma tiene un efecto *aditivo* sobre el ELO: un equipo con ELO similar pero mejor forma tiene ~5-8% más de probabilidad de victoria.

3. **`draw_rate_home` + `draw_rate_away`** - Cuando ambos equipos tienen draw rates altos, la probabilidad de empate aumenta considerablemente. El SHAP para la clase Draw muestra que esta interacción es la señal más fuerte para predecir empates.

4. **`gd_home` / `gd_away`** - El goal difference ponderado refleja la dominancia ofensiva y defensiva. Equipos con GD alto no solo ganan, sino que ganan con margen, lo que indica mayor calidad real.

**Factores secundarios pero significativos:**

5. **`h2h_diff`** - En partidos con larga historia de rivalidad, el H2H puede modular ±5% la probabilidad de victoria, capturando "maldiciones" o ventajas psicológicas históricas.

6. **`conf_diff`** - Cuando dos equipos de confederaciones muy diferentes se enfrentan (ej: UEFA vs OFC), la confederación amplifica el efecto del ELO.

7. **`wc_dna`** - El "ADN mundialista" tiene efecto especialmente en rondas avanzadas: equipos con experiencia en cuartos o semis rinden por encima de lo que su ELO predice en esos contextos de alta presión.

8. **`altitude_shock`** - En partidos en Ciudad de México o Guadalajara, equipos no adaptados a la altitud tienen una penalización real que el modelo captura gracias a los datos de las sedes del WC2026.

### SHAP por Clase: ¿Qué hace que el modelo prediga cada resultado?

**Para predecir Home Win:**
- `elo_diff` alto (local claramente superior)
- `form_home` alta + `form_away` baja
- `clean_sheet_home` alto (defensa sólida)
- `wc_dna_home` alto

**Para predecir Draw:**
- `elo_diff` cercano a 0 (equipos parejos)
- `draw_rate_home` y `draw_rate_away` ambos altos
- `form_comp` similares en ambos equipos
- Partidos competitivos donde ninguno se puede permitir arriesgar

**Para predecir Away Win:**
- `elo_diff` muy negativo (visitante claramente superior)
- `form_away` alta
- `penalty_diff` favorece al visitante (relevante en KO)
- `upset_rate` del local alto (equipo local propenso a sorpresas)

### El Árbol Individual - Cómo "razona" el RF

Un árbol de profundidad 3 del Random Forest toma decisiones del estilo:

```
¿elo_diff > 120?
    ├── Sí → ¿form_home > 0.65?
    │           ├── Sí → Home Win (alta confianza)
    │           └── No → ¿draw_rate_avg > 0.30? → Draw / Home Win
    └── No → ¿elo_diff < -80?
                ├── Sí → Away Win
                └── No → ¿draw_rate_avg > 0.35? → Draw / equipos muy parejos
```

El RF promedia cientos de estos árboles con distintas subselecciones de features y partidos, produciendo una predicción estable y robusta.

### Partial Dependence Plots (PDP) e ICE Curves

Los PDP muestran el efecto marginal de cada feature sobre la predicción manteniendo el resto constante:

- **PDP de `elo_diff`:** Curva aproximadamente sigmoidal. Por debajo de -100 puntos, la probabilidad de Away Win supera el 50%. Por encima de +100, domina Home Win. En el rango [-100, +100], los tres resultados compiten y el empate alcanza su máxima probabilidad.

- **PDP de `draw_rate_avg`:** Monotónicamente creciente para la clase Draw. Un draw rate medio del 40% (ambos equipos) incrementa la probabilidad de empate en ~15 puntos porcentuales respecto al 20%.

Las **ICE curves** (Individual Conditional Expectation) muestran que este efecto no es uniforme: para partidos donde `elo_diff` es extremo, `draw_rate` apenas importa (el partido ya está "decidido" por la diferencia de nivel). Solo en partidos parejos `draw_rate` tiene impacto real.

---

## 11. Simulación Monte Carlo WC2026

Con el modelo entrenado y calibrado, se simula el torneo completo **1.000 veces**.

### Metodología

1. **Pre-caching de probabilidades:** Para cada emparejamiento posible del torneo, se calculan las probabilidades (Home Win, Draw, Away Win) *una sola vez* y se almacenan en un diccionario. Las 1.000 simulaciones consultan este cache en O(1) en lugar de recalcular el modelo en cada iteración. Esto supone un **speedup de ~10x**.

2. **Fase de Grupos (12 grupos × 4 equipos):** Se simulan los 6 partidos de cada grupo. Según el resultado (victoria = 3 puntos, empate = 1, derrota = 0), se determina la clasificación. En caso de empate en puntos, se usan criterios de desempate (diferencia de goles, goles a favor).

3. **Fases Eliminatorias:** Los clasificados de cada grupo se emparejan según el bracket del torneo. En rondas KO, si hay empate en 90 minutos, se simula una tanda de penales usando `penalty_strength` de cada equipo.

4. **Factor de Altitud:** Los partidos asignados a Ciudad de México (2.240m) o Guadalajara (1.566m) aplican el `altitude_shock` correspondiente, ajustando las probabilidades de los equipos no adaptados.

5. **Estadísticas de salida:** Tras las 1.000 simulaciones se computan para cada equipo:
   - `p_champion`: Probabilidad de ser campeón
   - `p_top4`: Probabilidad de llegar a semifinales
   - `p_top8`: Probabilidad de llegar a cuartos
   - `p_qualified`: Probabilidad de superar la fase de grupos

### Resultados Destacados

- Los **grandes favoritos** según el modelo son los equipos con mayor ELO, mejor forma reciente y mayor experiencia mundialista (wc_dna alto).
- El **"Grupo de la Muerte"** se identifica mediante la entropía de la distribución de ELO en el grupo: un grupo donde cuatro equipos tienen ELO similar tiene entropía máxima (ningún resultado es predecible), mientras un grupo con un equipo dominante tiene entropía baja.
- El **factor de altitud** beneficia especialmente a selecciones sudamericanas y asiáticas en los partidos jugados en México.

### Persistencia en Oracle ADB

Los resultados de la simulación se escriben de vuelta a Oracle ADB en la tabla `WC2026_PREDICTIONS`, lo que permite:
- Consultar las predicciones desde un dashboard APEX
- Actualizar las probabilidades si se incorporan datos nuevos (partidos de mayo-junio 2026)
- Compartir resultados con otros componentes del sistema sin necesidad de reejecutar el notebook

---

## 12. Decisiones de Diseño - Resumen Razonado

| Decisión | Alternativa descartada | Razón para elegir esta |
|----------|----------------------|------------------------|
| **Split temporal** | Split aleatorio | Respeta causalidad; evita data leakage en serie temporal |
| **Regresión ELO 2% anual** | ELO puro acumulativo | Evita que ELO de décadas pasadas domine sobre rendimiento actual |
| **Half-life 3 años** | 1 año / 10 años | Cubre una generación de jugadores sin diluir la forma actual |
| **Threshold 0.38 para Draw** | Threshold 0.33 (default) | Trade-off explícito: más F1 en Draw a costa de -1.3% accuracy global |
| **Isotonic Regression** | Platt Scaling | No paramétrica; mejor en distribuciones multi-class no-normales |
| **TimeSeriesSplit** | KFold estándar | Evita que el future vea el past en validación cruzada |
| **Stacking vs promedio** | Voting / Bagging | El meta-learner aprende pesos óptimos, no asume pesos iguales |
| **tanh en GD** | GD crudo | Comprime outliers (8-0 ≠ 8× más informativo que 1-0) |
| **Vectorización rolling** | Iteración partido a partido | Speedup 10-20x; crítico con 48.000+ partidos |
| **Pre-caching MC** | Recálculo en cada simulación | Speedup 10x en las 1.000 iteraciones de Monte Carlo |

---

## 13. Limitaciones Honestas

El modelo es sólido dentro de sus restricciones, pero tiene limitaciones importantes que deben tenerse en cuenta:

**Datos no disponibles:**
- **Lesiones y alineaciones:** Si el mejor delantero del equipo está lesionado, el modelo no lo sabe. Una lesión de última hora puede invalidar completamente una predicción basada en historial.
- **Valor de mercado / ratings individuales:** Sin Transfermarkt o FIFA ratings de la plantilla actual, la calidad individual de los jugadores no está capturada.
- **Cuotas de apuestas:** Los mercados de apuestas agregan información de miles de expertos y tienen señal predictiva. No usarlas es una fuente de underperformance respecto a sistemas profesionales.

**Limitaciones estructurales:**
- **Empates:** El recall de los empates es estructuralmente bajo (~49%). El fútbol tiene una aleatoriedad intrínseca en la generación de empates que supera la capacidad predictiva de los datos históricos.
- **WC2026 es extrapolación OOD:** El formato de 48 equipos nunca se ha jugado. El modelo extrapola fuera de distribución: los patrones aprendidos de Mundiales de 32 equipos pueden no transferirse perfectamente.
- **Datos hasta enero 2026:** Los partidos de mayo-junio 2026 (previos al torneo) no están incluidos. Son los más relevantes temporalmente.

**Advertencia honesta:** El objetivo no es adivinar el resultado de cada partido, sino **cuantificar correctamente la incertidumbre**. Un modelo bien calibrado con 53% de accuracy es más útil que uno mal calibrado con 58%.

---

## 14. Estructura del Repositorio

```
paso5/
├── WC2026_ML.ipynb          # Notebook principal con todo el pipeline
├── WC2026_ML.html           # Versión HTML del notebook (visualización sin ejecutar)
├── data/                    # Datos locales (CSV backup de las tablas Oracle)
├── models/                  # Modelos serialziados (.pkl)
├── rf_tree_sample.png       # Visualización de árbol individual del RF
├── .env                     # Credenciales Oracle (NO commitear)
├── pyproject.toml           # Dependencias del proyecto
├── uv.lock                  # Lock file de dependencias
└── README.md                # Este archivo
```

---

## 15. Cómo Reproducir

### Requisitos

```bash
# Instalar dependencias con uv
uv sync

# O con pip
pip install xgboost lightgbm scikit-learn shap plotly oracledb python-dotenv scipy numpy pandas
```

### Configuración

Crear archivo `.env` en la raíz del proyecto con las credenciales de Oracle ADB:

```
ORACLE_USER=...
ORACLE_PASSWORD=...
ORACLE_DSN=...
ORACLE_WALLET_DIR=...
```

### Ejecución

```bash
# Ejecutar el notebook completo
jupyter nbconvert --to notebook --execute WC2026_ML.ipynb

# O abrirlo interactivamente
jupyter lab WC2026_ML.ipynb
```

El notebook ejecuta el pipeline completo en orden:
1. Conexión a Oracle ADB y carga de datos
2. EDA y visualizaciones
3. Feature engineering (ELO, time decay, rolling features, etc.)
4. Entrenamiento del ensemble stacking
5. Calibración de probabilidades
6. Evaluación y métricas
7. Backtesting WC2018 / WC2022
8. Interpretación SHAP y RF
9. Simulación Monte Carlo WC2026
10. Write-back de predicciones a Oracle ADB

**Reproducibilidad:** El notebook usa `random_state=42` en todos los modelos y operaciones estocásticas para garantizar resultados idénticos en cada ejecución.

---

*Oracle Hackathon 4.0 · Abril 2026 · Equipo MXNJ*
