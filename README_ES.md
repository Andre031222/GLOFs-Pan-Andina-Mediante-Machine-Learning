# Evaluacion de Susceptibilidad Pan-Andina a Inundaciones por Vaciamiento de Lagos Glaciares Mediante Aprendizaje Automatico e Imagenes Sentinel-2 Multitemporales

**Estado:** Manuscrito enviado para revision — Mayo 2026
**Revista objetivo:** IJACSA — International Journal of Advanced Computer Science and Applications (https://thesai.org/Publications/IJACSA)
**Institucion:** Facultad de Ingenieria Estadistica e Informatica, Universidad Nacional del Altiplano (UNAP), Puno, Peru

---

## Descripcion general

Este repositorio contiene el pipeline de investigacion completo, modelos entrenados, figuras y el manuscrito LaTeX para una evaluacion de susceptibilidad a GLOFs (Glacial Lake Outburst Floods) a escala pan-Andina. El estudio compila un inventario multitemporal de lagos glaciares a partir de nueve anos de imagenes Sentinel-2 (2017-2025) en diez cordilleras de cuatro paises, y entrena un modelo de aprendizaje automatico para identificar lagos con susceptibilidad elevada a GLOF. El algoritmo Balanced Random Forest obtuvo ROC-AUC = 0.821 (IC 95%: [0.737, 0.901]) bajo validacion cruzada out-of-fold, con transferabilidad geografica confirmada mediante validacion Leave-One-Country-Out en tres paises.

Este es el primer modelo de susceptibilidad a GLOFs de escala pan-Andina validado a traves de fronteras nacionales utilizando datos satelitales multitemporales.

---

## Resumen

Las inundaciones por vaciamiento de lagos glaciares (GLOFs) representan un peligro critico en los Andes tropicales y subtropicales. Este estudio presenta la primera evaluacion pan-Andina de susceptibilidad a GLOFs, integrando un inventario multitemporal de lagos glaciares Sentinel-2 L2A (12,588 observaciones lago-ano; 2017-2025) en diez cordilleras de Peru, Bolivia, Ecuador y Chile, con 57 caracteristicas morfometricas, topograficas, hidrologicas y temporales por observacion. Se entrenaron seis clasificadores bajo validacion cruzada out-of-fold (OOF) para abordar el desbalance extremo de clases (1:787). El Balanced Random Forest obtuvo la mayor discriminacion (ROC-AUC = 0.821; IC 95%: [0.737, 0.901]; Lift = 8.5x sobre la linea base aleatoria). La validacion Leave-One-Country-Out (LOCO) confirmo la transferabilidad geografica, con todos los paises retenidos alcanzando Lift > 3.7x. Seis procedimientos estadisticos complementarios (test de permutacion, DeLong, I de Moran, KS, Mann-Whitney U, jackknife) confirmaron la robustez del modelo. El analisis SHAP identifico la proximidad al glaciar y la morfometria del lago como los conductores dominantes de susceptibilidad, consistentes con la teoria de inestabilidad de diques de morrena. La calibracion isotonica post-hoc mapea los scores crudos a estimaciones de probabilidad operativa. El mapa de susceptibilidad resultante prioriza aproximadamente 5,030 lagos de alto riesgo en la region de estudio.

**Palabras clave:** susceptibilidad GLOF; aprendizaje automatico; Sentinel-2; lagos glaciares; Andes; Balanced Random Forest; SHAP; Leave-One-Country-Out; peligros criosfericos

---

## Resultados principales

### Rendimiento del modelo (validacion OOF)

| Clasificador | ROC-AUC | Lift | MCC |
|---|---|---|---|
| **Balanced Random Forest** | **0.821** | **8.5x** | 0.038 |
| LightGBM | 0.793 | 5.0x | -0.002 |
| XGBoost | 0.783 | 3.3x | 0.026 |
| Random Forest | 0.783 | 4.1x | 0.029 |
| Easy Ensemble | 0.701 | 4.8x | 0.032 |
| Regresion Logistica | 0.698 | 18.7x* | 0.018 |

*El Lift de Regresion Logistica esta inflado por un umbral ultra-conservador; solo recupera Recall = 12%.

IC Bootstrap 95% para BalancedRF: [0.737, 0.901]. Test de permutacion p < 0.0002.

### Validacion geografica (LOCO-CV)

| Pais retenido | n_test | n_GLOF | ROC-AUC | Lift | FPR@0.304 |
|---|---|---|---|---|---|
| Ecuador | 273 | 1* | 0.993 | 68.2x | 37.7% |
| Bolivia | 180 | 1* | 0.740 | 3.7x | 36.1% |
| Peru | 10,546 | 14 | 0.707 | 4.2x | 42.6% |
| Chile | 1,589 | 0 | dominio null | --- | 29.6% |

*Un solo label positivo en test: ROC-AUC es solo indicativo.

### Validacion estadistica (seis procedimientos)

| Test | Resultado | Interpretacion |
|---|---|---|
| Test de permutacion (5,000 iteraciones) | p < 0.0002 | Rechaza hipotesis nula |
| Test DeLong (BRF vs LightGBM) | z=0.94, p=0.350 | Diferencia no significativa |
| I de Moran (residuos OOF) | I=0.003, p=0.805 | Sin autocorrelacion espacial |
| Test KS (distribuciones de scores) | D=0.541, p=6.5e-5 | Separacion significativa |
| Mann-Whitney U (7 features) | p<0.05 (Bonferroni x8) | Features dominantes confirmadas |
| Jackknife (dejar-un-GLOF-fuera) | Rango AUC 0.815-0.845, caida maxima 0.012 | Ninguna etiqueta domina el resultado |

### Calibracion de scores

Calibracion isotonica post-hoc (compute_calibration.py):
- Brier Skill Score: BSS = -139.8 (crudo) -> BSS = +0.008 (calibrado)
- Brier score: 0.179 -> 0.001
- KS D = 0.541 preservado (el ranking no cambia)

### Mapa de susceptibilidad

- Umbral optimo de Youden: t = 0.304
- Lagos de alto riesgo: 5,030 de 12,588 (40.0%)
- Cordilleras de mayor riesgo: Raura 55.7%, Huayhuash 54.0%, Blanca 51.5%
- Cordilleras Blanca, Huayhuash y Raura concentran aproximadamente el 50% de los lagos de alto riesgo

### Conductores SHAP dominantes

1. Distancia al glaciar (m) -- proximidad a la fuente de hielo; pico de riesgo no lineal a 2-15 km
2. Perimetro del lago (m) -- complejidad morfologica
3. Incertidumbre del ensemble de profundidad -- proxy de inestabilidad estructural
4. Profundidad media (m) -- potencial de volumen de agua
5. Elevacion maxima (m) -- potencial de energia gravitacional

---

## Estructura del repositorio

```
03.-ML_Based Pan_Andean/
|
|
|-- figures/
|   |-- publication/                   Figuras de publicacion (600 DPI PNG + PDF)
|   |   |-- fig1_susceptibility_map.{pdf,png}          Mapa de susceptibilidad pan-Andino
|   |   |-- fig2_inventory_glof_context.{pdf,png}      Inventario de lagos y contexto GLOF
|   |   |-- fig3_susceptibility_distribution.{pdf,png} Distribuciones de scores
|   |   |-- fig4_model_performance.{pdf,png}           Curvas ROC, PR, LOCO, Top-K
|   |   |-- fig5_shap_risk_drivers.{pdf,png}           SHAP beeswarm y dependencia
|   |   |-- fig6_robustness_analysis.{pdf,png}         IC bootstrap, permutacion, lift/recall
|   |   |-- fig7_supplementary_validation.{pdf,png}    Test KS, calibracion, mapa top-100
|   |   `-- areas/                                     10 mapas de zoom por cordillera
|   `-- panels/                        Sub-paneles intermedios (generados por los scripts)
|
|-- models/
|   |-- best_model.joblib              BalancedRandomForest (n_estimators=500, final)
|   |-- best_model_calibrated.joblib   Version con calibracion isotonica
|   |-- easyensemble.joblib            Clasificador competidor
|   |-- lightgbm.joblib                Clasificador competidor
|   |-- logreg.joblib                  Clasificador competidor
|   |-- randomforest.joblib            Clasificador competidor
|   |-- xgboost.joblib                 Clasificador competidor
|   |-- scaler.joblib                  Escalador de features
|   |-- oof_curves.pkl                 Curvas ROC/PR OOF para todos los clasificadores
|   |-- oof_predictions.json           y_true, y_score_brf, y_score_lgbm (12,588 filas)
|   |-- loco_validation.csv            Resultados LOCO-CV por pais
|   |-- model_comparison.csv           Metricas OOF para los 6 clasificadores
|   |-- robustness_results.json        IC Bootstrap (2,000 remuestras) + test de permutacion
|   |-- statistical_tests.json         DeLong, I de Moran, KS, Mann-Whitney
|   |-- delong_test.json               Resultado del test pairwise DeLong
|   |-- jackknife_results.json         AUC jackknife (dejar-un-GLOF-fuera, 16 estimaciones)
|   |-- calibrated_oof.json            Scores OOF calibrados post-hoc
|   |-- extra_validation.json          Metricas de validacion prospectiva adicionales
|   |-- improvements.json              Analisis de mejora y robustez
|   |-- feature_names.json             57 nombres de features en orden
|   `-- optimal_thresholds.json        Umbrales Youden y otros
|
|-- scripts/
|   |-- generate_fig01.py              Fig 1: Mapa de susceptibilidad pan-Andino
|   |-- generate_fig01_areas.py        10 mapas de zoom por cordillera (Apendice A)
|   |-- generate_fig02.py              Fig 2: Inventario y contexto GLOF
|   |-- generate_fig03.py              Fig 3: Distribuciones de susceptibilidad
|   |-- generate_fig04.py              Fig 4: Rendimiento del modelo (ROC, PR, LOCO, Top-K)
|   |-- generate_fig05.py              Fig 5: Conductores de riesgo SHAP
|   |-- generate_fig06.py              Fig 6: Analisis de robustez
|   |-- generate_fig07.py              Fig 7: Validacion suplementaria
|   |-- compute_jackknife.py           Analisis de estabilidad AUC jackknife
|   |-- compute_calibration.py         Calibracion isotonica + Platt post-hoc
|   |-- compute_statistical_tests.py   Tests DeLong, I de Moran, Mann-Whitney U
|   |-- compute_extra_validation.py    Validacion prospectiva (top-100 lagos)
|   |-- compute_improvements.py        Analisis de mejora y robustez
|   |-- recompute_stats.py             Recalcula todas las estadisticas desde predicciones OOF
|   |-- add_dist_glacier_to_features.py  Agrega distancia al glaciar post-hoc
|   `-- compute_dist_glacier.py        Calcula distancia al glaciar para todos los lagos-ano
|
|-- notebooks/
|   |-- 00_environment_setup.ipynb
|   |-- 01-10_download_*.ipynb         Descargas Sentinel-2 + DEM por cordillera
|   |-- 11_dem_processing.ipynb        Derivados de terreno (pendiente, TRI, freeboard)
|   |-- 12_lake_detection.ipynb        Umbral MNDWI + filtrado morfologico
|   |-- 13_feature_extraction.ipynb    Construccion de matriz de 57 features por lago-ano
|   |-- 14_historical_glofs.ipynb      Inventario GLOF + asignacion de etiquetas (buffer 5 km)
|   |-- 15_model_training.ipynb        Entrenamiento OOF de 6 clasificadores + LOCO-CV
|   |-- 16_threshold_analysis.ipynb    Umbral Youden, seleccion del punto de operacion
|   |-- 17_shap_interpretation.ipynb   Analisis SHAP TreeExplainer
|   |-- 18_visualization.ipynb         Figuras de publicacion
|   `-- 19_robustness_analysis.ipynb   IC bootstrap, test de permutacion, autocorrelacion espacial
|
|-- data/
|   |-- raw/                           Archivos JSON de metadatos por area (10 cordilleras),
|   |                                  contornos, tiles DEM
|   `-- processed/                     Matrices de features, datasets etiquetados,
|                                      importancias SHAP, resumen de terreno
|
|-- src/
|   |-- download.py                    Utilidades de descarga via Planetary Computer STAC API
|   |-- data_validator.py              Funciones de validacion de calidad de datos
|   |-- visualization.py               Utilidades de visualizacion compartidas
|   `-- gpu_utils.py                   Helpers para seleccion GPU/CPU
|
|-- glof/                              Entorno virtual Python (Python 3.11.9, Windows)
|-- requirements.txt                   Lista completa de dependencias
|-- requirements_win.txt               Lista compatible con Windows (sin cupy)
|-- activate_env.sh                    Script de activacion del entorno
`-- README_ES.md                       Este archivo
```

---

## Configuracion del entorno

### Entorno virtual Python

El proyecto usa un entorno virtual dedicado ubicado en `glof/` en la raiz del proyecto.

**Windows (PowerShell o Git Bash):**
```bash
# Activar
source glof/Scripts/activate

# O mediante el script auxiliar
bash activate_env.sh
```

**Nota:** cupy no esta instalado en Windows sin una GPU CUDA configurada. Usar `requirements_win.txt` para instalar todas las dependencias sin cupy.

```bash
pip install -r requirements_win.txt
```

### Dependencias principales

| Paquete | Version | Proposito |
|---|---|---|
| scikit-learn | >= 1.3 | Pipeline de aprendizaje automatico |
| imbalanced-learn | >= 0.11 | BalancedRandomForest, SMOTETomek |
| shap | 0.49.1 | SHAP TreeExplainer |
| geopandas | >= 0.14 | Operaciones espaciales |
| lightgbm | ultima | Clasificador LightGBM |
| xgboost | ultima | Clasificador XGBoost |
| contextily | 1.7 | Tiles de mapa base (CartoDB Voyager) |
| matplotlib | >= 3.7 | Figuras |
| pandas / numpy | ultima | Procesamiento de datos |
| joblib | ultima | Serializacion de modelos |

---

## Orden de ejecucion del pipeline

El pipeline completo se ejecuta a traves de 19 notebooks Jupyter en secuencia. Cada notebook guarda las salidas que consume el siguiente.

```
NB00: Verificacion del entorno
NB01-10: Descargas Sentinel-2 + DEM (por cordillera, 10 areas)
NB11: Procesamiento DEM -> pendiente, TRI, freeboard, acumulacion de flujo
NB12: Deteccion de lagos -> 12,588 observaciones lago-ano
NB13: Extraccion de features -> matriz de 57 features
NB14: Emparejamiento GLOF historico -> 16 etiquetas positivas (buffer 5,000 m)
NB15: Entrenamiento del modelo -> 6 clasificadores, OOF + LOCO-CV
NB16: Analisis de umbral -> Youden t = 0.304
NB17: Interpretacion SHAP -> importancia de features, graficos de dependencia
NB18: Visualizacion -> figuras de publicacion
NB19: Analisis de robustez -> IC bootstrap, test de permutacion, I de Moran
```

Tiempo de ejecucion total estimado: 2-4 horas en CPU (se recomienda GPU para NB12-NB13).

### Regenerar figuras

Cada figura tiene un script dedicado en `scripts/`. Ejecutar desde la raiz del proyecto con el entorno virtual activado:

```bash
python scripts/generate_fig01.py        # Mapa pan-Andino (requiere dataset completo)
python scripts/generate_fig01_areas.py  # 10 mapas de zoom por cordillera
python scripts/generate_fig02.py
python scripts/generate_fig03.py
python scripts/generate_fig04.py
python scripts/generate_fig05.py
python scripts/generate_fig06.py
python scripts/generate_fig07.py
```

Las figuras se guardan en `figures/publication/` a 600 DPI.

### Recalcular resultados estadisticos

Si las predicciones OOF cambian, recalcular todas las estadisticas:

```bash
python scripts/recompute_stats.py
python scripts/compute_jackknife.py
python scripts/compute_calibration.py
python scripts/compute_statistical_tests.py
```

## Area de estudio

| Cordillera | Pais | Latitud | n_lagos | n_GLOF | Alto riesgo (%) |
|---|---|---|---|---|---|
| Blanca | Peru | 8-10 S | 3,083 | 9 | 51.5 |
| Vilcanota | Peru | 13-15 S | 2,770 | 0 | 34.9 |
| Central | Peru | 11-12 S | 2,239 | 0 | 28.9 |
| Andes Centrales | Chile | 33-35 S | 1,589 | 0 | 29.6 |
| Raura | Peru | 10-11 S | 1,361 | 2 | 55.7 |
| Urubamba | Peru | 13 S | 430 | 0 | 35.8 |
| Huanzo | Peru | 15 S | 385 | 0 | 32.7 |
| Huayhuash | Peru | 10 S | 278 | 3 | 54.0 |
| Antisana | Ecuador | 0.3 N-S | 273 | 1 | 37.7 |
| Real | Bolivia | 15-17 S | 180 | 1 | 36.1 |
| **Total** | | | **12,588** | **16** | **40.0** |

Rango latitudinal: 0.3 N a 35.5 S. Inventario GLOF: 52 eventos historicos (1932-2021), 16 emparejados dentro de 5,000 m de una observacion lago-ano (razon de clases 1:787).

---

## Ingenieria de features

57 features organizadas en cinco grupos:

| Grupo | Cantidad | Ejemplos |
|---|---|---|
| Morfometricas | 10 | area, perimetro, compacidad, elongacion, dimension fractal |
| Topograficas | 13 | elevacion (media/min/max/std), pendiente, TRI, freeboard, altura de represa |
| Hidrologicas/Profundidad | 12 | profundidad de 4 modelos empiricos (Cook, Huggel, Yao, O'Connor), media/std del ensemble, volumen |
| Temporales | 12 | area anual 2017-2025, tasa de crecimiento, cambio total, crecimiento relativo |
| Ingenieria | 10 | transformaciones logaritmicas, interaccion pendiente x log(area), indicador binario de ausencia de glaciar |

Los valores faltantes se imputaron con medianas por columna calculadas dentro de cada fold de entrenamiento (sin fuga de datos del conjunto de prueba). Las features de profundidad tuvieron la mayor tasa de valores faltantes (17.5%), reflejando la ausencia de sondeos empiricos para lagos pequenos o recientemente formados.

---

## Pipeline de aprendizaje automatico

- **Estrategia de desbalance:** BalancedRandomForest (bootstrap balanceado interno por arbol) con SMOTETomek (preprocesamiento externo, aplicado solo a folds de entrenamiento)
- **Validacion:** Validacion cruzada out-of-fold (OOF) estratificada de 5 folds
- **Validacion geografica:** Leave-One-Country-Out (LOCO-CV) para Ecuador, Peru, Bolivia
- **Hiperparametros:** Valores por defecto de scikit-learn/imbalanced-learn; sin ajuste adicional (para no inflar estimaciones de rendimiento con n+ = 16)
- **Interpretabilidad:** SHAP TreeExplainer (descomposiciones exactas de Shapley para ensembles de arboles)
- **Calibracion:** Regresion isotonica post-hoc sobre scores OOF

---

## Limitaciones del estudio

1. El inventario GLOF emparejado contiene solo 16 etiquetas positivas (de 52 eventos historicos). Los 36 restantes no se emparejaron porque son anteriores a la base temporal Sentinel-2 de 2017 o tienen coordenadas correspondientes a zonas de impacto aguas abajo en lugar de centroides del lago fuente, lo cual es una caracteristica sistematica de los catalogos historicos de GLOFs.
2. La resolucion de Sentinel-2 a 10 m no resuelve caracteristicas sub-pixel de diques de morrena con nucleo de hielo.
3. El modelo captura susceptibilidad estructural, no probabilidad de evento dinamico (no se incluyen disparadores de sismicidad o precipitacion).
4. El alcance temporal (2017-2025) excluye lagos formados y vaciados antes de la era Sentinel-2.
5. No se ha realizado una evaluacion de precision basada en campo. La validacion prospectiva usando los 100 lagos mejor puntuados muestra perfiles morfometricos consistentes con potencial de peligro elevado, pero la confirmacion directa en campo requeriria levantamientos de al menos 50 lagos por cordillera.

---

## Autores y afiliaciones

| Autor | Rol | Email | ORCID |
|---|---|---|---|
| Richar Andre Vilca Solorzano (autor de correspondencia) | Lider: diseno del estudio, pipeline ML, manuscrito | 75521963@est.unap.edu.pe | 0009-0003-2385-5263 |
| Dina Maribel Yana-Yucra | Procesamiento de datos, validacion espacial | maribelbel201314@gmail.com | 0009-0003-6218-2735 |
| Milton Vladimir Mamani Calisaya | Analisis estadistico, interpretacion | mmamanic@unap.edu.pe | 0000-0002-0676-0989 |
| Fred Torres-Cruz | Supervision, revision del manuscrito | ftorres@unap.edu.pe | 0000-0003-0834-6834 |

Facultad de Ingenieria Estadistica e Informatica
Universidad Nacional del Altiplano (UNAP), Puno, Peru

---

## Cita

Vilca Solorzano, R.A., Yana-Yucra, D.M., Mamani Calisaya, M.V., y Torres-Cruz, F. (2026).
Machine Learning-Based Pan-Andean Glacial Lake Outburst Flood Susceptibility Assessment
Using Multitemporal Sentinel-2 Imagery. International Journal of Advanced Computer Science
and Applications (IJACSA). DOI: [asignado al aceptarse]

---

## Disponibilidad de datos y codigo

Codigo, modelos entrenados y scripts de generacion de figuras:
https://github.com/Andre031222/GLOFs-Pan-Andina-Mediante-Machine-Learning
Licencia: MIT

Imagenes Sentinel-2: ESA Copernicus Open Access Hub (acceso libre)
DEM NASADEM: NASA EARTHDATA (acceso libre)
Contornos de glaciares RGI v7.0: GLIMS/NSIDC (acceso libre)
Inventario GLOF: GloFLOD (Lutzow et al., 2023), Emmer et al. (2022), INAIGEM (2018)

---

## Licencia

Codigo: Licencia MIT
Datos derivados de ESA Copernicus: sujetos a los terminos de acceso abierto de Copernicus
Datos del inventario GLOF: derivados de bases de datos publicadas de acceso abierto (ver citas)

---

*Ultima actualizacion: Mayo 2026*
