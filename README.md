# Pan-Andean Glacial Lake Outburst Flood Susceptibility Assessment Using Machine Learning and Multitemporal Sentinel-2 Imagery

**Status:** Manuscript submitted for review — May 2026
**Target journal:** IJACSA — International Journal of Advanced Computer Science and Applications (https://thesai.org/Publications/IJACSA)
**Manuscript:** `ijacsa_paper/manuscript_ijacsa.tex` (20 pages, compiled)
**Blind version:** `ijacsa_paper/manuscript_ijacsa_blind.tex` (19 pages, double-blind submission copy)
**Institution:** Faculty of Statistical and Computer Engineering, Universidad Nacional del Altiplano (UNAP), Puno, Peru

---

## Overview

This repository contains the full research pipeline, trained models, figures, and LaTeX manuscript for a pan-Andean glacial lake outburst flood (GLOF) susceptibility assessment. The study compiles a multitemporal glacial lake inventory from nine years of Sentinel-2 imagery (2017-2025) across ten glacial cordilleras in four countries, and trains a machine learning model to identify lakes with elevated GLOF susceptibility. The Balanced Random Forest algorithm achieved ROC-AUC = 0.821 (95% CI: [0.737, 0.901]) under out-of-fold cross-validation, with geographic generalizability confirmed by Leave-One-Country-Out validation across three countries.

This is the first pan-Andean, machine-learning-based GLOF susceptibility model validated across national boundaries using multitemporal satellite data.

---

## Abstract

Glacial lake outburst floods (GLOFs) represent a major cryospheric hazard in the tropical and subtropical Andes, yet no validated machine learning susceptibility model had previously spanned the full Andean arc. This study presents the first pan-Andean GLOF susceptibility assessment, integrating a multitemporal Sentinel-2 L2A glacial lake inventory (12,588 lake-year observations; 2017-2025) across ten cordilleras in Peru, Bolivia, Ecuador, and Chile with 57 morphometric, topographic, hydrological, and temporal features per lake-year. Six classifiers were trained under out-of-fold (OOF) cross-validation to address extreme class imbalance (1:787). The Balanced Random Forest achieved the highest discrimination (ROC-AUC = 0.821; 95% CI: [0.737, 0.901]; Lift = 8.5x above random baseline). Leave-One-Country-Out (LOCO) validation confirmed geographic generalizability, with all withheld countries attaining Lift > 3.7x. Six complementary statistical procedures (permutation test, DeLong, Moran's I, KS test, Mann-Whitney U, jackknife) confirmed robustness. SHAP analysis identified glacier proximity and lake morphometry as dominant susceptibility drivers, consistent with moraine dam instability theory. Post-hoc isotonic calibration maps raw scores to operational probability estimates. The resulting susceptibility map prioritizes approximately 5,030 high-risk lakes across the study region.

**Keywords:** GLOF susceptibility; machine learning; Sentinel-2; glacial lakes; Andes; Balanced Random Forest; SHAP; Leave-One-Country-Out; cryospheric hazards

---

## Key Results

### Model Performance (OOF Cross-Validation)

| Classifier | ROC-AUC | Lift | MCC |
|---|---|---|---|
| **Balanced Random Forest** | **0.821** | **8.5x** | 0.038 |
| LightGBM | 0.793 | 5.0x | -0.002 |
| XGBoost | 0.783 | 3.3x | 0.026 |
| Random Forest | 0.783 | 4.1x | 0.029 |
| Easy Ensemble | 0.701 | 4.8x | 0.032 |
| Logistic Regression | 0.698 | 18.7x* | 0.018 |

*LR Lift is inflated by an ultra-conservative threshold recovering only Recall = 12%.

Bootstrap 95% CI for BalancedRF: [0.737, 0.901]. Permutation test p < 0.0002.

### Geographic Validation (LOCO-CV)

| Country withheld | n_test | n_GLOF | ROC-AUC | Lift | FPR@0.304 |
|---|---|---|---|---|---|
| Ecuador | 273 | 1* | 0.993 | 68.2x | 37.7% |
| Bolivia | 180 | 1* | 0.740 | 3.7x | 36.1% |
| Peru | 10,546 | 14 | 0.707 | 4.2x | 42.6% |
| Chile | 1,589 | 0 | null-class | --- | 29.6% |

*Single positive test label: ROC-AUC is directional only.

### Statistical Validation (Six Procedures)

| Test | Result | Interpretation |
|---|---|---|
| Permutation test (5,000 shuffles) | p < 0.0002 | Rejects null hypothesis |
| DeLong test (BRF vs LightGBM) | z=0.94, p=0.350 | No significant difference |
| Moran's I (OOF residuals) | I=0.003, p=0.805 | No spatial autocorrelation |
| KS test (score distributions) | D=0.541, p=6.5e-5 | Significant separation |
| Mann-Whitney U (7 features) | p<0.05 (Bonferroni x8) | Dominant features confirmed |
| Jackknife (leave-one-GLOF-out) | AUC range 0.815-0.845, max drop 0.012 | No single label dominates |

### Score Calibration

Post-hoc isotonic regression calibration (compute_calibration.py):
- Brier Skill Score: BSS = -139.8 (raw) -> BSS = +0.008 (calibrated)
- Brier score: 0.179 -> 0.001
- KS D = 0.541 preserved (ranking unchanged)

### Susceptibility Map

- Youden optimal threshold: t = 0.304
- High-risk lakes: 5,030 of 12,588 (40.0%)
- Highest-risk cordilleras: Raura 55.7%, Huayhuash 54.0%, Blanca 51.5%
- Cordilleras Blanca, Huayhuash, and Raura together contain approximately 50% of high-risk lakes

### Dominant SHAP Drivers

1. Glacier distance (m) -- proximity to ice source; non-linear risk peak at 2-15 km
2. Lake perimeter (m) -- morphological complexity
3. Depth ensemble uncertainty -- proxy for structural instability
4. Mean depth (m) -- water volume potential
5. Maximum elevation (m) -- gravitational energy potential

---

## Repository Structure

```
03.-ML_Based Pan_Andean/
|
|-- ijacsa_paper/
|   |-- manuscript_ijacsa.tex          Main LaTeX manuscript (IJACSA/IEEEtran, 20 pages)
|   |-- manuscript_ijacsa.pdf          Compiled PDF for authors (41 MB)
|   |-- manuscript_ijacsa.bbl          Compiled bibliography (95 bibliography items)
|   |-- manuscript_ijacsa_blind.tex    Double-blind version for IJACSA submission
|   |-- manuscript_ijacsa_blind.pdf    Compiled blind PDF (19 pages)
|   |-- manuscript_ijacsa_blind.bbl    Bibliography copy for blind version
|   |-- references_glof.bib            BibTeX database (109 entries)
|   |-- cover_letter.tex               Submission cover letter
|   `-- cover_letter.pdf               Compiled cover letter
|
|-- figures/
|   |-- publication/                   Publication-quality figures (600 DPI PNG + PDF)
|   |   |-- fig1_susceptibility_map.{pdf,png}          Pan-Andean susceptibility map
|   |   |-- fig2_inventory_glof_context.{pdf,png}      Lake inventory and GLOF context
|   |   |-- fig3_susceptibility_distribution.{pdf,png} Susceptibility score distributions
|   |   |-- fig4_model_performance.{pdf,png}           ROC, PR, LOCO, Top-K curves
|   |   |-- fig5_shap_risk_drivers.{pdf,png}           SHAP beeswarm and dependence plots
|   |   |-- fig6_robustness_analysis.{pdf,png}         Bootstrap CI, permutation, lift/recall
|   |   |-- fig7_supplementary_validation.{pdf,png}    KS test, calibration, top-100 map
|   |   `-- areas/                                     10 individual cordillera zoom maps
|   `-- panels/                        Intermediate sub-panels (generated by figure scripts)
|
|-- models/
|   |-- best_model.joblib              BalancedRandomForest (n_estimators=500, final)
|   |-- best_model_calibrated.joblib   Isotonic-calibrated version
|   |-- easyensemble.joblib            Competing classifier
|   |-- lightgbm.joblib                Competing classifier
|   |-- logreg.joblib                  Competing classifier
|   |-- randomforest.joblib            Competing classifier
|   |-- xgboost.joblib                 Competing classifier
|   |-- scaler.joblib                  Feature scaler
|   |-- oof_curves.pkl                 OOF ROC/PR curves for all classifiers
|   |-- oof_predictions.json           y_true, y_score_brf, y_score_lgbm (12,588 rows)
|   |-- loco_validation.csv            LOCO-CV results per country
|   |-- model_comparison.csv           OOF metrics for all 6 classifiers
|   |-- robustness_results.json        Bootstrap CI (2,000 resamples) + permutation test
|   |-- statistical_tests.json         DeLong, Moran's I, KS test, Mann-Whitney results
|   |-- delong_test.json               DeLong pairwise test result
|   |-- jackknife_results.json         Leave-one-GLOF-out AUC stability (16 estimates)
|   |-- calibrated_oof.json            Post-hoc calibrated OOF scores
|   |-- extra_validation.json          Additional prospective validation metrics
|   |-- improvements.json             Improvement analysis results
|   |-- feature_names.json             57 feature names in order
|   |-- optimal_thresholds.json        Youden and other thresholds
|
|-- scripts/
|   |-- generate_fig01.py              Fig 1: Pan-Andean susceptibility map
|   |-- generate_fig01_areas.py        10 individual cordillera zoom maps (Appendix A)
|   |-- generate_fig02.py              Fig 2: Inventory and GLOF context
|   |-- generate_fig03.py              Fig 3: Susceptibility distributions
|   |-- generate_fig04.py              Fig 4: Model performance (ROC, PR, LOCO, Top-K)
|   |-- generate_fig05.py              Fig 5: SHAP risk drivers
|   |-- generate_fig06.py              Fig 6: Robustness analysis
|   |-- generate_fig07.py              Fig 7: Supplementary validation
|   |-- compute_jackknife.py           Leave-one-GLOF-out AUC stability analysis
|   |-- compute_calibration.py         Post-hoc isotonic + Platt calibration
|   |-- compute_statistical_tests.py   DeLong, Moran's I, Mann-Whitney U tests
|   |-- compute_extra_validation.py    Prospective validation (top-100 lakes)
|   |-- compute_improvements.py        Improvement and robustness analysis
|   |-- recompute_stats.py             Recompute all statistical results from OOF predictions
|   |-- add_dist_glacier_to_features.py  Add glacier distance feature post-hoc
|   `-- compute_dist_glacier.py        Compute distance-to-glacier for all lake-years
|
|-- notebooks/
|   |-- 00_environment_setup.ipynb
|   |-- 01-10_download_*.ipynb         Per-cordillera Sentinel-2 + DEM downloads
|   |-- 11_dem_processing.ipynb        Terrain derivatives (slope, TRI, freeboard)
|   |-- 12_lake_detection.ipynb        MNDWI thresholding + morphological filtering
|   |-- 13_feature_extraction.ipynb    57-feature matrix construction per lake-year
|   |-- 14_historical_glofs.ipynb      GLOF inventory + spatial label assignment (5 km buffer)
|   |-- 15_model_training.ipynb        6-classifier OOF training + LOCO-CV
|   |-- 16_threshold_analysis.ipynb    Youden threshold, operating point selection
|   |-- 17_shap_interpretation.ipynb   SHAP TreeExplainer analysis
|   |-- 18_visualization.ipynb         Publication figures
|   `-- 19_robustness_analysis.ipynb   Bootstrap CI, permutation test, spatial autocorrelation
|
|-- data/
|   |-- raw/                           Study-area metadata JSON files (10 cordilleras),
|   |                                  boundaries, DEM tiles
|   `-- processed/                     Feature matrices, labeled datasets, SHAP importances,
|                                      terrain summary, threshold summary
|
|-- src/
|   |-- download.py                    Planetary Computer STAC API download utilities
|   |-- data_validator.py              Data quality validation functions
|   |-- visualization.py               Shared visualization utilities
|   `-- gpu_utils.py                   GPU/CPU device selection helpers
|
|-- glof/                              Python virtual environment (Python 3.11.9, Windows)
|-- requirements.txt                   Full dependency list
|-- requirements_win.txt               Windows-compatible list (without cupy)
|-- activate_env.sh                    Environment activation helper
|-- INVESTIGACION_ARTICULO.md          Detailed research plan (Spanish)
|-- JOURNAL_TARGETS.md                 Target journal analysis and submission checklist
`-- README.md                          This file
```

---

## Environment Setup

### Python Virtual Environment

The project uses a dedicated virtual environment located at `glof/` in the project root.

**Windows (PowerShell or Git Bash):**
```bash
# Activate
source glof/Scripts/activate

# Or via the helper script
bash activate_env.sh
```

**Note:** cupy is not installed on Windows without a configured CUDA GPU. Use `requirements_win.txt` to install all dependencies except cupy.

```bash
pip install -r requirements_win.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | >= 1.3 | Machine learning pipeline |
| imbalanced-learn | >= 0.11 | BalancedRandomForest, SMOTETomek |
| shap | 0.49.1 | SHAP TreeExplainer |
| geopandas | >= 0.14 | Spatial operations |
| lightgbm | latest | LightGBM classifier |
| xgboost | latest | XGBoost classifier |
| contextily | 1.7 | Basemap tiles (CartoDB Voyager) |
| matplotlib | >= 3.7 | Figures |
| pandas / numpy | latest | Data processing |
| joblib | latest | Model serialization |

---

## Pipeline Execution Order

The full pipeline runs through 19 Jupyter notebooks in sequence. Each notebook saves outputs consumed by the next.

```
NB00: Environment verification
NB01-10: Sentinel-2 + DEM downloads (per cordillera, 10 areas)
NB11: DEM processing -> slope, TRI, freeboard, flow accumulation
NB12: Lake detection -> 12,588 lake-year observations
NB13: Feature extraction -> 57-feature matrix
NB14: Historical GLOF matching -> 16 positive labels (buffer 5,000 m)
NB15: Model training -> 6 classifiers, OOF + LOCO-CV
NB16: Threshold analysis -> Youden t = 0.304
NB17: SHAP interpretation -> feature importance, dependence plots
NB18: Visualization -> publication figures
NB19: Robustness analysis -> bootstrap CI, permutation test, Moran's I
```

Estimated total runtime: 2-4 hours on CPU (GPU recommended for NB12-NB13).

### Regenerating Figures

Each figure has a dedicated script in `scripts/`. Run from the project root with the venv activated:

```bash
python scripts/generate_fig01.py        # Pan-Andean map (requires full dataset)
python scripts/generate_fig01_areas.py  # 10 cordillera zoom maps
python scripts/generate_fig02.py
python scripts/generate_fig03.py
python scripts/generate_fig04.py
python scripts/generate_fig05.py
python scripts/generate_fig06.py
python scripts/generate_fig07.py
```

Figures are written to `figures/publication/` at 600 DPI.

### Recomputing Statistical Results

If OOF predictions change, recompute all statistics:

```bash
python scripts/recompute_stats.py
python scripts/compute_jackknife.py
python scripts/compute_calibration.py
python scripts/compute_statistical_tests.py
```

### Compiling the Manuscript (MiKTeX / TeX Live)

From `ijacsa_paper/`:

```bash
pdflatex -interaction=nonstopmode manuscript_ijacsa.tex
bibtex manuscript_ijacsa
pdflatex -interaction=nonstopmode manuscript_ijacsa.tex
pdflatex -interaction=nonstopmode manuscript_ijacsa.tex
```

For the double-blind review copy (same commands, uses the shared .bbl):

```bash
cp manuscript_ijacsa.bbl manuscript_ijacsa_blind.bbl
pdflatex -interaction=nonstopmode manuscript_ijacsa_blind.tex
pdflatex -interaction=nonstopmode manuscript_ijacsa_blind.tex
```

---

## Study Area

| Cordillera | Country | Latitude | n_lakes | n_GLOF | High-risk (%) |
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

Latitudinal range: 0.3 N to 35.5 S. GLOF inventory: 52 historical events (1932-2021), 16 matched within 5,000 m of a lake-year observation (class ratio 1:787).

---

## Feature Engineering

57 features organized into five groups:

| Group | Count | Examples |
|---|---|---|
| Morphometric | 10 | area, perimeter, compactness, elongation, fractal dimension |
| Topographic | 13 | elevation (mean/min/max/std), slope, TRI, freeboard, dam height |
| Hydrological/Depth | 12 | depth from 4 empirical models (Cook, Huggel, Yao, O'Connor), ensemble mean/std, volume |
| Temporal | 12 | annual area 2017-2025, growth rate, total change, relative growth |
| Engineered | 10 | log-transforms, slope x log(area) interaction, binary glacier-absence indicator |

Missing values were imputed with column-wise medians computed within each training fold (no leakage from test data). Depth features had the highest missingness (17.5%), reflecting the absence of empirical soundings for small or recently formed lakes.

---

## Machine Learning Pipeline

- **Imbalance strategy:** BalancedRandomForest (internal bootstrap per-tree) with SMOTETomek (external preprocessing pipeline, applied only to training folds)
- **Validation:** Stratified 5-fold out-of-fold (OOF) cross-validation
- **Geographic validation:** Leave-One-Country-Out (LOCO-CV) for Ecuador, Peru, Bolivia
- **Hyperparameters:** Default scikit-learn/imbalanced-learn defaults; no additional tuning (to avoid inflating performance estimates on n+ = 16)
- **Interpretability:** SHAP TreeExplainer (exact Shapley decompositions for tree ensembles)
- **Calibration:** Isotonic regression post-hoc on OOF scores

---

## Manuscript Status and Compilation

Both PDF files compile cleanly with MiKTeX (Windows) and TeX Live (Linux/Mac):

| File | Pages | Overfull warnings | Undefined references |
|---|---|---|---|
| manuscript_ijacsa.pdf | 20 | 0 | 0 |
| manuscript_ijacsa_blind.pdf | 19 | 0 | 0 |

Remaining warnings are cosmetic Underfull spacing only (22 entries, caused by narrow two-column format and long technical terms that resist hyphenation). No text protrudes outside any column boundary.

The bibliography contains 109 BibTeX entries; 95 are cited in the manuscript. BibTeX key naming convention: AuthorYear (e.g., Emmer2022, RGI2023).

---

## Limitations

1. The matched GLOF inventory contains only 16 positive labels (of 52 historical events). The remaining 36 are unmatched because they either pre-date the 2017 Sentinel-2 baseline or have coordinates that correspond to downstream impact zones rather than source-lake centroids, which is a systematic characteristic of historical GLOF catalogs.
2. Sentinel-2 at 10 m resolution does not resolve sub-pixel features of ice-cored moraine dams.
3. The model captures structural susceptibility, not dynamic event probability (no seismicity or precipitation triggers are included).
4. The temporal scope (2017-2025) excludes lakes formed and drained before the Sentinel-2 era.
5. No formal field-based accuracy assessment has been conducted. Prospective validation using the top-100 scored non-GLOF lakes shows morphometric profiles consistent with elevated hazard potential, but direct field confirmation would require targeted surveys across at least 50 lakes per cordillera.

---

## Authors and Affiliations

| Author | Role | Email | ORCID |
|---|---|---|---|
| Richar Andre Vilca Solorzano (corresponding) | Lead: study design, ML pipeline, manuscript | 75521963@est.unap.edu.pe | 0009-0003-2385-5263 |
| Dina Maribel Yana-Yucra | Data processing, spatial validation | maribelbel201314@gmail.com | 0009-0003-6218-2735 |
| Milton Vladimir Mamani Calisaya | Statistical analysis, interpretation | mmamanic@unap.edu.pe | 0000-0002-0676-0989 |
| Fred Torres-Cruz | Supervision, manuscript revision | ftorres@unap.edu.pe | 0000-0003-0834-6834 |

Faculty of Statistical and Computer Engineering
Universidad Nacional del Altiplano (UNAP), Puno, Peru

---

## Citation

Vilca Solorzano, R.A., Yana-Yucra, D.M., Mamani Calisaya, M.V., and Torres-Cruz, F. (2026).
Machine Learning-Based Pan-Andean Glacial Lake Outburst Flood Susceptibility Assessment
Using Multitemporal Sentinel-2 Imagery. International Journal of Advanced Computer Science
and Applications (IJACSA). DOI: [assigned upon acceptance]

---

## Data and Code Availability

Code, trained models, and figure-generation scripts:
https://github.com/Andre031222/GLOFs-Pan-Andina-Mediante-Machine-Learning
License: MIT

Sentinel-2 imagery: ESA Copernicus Open Access Hub (free, open access)
NASADEM DEM: NASA EARTHDATA (free, open access)
RGI v7.0 glacier outlines: GLIMS/NSIDC (free, open access)
GLOF inventory: GloFLOD (Lutzow et al., 2023), Emmer et al. (2022), INAIGEM (2018)

---

## License

Code: MIT License
Data derived from ESA Copernicus: subject to Copernicus open-access terms
GLOF inventory data: derived from published open-access databases (see citations)

---

*Last updated: May 2026*
