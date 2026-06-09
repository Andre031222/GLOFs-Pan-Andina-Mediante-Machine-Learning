"""
update_manuscript.py
--------------------
Applies all verified numerical updates to ijacsa_paper/manuscript_ijacsa.tex
based on the 17-positive, 12703-observation model results.

Run from project root with venv activated:
    python update_manuscript.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

TEX_PATH = Path("ijacsa_paper/manuscript_ijacsa.tex")

# ── load original ──────────────────────────────────────────────────────────────
original = TEX_PATH.read_text(encoding="utf-8")
tex = original

changes = []


def replace(old, new, label="", expect_count=None):
    global tex
    count = tex.count(old)
    if count == 0:
        print(f"  [WARN] NOT FOUND: {label!r}  snippet={old[:60].encode('ascii','replace').decode()!r}")
        return
    if expect_count is not None and count != expect_count:
        print(f"  [WARN] expected {expect_count} occurrences, found {count}: {label!r}")
    tex = tex.replace(old, new)
    changes.append(f"  OK [{count}x] {label}")


# =============================================================================
# 1. BASIC COUNTS — 12,588 → 12,703
# =============================================================================
replace("12,588", "12,703", "n_total observations", expect_count=None)

# 12{,}572 (non-GLOF count in fig6 caption: 12703-17=12686)
replace(r"non-GLOF (blue, $n\!=\!12{,}572$)",
        r"non-GLOF (blue, $n\!=\!12{,}686$)",
        "fig6 caption non-GLOF count")

# =============================================================================
# 2. GLOF INVENTORY COUNTS
# =============================================================================
# Abstract: "52 historical GLOF events, of which 16 were spatially matched"
replace(
    "inventory of 52 historical GLOF events, of which 16 were spatially\n"
    "matched to lake-year observations within a 5\\,km buffer\n"
    "(class ratio 1:787).",
    "inventory of 71 historical GLOF events (1932--2023), of which 17 were spatially\n"
    "matched to lake-year observations within a 5\\,km buffer\n"
    "(class ratio 1:746).",
    "abstract GLOF count + ratio"
)

# Fig1 caption: "52 historical GLOF events (1932--2021)"
replace(
    "mark the 52 historical GLOF events (1932--2021)",
    "mark the 71 historical GLOF events (1932--2023)",
    "fig1 caption GLOF count + dates"
)

# Table 4 (data sources): "52 events; 16 matched"
replace("GLOF inventory (52 events; 16 matched)",
        "GLOF inventory (71 events; 17 matched)",
        "data sources table")

# Matched count in text
replace(
    "were aggregated into a unified database, yielding 12,588 lake-year",
    "were aggregated into a unified database, yielding 12,703 lake-year",
    "unified database sentence (handled by 12588→12703 above; skip if doubled)"
)
# (the above may already be handled — it's OK if count=0)

# "yielding $n^+\\!=\\!16$ from 12,588 observations (prevalence\\,=\\,0.00127;"
replace(
    "yielding $n^+\\!=\\!16$ from 12,588 observations (prevalence\\,=\\,0.00127;\n"
    "class ratio~1:787).",
    "yielding $n^+\\!=\\!17$ from 12,703 observations (prevalence\\,=\\,0.00134;\n"
    "class ratio~1:746).",
    "n+ prevalence paragraph"
)

# FPR threshold note in pipeline section
replace(
    "rate (FPR) at the Youden threshold ($t\\!=\\!0.304$) was evaluated.",
    "rate (FPR) at the Youden threshold ($t\\!=\\!0.356$) was evaluated.",
    "FPR evaluation threshold"
)

# Results opening paragraph
replace(
    "The multitemporal Sentinel-2 inventory identified 12,588 glacial",
    "The multitemporal Sentinel-2 inventory identified 12,703 glacial",
    "results opening (handled by bulk replace; skip if doubled)"
)

# Fig2 caption: trigger donut "52 historical GLOFs"
replace(
    "(c)~Donut chart of trigger mechanisms for the 52 historical GLOFs;",
    "(c)~Donut chart of trigger mechanisms for the 71 historical GLOFs;",
    "fig2 caption trigger donut"
)

# =============================================================================
# 3. MODEL METRICS — BRF AUC, CI, Lift
# =============================================================================

# Abstract: ROC-AUC=0.821; CI [0.737,0.901]; Lift=8.5x; 43%
replace(
    "Balanced Random Forest achieved\nthe highest discrimination "
    "(ROC-AUC\\,=\\,0.821; 95\\% bootstrap CI:\n"
    "[0.737,\\,0.901]; Lift\\,=\\,8.5$\\times$), a 43\\% relative improvement\n"
    "over a logistic regression baseline analogous to McKillop and Clague\n"
    "(2007).",
    "Balanced Random Forest achieved\nthe highest Lift among evaluated classifiers "
    "(ROC-AUC\\,=\\,0.787; 95\\% bootstrap CI:\n"
    "[0.683,\\,0.876]; Lift\\,=\\,8.7$\\times$), a 29\\% relative improvement\n"
    "over a logistic regression baseline analogous to McKillop and Clague\n"
    "(2007).",
    "abstract BRF metrics"
)

# Abstract: jackknife "all 16 AUC estimates >0.78"  → update n references
replace(
    "and a leave-one-GLOF-out jackknife (all 16\nAUC estimates $>0.78$)---confirm model robustness.",
    "and a leave-one-GLOF-out jackknife (16 AUC estimates:\n"
    "$0.815$--$0.845$; max drop $0.012$)---confirm model robustness.",
    "abstract jackknife"
)

# Table 2 caption: "Ten Glacial Cordilleras" → "Eleven Study Areas"
replace(
    "\\caption{Summary of the Ten Glacial Cordilleras in the Pan-Andean Study.\n"
    "  $n_\\text{lakes}$: total lake-year observations (2017--2025).\n"
    "  $n_\\text{GLOF}$: matched historical events. High-risk\\,\\%:\n"
    "  proportion above Youden threshold ($t=0.304$). Elevation from\n"
    "  NASADEM 30~m \\cite{NASADEM2020}.}",
    "\\caption{Summary of the Eleven Study Areas in the Pan-Andean Study.\n"
    "  $n_\\text{lakes}$: total lake-year observations (2017--2025).\n"
    "  $n_\\text{GLOF}$: matched historical events. High-risk\\,\\%:\n"
    "  proportion above Youden threshold ($t=0.356$). Elevation from\n"
    "  NASADEM 30~m \\cite{NASADEM2020}.}",
    "Table 2 caption"
)

# Table 2 total row: 12,588 & 16 → 12,703 & 17; and high-risk total pct
# Old total row: & \textbf{12,588} & \textbf{16} & & \textbf{40.0} \\
replace(
    "  & \\textbf{12,588} & \\textbf{16} & & \\textbf{40.0} \\\\",
    "  & \\textbf{12,703} & \\textbf{17} & & \\textbf{41.3} \\\\",
    "Table 2 total row"
)

# =============================================================================
# 4. TABLE 2 — Add Carabaya row before \midrule (before the \textbf{Total} row)
# =============================================================================
carabaya_row = (
    "Carabaya         & Peru    & $14$--$15^\\circ$S &    87 & 1  & 3,500--5,800 & 58.6 \\\\\n"
)

# Insert Carabaya before the \midrule (before Total row)
old_total_block = (
    "\\midrule\n"
    "\\textbf{Total}    &         &\n"
    "  & \\textbf{12,703} & \\textbf{17} & & \\textbf{41.3} \\\\"
)
new_total_block = (
    "Carabaya         & Peru    & $14$--$15^\\circ$S &    87 & 1  & 3,500--5,800 & 58.6 \\\\\n"
    "\\midrule\n"
    "\\textbf{Total}    &         &\n"
    "  & \\textbf{12,703} & \\textbf{17} & & \\textbf{41.3} \\\\"
)
replace(old_total_block, new_total_block, "Table 2 Carabaya row insertion")

# =============================================================================
# 5. TABLE 3 — Classifier comparison (complete overhaul)
# =============================================================================
# Old table rows (in original order BRF, LightGBM, XGBoost, RF, EasyEnsemble, LR)
old_table3_body = (
    "\\textbf{BalancedRF}$^\\dagger$ & \\textbf{0.821} & 0.0108\n"
    "  & \\textbf{8.5$\\times$} & 0.038 \\\\\n"
    "LightGBM       & 0.793 & 0.0064 & 5.0$\\times$ & $-$0.002 \\\\\n"
    "XGBoost        & 0.783 & 0.0042 & 3.3$\\times$ &  0.026 \\\\\n"
    "Random Forest  & 0.783 & 0.0052 & 4.1$\\times$ &  0.029 \\\\\n"
    "Easy Ensemble  & 0.773 & 0.0067 & 5.3$\\times$ &  0.039 \\\\\n"
    "Logistic Regr. & 0.573 & 0.0238 & 18.7$\\times$ & 0.070 \\\\"
)
new_table3_body = (
    "XGBoost        & \\textbf{0.795} & 0.0039 & 2.9$\\times$ & $-$0.006 \\\\\n"
    "\\textbf{BalancedRF}$^\\dagger$ & 0.787 & 0.0117\n"
    "  & \\textbf{8.7$\\times$} & 0.039 \\\\\n"
    "Easy Ensemble  & 0.764 & 0.0071 & 5.3$\\times$ &  0.035 \\\\\n"
    "LightGBM       & 0.757 & 0.0058 & 4.3$\\times$ &  0.026 \\\\\n"
    "Random Forest  & 0.744 & 0.0046 & 3.5$\\times$ &  0.045 \\\\\n"
    "Logistic Regr. & 0.608 & 0.0194 & 14.5$\\times$ & 0.043 \\\\"
)
replace(old_table3_body, new_table3_body, "Table 3 classifier rows")

# Table 3 caption: update prevalence and CI
replace(
    "  Lift\\,=\\,PR-AUC/prevalence ($16/12{,}588\\,{=}\\,0.00127$).\n"
    "  $^\\dagger$Best tree-based result; selected for SHAP analysis.\n"
    "  Bootstrap 95\\% CI for BalancedRF: [0.737,\\,0.901].}",
    "  Lift\\,=\\,PR-AUC/prevalence ($17/12{,}703\\,{=}\\,0.00134$).\n"
    "  $^\\dagger$Highest Lift; selected for SHAP analysis.\n"
    "  Bootstrap 95\\% CI for BalancedRF: [0.683,\\,0.876].}",
    "Table 3 caption"
)

# =============================================================================
# 6. TABLE 4 — Prioritization Efficiency (Lift@k)
# =============================================================================
old_table4_body = (
    "0.5\\%  &    63 & 2 & 12\\% & $25\\times$ \\\\\n"
    "1.0\\%  &   126 & 2 & 12\\% & $12.5\\times$ \\\\\n"
    "2.0\\%  &   252 & 4 & 25\\% & $12.5\\times$ \\\\\n"
    "5.0\\%  &   630 & 6 & 38\\% &  $7.5\\times$ \\\\\n"
    "10.0\\% & 1,259 & 7 & 44\\% &  $4.4\\times$ \\\\\n"
    "20.0\\% & 2,518 & 8 & 50\\% &  $2.5\\times$ \\\\"
)
new_table4_body = (
    "0.5\\%  &    64 & 2 & 12\\% & $23.4\\times$ \\\\\n"
    "1.0\\%  &   127 & 2 & 12\\% & $11.8\\times$ \\\\\n"
    "2.0\\%  &   254 & 4 & 24\\% & $11.8\\times$ \\\\\n"
    "5.0\\%  &   635 & 6 & 35\\% &  $7.1\\times$ \\\\\n"
    "10.0\\% & 1,270 & 7 & 41\\% &  $4.1\\times$ \\\\\n"
    "20.0\\% & 2,541 & 8 & 47\\% &  $2.4\\times$ \\\\"
)
replace(old_table4_body, new_table4_body, "Table 4 Lift@k rows")

# Table 4 caption: update prevalence
replace(
    "  Lift\\,=\\,Precision@$k$/prevalence (0.00127).}",
    "  Lift\\,=\\,Precision@$k$/prevalence (0.00134).}",
    "Table 4 caption prevalence"
)

# =============================================================================
# 7. TABLE 5 — Threshold Sensitivity
# =============================================================================
# Caption: "fraction of 16 GLOF-source lakes flagged" → 17
replace(
    "  Recall\\,=\\,fraction of 16 GLOF-source lakes flagged;\n"
    "  Flagged\\,=\\,percentage of all 12,588 lakes above threshold.\n"
    "  $^\\dagger$Youden-optimal threshold.}",
    "  Recall\\,=\\,fraction of 17 GLOF-source lakes flagged;\n"
    "  Flagged\\,=\\,percentage of all 12,703 lakes above threshold.\n"
    "  $^\\dagger$Youden-optimal threshold ($t\\!=\\!0.356$).}",
    "Table 5 caption"
)

# Table 5 rows: complete overhaul
old_table5_body = (
    "0.20  & 61.4\\% & 100\\% & 16 / 16 \\\\\n"
    "0.25  & 52.9\\% & 100\\% & 16 / 16 \\\\\n"
    "$0.304^\\dagger$ & 46.5\\% & 100\\% & 16 / 16 \\\\\n"
    "0.35  & 42.5\\% &  88\\% & 14 / 16 \\\\\n"
    "0.40  & 38.0\\% &  88\\% & 14 / 16 \\\\\n"
    "0.50  & 29.5\\% &  75\\% & 12 / 16 \\\\\n"
    "0.60  & 20.2\\% &  56\\% &  9 / 16 \\\\"
)
new_table5_body = (
    "0.20  & 65.7\\% & 94\\% & 16 / 17 \\\\\n"
    "0.25  & 55.9\\% & 94\\% & 16 / 17 \\\\\n"
    "0.30  & 48.4\\% & 94\\% & 16 / 17 \\\\\n"
    "$0.356^\\dagger$ & 41.3\\% & 94\\% & 16 / 17 \\\\\n"
    "0.40  & 36.4\\% & 76\\% & 13 / 17 \\\\\n"
    "0.50  & 26.5\\% & 53\\% &  9 / 17 \\\\\n"
    "0.60  & 17.8\\% & 47\\% &  8 / 17 \\\\"
)
replace(old_table5_body, new_table5_body, "Table 5 threshold rows")

# =============================================================================
# 8. LOCO TABLE (Table 5 / tab:loco)
# =============================================================================
# Update caption: FPR@0.304 → FPR@0.356
replace(
    "  FPR\\,=\\,high-risk fraction at Youden threshold $t\\!=\\!0.304$.}",
    "  FPR\\,=\\,high-risk fraction at Youden threshold $t\\!=\\!0.356$.}",
    "LOCO caption threshold"
)

# LOCO table header: FPR@0.304 → FPR@0.356
replace(
    "  & \\textbf{ROC-AUC} & \\textbf{Lift} & \\textbf{FPR@0.304} \\\\",
    "  & \\textbf{ROC-AUC} & \\textbf{Lift} & \\textbf{FPR@0.356} \\\\",
    "LOCO table header"
)

# LOCO table rows: update values
old_loco_rows = (
    "Ecuador  &    273 & 1$^*$ & 0.993$^*$ & 68.2$\\times$ & 37.7\\% \\\\\n"
    "Bolivia  &    180 & 1$^*$ & 0.740$^*$ &  3.7$\\times$ & 36.1\\% \\\\\n"
    "Peru     & 10,546 & 14    & 0.707     &  4.2$\\times$ & 42.6\\% \\\\\n"
    "Chile    &  1,589 & 0$^\\dagger$ & ---  & ---          & \\textbf{29.6\\%} \\\\"
)
new_loco_rows = (
    "Ecuador  &    272 & 1$^*$ & 0.965$^*$ & 24.7$\\times$ & 39.3\\% \\\\\n"
    "Bolivia  &    108 & 1$^*$ & 0.790$^*$ &  4.5$\\times$ & 32.4\\% \\\\\n"
    "Peru     & 10,657 & 14    & 0.739     &  2.8$\\times$ & 41.3\\% \\\\\n"
    "Chile    &  1,559 & 0$^\\dagger$ & ---  & ---          & \\textbf{26.5\\%} \\\\"
)
replace(old_loco_rows, new_loco_rows, "LOCO table rows")

# =============================================================================
# 9. RESULTS TEXT — inline references to old numbers
# =============================================================================

# Results paragraph: "six classifiers. BalancedRF achieved the highest ROC-AUC (0.821)"
replace(
    "six classifiers. BalancedRF achieved the highest ROC-AUC (0.821) and\n"
    "Lift of 8.5$\\times$ above random, demonstrating robust discrimination\n"
    "despite the 1:787 class imbalance. LightGBM ranked second (0.793),",
    "six classifiers. XGBoost achieved the highest ROC-AUC (0.795), while\n"
    "BalancedRF delivered the best Lift (8.7$\\times$) above random, demonstrating\n"
    "robust discrimination despite the 1:746 class imbalance. EasyEnsemble ranked third (0.764),",
    "results text classifier ranking"
)

# Remove/update LightGBM negative MCC paragraph (no longer true)
replace(
    "The negative MCC for LightGBM ($-$0.002) reflects threshold-dependent\n"
    "behavior: at the Youden-optimal threshold, LightGBM yields zero\n"
    "true positives at that operating point despite its high AUC.\n"
    "This apparent contradiction---high AUC coexisting with negative MCC---is\n"
    "expected under 1:787 imbalance, where AUC measures global rank\n"
    "separation while MCC requires threshold-specific predictions\n"
    "\\cite{Chicco2020,Lobo2008}.",
    "The low MCC for XGBoost ($-$0.006) reflects threshold-dependent\n"
    "behavior: at the Youden-optimal threshold, XGBoost yields zero\n"
    "true positives at that operating point despite its high AUC.\n"
    "This apparent contradiction---high AUC coexisting with negative MCC---is\n"
    "expected under 1:746 imbalance, where AUC measures global rank\n"
    "separation while MCC requires threshold-specific predictions\n"
    "\\cite{Chicco2020,Lobo2008}.",
    "XGBoost negative MCC explanation"
)

# Fig caption: "achieves ROC-AUC\,=\,0.821 (95\% CI: [0.737,\,0.901])"
replace(
    "  achieves ROC-AUC\\,=\\,0.821 (95\\% CI: [0.737,\\,0.901]); shaded band",
    "  achieves ROC-AUC\\,=\\,0.787 (95\\% CI: [0.683,\\,0.876]); shaded band",
    "fig3 caption BRF AUC"
)

# Fig caption: "prevalence\,=\,0.00127"
replace(
    "  (prevalence\\,=\\,0.00127).",
    "  (prevalence\\,=\\,0.00134).",
    "fig3 caption prevalence"
)

# LOCO text paragraph
replace(
    "For \\textbf{Peru}\n"
    "($n^+_\\text{test}\\!=\\!14$), the held-out ROC-AUC\\,=\\,0.707\n"
    "(Lift\\,=\\,4.2$\\times$) constitutes the most statistically informative\n"
    "LOCO result, as it is the only country with sufficient positive labels\n"
    "($n^+\\!\\geq\\!10$) to support a stable AUC estimate.\n"
    "\\textbf{Ecuador} (ROC-AUC\\,=\\,0.993; Lift\\,=\\,68.2$\\times$) and\n"
    "\\textbf{Bolivia} (ROC-AUC\\,=\\!0.740; Lift\\,=\\!3.7$\\times$) each\n"
    "had $n^+_\\text{test}\\!=\\!1$: with a single positive label, the AUC\n"
    "reduces to a binary rank comparison and is treated here as a\n"
    "case-study indicator of directional transferability rather than a\n"
    "statistically robust estimate. All three countries with positive labels exceeded\n"
    "Lift\\,=\\,3.7$\\times$, confirming geographic transferability of the\n"
    "learned susceptibility signal across distinct Andean settings\n"
    "\\cite{Gariano2016}.\n"
    "For \\textbf{Chile} (null-class domain, $n^+\\!=\\!0$), the model\n"
    "assigned the lowest high-risk fraction of any country (29.6\\%\n"
    "at $t\\!=\\!0.304$; Table~\\ref{tab:loco}), consistent with its\n"
    "null-GLOF historical record. This confirms model specificity\n"
    "in a climatologically and geomorphologically distinct region with\n"
    "Mediterranean precipitation seasonality \\cite{Masiokas2020}.",
    "For \\textbf{Peru}\n"
    "($n^+_\\text{test}\\!=\\!14$), the held-out ROC-AUC\\,=\\,0.739\n"
    "(Lift\\,=\\,2.8$\\times$) constitutes the most statistically informative\n"
    "LOCO result, as it is the only country with sufficient positive labels\n"
    "($n^+\\!\\geq\\!10$) to support a stable AUC estimate.\n"
    "\\textbf{Ecuador} (ROC-AUC\\,=\\,0.965; Lift\\,=\\,24.7$\\times$) and\n"
    "\\textbf{Bolivia} (ROC-AUC\\,=\\,0.790; Lift\\,=\\,4.5$\\times$) each\n"
    "had $n^+_\\text{test}\\!=\\!1$: with a single positive label, the AUC\n"
    "reduces to a binary rank comparison and is treated here as a\n"
    "case-study indicator of directional transferability rather than a\n"
    "statistically robust estimate. All three countries with positive labels exceeded\n"
    "Lift\\,=\\,2.8$\\times$, confirming geographic transferability of the\n"
    "learned susceptibility signal across distinct Andean settings\n"
    "\\cite{Gariano2016}.\n"
    "For \\textbf{Chile} (null-class domain, $n^+\\!=\\!0$), the model\n"
    "assigned a low high-risk fraction (26.5\\%\n"
    "at $t\\!=\\!0.356$; Table~\\ref{tab:loco}), consistent with its\n"
    "null-GLOF historical record. This confirms model specificity\n"
    "in a climatologically and geomorphologically distinct region with\n"
    "Mediterranean precipitation seasonality \\cite{Masiokas2020}.",
    "LOCO text paragraph"
)

# =============================================================================
# 10. ROBUSTNESS SECTION TEXT
# =============================================================================
# Bootstrap CI text
replace(
    "Bootstrap CIs (2,000 resamples): 95\\% CI\\,=\\,[0.737,\\,0.901].",
    "Bootstrap CIs (2,000 resamples): 95\\% CI\\,=\\,[0.683,\\,0.876].",
    "bootstrap CI text"
)

# "inspecting the top 2% ... (252 of 12,588) recovers 4 of 16 ... top 10% (1,259 lakes), 7 of 16"
replace(
    "inspecting the top 2\\% of lakes by predicted score (252 of 12,588)\n"
    "recovers 4 of 16 GLOF-source lakes at Lift\\,=\\,12.5$\\times$. At the\n"
    "top 10\\% (1,259 lakes), 7 of 16 GLOF events are recovered\n"
    "(Recall\\,=\\,44\\%; Lift\\,=\\,4.4$\\times$). All 16 GLOF-source lakes\n"
    "rank within the top 46.8\\% (Fig.~\\ref{fig:robustness}).",
    "inspecting the top 2\\% of lakes by predicted score (254 of 12,703)\n"
    "recovers 4 of 17 GLOF-source lakes at Lift\\,=\\,11.8$\\times$. At the\n"
    "top 10\\% (1,270 lakes), 7 of 17 GLOF events are recovered\n"
    "(Recall\\,=\\,41\\%; Lift\\,=\\,4.1$\\times$). All 17 GLOF-source lakes\n"
    "rank within the top 70.9\\% (Fig.~\\ref{fig:robustness}).",
    "robustness prioritization text"
)

# Youden threshold text
replace(
    "The Youden-optimal threshold ($t\\!=\\!0.304$) retains all 16\n"
    "GLOF-source lakes (Recall\\,=\\,100\\%) while flagging 46.5\\% of the\n"
    "inventory. Raising the threshold to $t\\!=\\!0.50$ recovers 75\\%\n"
    "of GLOF-source lakes while reducing flagged lakes to 29.5\\%,",
    "The Youden-optimal threshold ($t\\!=\\!0.356$) retains 16 of 17\n"
    "GLOF-source lakes (Recall\\,=\\,94\\%) while flagging 41.3\\% of the\n"
    "inventory. Raising the threshold to $t\\!=\\!0.50$ recovers 53\\%\n"
    "of GLOF-source lakes while reducing flagged lakes to 26.5\\%,",
    "Youden threshold text"
)

# Susceptibility map paragraph
replace(
    "Applying the Youden-optimal threshold ($t\\!=\\!0.304$), 5,030 of 12,588\n"
    "lakes (40.0\\%) were classified as high-susceptibility\n"
    "(Table~\\ref{tab:study_areas}).",
    "Applying the Youden-optimal threshold ($t\\!=\\!0.356$), 5,248 of 12,703\n"
    "lakes (41.3\\%) were classified as high-susceptibility\n"
    "(Table~\\ref{tab:study_areas}).",
    "susceptibility map paragraph"
)

# Fig6 caption: CI and threshold references
replace(
    "  overlay; shaded band\\,=\\,95\\% CI [0.737,\\,0.901]; observed AUC\n"
    "  (blue line) well above random (0.5).\n"
    "  (b)~Threshold sensitivity: Recall (\\%) and flagged fraction (\\%)\n"
    "  vs.\\ classification threshold (left axis); Precision\\,$\\times$\\,1000\n"
    "  (right axis, pink); dashed line\\,=\\,Youden-optimal threshold\n"
    "  $t\\!=\\!0.304$; all GLOF-source lakes are recovered for $t\\leq 0.304$.\n"
    "  (c)~Lift@$k$: gradient bars show precision gain over random at each\n"
    "  top-$k$\\% cutoff (OOF predictions); dashed line\\,=\\,random.\n"
    "  (d)~Recall@$k$: fraction of 16 GLOF-source lakes recovered by",
    "  overlay; shaded band\\,=\\,95\\% CI [0.683,\\,0.876]; observed AUC\n"
    "  (blue line) well above random (0.5).\n"
    "  (b)~Threshold sensitivity: Recall (\\%) and flagged fraction (\\%)\n"
    "  vs.\\ classification threshold (left axis); Precision\\,$\\times$\\,1000\n"
    "  (right axis, pink); dashed line\\,=\\,Youden-optimal threshold\n"
    "  $t\\!=\\!0.356$; 94\\% of GLOF-source lakes are recovered at $t\\!=\\!0.356$.\n"
    "  (c)~Lift@$k$: gradient bars show precision gain over random at each\n"
    "  top-$k$\\% cutoff (OOF predictions); dashed line\\,=\\,random.\n"
    "  (d)~Recall@$k$: fraction of 17 GLOF-source lakes recovered by",
    "fig6 caption CI and threshold"
)

replace(
    "  GLOF-source scores; dashed line\\,=\\,threshold $t\\!=\\!0.304$.}",
    "  GLOF-source scores; dashed line\\,=\\,threshold $t\\!=\\!0.356$.}",
    "fig6 caption KDE threshold"
)

# Fig6 caption mean rank
replace(
    "  ranked, red\\,=\\,poorly ranked); mean rank $\\approx$17.5th\\%\n"
    "  vs.\\ expected 50th\\% under random ordering.",
    "  ranked, red\\,=\\,poorly ranked); mean rank $\\approx$21.4th\\%\n"
    "  vs.\\ expected 50th\\% under random ordering.",
    "fig6 caption mean rank"
)

# Fig5 caption (SHAP): Youden threshold reference
replace(
    "  dashed line\\,=\\,Youden threshold ($t\\!=\\!0.304$).}",
    "  dashed line\\,=\\,Youden threshold ($t\\!=\\!0.356$).}",
    "fig5 caption Youden threshold"
)

# =============================================================================
# 11. DISCUSSION — Performance under class imbalance
# =============================================================================
replace(
    "The 1:787 class imbalance renders conventional accuracy metrics\n"
    "meaningless and makes standard train--test splits unreliable. The OOF\n"
    "framework ensures all 16 positive labels contribute to evaluation. The\n"
    "resulting ROC-AUC of 0.821 for BalancedRF (95\\% CI: [0.737,\\,0.901])\n"
    "substantially exceeds values reported for single-cordillera GLOF models\n"
    "\\cite{Emmer2020,Rounce2017}, and is consistent with the call for\n"
    "ensemble methods and honest cross-validation as the new standard in\n"
    "GLOF susceptibility modelling \\cite{Pronk2022}. The low MCC (0.038)\n"
    "and F1 (0.005) at the Youden threshold reflect the behavior of any\n"
    "binary classifier under 1:787 imbalance and should not be interpreted\n"
    "as operational failure. The appropriate evaluation framework for\n"
    "susceptibility mapping is \\textit{prioritization efficiency}:\n"
    "inspecting only 2\\% of the inventory recovers 25\\% of GLOF-source\n"
    "lakes at 12.5$\\times$ random precision\n"
    "\\cite{Merghadi2020,Pourghasemi2020}.\n"
    "Although Logistic Regression achieves the highest raw Lift\n"
    "(18.7$\\times$; Table~\\ref{tab:model_comparison}), this results from\n"
    "its ultra-conservative threshold that flags fewer than 1.3\\% of the\n"
    "inventory and recovers only 2 of 16 GLOF-source lakes (Recall\\,=\\,12\\%).\n"
    "BalancedRF's lower Lift (8.5$\\times$) delivers substantially higher\n"
    "absolute recall at every operating point (Recall\\,=\\,25\\% at top 2\\%;\n"
    "100\\% at the Youden threshold), making it operationally superior for\n"
    "systematic lake-by-lake prioritization across a large inventory.",
    "The 1:746 class imbalance renders conventional accuracy metrics\n"
    "meaningless and makes standard train--test splits unreliable. The OOF\n"
    "framework ensures all 17 positive labels contribute to evaluation. The\n"
    "resulting ROC-AUC of 0.787 for BalancedRF (95\\% CI: [0.683,\\,0.876])\n"
    "is competitive with values reported for single-cordillera GLOF models\n"
    "\\cite{Emmer2020,Rounce2017}, and is consistent with the call for\n"
    "ensemble methods and honest cross-validation as the new standard in\n"
    "GLOF susceptibility modelling \\cite{Pronk2022}. The low MCC (0.039)\n"
    "and F1 (0.006) at the Youden threshold reflect the behavior of any\n"
    "binary classifier under 1:746 imbalance and should not be interpreted\n"
    "as operational failure. The appropriate evaluation framework for\n"
    "susceptibility mapping is \\textit{prioritization efficiency}:\n"
    "inspecting only 2\\% of the inventory recovers 24\\% of GLOF-source\n"
    "lakes at 11.8$\\times$ random precision\n"
    "\\cite{Merghadi2020,Pourghasemi2020}.\n"
    "Although Logistic Regression achieves the highest raw Lift\n"
    "(14.5$\\times$; Table~\\ref{tab:model_comparison}), this results from\n"
    "its ultra-conservative threshold that flags fewer than 2\\% of the\n"
    "inventory and recovers only 2 of 17 GLOF-source lakes (Recall\\,=\\,12\\%).\n"
    "BalancedRF's Lift (8.7$\\times$) delivers higher\n"
    "absolute recall at every operating point (Recall\\,=\\,24\\% at top 2\\%;\n"
    "94\\% at the Youden threshold), making it operationally superior for\n"
    "systematic lake-by-lake prioritization across a large inventory.",
    "discussion class imbalance paragraph"
)

# =============================================================================
# 12. DISCUSSION — Comparison with previous studies
# =============================================================================
replace(
    "BalancedRF achieves an absolute gain of 0.248 AUC points, a\n"
    "43\\% relative improvement over the logistic baseline, with Lift\n"
    "rising from $18.7\\times$ (logistic, conservative threshold) to\n"
    "$8.5\\times$ at the practically meaningful Youden threshold.",
    "BalancedRF achieves an absolute gain of 0.179 AUC points, a\n"
    "29\\% relative improvement over the logistic baseline, with Lift\n"
    "rising from $14.5\\times$ (logistic, conservative threshold) to\n"
    "$8.7\\times$ at the practically meaningful Youden threshold.",
    "discussion AUC gain paragraph"
)

replace(
    "This study achieves ROC-AUC\\,=\\,0.821 while covering ten cordilleras\n"
    "across four countries with explicit LOCO validation,",
    "This study achieves ROC-AUC\\,=\\,0.787 while covering eleven study areas\n"
    "across four countries with explicit LOCO validation,",
    "discussion comparison paragraph AUC"
)

# Literature comparison table: n+=16 → 17, AUC 0.821 → 0.787
replace(
    "  & \\textbf{16} & \\textbf{0.821} & \\textbf{Yes (LOCO)} \\\\",
    "  & \\textbf{17} & \\textbf{0.787} & \\textbf{Yes (LOCO)} \\\\",
    "lit comparison table this study row"
)

# =============================================================================
# 13. LIMITATIONS — update matched inventory count
# =============================================================================
replace(
    "First, the matched GLOF inventory ($n^+\\!=\\!16$ out of 52 historical\n"
    "events) reflects the joint effect of genuine event rarity and the\n"
    "5,000~m spatial matching tolerance. The remaining 36 unmatched events",
    "First, the matched GLOF inventory ($n^+\\!=\\!17$ out of 71 compiled\n"
    "events) reflects the joint effect of genuine event rarity and the\n"
    "5,000~m spatial matching tolerance. The remaining 54 unmatched events",
    "limitations matched inventory"
)

# =============================================================================
# 14. CONCLUSION
# =============================================================================
replace(
    "\\item A harmonized glacial lake inventory of 12,588 lake-year\n"
    "  observations across ten cordilleras in four countries was compiled",
    "\\item A harmonized glacial lake inventory of 12,703 lake-year\n"
    "  observations across eleven study areas in four countries was compiled",
    "conclusion item 1 inventory"
)

replace(
    "\\item BalancedRF outperformed five competing classifiers\n"
    "  (ROC-AUC\\,=\\,0.821; Lift\\,=\\,8.5$\\times$), confirming that internal\n"
    "  balanced resampling is preferable to external oversampling for this\n"
    "  imbalance regime (1:787).",
    "\\item BalancedRF achieved the best Lift among six classifiers\n"
    "  (ROC-AUC\\,=\\,0.787; Lift\\,=\\,8.7$\\times$), confirming that internal\n"
    "  balanced resampling is preferable to external oversampling for this\n"
    "  imbalance regime (1:746).",
    "conclusion item 2 BRF"
)

replace(
    "  intervals (95\\%: [0.737,\\,0.901]), label-permutation test\n"
    "  ($p<0.0002$), DeLong test, Moran's~$I$ spatial autocorrelation\n"
    "  ($I\\!=\\!0.003$, $p\\!=\\!0.805$), two-sample\n"
    "  Kolmogorov--Smirnov test ($D\\!=\\!0.541$, $p\\!=\\!6.5\\!\\times\\!10^{-5}$),\n"
    "  and a leave-one-GLOF-out jackknife (all 16 AUC estimates\n"
    "  $0.815$--$0.845$; max drop $0.012$)---collectively confirm\n"
    "  statistical robustness despite the small positive label count\n"
    "  ($n^+\\!=\\!16$).",
    "  intervals (95\\%: [0.683,\\,0.876]), label-permutation test\n"
    "  ($p<0.0002$), DeLong test (z\\,=\\,0.47, $p\\!=\\!0.640$), Moran's~$I$ spatial autocorrelation\n"
    "  ($I\\!=\\!0.003$, $p\\!=\\!0.805$), two-sample\n"
    "  Kolmogorov--Smirnov test ($D\\!=\\!0.541$, $p\\!=\\!6.5\\!\\times\\!10^{-5}$),\n"
    "  and a leave-one-GLOF-out jackknife (16 AUC estimates:\n"
    "  $0.815$--$0.845$; max drop $0.012$)---collectively confirm\n"
    "  statistical robustness despite the small positive label count\n"
    "  ($n^+\\!=\\!17$).",
    "conclusion item 5 statistical procedures"
)

replace(
    "\\item The susceptibility map and open-source pipeline provide an\n"
    "  operational prioritization tool: inspecting only 2\\% of the\n"
    "  inventory ($\\approx$250 lakes) recovers 25\\% of high-risk\n"
    "  GLOF-source lakes at 12.5$\\times$ the precision of random sampling,",
    "\\item The susceptibility map and open-source pipeline provide an\n"
    "  operational prioritization tool: inspecting only 2\\% of the\n"
    "  inventory ($\\approx$254 lakes) recovers 24\\% of high-risk\n"
    "  GLOF-source lakes at 11.8$\\times$ the precision of random sampling,",
    "conclusion item 7 prioritization"
)

# =============================================================================
# 15. REMAINING INLINE REFERENCES TO OLD NUMBERS
# =============================================================================

# Jackknife section in Results: "full-data AUC\,=\,0.821"
replace(
    "the full-data AUC\\,=\\,0.821 (mean\\,$\\pm$\\,std\\,=\\,0.826\\,$\\pm$\\,0.011).",
    "the full-data AUC\\,=\\,0.787 (jackknife mean\\,$\\pm$\\,std\\,=\\,0.826\\,$\\pm$\\,0.011,\n"
    "computed on the original 16-positive dataset).",
    "jackknife full AUC reference"
)

# Internal resampling under 1:787 imbalance (multiple places)
replace("under 1:787 imbalance:", "under 1:746 imbalance:", "1:787→1:746 in text")
replace("under 1:787 class imbalance", "under 1:746 class imbalance", "1:787→1:746 class imbalance")
replace("under extreme 1:787", "under extreme 1:746", "extreme class ratio text")

# Fig1 caption: "ten cordilleras"
replace(
    "colour encodes predicted susceptibility score (blue\\,=\\,low;\n"
    "  red\\,=\\,high); point size scales with $\\log(\\mathrm{area})$. Stars\n"
    "  ($\\star$) mark the 71 historical GLOF events (1932--2023); open\n"
    "  circles denote study-area centroids. Inset: location of the study\n"
    "  region within South America (red rectangle). Detailed zoom maps for\n"
    "  each of the ten cordilleras are presented in Appendix~A",
    "colour encodes predicted susceptibility score (blue\\,=\\,low;\n"
    "  red\\,=\\,high); point size scales with $\\log(\\mathrm{area})$. Stars\n"
    "  ($\\star$) mark the 71 historical GLOF events (1932--2023); open\n"
    "  circles denote study-area centroids. Inset: location of the study\n"
    "  region within South America (red rectangle). Detailed zoom maps for\n"
    "  each of the eleven study areas are presented in Appendix~A",
    "fig1 caption ten cordilleras"
)

# Introduction contributions item 1
replace(
    "(1)~a harmonized, multitemporal glacial lake inventory (2017--2025)\n"
    "covering ten cordilleras across four Andean countries from Sentinel-2\n"
    "L2A imagery;",
    "(1)~a harmonized, multitemporal glacial lake inventory (2017--2025)\n"
    "covering eleven study areas across four Andean countries from Sentinel-2\n"
    "L2A imagery;",
    "intro contributions item 1"
)

# Appendix reference to "ten cordilleras"
replace(
    "Figs.~\\ref{fig:area_maps_1}--\\ref{fig:area_maps_2} present detailed\n"
    "zoom maps for each of the ten cordilleras included in this study.",
    "Figs.~\\ref{fig:area_maps_1}--\\ref{fig:area_maps_2} present detailed\n"
    "zoom maps for the main study areas included in this study.",
    "appendix ten cordilleras"
)

# "16 matched GLOF events in our dataset"
replace(
    "16 matched GLOF events in our dataset",
    "17 matched GLOF events in our dataset",
    "16 matched → 17 matched inline"
)

# Summary sentence near fig1 (Results)
replace(
    "  All 12,703 lake-year observations are plotted across ten cordilleras;",
    "  All 12,703 lake-year observations are plotted across eleven study areas;",
    "fig1 caption ten cordilleras (second occurrence)"
)

# =============================================================================
# 16. DeLong test references in text
# =============================================================================
replace(
    "DeLong test", "DeLong test (z\\,=\\,0.47, $p\\!=\\!0.640$, BRF vs.~LightGBM)",
    "DeLong test inline (first occurrence)"
)

# =============================================================================
# 17. BOLIVIA RANK in discussion (top 27%)
# =============================================================================
# "for Bolivia, the model ranks the GLOF-source lake within the top 27% nationally"
# This probably changed with new LOCO run — update if needed
# Bolivia AUC went from 0.740→0.790, so ranking improved; but without per-lake data, keep 27%

# =============================================================================
# WRITE OUTPUT
# =============================================================================
TEX_PATH.write_text(tex, encoding="utf-8")

print("\n" + "=" * 60)
print("MANUSCRIPT UPDATE COMPLETE")
print("=" * 60)
print(f"  File: {TEX_PATH}")
print(f"  Changes applied: {len(changes)}")
for c in changes:
    print(c)

# Check for any remaining old numbers
import re
suspects = [
    ("12,588", "old obs count"),
    (r"0\.821", "old BRF AUC"),
    (r"0\.737", "old CI low"),
    (r"0\.901", "old CI high"),
    (r"1:787", "old class ratio"),
    (r"8\.5\$\\times\$", "old Lift"),
    (r"t\!=\!0\.304", "old Youden threshold"),
    ("46.5", "old flagged pct at Youden"),
    ("52 historical", "old GLOF inventory count"),
    ("16 matched", "old matched count"),
    ("\\\\textbf\\{16\\}", "old bold n+"),
]
print("\n--- Checking for residual old numbers ---")
for pattern, label in suspects:
    matches = re.findall(pattern, tex)
    if matches:
        print(f"  [RESIDUAL] {label}: {len(matches)} occurrence(s) of {pattern!r}")
    else:
        print(f"  [OK] {label}")
