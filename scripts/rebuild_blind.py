"""
rebuild_blind.py
Regenerates the double-blind version from the updated main manuscript.
"""
from pathlib import Path

main = Path("ijacsa_paper/manuscript_ijacsa.tex").read_text(encoding="utf-8")
blind = Path("ijacsa_paper/manuscript_ijacsa_blind.tex").read_text(encoding="utf-8")

# Split at \begin{abstract}
SPLIT = r"\begin{abstract}"
split_main = main.index(SPLIT)
split_blind = blind.index(SPLIT)

# New blind = blind-specific header/author block + main body from abstract onward
new_blind = blind[:split_blind] + main[split_main:]

# Anonymize acknowledgment
old_ack = (
    "The authors thank the European Space Agency (ESA) for free access to\n"
    "Sentinel-2 Level-2A imagery through the Copernicus Open Access Hub,\n"
    "and the open-source community behind scikit-learn, imbalanced-learn,\n"
    "SHAP, GeoPandas, and Matplotlib. The authors thank the Faculty of\n"
    "Statistical and Computer Engineering at Universidad Nacional del\n"
    "Altiplano (UNAP), Puno, Per\\'{u}, for institutional support.\n"
    "\n"
    "\\textbf{Author contributions:}\n"
    "R.A.V.S.\\ led the study design, data processing, ML pipeline, and\n"
    "manuscript writing.\n"
    "D.M.Y.Y.\\ contributed to data processing and spatial validation.\n"
    "M.V.M.C.\\ contributed to statistical analysis and result\n"
    "interpretation.\n"
    "F.T.C.\\ supervised the research and contributed to manuscript revision.\n"
    "\n"
    "\\textbf{Data and code availability:}\n"
    "The machine learning pipeline, trained models, and figure-generation\n"
    "notebooks are available at \\url{https://github.com/Andre031222/GLOFs-Pan-Andina-Mediante-Machine-Learning}\n"
    "under an MIT licence."
)
new_ack = (
    "The authors thank the European Space Agency (ESA) for free access to\n"
    "Sentinel-2 Level-2A imagery through the Copernicus Open Access Hub,\n"
    "and the open-source community behind scikit-learn, imbalanced-learn,\n"
    "SHAP, GeoPandas, and Matplotlib.\n"
    "[Institutional acknowledgments omitted for blind review.]\n"
    "\n"
    "\\textbf{Author contributions:}\n"
    "[Author contributions omitted for blind review.]\n"
    "\n"
    "\\textbf{Data and code availability:}\n"
    "The machine learning pipeline, trained models, and figure-generation\n"
    "notebooks will be made available upon acceptance.\n"
    "[Repository URL omitted for blind review.]"
)

if old_ack in new_blind:
    new_blind = new_blind.replace(old_ack, new_ack)
    print("Acknowledgment anonymized OK")
else:
    print("WARNING: acknowledgment block not found literally — attempting partial match")
    # Try to find and replace just the URL line
    url_line = "notebooks are available at \\url{https://github.com/Andre031222/GLOFs-Pan-Andina-Mediante-Machine-Learning}\nunder an MIT licence."
    anon_url = "notebooks will be made available upon acceptance.\n[Repository URL omitted for blind review.]"
    if url_line in new_blind:
        new_blind = new_blind.replace(url_line, anon_url)
        print("  -> URL line anonymized")
    # Also anonymize author contributions
    contrib_pattern = "R.A.V.S.\\ led the study design"
    if contrib_pattern in new_blind:
        # Find the full author contrib section
        idx = new_blind.index("\\textbf{Author contributions:}")
        end_idx = new_blind.index("\\textbf{Data and code availability:}")
        new_blind = (new_blind[:idx]
                    + "\\textbf{Author contributions:}\n[Author contributions omitted for blind review.]\n\n"
                    + new_blind[end_idx:])
        print("  -> Author contributions anonymized")
    # UNAP acknowledgment
    unap_line = "and the open-source community behind scikit-learn, imbalanced-learn,\nSHAP, GeoPandas, and Matplotlib. The authors thank the Faculty of\nStatistical and Computer Engineering at Universidad Nacional del\nAltiplano (UNAP), Puno, Per\\'{u}, for institutional support."
    anon_unap = "and the open-source community behind scikit-learn, imbalanced-learn,\nSHAP, GeoPandas, and Matplotlib.\n[Institutional acknowledgments omitted for blind review.]"
    if unap_line in new_blind:
        new_blind = new_blind.replace(unap_line, anon_unap)
        print("  -> UNAP acknowledgment anonymized")

Path("ijacsa_paper/manuscript_ijacsa_blind.tex").write_text(new_blind, encoding="utf-8")
print(f"Blind version written: {len(new_blind.splitlines())} lines")
