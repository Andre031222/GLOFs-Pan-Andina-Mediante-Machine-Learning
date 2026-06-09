#!/usr/bin/env python3
"""
scripts/compute_jackknife.py
-----------------------------
Leave-one-GLOF-out (jackknife) AUC stability analysis.

For each of the n+=16 GLOF-positive observations, temporarily sets its
label to 0 and recomputes ROC-AUC from existing OOF scores (no retraining
required). Reports mean ± std of the 16 jackknife AUCs and saves to
models/jackknife_results.json.

If AUC remains stable across all 16 leave-one-out iterations, it confirms
that no single GLOF event drives the discriminative performance.

Run from project root:
    glof/Scripts/python scripts/compute_jackknife.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'

def main():
    print("Jackknife AUC Stability Analysis (Leave-One-GLOF-Out)")
    print("=" * 55)

    # Load OOF predictions
    oof_path = MODELS_DIR / 'oof_predictions.json'
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")

    with open(oof_path) as f:
        oof = json.load(f)

    y_true  = np.array(oof['y_true'],       dtype=int)
    y_score = np.array(oof['y_score_brf'],  dtype=float)

    pos_idx = np.where(y_true == 1)[0]
    n_pos   = len(pos_idx)
    n_tot   = len(y_true)

    # Full AUC baseline
    auc_full = roc_auc_score(y_true, y_score)
    print(f"  Full OOF AUC       : {auc_full:.4f}  (n+={n_pos}, n={n_tot})")
    print()

    # Leave-one-GLOF-out
    jack_aucs = []
    for i, idx in enumerate(pos_idx):
        y_jack = y_true.copy()
        y_jack[idx] = 0           # mask out this GLOF event
        try:
            auc_j = roc_auc_score(y_jack, y_score)
        except ValueError:
            auc_j = float('nan')
        jack_aucs.append(float(auc_j))
        _dir = "+" if auc_j > auc_full else "-"
        print(f"  Leave-out GLOF #{i+1:02d} (obs {idx:5d}) : AUC = {auc_j:.4f}  "
              f"({_dir}{abs(auc_j - auc_full):.4f})")

    valid = [a for a in jack_aucs if not np.isnan(a)]
    mean_j = float(np.mean(valid))
    std_j  = float(np.std(valid))
    min_j  = float(np.min(valid))
    max_j  = float(np.max(valid))

    print()
    print(f"  Jackknife summary:")
    print(f"    Mean AUC = {mean_j:.4f} ± {std_j:.4f}")
    print(f"    Range    = [{min_j:.4f}, {max_j:.4f}]")
    print(f"    Max drop from full AUC: {auc_full - min_j:.4f}")
    print(f"    All jackknife AUCs > 0.75: {all(a > 0.75 for a in valid)}")
    print()

    results = {
        'auc_full':        auc_full,
        'n_pos':           n_pos,
        'n_total':         n_tot,
        'jackknife_aucs':  jack_aucs,
        'jackknife_mean':  mean_j,
        'jackknife_std':   std_j,
        'jackknife_min':   min_j,
        'jackknife_max':   max_j,
        'max_drop':        float(auc_full - min_j),
        'all_above_0_75':  bool(all(a > 0.75 for a in valid)),
    }

    out_path = MODELS_DIR / 'jackknife_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {out_path}")


if __name__ == '__main__':
    main()
