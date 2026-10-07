from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from core import FEATURES, RANDOM_STATE, load_clean_data

# Paste here the hyperparameters found in the notebook (search.best_params_).
RF_PARAMS = dict(n_estimators=300, max_depth=24, min_samples_leaf=2, max_features="log2")

MODEL_PATH = Path("model/wine_rf.joblib")


def main():
    wine = load_clean_data()
    X = wine[FEATURES]
    y = (wine["quality"] >= 7).astype(int)

    # Same split as the notebook: stratified by target and wine type
    strata = y.astype(str) + "_" + wine["type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=strata
    )

    model = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **RF_PARAMS
    )

    # Screening threshold chosen on out-of-fold predictions of the TRAINING data (F2)
    oof = cross_val_predict(
        clone(model), X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
    )[:, 1]
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f2 = [fbeta_score(y_train, oof >= t, beta=2, zero_division=0) for t in grid]
    best_threshold = float(grid[int(np.argmax(f2))])

    model.fit(X_train, y_train)

    # Held-out evaluation and operating points (workload vs. coverage)
    proba = model.predict_proba(X_test)[:, 1]
    y_arr = y_test.values
    rows = []
    for t in grid:
        flagged = proba >= t
        rows.append({
            "threshold": float(t),
            "flagged_pct": flagged.mean() * 100,
            "precision": precision_score(y_arr, flagged, zero_division=0),
            "recall": recall_score(y_arr, flagged, zero_division=0),
        })

    high = wine[y == 1]
    rest = wine[y == 0]

    meta = {
        "features": FEATURES,
        "best_threshold": best_threshold,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sklearn_version": sklearn.__version__,
        "test": {
            "roc_auc": roc_auc_score(y_arr, proba),
            "pr_auc": average_precision_score(y_arr, proba),
            "baseline_pr_auc": float(y_arr.mean()),
        },
        "operating_points": pd.DataFrame(rows),
        "ranges": {
            f: {"min": float(wine[f].min()), "max": float(wine[f].max()), "median": float(wine[f].median())}
            for f in FEATURES
        },
        "type_medians": {
            t: wine.loc[wine["type"] == t, FEATURES].median().to_dict() for t in ("red", "white")
        },
        "medians_high": high[FEATURES].median().to_dict(),
        "medians_rest": rest[FEATURES].median().to_dict(),
        "importances": dict(zip(FEATURES, model.feature_importances_)),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, MODEL_PATH)

    print(f"Saved {MODEL_PATH}")
    print(f"Test ROC-AUC {meta['test']['roc_auc']:.3f} | PR-AUC {meta['test']['pr_auc']:.3f} "
          f"(baseline {meta['test']['baseline_pr_auc']:.3f}) | threshold {best_threshold:.2f}")


if __name__ == "__main__":
    main()
