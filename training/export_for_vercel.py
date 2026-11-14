"""
Train a Logistic Regression on the claims data and export it as
../public/model.json so the website can score claims in the browser.

Run (from the training/ folder):
    python export_for_vercel.py --data data/insurance_claims.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from fraud_detection import RANDOM_STATE, build_preprocessor, load_data

# Fields shown on the website (all other fields use typical default values)
UI_NUMERIC = ["total_claim_amount", "policy_annual_premium", "days_bind_to_incident",
              "months_as_customer", "witnesses", "number_of_vehicles_involved",
              "bodily_injuries"]
UI_CATEGORICAL = ["incident_severity", "incident_type", "police_report_available",
                  "property_damage", "collision_type", "authorities_contacted"]


def main(data_path, out_path, demo=False):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipe = Pipeline([("pre", build_preprocessor(X)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])

    # Decision threshold from cross-validated predictions on training data (favours recall, F2)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f2 = 5 * prec * rec / np.maximum(4 * prec + rec, 1e-9)
    threshold = float(thr[np.argmax(f2[:-1])])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = proba >= threshold
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 3),
        "recall": round(float(recall_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
        "fraud_rate": round(float(y.mean()), 3),
        "n_claims": int(len(X)),
    }
    print(metrics)

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    num_cols = list(pre.transformers_[0][2])
    cat_cols = list(pre.transformers_[1][2])
    scaler = pre.named_transformers_["num"].named_steps["sc"]
    ohe = pre.named_transformers_["cat"].named_steps["oh"]

    defaults = {c: float(X_train[c].median()) for c in num_cols}
    defaults.update({c: str(X_train[c].mode()[0]) for c in cat_cols})

    model = {
        "demo": demo,
        "metrics": metrics,
        "threshold": threshold,
        "intercept": float(clf.intercept_[0]),
        "num_cols": num_cols,
        "num_mean": scaler.mean_.tolist(),
        "num_scale": scaler.scale_.tolist(),
        "cat_cols": cat_cols,
        "cat_categories": [list(map(str, c)) for c in ohe.categories_],
        "coef": clf.coef_[0].tolist(),  # numeric coefficients first, then one-hot coefficients
        "defaults": defaults,
        "ui_numeric": [c for c in UI_NUMERIC if c in num_cols],
        "ui_categorical": [c for c in UI_CATEGORICAL if c in cat_cols],
        "ui_ranges": {c: [float(X[c].min()), float(X[c].max())]
                      for c in UI_NUMERIC if c in num_cols},
    }
    Path(out_path).write_text(json.dumps(model))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/insurance_claims.csv")
    ap.add_argument("--out", default="../public/model.json")
    ap.add_argument("--demo", action="store_true", help="mark model as demo (synthetic data)")
    a = ap.parse_args()
    main(a.data, a.out, a.demo)
