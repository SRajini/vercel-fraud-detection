"""
Insurance Claims Fraud Detection
Run:  python fraud_detection.py --data data/insurance_claims.csv
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score,
                             classification_report, fbeta_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
OUT = Path("outputs")
DROP_COLS = ["policy_number", "insured_zip", "incident_location",
             "policy_bind_date", "incident_date", "_c39", "fraud_reported"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from raw claim columns."""
    df = df.copy()
    bind = pd.to_datetime(df["policy_bind_date"], errors="coerce")
    inc = pd.to_datetime(df["incident_date"], errors="coerce")
    df["days_bind_to_incident"] = (inc - bind).dt.days
    df["claim_to_premium"] = df["total_claim_amount"] / df["policy_annual_premium"].replace(0, np.nan)
    df["vehicle_age"] = inc.dt.year - df["auto_year"]
    return df


def load_data(path: str):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])  # keep "None" as a real category
    df = df.replace("?", "Unknown")  # dataset uses '?' for missing values
    y = (df["fraud_reported"] == "Y").astype(int)
    df = engineer_features(df)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def make_models(X):
    pre = lambda: build_preprocessor(X)
    return {
        "LogisticRegression": Pipeline([("pre", pre()), ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced"))]),
        "RandomForest": Pipeline([("pre", pre()), ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "GradientBoosting": Pipeline([("pre", pre()), ("clf", GradientBoostingClassifier(
            random_state=RANDOM_STATE))]),
    }


def main(data_path: str):
    OUT.mkdir(exist_ok=True)
    X, y = load_data(data_path)
    print(f"Rows: {len(X)}, Features: {X.shape[1]}")
    print(f"Fraud rate: {y.mean():.1%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # ---- Compare models with stratified cross-validation (PR-AUC & ROC-AUC)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = make_models(X)
    results = {}
    for name, model in models.items():
        roc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        pr = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision").mean()
        results[name] = (roc, pr)
        print(f"{name:20s} CV ROC-AUC={roc:.3f}  CV PR-AUC={pr:.3f}")
    pd.DataFrame(results, index=["cv_roc_auc", "cv_pr_auc"]).T.to_csv(OUT / "model_comparison.csv")

    best_name = max(results, key=lambda k: results[k][1])
    best = models[best_name]
    print(f"\nBest model (by PR-AUC): {best_name}")

    # ---- Choose a threshold on training data only (favour recall: F2 score)
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f2 = (5 * prec * rec) / np.maximum(4 * prec + rec, 1e-9)
    threshold = float(thr[np.argmax(f2[:-1])])
    print(f"Chosen threshold (max F2 on CV predictions): {threshold:.3f}")

    # ---- Final fit and test evaluation
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    print("\n=== Test set results ===")
    print(classification_report(y_test, pred, target_names=["Not fraud", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_test, proba):.3f}")
    print(f"F2:      {fbeta_score(y_test, pred, beta=2):.3f}")

    # ---- Plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax[0], colorbar=False)
    ax[0].set_title("Confusion matrix")
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax[1])
    ax[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax[2])
    ax[2].set_title("Precision-recall curve")
    plt.tight_layout()
    plt.savefig(OUT / "evaluation.png", dpi=150)
    plt.close()

    # ---- Explainability: permutation importance on raw columns
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                                 n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
              .sort_values("importance", ascending=False))
    imp_df.to_csv(OUT / "feature_importance.csv", index=False)
    top = imp_df.head(12).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top features (permutation importance, drop in ROC-AUC)")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close()
    print("\nTop 10 features:\n", imp_df.head(10).to_string(index=False))

    # ---- Save model + defaults for the Streamlit app
    defaults = {c: (X_train[c].median() if pd.api.types.is_numeric_dtype(X_train[c])
                    else X_train[c].mode()[0]) for c in X.columns}
    options = {c: sorted(X_train[c].dropna().unique().tolist())
               for c in X.select_dtypes(exclude="number").columns}
    joblib.dump({"model": best, "features": X.columns.tolist(), "defaults": defaults,
                 "options": options, "threshold": threshold, "name": best_name},
                OUT / "fraud_model.joblib")
    print(f"\nSaved model and charts to {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/insurance_claims.csv")
    main(ap.parse_args().data)
