import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report,
    roc_auc_score, precision_recall_curve
)
import joblib

from src.features import FEATURE_NAMES

RESULTS_DIR = Path("/home/claude/project/results")


def load_rows(split):
    rows = []
    with open(RESULTS_DIR / f"claims_{split}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_xy(rows):
    X = np.array([[r["features"][name] for name in FEATURE_NAMES] for r in rows])
    y = np.array([r["gold_label"] for r in rows])
    return X, y


def best_threshold(y_true, scores):
    """Pick the probability threshold that maximizes F1 on the hallucination class."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best_i = np.nanargmax(f1[:-1]) if len(thr) else 0
    return thr[best_i] if len(thr) else 0.5, f1[best_i] if len(thr) else 0.0


def response_level_eval(rows, y_true, y_pred):
    """Group claim-level predictions back up to the response level: a response
    is flagged if it contains >=1 predicted-hallucinated claim; gold likewise."""
    by_resp_true, by_resp_pred = {}, {}
    for r, t, p in zip(rows, y_true, y_pred):
        key = (r["source_id"], r["response_id"])
        by_resp_true[key] = by_resp_true.get(key, 0) | int(t)
        by_resp_pred[key] = by_resp_pred.get(key, 0) | int(p)
    keys = list(by_resp_true.keys())
    yt = np.array([by_resp_true[k] for k in keys])
    yp = np.array([by_resp_pred[k] for k in keys])
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    return {"n_responses": len(keys), "precision": p, "recall": r, "f1": f1,
            "pct_flagged_gold": float(yt.mean()), "pct_flagged_pred": float(yp.mean())}


def main():
    train_rows = load_rows("train")
    test_rows = load_rows("test")

    # held-out validation split from train, for threshold tuning / model selection
    tr_rows, val_rows = train_test_split(
        train_rows, test_size=0.15, random_state=13,
        stratify=[r["gold_label"] for r in train_rows]
    )

    Xtr, ytr = to_xy(tr_rows)
    Xval, yval = to_xy(val_rows)
    Xtest, ytest = to_xy(test_rows)

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "gboost": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.08, random_state=0
        ),
    }

    results = {}
    best_name, best_val_f1 = None, -1
    fitted = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        val_scores = model.predict_proba(Xval)[:, 1]
        thr, val_f1 = best_threshold(yval, val_scores)
        results[name] = {"val_f1": float(val_f1), "threshold": float(thr)}
        print(f"{name}: val F1(hallucination)={val_f1:.3f} @ threshold={thr:.3f}")
        if val_f1 > best_val_f1:
            best_val_f1, best_name = val_f1, name

    print(f"\n==> Selected model: {best_name}")
    model = fitted[best_name]
    thr = results[best_name]["threshold"]

    # refit best model on train+val (all of train) before final test evaluation
    model.fit(Xtr, ytr)
    test_scores = model.predict_proba(Xtest)[:, 1]
    test_pred = (test_scores >= thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(ytest, test_pred, average="binary", zero_division=0)
    auc = roc_auc_score(ytest, test_scores)
    report = classification_report(ytest, test_pred, target_names=["supported", "hallucinated"], zero_division=0)
    print(f"\n=== CLAIM-LEVEL TEST RESULTS ({best_name}) ===")
    print(f"precision={p:.3f} recall={r:.3f} f1={f1:.3f} auc={auc:.3f}")
    print(report)

    resp_metrics = response_level_eval(test_rows, ytest, test_pred)
    print("=== RESPONSE-LEVEL TEST RESULTS ===")
    print(resp_metrics)

    # per task_type breakdown
    task_breakdown = {}
    for tt in ["Summary", "QA", "Data2txt"]:
        idxs = [i for i, r in enumerate(test_rows) if r["task_type"] == tt]
        if not idxs:
            continue
        yt = ytest[idxs]
        yp = test_pred[idxs]
        p_, r_, f1_, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        task_breakdown[tt] = {"n": len(idxs), "precision": p_, "recall": r_, "f1": f1_,
                               "positive_rate": float(yt.mean())}
    print("=== PER-TASK-TYPE BREAKDOWN ===")
    for k, v in task_breakdown.items():
        print(k, v)

    # feature importance (gboost) / coefficients (logreg) for the report
    if best_name == "gboost":
        importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    else:
        importances = dict(zip(FEATURE_NAMES, model.coef_[0].tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: -abs(kv[1])))
    print("=== FEATURE IMPORTANCE ===")
    for k, v in importances.items():
        print(f"  {k}: {v:.4f}")

    # save everything needed for error analysis + demo + writeup
    joblib.dump(model, RESULTS_DIR / "model.joblib")
    summary = {
        "model_selection": results,
        "selected_model": best_name,
        "threshold": float(thr),
        "claim_level_test": {"precision": p, "recall": r, "f1": f1, "auc": float(auc),
                              "n_test_claims": len(ytest), "n_positive": int(ytest.sum())},
        "response_level_test": resp_metrics,
        "per_task_type": task_breakdown,
        "feature_importance": importances,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # save per-claim predictions for error analysis
    with open(RESULTS_DIR / "test_predictions.jsonl", "w") as f:
        for row, score, pred in zip(test_rows, test_scores, test_pred):
            out = dict(row)
            out["pred_score"] = float(score)
            out["pred_label"] = int(pred)
            f.write(json.dumps(out) + "\n")

    print("\nSaved model.joblib, metrics.json, test_predictions.jsonl to results/")


if __name__ == "__main__":
    main()
