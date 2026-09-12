import json
import joblib
import numpy as np

from src.features import FEATURE_NAMES

model = joblib.load("/home/claude/project/results/model.joblib")
metrics = json.load(open("/home/claude/project/results/metrics.json"))

X0 = np.zeros((1, len(FEATURE_NAMES)))
init_raw = float(model._raw_predict_init(X0)[0][0])

trees = []
for stage in model.estimators_:
    est = stage[0]
    t = est.tree_
    trees.append({
        "feature": t.feature.tolist(),          # -2 = leaf
        "threshold": t.threshold.tolist(),
        "value": t.value[:, 0, 0].tolist(),      # leaf/node raw value
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
    })

export = {
    "feature_names": FEATURE_NAMES,
    "learning_rate": model.learning_rate,
    "init_raw": init_raw,
    "threshold": metrics["threshold"],
    "trees": trees,
}

out_path = "/home/claude/project/demo/model_export.json"
with open(out_path, "w") as f:
    json.dump(export, f)

print(f"Exported {len(trees)} trees, {len(FEATURE_NAMES)} features -> {out_path}")
import os
print("size:", os.path.getsize(out_path), "bytes")
