import json, pathlib
import joblib, numpy as np

res = json.loads(pathlib.Path("artifacts/phase5_results.json").read_text())
label_map = res["label_map"]

b = joblib.load("deploy/model_bundle.joblib")
clf = b["clf_bootstrap"]

missing = [c for c in clf.classes_ if c not in label_map]
if missing:
    raise SystemExit(f"label_map does not cover: {missing}")

clf.classes_ = np.array([label_map[c] for c in clf.classes_])
b["classes_bootstrap"] = list(clf.classes_)
b["label_map"] = label_map
b["offline_artifacts"] = bool(res.get("offline"))

joblib.dump(b, "deploy/model_bundle.joblib", compress=3)
print("label_map applied:", label_map)
print("bootstrap classes now:", list(clf.classes_))
print("ceiling classes:     ", b["classes_ceiling"])
print("offline artifacts:   ", b["offline_artifacts"])