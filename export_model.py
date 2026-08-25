"""Build a serving bundle from the artifacts the pipeline already writes.

Run this after `python run_pipeline.py --offline` (or a live run) has populated
artifacts/. It reads only CSV/JSON artifacts named in config.py, so it does not
depend on any function signature inside src/.

    python export_model.py

Writes deploy/model_bundle.joblib containing:
    vectorizer      TF-IDF fitted on the cleaned corpus
    clf_bootstrap   trained on LLM-derived labels, zero human annotation
    clf_ceiling     trained on true labels -- the upper bound, for contrast
    classes_*       label names for each model
    metrics         whatever phase5_results.json holds
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

OUT = ROOT / "deploy" / "model_bundle.joblib"

# Both models are trained on the full corpus. Held-out scoring already happened
# in Phase 5; re-deriving accuracy here would only invite someone to quote a
# number that was never computed under the blind protocol. The metrics shown in
# the demo come from phase5_results.json, unmodified.


def _pick(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    """Resolve a column by trying known names. Fails loudly rather than guessing."""
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"\nCould not find the {what} column in the artifact.\n"
        f"Columns present: {list(df.columns)}\n"
        f"Add the right name to the candidate list in export_model.py:_pick().\n"
    )


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"\nMissing artifact: {path}\n"
            f"{hint}\n"
        )


def main() -> None:
    _require(C.P_BOOTSTRAPPED, "Run `python run_pipeline.py --offline` first.")
    _require(C.P_QUARANTINE, "The quarantined true labels are needed for the ceiling model.")

    boot = pd.read_csv(C.P_BOOTSTRAPPED)
    text_col = _pick(boot, ["text", "clean_text", "cleaned", "content", "document"], "text")
    boot_col = _pick(
        boot,
        ["bootstrapped_label", "llm_label", "predicted_label", "label", "assigned_label"],
        "bootstrapped label",
    )

    texts = boot[text_col].astype(str)
    y_boot = boot[boot_col].astype(str)

    truth = pd.read_csv(C.P_QUARANTINE)
    true_col = _pick(truth, [C.LABEL_COL, "true_label", "label", "Class Index"], "true label")
    if len(truth) != len(boot):
        raise SystemExit(
            f"\nRow mismatch: {len(boot)} bootstrapped rows vs {len(truth)} quarantined labels.\n"
            f"These artifacts came from different runs. Re-run the pipeline end to end.\n"
        )
    y_true = truth[true_col].map(lambda v: C.CLASS_NAMES.get(v, v)).astype(str)

    print(f"corpus: {len(texts)} rows")
    print(f"bootstrapped classes: {sorted(y_boot.unique())}")
    print(f"true classes:         {sorted(y_true.unique())}")

    # TF-IDF, not MiniLM. Your own ablation puts these 0.4 points apart (0.728 vs
    # 0.731) and drops torch from the runtime image. It is also the only path
    # your verification table lists as actually executed.
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    print(f"vocabulary: {len(vec.vocabulary_)} terms")

    def fit(y):
        return LogisticRegression(
            max_iter=2000, random_state=C.RANDOM_STATE
        ).fit(X, y)

    clf_boot = fit(y_boot)
    clf_ceil = fit(y_true)

    metrics = {}
    if C.P_RESULTS.exists():
        metrics = json.loads(C.P_RESULTS.read_text())
    else:
        print("warning: phase5_results.json missing, demo will omit the metrics panel")

    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vec,
            "clf_bootstrap": clf_boot,
            "clf_ceiling": clf_ceil,
            "classes_bootstrap": list(clf_boot.classes_),
            "classes_ceiling": list(clf_ceil.classes_),
            "metrics": metrics,
            "k": C.K_FINAL,
            "embedding": "tfidf",
            "n_rows": int(len(texts)),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        OUT,
        compress=3,
    )
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
