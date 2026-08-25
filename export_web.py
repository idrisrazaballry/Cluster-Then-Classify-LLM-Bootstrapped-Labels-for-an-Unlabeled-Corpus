"""Export the serving bundle to JSON so the demo can run in a browser.

    python export_web.py

Writes docs/model.json. Before writing, it reimplements the TF-IDF transform
and the softmax in plain Python -- no scikit-learn -- and checks that the
reimplementation reproduces sklearn's predictions on sample text. That check is
the point of this script: the JS in index.html is a direct translation of the
same arithmetic, so if the Python reimplementation agrees, the browser will too.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import joblib

ROOT = Path(__file__).parent
BUNDLE = ROOT / "deploy" / "model_bundle.joblib"
OUT = ROOT / "docs" / "model.json"

# sklearn's default token pattern. Lowercase, then words of 2+ word chars.
TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    """Unigrams + bigrams, matching TfidfVectorizer(ngram_range=(1, 2))."""
    words = TOKEN.findall(text.lower())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def vectorize(text: str, vocab: dict[str, int], idf: list[float]) -> dict[int, float]:
    """sublinear_tf=True, then idf, then L2. Returns index -> weight."""
    counts: dict[int, int] = {}
    for tok in tokenize(text):
        j = vocab.get(tok)
        if j is not None:
            counts[j] = counts.get(j, 0) + 1

    vec = {j: (1.0 + math.log(c)) * idf[j] for j, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        vec = {j: v / norm for j, v in vec.items()}
    return vec


def predict(vec, coef, intercept, classes):
    scores = [
        sum(w * coef[k][j] for j, w in vec.items()) + intercept[k]
        for k in range(len(classes))
    ]
    hi = max(scores)
    exp = [math.exp(s - hi) for s in scores]
    total = sum(exp)
    return {c: e / total for c, e in zip(classes, exp)}


SAMPLES = [
    "Red Sox complete stunning comeback to reach the World Series after four straight wins.",
    "Peace talks stall as delegates fail to agree on a troop withdrawal timetable.",
    "Researchers unveil a telescope array capable of imaging planets around nearby stars.",
    "Chip maker beats quarterly earnings expectations as demand for AI accelerators surges.",
    "Regulators open an antitrust probe into the search giant's advertising business.",
    "Shares tumbled after the airline warned of weaker demand across its transatlantic routes.",
]


def main() -> None:
    if not BUNDLE.exists():
        raise SystemExit(f"\nMissing {BUNDLE}. Run export_model.py first.\n")

    b = joblib.load(BUNDLE)
    vec_sk = b["vectorizer"]
    boot, ceil = b["clf_bootstrap"], b["clf_ceiling"]

    if list(boot.classes_) != list(ceil.classes_):
        raise SystemExit(
            f"\nClass names differ between the two models:\n"
            f"  bootstrapped: {list(boot.classes_)}\n"
            f"  ceiling:      {list(ceil.classes_)}\n"
            f"Run fix_bundle.py to apply label_map before exporting.\n"
        )

    vocab = {t: int(i) for t, i in vec_sk.vocabulary_.items()}
    idf = [round(float(v), 7) for v in vec_sk.idf_]
    classes = [str(c) for c in boot.classes_]

    def coefs(clf):
        return [[round(float(v), 7) for v in row] for row in clf.coef_]

    model = {
        "classes": classes,
        "vocab": vocab,
        "idf": idf,
        "boot": {"coef": coefs(boot), "intercept": [float(v) for v in boot.intercept_]},
        "ceil": {"coef": coefs(ceil), "intercept": [float(v) for v in ceil.intercept_]},
        "metrics": b.get("metrics", {}),
    }

    # --- verification -----------------------------------------------------
    print("verifying the reimplementation against scikit-learn\n")
    worst = 0.0
    mismatches = 0
    for text in SAMPLES:
        v = vectorize(text, vocab, idf)
        for name, clf in (("boot", boot), ("ceil", ceil)):
            mine = predict(v, model[name]["coef"], model[name]["intercept"], classes)
            theirs = dict(zip(classes, clf.predict_proba(vec_sk.transform([text]))[0]))
            delta = max(abs(mine[c] - theirs[c]) for c in classes)
            worst = max(worst, delta)
            if max(mine, key=mine.get) != max(theirs, key=theirs.get):
                mismatches += 1
                print(f"  MISMATCH {name}: {text[:50]}")
    print(f"  max probability difference: {worst:.2e}")
    print(f"  label mismatches: {mismatches}\n")

    if mismatches or worst > 1e-4:
        raise SystemExit(
            "The reimplementation does not match scikit-learn, so the browser\n"
            "version would give different answers. Not writing model.json.\n"
        )

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(model, separators=(",", ":")))
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({mb:.1f} MB, {len(vocab)} terms)")
    if mb > 8:
        print("\nThat is large for a web page. Tell me the size and I'll add pruning.")


if __name__ == "__main__":
    main()