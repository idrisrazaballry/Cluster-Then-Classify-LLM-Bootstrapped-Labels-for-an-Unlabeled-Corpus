"""Cluster Then Classify -- side-by-side demo.

Two classifiers, same input. One was trained on labels an LLM invented from
clusters, with zero human annotation. The other was trained on the true labels
and represents the ceiling. Where they disagree is the price of not annotating.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import joblib

HERE = Path(__file__).parent
BUNDLE = joblib.load(HERE / "model_bundle.joblib")

VEC = BUNDLE["vectorizer"]
CLF_BOOT = BUNDLE["clf_bootstrap"]
CLF_CEIL = BUNDLE["clf_ceiling"]
METRICS = BUNDLE.get("metrics", {})
OFFLINE = BUNDLE.get("offline_artifacts", False)


# --- text cleaning ---------------------------------------------------------
# Input has to be cleaned exactly the way the corpus was, or the vectorizer sees
# tokens it was never fitted on. src/data.py is copied into this folder; the
# resolver below finds the cleaning function whatever it is called.

sys.path.insert(0, str(HERE))


def _resolve_cleaner():
    try:
        import data  # src/data.py, copied alongside this file
    except ImportError:
        print("WARNING: data.py not found. Input will not be cleaned and "
              "predictions will be worse than the reported numbers.")
        return lambda s: s
    for name in ("clean_text", "clean", "clean_string", "normalise", "normalize"):
        fn = getattr(data, name, None)
        if callable(fn):
            print(f"using data.{name}() for input cleaning")
            return fn
    print("WARNING: no cleaning function found in data.py. Set it explicitly "
          "in _resolve_cleaner().")
    return lambda s: s


CLEAN = _resolve_cleaner()


# --- prediction ------------------------------------------------------------

def classify(text: str):
    text = (text or "").strip()
    if not text:
        return {}, {}, "Paste a news headline or story above to compare the two models."

    X = VEC.transform([CLEAN(text)])

    boot = dict(zip(CLF_BOOT.classes_, CLF_BOOT.predict_proba(X)[0]))
    ceil = dict(zip(CLF_CEIL.classes_, CLF_CEIL.predict_proba(X)[0]))

    top_boot = max(boot, key=boot.get)
    top_ceil = max(ceil, key=ceil.get)
    conf_boot = boot[top_boot]

    if top_boot == top_ceil:
        verdict = (
            f"**Both models say {top_ceil}.** Agreement on a clear-cut story is "
            f"the common case — it is the 82% of ceiling that bootstrapping "
            f"recovers for free."
        )
    else:
        verdict = (
            f"**They disagree.** The bootstrapped model says {top_boot} "
            f"({conf_boot:.0%} confident); the model trained on true labels says "
            f"{top_ceil}. This is the gap annotation would have closed — and "
            f"note how confident the bootstrapped model is while being wrong."
        )

    return boot, ceil, verdict


# --- interface -------------------------------------------------------------

INTRO = """
# Cluster Then Classify

Nobody labelled this training data. The documents were clustered, an LLM was
shown the cluster centres and asked to name them without being told what the
taxonomy was, and those names became the training labels.

The left model is the result. The right model is the same architecture trained
on the real labels — the ceiling you are paying an annotator to reach.
**Try to make them disagree.**
"""

NOTE = """
---
Business and Sci/Tech entangle: a chip maker's earnings is honestly both, and
clustering cannot draw a line the taxonomy itself draws badly. The last two
examples are built to trip it.

A classifier trained on bootstrapped labels scores **0.953** against its own
held-out split while being **0.728** accurate against ground truth. That 95% is
fidelity to KMeans, not accuracy — the most seductive number in the project,
and it means nothing.
"""

EXAMPLES = [
    "Red Sox complete stunning comeback to reach the World Series after four straight wins.",
    "Peace talks stall as delegates fail to agree on troop withdrawal timetable.",
    "Researchers unveil a telescope array capable of imaging planets around nearby stars.",
    "Chip maker beats quarterly earnings expectations as demand for AI accelerators surges.",
    "Regulators open antitrust probe into the search giant's advertising business.",
]


def build():
    with gr.Blocks(title="Cluster Then Classify", theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)
        if OFFLINE:
            gr.Markdown(
                "> **Provenance warning.** This bundle was built from offline "
                "stub artifacts. The cluster names were generated from top "
                "terms, not by an LLM, and the accuracy figures below were "
                "produced without an API call. Re-run the pipeline without "
                "`--offline` before treating any of this as a result."
            )

        inp = gr.Textbox(
            lines=4,
            label="News headline or story",
            placeholder="Paste a headline...",
        )
        btn = gr.Button("Compare the two models", variant="primary")

        verdict = gr.Markdown()

        with gr.Row():
            out_boot = gr.Label(
                num_top_classes=4,
                label="Bootstrapped — 0 human labels",
            )
            out_ceil = gr.Label(
                num_top_classes=4,
                label="Ceiling — 100% true labels",
            )

        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=inp)
        gr.Markdown(NOTE)

        if METRICS:
            gr.Markdown("### Reported results\n```json\n"
                        + __import__("json").dumps(METRICS, indent=2)[:1500]
                        + "\n```")

        for ev in (btn.click, inp.submit):
            ev(classify, inputs=inp, outputs=[out_boot, out_ceil, verdict])

    return demo


if __name__ == "__main__":
    build().launch()
