"""Cluster Then Classify -- side-by-side demo."""

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

sys.path.insert(0, str(HERE))


def _resolve_cleaner():
    try:
        import data
    except Exception as exc:
        print(f"WARNING: could not import data.py ({exc}). Input will not be cleaned.")
        return lambda s: s
    for name in ("clean_field", "clean_text", "clean", "clean_string"):
        fn = getattr(data, name, None)
        if callable(fn):
            try:
                probe = fn("Test (Reuters) - a headline.")
                if isinstance(probe, str):
                    print(f"using data.{name}() for input cleaning")
                    return fn
            except Exception:
                import pandas as pd

                def wrapped(s, _fn=fn):
                    return str(_fn(pd.Series([s])).iloc[0])

                try:
                    wrapped("Test (Reuters) - a headline.")
                    print(f"using data.{name}() via Series wrapper")
                    return wrapped
                except Exception as exc:
                    print(f"WARNING: data.{name}() unusable ({exc}).")
    print("WARNING: no usable cleaning function found in data.py.")
    return lambda s: s


CLEAN = _resolve_cleaner()


def classify(text: str):
    text = (text or "").strip()
    if not text:
        return {}, {}, "Enter a headline above to compare the two models."

    X = VEC.transform([CLEAN(text)])
    boot = dict(zip(CLF_BOOT.classes_, CLF_BOOT.predict_proba(X)[0]))
    ceil = dict(zip(CLF_CEIL.classes_, CLF_CEIL.predict_proba(X)[0]))

    top_boot = max(boot, key=boot.get)
    top_ceil = max(ceil, key=ceil.get)

    if top_boot == top_ceil:
        verdict = (
            f"**Both models say {top_ceil}.** Agreement is the common case — "
            f"this is the 81% of ceiling that bootstrapping reaches without "
            f"a single annotated example."
        )
    else:
        verdict = (
            f"**They disagree.** The bootstrapped model says {top_boot} "
            f"({boot[top_boot]:.0%} confident); the model trained on true "
            f"labels says {top_ceil}. Business stories are where this happens "
            f"most — see the note below."
        )
    return boot, ceil, verdict


INTRO = """
# Cluster Then Classify

Nobody labelled this training data. The documents were clustered, Gemini was
shown each cluster's top terms and asked to name it without being told what the
taxonomy was, and those names became the training labels.

The left model is the result. The right model is the same architecture trained
on the real labels — the ceiling you would pay an annotator to reach.
**Try to make them disagree.**
"""

NOTE = """
---
**Where it fails.** Business is the weak class: recall 0.36 against precision
0.87. The model finds barely a third of Business stories, though it is usually
right when it does. The cause sits upstream of the classifier — k-means splits
Business across two clusters, and 760 of those documents landed in the cluster
Gemini named World Politics. No labelling strategy recovers a distinction the
clustering never drew.

**The seductive number.** The classifier scores 0.955 against its own held-out
split while being 0.716 accurate against ground truth. That 95.5% is fidelity
to k-means, not accuracy — it measures how faithfully the classifier reproduces
the clustering, mistakes included.

**What the LLM actually bought.** Swapping Gemini's labels for deterministic
top-term strings changes accuracy by exactly zero, because renaming a cluster
moves no documents. The LLM contributes readable class names derived without
ground truth. The 81% of ceiling comes from the clustering.
"""

EXAMPLES = [
    "Red Sox complete stunning comeback to reach the World Series after four straight wins.",
    "Peace talks stall as delegates fail to agree on a troop withdrawal timetable.",
    "Researchers unveil a telescope array capable of imaging planets around nearby stars.",
    "Chip maker beats quarterly earnings expectations as demand for AI accelerators surges.",
    "Regulators open an antitrust probe into the search giant's advertising business.",
]


def _summary():
    rows = []
    for r in METRICS.get("ablation", []):
        rows.append(
            f"| {r['condition']} | {r['labels used']} | "
            f"{r['accuracy']:.3f} | {r['% of ceiling'] * 100:.1f}% |"
        )
    if not rows:
        return ""
    head = "| Condition | Labels used | Accuracy | % of ceiling |\n|---|---|---|---|"
    return "### Ablation\n" + head + "\n" + "\n".join(rows)


def build():
    with gr.Blocks(title="Cluster Then Classify") as demo:
        gr.Markdown(INTRO)
        inp = gr.Textbox(lines=4, label="News headline or story",
                         placeholder="Paste a headline...")
        btn = gr.Button("Compare the two models", variant="primary")
        verdict = gr.Markdown()
        with gr.Row():
            out_boot = gr.Label(num_top_classes=4,
                                label="Bootstrapped — 0 human labels")
            out_ceil = gr.Label(num_top_classes=4,
                                label="Ceiling — 100% true labels")
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=inp)
        gr.Markdown(NOTE)
        summary = _summary()
        if summary:
            gr.Markdown(summary)
        for ev in (btn.click, inp.submit):
            ev(classify, inputs=inp, outputs=[out_boot, out_ceil, verdict])
    return demo


if __name__ == "__main__":
    build().launch()