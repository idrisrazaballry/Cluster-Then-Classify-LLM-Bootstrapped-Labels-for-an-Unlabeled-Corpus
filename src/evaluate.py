"""Phase 5 -- the reveal.

The only module that opens the true labels. Everything upstream was decided
blind, which is the sole reason these numbers mean anything.

The headline result of the project is NOT accuracy. It is accuracy relative to
a model trained on real labels -- the "recovery" column of the ablation table.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, classification_report,
                             confusion_matrix, normalized_mutual_info_score)

import config as C
from classify import train_emb, train_text


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------
def align(pred, y_true, allow_surplus=True):
    """Map arbitrary cluster/label names onto true classes optimally.

    Clusters are unordered and the LLM's names are free-form, so "Cluster 3" or
    "Markets And Tech" has to be matched to a true class before any accuracy
    figure is possible. Hungarian on the contingency matrix gives the assignment
    that maximises agreement -- do not eyeball this, and do not match on name
    similarity, which would silently reward the LLM for guessing the dataset.

    Hungarian is strictly one-to-one, so when k exceeds the number of true
    classes it leaves surplus clusters unmapped. Unmapped predictions can then
    never match anything, which silently deflates accuracy and produces phantom
    classes with zero support in the report. With allow_surplus, leftovers are
    assigned to their own majority true class -- the standard many-to-one
    cluster-accuracy convention, and the only fair reading when the clustering
    legitimately splits a class in two.

    Returns (mapping, accuracy, contingency).
    """
    cm = pd.crosstab(pd.Series(pred, name="pred"), pd.Series(y_true, name="true"))
    rows, cols = linear_sum_assignment(-cm.values)
    mapping = {cm.index[r]: cm.columns[c] for r, c in zip(rows, cols)}

    if allow_surplus:
        for cat in cm.index:
            if cat not in mapping:
                mapping[cat] = cm.loc[cat].idxmax()

    hits = sum(cm.loc[cat, tgt] for cat, tgt in mapping.items() if tgt in cm.columns)
    return mapping, hits / len(y_true), cm


def clustering_quality(labels, y_true):
    """ARI and NMI are name-free -- they measure whether the partition recovered
    the true structure at all, independent of any label assignment."""
    return {
        "ARI": float(adjusted_rand_score(y_true, labels)),
        "NMI": float(normalized_mutual_info_score(y_true, labels)),
    }


# --------------------------------------------------------------------------
# ablations
# --------------------------------------------------------------------------
def run_ablations(texts, emb, y_boot, y_true, tr, te, n_hybrid=None, seed=None):
    """The core result table.

    Every condition trains on a different label source and is evaluated against
    the SAME true test labels, so the rows are directly comparable.

      ceiling          100% true labels -- best achievable, the denominator
      bootstrapped     LLM labels only, zero human annotation -- the method
      random           shuffled labels -- sanity floor
      hybrid           n_hybrid true labels + bootstrapped for the rest
      small_supervised n_hybrid true labels only

    small_supervised is the row a sceptical reader reaches for first: if hand
    labelling 100 rows beats the whole pipeline, you need to know and say so.
    """
    n_hybrid = n_hybrid or C.HYBRID_N_TRUE
    seed = C.RANDOM_STATE if seed is None else seed
    texts = np.asarray(texts)
    y_boot = np.asarray(y_boot)
    y_true = np.asarray(y_true)
    rng = np.random.RandomState(seed)

    # Bootstrapped labels live in their own namespace and must be mapped onto
    # true classes. Fit that mapping on the TRAINING split only -- deriving it
    # from all rows lets test-set ground truth influence the mapping, which is
    # leakage, and it flatters the bootstrapped conditions specifically.
    boot_map, _, _ = align(y_boot[tr], y_true[tr])
    for cat in np.unique(y_boot):                    # unseen in train -> identity
        boot_map.setdefault(cat, cat)

    def score(model, is_emb, needs_map):
        pred = model.predict(emb[te] if is_emb else texts[te])
        if needs_map:
            pred = np.array([boot_map.get(p, p) for p in pred])
        return float((pred == y_true[te]).mean()), pred

    results, preds = {}, {}

    # --- ceiling ---------------------------------------------------------
    for tag, is_emb, fn in [("tfidf", False, train_text), ("emb", True, train_emb)]:
        m = fn(texts if not is_emb else emb, y_true, tr)
        acc, p = score(m, is_emb, needs_map=False)
        results[f"ceiling_{tag}"] = acc
        preds[f"ceiling_{tag}"] = p

    # --- bootstrapped ----------------------------------------------------
    for tag, is_emb, fn in [("tfidf", False, train_text), ("emb", True, train_emb)]:
        m = fn(texts if not is_emb else emb, y_boot, tr)
        acc, p = score(m, is_emb, needs_map=True)
        results[f"bootstrapped_{tag}"] = acc
        preds[f"bootstrapped_{tag}"] = p

    # --- random floor ----------------------------------------------------
    y_rand = rng.permutation(y_true)
    m = train_text(texts, y_rand, tr)
    results["random_tfidf"] = float((m.predict(texts[te]) == y_true[te]).mean())

    # --- hybrid ----------------------------------------------------------
    # n_hybrid training rows get their true label; the rest keep the mapped
    # bootstrapped label. Mixed into one namespace so one model trains on both.
    hyb_idx = rng.choice(tr, size=min(n_hybrid, len(tr)), replace=False)
    y_mixed = np.array([boot_map.get(v, v) for v in y_boot], dtype=object)
    y_mixed[hyb_idx] = y_true[hyb_idx]
    m = train_text(texts, y_mixed, tr)
    results["hybrid_tfidf"] = float((m.predict(texts[te]) == y_true[te]).mean())

    # --- small supervised -------------------------------------------------
    m = train_text(texts, y_true, hyb_idx)
    results["small_supervised_tfidf"] = float((m.predict(texts[te]) == y_true[te]).mean())

    return results, preds, boot_map


def ablation_table(results, n_hybrid=None):
    n_hybrid = n_hybrid or C.HYBRID_N_TRUE
    ceiling = results["ceiling_tfidf"]
    order = [
        ("ceiling_tfidf", "Ceiling (tf-idf)", "100% true labels"),
        ("ceiling_emb", "Ceiling (embeddings)", "100% true labels"),
        ("bootstrapped_tfidf", "Bootstrapped (tf-idf)", "LLM labels, 0 human"),
        ("bootstrapped_emb", "Bootstrapped (embeddings)", "LLM labels, 0 human"),
        ("hybrid_tfidf", f"Hybrid ({n_hybrid} true + rest)", f"{n_hybrid} human labels"),
        ("small_supervised_tfidf", f"Small supervised ({n_hybrid})", f"{n_hybrid} human labels"),
        ("random_tfidf", "Random floor", "shuffled labels"),
    ]
    rows = [{"condition": name, "labels used": desc,
             "accuracy": results[k], "% of ceiling": results[k] / ceiling}
            for k, name, desc in order if k in results]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# error analysis
# --------------------------------------------------------------------------
def error_analysis(labels, y_true, class_names=None):
    """Distinguish MERGING from SPLITTING. Different causes, different fixes,
    and naming them correctly is what makes a writeup read as analysis."""
    names = class_names or C.CLASS_NAMES
    y_named = pd.Series(y_true).map(names).fillna(pd.Series(y_true).astype(str))
    cm = pd.crosstab(pd.Series(labels, name="cluster"), y_named)

    merged, split = [], []
    for cl in cm.index:                              # one cluster, many classes
        row = cm.loc[cl]
        share = row / row.sum()
        big = share[share > 0.25]
        if len(big) > 1:
            merged.append({"cluster": cl,
                           "classes": list(big.index),
                           "shares": [round(float(v), 3) for v in big.values]})

    for cls in cm.columns:                           # one class, many clusters
        col = cm[cls]
        share = col / col.sum()
        big = share[share > 0.25]
        if len(big) > 1:
            split.append({"class": cls,
                          "clusters": [int(i) for i in big.index],
                          "shares": [round(float(v), 3) for v in big.values]})

    return {"contingency": cm, "merged": merged, "split": split}


def cost_comparison(n_docs, n_api_calls, hourly_rate=25.0, docs_per_hour=200,
                    cost_per_call=0.002):
    """Rough, and label it rough in the writeup. The order of magnitude is the
    point, not the second decimal."""
    api = n_api_calls * cost_per_call
    human = (n_docs / docs_per_hour) * hourly_rate
    return {"api_cost_usd": round(api, 4),
            "human_cost_usd": round(human, 2),
            "human_hours": round(n_docs / docs_per_hour, 1),
            "ratio": round(human / api, 1) if api else None}


def full_report(labels, y_boot, y_true, texts, emb, tr, te, class_names=None):
    names = class_names or C.CLASS_NAMES
    y_named = pd.Series(y_true).map(names).fillna(pd.Series(y_true).astype(str)).values

    quality = clustering_quality(labels, y_true)
    _, cluster_acc, _ = align(labels, y_named)
    _, boot_label_acc, _ = align(np.asarray(y_boot), y_named)

    results, preds, boot_map = run_ablations(texts, emb, y_boot, y_named, tr, te)
    table = ablation_table(results)
    errors = error_analysis(labels, y_true, names)

    report = classification_report(
        y_named[te],
        np.array([boot_map.get(p, p) for p in preds["bootstrapped_tfidf"]]),
        zero_division=0,
    )

    return {
        "clustering_quality": quality,
        "cluster_alignment_accuracy": cluster_acc,
        "bootstrapped_label_accuracy": boot_label_acc,
        "ablation": table,
        "errors": errors,
        "classification_report": report,
        "label_map": {str(k): str(v) for k, v in boot_map.items()},
    }
