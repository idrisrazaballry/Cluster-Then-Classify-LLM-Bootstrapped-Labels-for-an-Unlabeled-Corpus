#!/usr/bin/env python3
"""Cluster Then Classify -- end-to-end runner.

    python run_pipeline.py --offline          # no network, no API key, no cost
    python run_pipeline.py                    # MiniLM + Gemini, the real run
    python run_pipeline.py --k 6 --spot-check

--offline swaps MiniLM for TF-IDF+SVD and the LLM for a deterministic
stand-in, so you can verify the plumbing before spending anything. The Phase 5
numbers it prints are real for that weaker embedding; treat them as a floor.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

import config as C
import cluster as CL
import evaluate as EV
import label as LB
from data import load_and_clean
from embed import get_embeddings
from classify import make_split


def banner(msg):
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="TF-IDF embeddings + mock labeller; no network")
    ap.add_argument("--k", type=int, default=C.K_FINAL)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sweep", action="store_true", help="run the k sweep")
    ap.add_argument("--spot-check", action="store_true",
                    help="LLM propagation audit (~15 extra API calls)")
    ap.add_argument("--force-embed", action="store_true")
    args = ap.parse_args()

    # ---- Phase 0 -------------------------------------------------------
    banner("PHASE 0  load, clean, quarantine labels")
    work, y_true = load_and_clean(args.csv)

    # ---- Phase 1 -------------------------------------------------------
    banner("PHASE 1  embeddings")
    emb = get_embeddings(work["text"], backend="tfidf" if args.offline else "minilm",
                         force=args.force_embed)

    # ---- Phase 2 -------------------------------------------------------
    banner("PHASE 2  clustering")
    if args.sweep:
        sweep, _ = CL.sweep(emb)
        print(sweep.to_string(index=False))
        print("\nNote: silhouette usually climbs with k on text embeddings. It is")
        print("measuring granularity, not correctness. Read the samples instead.")

    labels = CL.fit_kmeans(emb, args.k)
    work["cluster"] = labels
    work.to_csv(C.P_CLUSTERS, index=False)
    np.save(C.P_CLUSTER_LABELS, labels)
    print(f"[phase2] k={args.k}")
    print(work["cluster"].value_counts().sort_index().to_string())

    terms = CL.distinctive_terms(work["text"], labels)
    for c, t in terms.items():
        print(f"[phase2]   cluster {c}: {', '.join(t[:8])}")

    # ---- Phase 3 -------------------------------------------------------
    banner("PHASE 3  LLM labelling")
    reps = CL.all_representatives(emb, work["text"], labels)
    sizes = {int(c): int((labels == c).sum()) for c in set(labels)}

    # Offline swaps only the transport. label_clusters / propagate / spot_check
    # below are the same functions a live run executes.
    llm = LB.get_labeller(offline=args.offline, terms=terms)
    if args.offline:
        print("[phase3] OFFLINE transport (no network); real labelling code path")
    llm_labels = LB.label_clusters(llm, reps, sizes)
    print(llm_labels[["cluster", "label", "coherent", "confidence"]].to_string(index=False))

    human_desc = json.load(open(C.P_HUMAN_DESC)) if C.P_HUMAN_DESC.exists() else None
    ok, problems = LB.audit_labels(llm_labels, human_desc)
    if not ok:
        print("\n[phase3] AUDIT FLAGS:")
        for p in problems:
            print("  -", p)
        print("  These do not stop the run, but read the clusters again before")
        print("  trusting the Phase 5 numbers.")

    boot = LB.propagate(work, llm_labels)
    print(f"\n[phase3] label distribution:\n{boot['llm_label'].value_counts().to_string()}")

    n_calls = 1
    if args.spot_check:
        _, agreement = LB.spot_check(llm, boot, llm_labels)
        n_calls += -(-C.SPOT_CHECK_N // C.SPOT_CHECK_BATCH)

    # ---- Phase 4 -------------------------------------------------------
    banner("PHASE 4  downstream classifier (trained on bootstrapped labels)")
    y_boot = boot["llm_label"].values
    tr, te = make_split(len(boot), y_boot)
    np.savez(C.P_SPLIT, train=tr, test=te)

    m_txt = EV.train_text(work["text"].values, y_boot, tr)
    fidelity = float((m_txt.predict(work["text"].values[te]) == y_boot[te]).mean())
    print(f"[phase4] fidelity to the clustering: {fidelity:.3f}")
    print("[phase4] This is NOT task accuracy -- the test split carries")
    print("[phase4] bootstrapped labels too. Phase 5 is what makes it meaningful.")

    # ---- Phase 5 -------------------------------------------------------
    banner("PHASE 5  the reveal")
    rep = EV.full_report(labels, y_boot, y_true, work["text"].values, emb, tr, te)

    q = rep["clustering_quality"]
    print(f"clustering ARI {q['ARI']:.3f} | NMI {q['NMI']:.3f}")
    print(f"cluster->class alignment accuracy : {rep['cluster_alignment_accuracy']:.1%}")
    print(f"bootstrapped label accuracy       : {rep['bootstrapped_label_accuracy']:.1%}")

    print("\n--- ABLATION ---")
    t = rep["ablation"].copy()
    t["accuracy"] = t["accuracy"].map("{:.3f}".format)
    t["% of ceiling"] = t["% of ceiling"].map("{:.1%}".format)
    print(t.to_string(index=False))
    rep["ablation"].to_csv(C.P_ABLATION, index=False)

    print("\n--- PER CLASS (bootstrapped model vs true labels) ---")
    print(rep["classification_report"])

    print("--- ERROR ANALYSIS ---")
    print(rep["errors"]["contingency"].to_string())
    for m in rep["errors"]["merged"]:
        print(f"MERGED : cluster {m['cluster']} mixes {m['classes']} {m['shares']}")
    for s in rep["errors"]["split"]:
        print(f"SPLIT  : class {s['class']} spread over clusters {s['clusters']} {s['shares']}")
    if not rep["errors"]["merged"] and not rep["errors"]["split"]:
        print("clean one-to-one recovery")

    cost = EV.cost_comparison(len(work), n_calls)
    print(f"\n--- COST (rough) ---")
    print(f"API      : ${cost['api_cost_usd']} over {n_calls} call(s)")
    print(f"Human    : ${cost['human_cost_usd']} ({cost['human_hours']}h at 200 docs/h)")
    print(f"Ratio    : {cost['ratio']}x cheaper")

    json.dump({
        "offline": args.offline, "k": args.k,
        "clustering_quality": q,
        "cluster_alignment_accuracy": rep["cluster_alignment_accuracy"],
        "bootstrapped_label_accuracy": rep["bootstrapped_label_accuracy"],
        "fidelity_to_clustering": fidelity,
        "ablation": rep["ablation"].to_dict("records"),
        "label_map": rep["label_map"],
        "cost": cost,
    }, open(C.P_RESULTS, "w"), indent=2)
    print(f"\nwrote {C.P_RESULTS}")


if __name__ == "__main__":
    main()
