#!/usr/bin/env python3
"""Preflight check for the three integrations that cannot be tested offline.

    python verify_setup.py            # imports + embeddings, no API call
    python verify_setup.py --llm      # also makes ONE tiny Gemini call

Run this before `python run_pipeline.py`. It takes under a minute and one
fraction of a cent, and it fails loudly at the exact line that is broken instead
of thirty minutes into a full run.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

PASS, FAIL, WARN = "  [ok]", "  [FAIL]", "  [warn]"
results = []


def check(name, fn, required=True):
    print(f"\n{name}")
    try:
        detail = fn()
        print(f"{PASS} {detail}")
        results.append((name, True, required))
        return True
    except Exception as exc:
        tag = FAIL if required else WARN
        print(f"{tag} {type(exc).__name__}: {exc}")
        results.append((name, False, required))
        return False


# --------------------------------------------------------------------------
def core_imports():
    import numpy, pandas, scipy, sklearn
    return (f"numpy {numpy.__version__}, pandas {pandas.__version__}, "
            f"sklearn {sklearn.__version__}, scipy {scipy.__version__}")


def data_loads():
    import config as C
    from data import load_and_clean
    work, y = load_and_clean(quarantine=False)
    assert len(work) == len(y) and len(work) > 0
    return f"{len(work)} rows cleaned, {len(set(y))} classes"


def sentence_transformers_works():
    """The real Phase 1 path. Downloads ~90MB the first time."""
    from sentence_transformers import SentenceTransformer

    import config as C
    m = SentenceTransformer(C.EMBED_MODEL)
    v = m.encode(["a short test sentence", "another one"],
                 normalize_embeddings=True)
    assert v.shape[0] == 2 and v.shape[1] > 0
    return f"{C.EMBED_MODEL} -> {v.shape[1]}-dim embeddings"


def umap_works():
    import numpy as np
    import umap
    x = np.random.RandomState(0).rand(200, 20)
    p = umap.UMAP(n_neighbors=10, random_state=42).fit_transform(x)
    return f"umap-learn OK, projected to {p.shape[1]}-d"


def hdbscan_works():
    import numpy as np
    from cluster import fit_hdbscan
    x = np.random.RandomState(0).rand(400, 10)
    lab, n, noise = fit_hdbscan(x, min_cluster_size=15, n_components=5)
    return f"density clustering OK ({n} clusters on random data)"


def llm_key_present():
    import os
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set.\n"
            '     PowerShell: $env:GOOGLE_API_KEY = "your-key"\n'
            '     bash/zsh:   export GOOGLE_API_KEY="your-key"'
        )
    return f"key present ({len(key)} chars, ...{key[-4:]})"


def llm_call_works():
    """One real call. This is the single most likely thing to be broken:
    the model id, the auth, or the response shape."""
    import config as C
    from label import call_llm, get_llm

    llm = get_llm()
    parsed, raw = call_llm(
        llm,
        'Return ONLY this JSON array and nothing else: '
        '[{"cluster": 0, "label": "Test", "description": "d", '
        '"coherent": true, "confidence": 1.0}]',
    )
    assert isinstance(parsed, list) and parsed[0]["label"]
    return f"{C.LLM_MODEL} responded, JSON parsed ({len(raw)} chars raw)"


def parser_robustness():
    """No network needed. Exercises the four response shapes seen in practice."""
    from label import parse_json, response_text

    obj = '[{"cluster": 0, "label": "A"}]'
    shapes = [obj, f"```json\n{obj}\n```", f"Sure!\n```\n{obj}\n```\nDone.",
              f"Here you go:\n{obj}"]
    for s in shapes:
        assert parse_json(s)[0]["label"] == "A", s

    class R:
        def __init__(self, c): self.content = c
    assert response_text(R("hi")) == "hi"
    assert response_text(R([{"type": "text", "text": "h"},
                            {"type": "text", "text": "i"}])) == "hi"
    return f"{len(shapes)} response shapes + 2 content shapes parsed"


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="make one real Gemini call (costs a fraction of a cent)")
    ap.add_argument("--skip-embed", action="store_true",
                    help="skip the sentence-transformers download")
    args = ap.parse_args()

    print("=" * 66)
    print("PREFLIGHT")
    print("=" * 66)

    check("1. core scientific stack", core_imports)
    check("2. data loading and cleaning", data_loads)
    check("3. JSON parser robustness (offline)", parser_robustness)

    if args.skip_embed:
        print("\n4. sentence-transformers\n  [skipped]")
    else:
        check("4. sentence-transformers (real Phase 1 path)",
              sentence_transformers_works)

    check("5. umap-learn (optional; PCA fallback exists)", umap_works, required=False)
    check("6. density clustering (sklearn HDBSCAN or fallback)", hdbscan_works,
          required=False)
    check("7. GOOGLE_API_KEY present", llm_key_present)

    if args.llm:
        check("8. live Gemini call", llm_call_works)
    else:
        print("\n8. live Gemini call\n  [skipped] re-run with --llm to test for real")

    print("\n" + "=" * 66)
    hard = [r for r in results if not r[1] and r[2]]
    soft = [r for r in results if not r[1] and not r[2]]
    if hard:
        print("BLOCKED. Fix these before running the pipeline:")
        for n, _, _ in hard:
            print(f"  - {n}")
    else:
        print("Ready. Run: python run_pipeline.py")
    if soft:
        print("\nOptional components missing (fallbacks will be used):")
        for n, _, _ in soft:
            print(f"  - {n}")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
