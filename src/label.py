"""Phase 3 -- turn clusters into a labelled training set using an LLM.

Design decisions that matter more than the code:

1. The prompt never states the taxonomy. Telling the model "these are news
   articles in four categories" hands it the answer and makes Phase 5
   meaningless. It must discover the categories from documents alone.

2. All clusters go in ONE call. Labelling clusters separately produces
   collisions -- two clusters both come back "Technology News" because neither
   call knew the other existed. One call forces mutual exclusivity, which is
   what a classification target requires.

3. temperature=0. Labelling is not a creative task and needs to be reproducible.
"""
import json
import re
import time

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------
def get_llm(model=None, temperature=None, api_key=None):
    import os

    from langchain_google_genai import ChatGoogleGenerativeAI

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set.\n"
            '  PowerShell:  $env:GOOGLE_API_KEY = "your-key"\n'
            '  bash/zsh:    export GOOGLE_API_KEY="your-key"'
        )
    return ChatGoogleGenerativeAI(
        model=model or C.LLM_MODEL,
        google_api_key=key,
        temperature=C.LLM_TEMPERATURE if temperature is None else temperature,
    )


def response_text(resp) -> str:
    """`.content` is normally a str; newer versions may return a list of blocks."""
    c = resp.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def parse_json(text):
    """Models wrap JSON in markdown fences and preamble however firmly you ask."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", t, re.DOTALL)      # outermost array
        if not m:
            raise
        return json.loads(m.group(0))


CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
]


def _explain(exc):
    """Turn the two failures that actually happen into actionable messages.

    Model ids get deprecated, and a stale one surfaces as a 404 buried in a
    stack trace that never mentions config.py.
    """
    msg = str(exc)
    if "404" in msg or "not found" in msg.lower() or "NotFound" in type(exc).__name__:
        return (f"\nThe model id in config.py appears to be invalid or retired.\n"
                f"  Current: LLM_MODEL = {C.LLM_MODEL!r}\n"
                f"  Try one of: {', '.join(CANDIDATE_MODELS)}\n"
                f"  Or list what your key can reach:\n"
                f"    import google.generativeai as genai\n"
                f"    genai.configure(api_key=...); "
                f"[print(m.name) for m in genai.list_models()]\n")
    if any(t in msg.lower() for t in ("api key", "permission", "unauthenticated", "401", "403")):
        return ("\nAuthentication failed. Check GOOGLE_API_KEY is set correctly and\n"
                "that the Generative Language API is enabled for that key.\n")
    if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
        return ("\nRate limited. The labelling call is one request; the spot check is\n"
                "~15. Wait a minute, or raise SPOT_CHECK_BATCH in config.py.\n")
    return ""


def call_llm(llm, prompt, retries=3):
    for attempt in range(retries):
        try:
            raw = response_text(llm.invoke(prompt))
            return parse_json(raw), raw
        except Exception as exc:
            hint = _explain(exc)
            if hint or attempt == retries - 1:
                # A bad model id or bad key will not fix itself on retry.
                if hint:
                    print(hint)
                    raise RuntimeError(f"{type(exc).__name__}: {exc}{hint}") from exc
                raise
            print(f"  [llm] attempt {attempt + 1} failed ({type(exc).__name__}); retrying")
            time.sleep(2 ** attempt)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
def truncate(s, n=280):
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."


def build_label_prompt(reps, sizes):
    blocks = []
    for c, docs in reps.items():
        listed = "\n".join(f"  {i + 1}. {truncate(d)}" for i, d in enumerate(docs))
        blocks.append(f"### CLUSTER {c}  (contains {sizes[c]} documents total)\n{listed}")
    corpus = "\n\n".join(blocks)

    return f"""You are analysing the output of an unsupervised clustering run on a
corpus of short text documents. Each cluster below is shown via the documents
closest to its centre.

Give every cluster a category label that could serve as a classification target.

Requirements:
- Labels must be MUTUALLY EXCLUSIVE. No two clusters may receive labels that
  overlap in meaning. If two clusters look similar, identify what actually
  separates them and let the labels express that difference.
- Labels must be SHORT: one to three words, title case.
- Base the label on what the documents are ABOUT -- not on their writing style,
  their length, or the publication they came from.
- Set "coherent" to false if a cluster has no single subject and reads as a
  grab-bag. Do not invent a label to be agreeable.
- "confidence" is your own 0-1 estimate that the label describes the cluster.

Return ONLY a JSON array. No prose, no markdown fences.

[
  {{"cluster": 0, "label": "...", "description": "one sentence", "coherent": true, "confidence": 0.0}}
]

{corpus}
"""


def build_spot_prompt(texts, taxonomy):
    listed = "\n".join(f"{i + 1}. {truncate(t, 240)}" for i, t in enumerate(texts))
    return f"""Assign each document below to exactly one category.

Categories:
{taxonomy}

Return ONLY a JSON array: [{{"n": 1, "label": "..."}}]
Use the category names exactly as written above.

Documents:
{listed}
"""


# --------------------------------------------------------------------------
# labelling
# --------------------------------------------------------------------------
def label_clusters(llm, reps, sizes, save=True):
    prompt = build_label_prompt(reps, sizes)
    parsed, raw = call_llm(llm, prompt)

    if save:
        C.P_LLM_RAW.write_text(raw, encoding="utf-8")
        json.dump(parsed, open(C.P_LLM_LABELS, "w"), indent=2)

    df = pd.DataFrame(parsed).sort_values("cluster").reset_index(drop=True)
    for col, default in [("coherent", True), ("confidence", 1.0), ("description", "")]:
        if col not in df.columns:
            df[col] = default
    return df


def audit_labels(llm_labels, human_desc=None):
    """Three checks, none of which need ground truth. Returns (ok, problems)."""
    problems = []

    flagged = llm_labels[(~llm_labels["coherent"].astype(bool))
                         | (llm_labels["confidence"] < 0.6)]
    if len(flagged):
        problems.append(
            f"{len(flagged)} cluster(s) flagged incoherent or low-confidence: "
            f"{list(flagged['cluster'])}"
        )

    names = llm_labels["label"].str.lower().str.strip()
    if names.nunique() < len(names):
        problems.append("label collision -- clusters received overlapping names; "
                        "lower k or strengthen the exclusivity instruction")

    if human_desc:
        missing = [c for c in llm_labels["cluster"] if str(c) not in human_desc]
        if missing:
            problems.append(f"no human description written for clusters {missing}")

    return (len(problems) == 0), problems


def propagate(work, llm_labels, save=True):
    """Every document inherits its cluster's label."""
    mapping = dict(zip(llm_labels["cluster"], llm_labels["label"]))
    out = work.copy()
    out["llm_label"] = out["cluster"].map(mapping)
    if save:
        out.to_csv(C.P_BOOTSTRAPPED, index=False)
    return out


def spot_check(llm, bootstrapped, llm_labels, n=None, batch=None, save=True):
    """Estimate propagation noise WITHOUT ground truth.

    Cluster-level labels assume every member belongs; boundary documents often
    do not. Ask the model to label individual documents against the taxonomy it
    just produced, then measure agreement with the propagated label. The
    disagreement rate is a label-noise figure you can report honestly having
    never touched the true labels.
    """
    n = n or C.SPOT_CHECK_N
    batch = batch or C.SPOT_CHECK_BATCH

    taxonomy = "\n".join(f"- {r.label}: {r.description}" for r in llm_labels.itertuples())
    sample = bootstrapped.sample(min(n, len(bootstrapped)), random_state=C.RANDOM_STATE)

    preds = []
    for start in range(0, len(sample), batch):
        chunk = sample.iloc[start:start + batch]
        out, _ = call_llm(llm, build_spot_prompt(chunk["text"].tolist(), taxonomy))
        got = {int(o["n"]): o["label"] for o in out}
        preds += [got.get(i + 1) for i in range(len(chunk))]
        print(f"  [spot] {start + len(chunk)}/{len(sample)}", end="\r")

    sample = sample.assign(llm_direct=preds)
    agreement = float((sample["llm_direct"] == sample["llm_label"]).mean())
    if save:
        sample.to_csv(C.P_SPOTCHECK, index=False)
    print(f"\n  [spot] propagated vs direct agreement: {agreement:.1%} "
          f"(implied label noise ~{1 - agreement:.1%})")
    return sample, agreement


# --------------------------------------------------------------------------
# offline stand-in
# --------------------------------------------------------------------------
class FakeLLM:
    """A stand-in transport, not a stand-in pipeline.

    The point of this class is that `--offline` runs the SAME functions as a
    live run -- label_clusters, propagate, spot_check, call_llm and its retry
    loop all execute for real. Only the network hop is replaced. A mock that
    short-circuits label_clusters() would leave that function untested, which
    defeats the purpose of having an offline mode at all.

    It deliberately misbehaves the way real models do:
      - wraps JSON in markdown fences on some calls
      - adds conversational preamble on others
      - raises once on the first call, to exercise the retry path
    """

    def __init__(self, terms=None, fail_once=True):
        self.terms = terms or {}
        self.calls = 0
        self._failed = not fail_once

    class _Resp:
        def __init__(self, content):
            self.content = content

    def _label_payload(self, prompt):
        ids = [int(m) for m in re.findall(r"### CLUSTER (\d+)", prompt)]
        out = []
        for c in ids:
            name = (" / ".join(self.terms[c][:2]).title()
                    if c in self.terms else f"Cluster {c}")
            out.append({"cluster": c, "label": name,
                        "description": f"offline stand-in label for cluster {c}",
                        "coherent": True, "confidence": 1.0})
        return out

    def _spot_payload(self, prompt):
        cats = re.findall(r"^- (.+?):", prompt, re.MULTILINE)
        docs = re.findall(r"^(\d+)\. ", prompt, re.MULTILINE)
        # Assign by crude keyword overlap so agreement is non-trivial but
        # imperfect -- a constant answer would make the spot check meaningless.
        body = prompt.split("Documents:")[-1].lower()
        lines = [l for l in body.split("\n") if l.strip()]
        out = []
        for i, _ in enumerate(docs):
            line = lines[i] if i < len(lines) else ""
            best = max(cats, key=lambda c: sum(
                w in line for w in c.lower().replace("/", " ").split()))
            out.append({"n": int(docs[i]), "label": best})
        return out

    def invoke(self, prompt):
        self.calls += 1
        if not self._failed:
            self._failed = True
            raise ConnectionError("simulated transient failure (exercises retry)")

        payload = (self._label_payload(prompt) if "### CLUSTER" in prompt
                   else self._spot_payload(prompt))
        body = json.dumps(payload, indent=2)

        style = self.calls % 3                      # vary the wrapping
        if style == 0:
            text = f"```json\n{body}\n```"
        elif style == 1:
            text = f"Here are the labels:\n\n{body}\n\nLet me know if you need changes."
        else:
            text = body
        return self._Resp(text)


def get_labeller(offline=False, terms=None):
    """One entry point so run_pipeline does not branch on offline/online."""
    return FakeLLM(terms=terms) if offline else get_llm()
