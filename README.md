# Cluster Then Classify

**Building a labelled dataset from an unlabelled corpus, with no human annotation.**

Cluster the documents, have an LLM name the clusters, propagate those names as
training labels, and train a cheap classifier that runs without the LLM at
inference. The question the project answers is not "does it work" but **how much
accuracy do you give up by not annotating.**

Corpus: AG News (7,600 rows, 4 classes). The true labels are quarantined at
Phase 0 and opened only at Phase 5, so every decision — the number of clusters,
the algorithm, the category names — is made blind. That ordering is the entire
methodology; without it the final number is unfalsifiable.

---

## Results

From `python run_pipeline.py --offline` — TF-IDF embeddings, deterministic
stand-in labeller. **These are floor numbers.** MiniLM plus a real LLM should
beat them; treat this as the smoke test that proves the plumbing.

| Condition | Labels used | Accuracy | % of ceiling |
|---|---|---|---|
| Ceiling (tf-idf) | 100% true labels | 0.884 | 100.0% |
| Ceiling (embeddings) | 100% true labels | 0.863 | 97.6% |
| **Bootstrapped (tf-idf)** | **LLM labels, 0 human** | **0.728** | **82.4%** |
| Bootstrapped (embeddings) | LLM labels, 0 human | 0.731 | 82.7% |
| Hybrid (100 true + rest) | 100 human labels | 0.730 | 82.6% |
| Small supervised (100) | 100 human labels | 0.600 | 67.8% |
| Random floor | shuffled labels | 0.249 | 28.1% |

Clustering ARI 0.419, NMI 0.442. Bootstrapped label accuracy 71.4%.

**Three things to read off this table.**

*Bootstrapping beats hand-labelling 100 rows* (82.4% vs 67.8% of ceiling). That
is the result the project exists to establish.

*The hybrid row barely moves* (82.6% vs 82.4%). Adding 100 true labels changed
almost nothing, which says the bottleneck is not annotation volume — it is the
cluster boundaries themselves. More human labels will not fix that; better
separation would.

*A classifier trained on bootstrapped labels scored 0.953 against its own
held-out split* while being only 0.728 accurate against ground truth. That 95%
is fidelity to KMeans, not accuracy. It is the most seductive number in the
project and it means nothing.

### Over-clustering helps

k does not have to equal the number of true classes. Surplus clusters map
many-to-one onto classes, letting a broad class split into the sub-topics that
actually cluster:

| k | Bootstrapped accuracy | % of ceiling |
|---|---|---|
| 4 | 0.728 | 82.4% |
| 6 | 0.673 | 76.6% |
| 8 | **0.807** | **91.8%** |

Worth testing properly with real embeddings before drawing conclusions, but the
direction is suggestive.

### Known failure mode

Business and Sci/Tech entangle. Cluster 0 holds 1,287 Sci/Tech and 400 Business
documents; Business itself splits across two clusters (36%/40%). A story about a
chip maker's quarterly earnings is honestly both, and no amount of tuning
resolves a boundary the taxonomy itself does not draw cleanly.

---

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"        # PowerShell: $env:GOOGLE_API_KEY="your-key"
```

Never hard-code the key in a notebook. It ends up in the file's history.

## Verification status

**The offline path is fully verified. The three third-party integrations are
not** — this sandbox has no network, so MiniLM, UMAP and the Gemini API were
never executed.

| Component | Status |
|---|---|
| Phase 0 cleaning + quarantine | run, all 5 artifact checks at zero |
| Phase 2 clustering, k sweep, representatives | run at k = 4, 6, 8 |
| Phase 3 labelling, propagation, spot check | run via a fake transport |
| Phase 3 retry loop + error explanations | run (404 / auth / 429 branches) |
| Phase 4 classifiers | run |
| Phase 5 alignment, ablations, error analysis | run |
| `run_pipeline.py --offline` | run end to end from empty artifacts |
| Notebooks 01 and 05 | executed cell by cell |
| **MiniLM embeddings** | **never run** |
| **UMAP projection** | **never run** (PCA fallback tested) |
| **Live Gemini call** | **never run** |

`--offline` swaps only the network transport. `label_clusters`, `propagate`,
`spot_check`, `call_llm` and its retry loop are the same functions a live run
executes — a mock that short-circuited them would leave them untested, which
would defeat the point of having an offline mode.

The one thing still unexercised is the HTTP request itself. Every likely failure
of that request is handled with a named message: a retired model id, a bad key,
and rate limiting each print what to change rather than a raw stack trace.

Run the preflight first — it fails at the exact broken line instead of thirty
minutes into a full run:

```bash
python verify_setup.py           # imports, cleaning, parser, embeddings
python verify_setup.py --llm     # plus one real API call, a fraction of a cent
```

## Running

```bash
python run_pipeline.py --offline        # no network, no API key, no cost
python run_pipeline.py                  # MiniLM + Gemini, the real run
python run_pipeline.py --k 8 --sweep --spot-check
```

Run `--offline` first. It exercises every stage using TF-IDF instead of MiniLM
and a deterministic stand-in instead of the API, so plumbing bugs surface before
you spend anything.

| Flag | Effect |
|---|---|
| `--offline` | TF-IDF embeddings, mock labeller, no network |
| `--k N` | number of clusters (default 4) |
| `--sweep` | k sweep with inertia and silhouette |
| `--spot-check` | LLM propagation audit, ~15 extra API calls |
| `--force-embed` | ignore the embedding cache |

## Layout

```
config.py                 all paths and hyperparameters
run_pipeline.py           end-to-end runner
src/
  data.py                 Phase 0  clean + quarantine labels
  embed.py                Phase 1  MiniLM, or TF-IDF fallback
  cluster.py              Phase 2  k sweep, KMeans, HDBSCAN, representatives
  label.py                Phase 3  LLM labelling, parsing, audit, spot check
  classify.py             Phase 4  downstream classifiers
  evaluate.py             Phase 5  alignment, ablations, error analysis
verify_setup.py           preflight check for the live integrations
notebooks/
  01_clean_embed_cluster.ipynb
  03_llm_labeling_and_classifier.ipynb
  05_reveal_and_ablations.ipynb
artifacts/                everything the pipeline writes
```

Use the notebooks to develop and to see the intermediate output; use
`run_pipeline.py` to reproduce a full run in one command.

---

## Design decisions worth defending in a writeup

**The prompt never names the taxonomy.** It does not say "news articles in four
categories." Telling the model that hands it the answer and makes Phase 5
meaningless. It discovers the categories from documents alone.

**All clusters go in one API call.** Labelling clusters separately produces
collisions — two clusters both come back "Technology News" because neither call
knew the other existed. One call forces mutual exclusivity.

**Representatives are centroid-nearest, not random.** Those documents are the
cluster's core. A random draw pulls in boundary cases and pushes the model
toward vague labels.

**The cluster→class map is fit on the training split only.** Deriving it from
all rows lets test-set ground truth influence the mapping. That is leakage, and
it inflates precisely the conditions the project is trying to defend.

**Hungarian alignment, never name matching.** If the LLM output "Sports" and you
matched it to the true class "Sports" by string equality, you would be rewarding
it for guessing which public dataset this is.

**Surplus clusters are assigned by majority class.** Hungarian is strictly
one-to-one, so at k > 4 it leaves clusters unmapped — those predictions can then
never match anything, silently deflating accuracy and producing phantom classes
with zero support. Many-to-one is the standard cluster-accuracy convention and
the only fair reading when clustering legitimately splits a class.

## Cleaning notes

Each rule in `src/data.py` was added because it changed the result:

- **Source markers** (`AP -`, `LOS ANGELES (Reuters) -`, `(AP)`) appear in three
  positions. Left in, clusters form around the news agency instead of the topic.
- **Title and Description are cleaned separately**, then joined. The prefix
  patterns anchor to the start of a field; after joining, the Description no
  longer starts the string and the patterns silently never fire.
- **Entities arrive missing their leading `&`** (`#39;`, `quot;`), so
  `html.unescape` passes over them untouched. Repair first, then unescape.
- **Unescaping exposes real markup.** Business stories embed Reuters quote links
  (`<A HREF="...FullQuote.aspx?ticker=SPLS.O">`), which inject `fullquote`,
  `aspx` and `reuters` as tokens — the Business cluster partly formed around
  Reuters markup.

Fixing these moved aligned accuracy from 61.3% to 65.6% on the TF-IDF dry run.
`src.data.audit()` checks all five artifact classes and every one of them caught
a real bug.

## Caveats

- Offline-mode numbers use TF-IDF, not MiniLM. Real embeddings should be better;
  nothing here has been run against the Gemini API.
- `umap-learn` and `hdbscan` are optional. Density clustering prefers sklearn's
  built-in HDBSCAN (1.3+); projection falls back to PCA when UMAP is missing.
  The standalone `hdbscan` package needs a C extension and often fails to
  install on Windows.
- 7,600 rows is the AG News *test* split. The 120k train split gives more data
  and probably cleaner clusters.
- The offline spot-check agreement figure is meaningless — the fake transport
  assigns categories by crude keyword overlap. Only a live run gives a real
  label-noise estimate.
- Cost figures are order-of-magnitude only. Do not present the second decimal as
  if it were measured.
