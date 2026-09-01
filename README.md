# Cluster Then Classify

**Building a labelled dataset from an unlabelled corpus, with no human annotation.**

[**Try the live demo →**](https://idrisrazaballry.github.io/Cluster-Then-Classify-LLM-Bootstrapped-Labels-for-an-Unlabeled-Corpus/)
Both classifiers run in your browser — TF-IDF and logistic regression
reimplemented in JavaScript, no inference server.

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

From `python run_pipeline.py` — one live Gemini call, no human labels.
Gemini saw each cluster's top terms and returned Software And Technology,
Sports News, Financial Markets, World Politics, without being told the taxonomy.

| Condition | Labels used | Accuracy | % of ceiling |
|---|---|---|---|
| Ceiling (tf-idf) | 100% true labels | 0.885 | 100.0% |
| Ceiling (LSA) | 100% true labels | 0.868 | 98.1% |
| **Bootstrapped (tf-idf)** | **LLM labels, 0 human** | **0.716** | **80.9%** |
| Bootstrapped (LSA) | LLM labels, 0 human | 0.714 | 80.7% |
| Hybrid (100 true + rest) | 100 human labels | 0.720 | 81.4% |
| Small supervised (100) | 100 human labels | 0.589 | 66.5% |
| Random floor | shuffled labels | 0.243 | 27.5% |

Clustering ARI 0.419, NMI 0.442. Bootstrapped label accuracy 71.4%.
API cost: $0.002, one call.

**Four things to read off this table.**

*Bootstrapping beats hand-labelling 100 rows* (80.9% vs 66.5% of ceiling). That
is the result the project exists to establish.

*The hybrid row barely moves* (81.4% vs 80.9%). Adding 100 true labels changed
almost nothing, which says the bottleneck is not annotation volume — it is the
cluster boundaries themselves. More human labels will not fix that; better
separation would.

*A classifier trained on bootstrapped labels scored 0.955 against its own
held-out split* while being only 0.716 accurate against ground truth. That 95.5%
is fidelity to KMeans, not accuracy. It is the most seductive number in the
project and it means nothing.

*The LLM contributes naming, not accuracy.* An earlier run used a deterministic
stand-in labeller that named clusters from their top terms — `Oil / Prices`
instead of `Financial Markets`. Bootstrapped label accuracy was 71.4% either
way, to sixteen decimal places, because renaming a cluster moves no documents.
What the LLM buys is human-readable class names derived without ground truth;
the 80.9% of ceiling comes from the clustering. Any claim that an LLM
*classified* anything here would be false.

### Known failure mode

Business is the weak class: recall 0.36 against precision 0.87. The classifier
finds barely a third of Business stories and is usually right when it does.

The cause is upstream. KMeans splits Business across two clusters — 36% into the
cluster that became Financial Markets, 40% into the one that became World
Politics — so 760 Business documents carry a World label into training. World's
precision drops to 0.57 as a result, while its recall reaches 0.90.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Business | 0.87 | 0.36 | 0.51 | 399 |
| Sci/Tech | 0.70 | 0.69 | 0.69 | 358 |
| Sports | 0.89 | 0.92 | 0.90 | 386 |
| World | 0.57 | 0.90 | 0.70 | 376 |

Business and Sci/Tech also blur — 400 Business documents sit in the Software And
Technology cluster, and a chip maker's quarterly earnings is honestly both — but
the Business/World split is the larger effect and the one that drives the
number. No labelling strategy recovers a distinction the clustering never drew.

### Over-clustering may help

k does not have to equal the number of true classes. Surplus clusters map
many-to-one onto classes, letting a broad class split into the sub-topics that
actually cluster:

| k | Bootstrapped accuracy | % of ceiling |
|---|---|---|
| 4 | 0.728 | 82.4% |
| 6 | 0.673 | 76.6% |
| 8 | **0.807** | **91.8%** |

**These three rows are from the offline stub run and have not been reproduced
live.** The k = 4 figure there was 0.728 against 0.716 in the live run, so treat
the direction as suggestive and the magnitudes as unverified until the sweep is
re-run.

---

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"        # PowerShell: $env:GOOGLE_API_KEY="your-key"
```

Never hard-code the key in a notebook. It ends up in the file's history.

## Verification status

| Component | Status |
|---|---|
| Phase 0 cleaning + quarantine | run, all 5 artifact checks at zero |
| Phase 2 clustering, k sweep, representatives | run at k = 4, 6, 8 |
| Phase 3 labelling, propagation, spot check | run |
| Phase 3 retry loop + error explanations | run (404 / auth / 429 branches) |
| Phase 4 classifiers | run |
| Phase 5 alignment, ablations, error analysis | run |
| `run_pipeline.py` end to end | run, live Gemini |
| Live Gemini call | run |
| Browser demo | deployed |
| **MiniLM embeddings** | **never run** |
| **UMAP projection** | **never run** (PCA fallback tested) |

**The rows labelled "embeddings" are LSA, not MiniLM.** Phase 1 loads a cached
100-dimensional matrix — TruncatedSVD over TF-IDF, not a sentence transformer,
which would be 384-dimensional. MiniLM has never executed in this environment.
Both `Ceiling (LSA)` and `Bootstrapped (LSA)` above should be read as a
dimensionality-reduction comparison, not as evidence about sentence embeddings.

Run the preflight first — it fails at the exact broken line instead of thirty
minutes into a full run:

```bash
python verify_setup.py           # imports, cleaning, parser, embeddings
python verify_setup.py --llm     # plus one real API call, a fraction of a cent
```

## Running

```bash
python run_pipeline.py --offline        # no network, no API key, no cost
python run_pipeline.py                  # the real run
python run_pipeline.py --k 8 --sweep --spot-check
```

Run `--offline` first. It exercises every stage using a deterministic stand-in
instead of the API, so plumbing bugs surface before you spend anything. Note
that `--offline` numbers are not results: the stand-in names clusters from their
top terms, and `artifacts/phase5_results.json` records `"offline": true` when a
run was produced that way. Check that flag before quoting any figure.

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
export_model.py           serving bundle from pipeline artifacts
export_web.py             bundle -> JSON for the browser demo
src/
  data.py                 Phase 0  clean + quarantine labels
  embed.py                Phase 1  MiniLM, or TF-IDF fallback
  cluster.py              Phase 2  k sweep, KMeans, HDBSCAN, representatives
  label.py                Phase 3  LLM labelling, parsing, audit, spot check
  classify.py             Phase 4  downstream classifiers
  evaluate.py             Phase 5  alignment, ablations, error analysis
verify_setup.py           preflight check for the live integrations
deploy/                   Gradio app for local use
docs/                     the published browser demo
notebooks/
  01_clean_embed_cluster.ipynb
  03_llm_labeling_and_classifier.ipynb
  05_reveal_and_ablations.ipynb
artifacts/                everything the pipeline writes (gitignored)
```

Use the notebooks to develop and to see the intermediate output; use
`run_pipeline.py` to reproduce a full run in one command.

---

## The demo

`docs/` serves a static page with no backend. `export_web.py` dumps the fitted
vectorizer's vocabulary and IDF weights plus both models' coefficients to JSON,
and the page reimplements the TF-IDF transform, the cleaning rules from
`src/data.py`, and the softmax in JavaScript.

Before writing anything, `export_web.py` reimplements the same arithmetic in
plain Python and checks it against scikit-learn on sample text. It refuses to
emit `model.json` if predictions disagree. Current agreement: max probability
difference 1.5e-08, zero label mismatches.

That the whole thing runs client-side is the point. The LLM is a training-time
cost. Inference is a sparse dot product, and it belongs in the visitor's
browser rather than on a server you pay for.

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

- MiniLM has never run here; the "LSA" rows are TruncatedSVD, not sentence
  embeddings.
- The k-sweep table is from the offline stub run and has not been reproduced
  live.
- 7,600 rows is the AG News *test* split. The 120k train split gives more data
  and probably cleaner clusters.
- Only one live labelling call has been made. Cluster naming is not
  deterministic across runs, and a second call could return different names —
  though on this clustering it would not change accuracy, since only the
  cluster→class mapping affects the score.
- Cost figures are order-of-magnitude only. The $0.002 API figure is real; the
  $948 annotation figure is a wage estimate against a task nobody performed, so
  the ratio between them is illustrative, not measured.
