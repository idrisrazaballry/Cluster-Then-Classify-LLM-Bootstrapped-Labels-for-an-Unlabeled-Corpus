"""Phase 2 -- clustering.

Nothing here consults the true labels. k is chosen by the human reading
representative documents, not by a metric; see `sweep` for why silhouette is
recorded but not trusted.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import config as C


def sweep(emb, ks=None):
    """KMeans across k. Returns a frame plus the labels for each k.

    Silhouette on high-dimensional text embeddings is a weak signal and tends to
    rise with k regardless of whether the extra clusters mean anything. On this
    corpus it rose monotonically from k=4 to k=20 while agreement with the true
    classes fell by more than half. Recorded for the writeup; not used to pick k.
    """
    ks = ks or C.K_SWEEP
    rows, labels_by_k = [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE)
        lab = km.fit_predict(emb)
        labels_by_k[k] = lab
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(emb, lab, sample_size=min(5000, len(emb)),
                                           random_state=C.RANDOM_STATE),
        })
    return pd.DataFrame(rows), labels_by_k


def fit_kmeans(emb, k):
    return KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit_predict(emb)


def fit_hdbscan(emb, min_cluster_size=80, min_samples=10, n_components=15):
    """Density clustering, marking outliers as noise (-1).

    Prefers sklearn's built-in HDBSCAN (1.3+). The standalone `hdbscan` package
    needs a C extension and frequently fails to install on Windows, so it is
    only a fallback here.

    Reduces dimensionality first when UMAP is available -- density estimates
    degrade badly in 384 dimensions. Falls back to PCA, which is weaker for this
    but always installed.

    Returns (labels, n_clusters, noise_fraction).
    """
    dens = _reduce(emb, n_components)

    try:
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                      cluster_selection_method="eom", copy=True).fit_predict(dens)
    except ImportError:                              # sklearn < 1.3
        import hdbscan
        lab = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                              min_samples=min_samples, metric="euclidean",
                              cluster_selection_method="eom").fit_predict(dens)

    n = len(set(lab)) - (1 if -1 in lab else 0)
    return lab, n, float((lab == -1).mean())


def _reduce(emb, n_components):
    """UMAP if installed, PCA otherwise. UMAP is better; PCA always works."""
    try:
        import umap
        return umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.0,
                         metric="cosine", random_state=C.RANDOM_STATE).fit_transform(emb)
    except Exception as exc:
        from sklearn.decomposition import PCA
        print(f"[cluster] UMAP unavailable ({type(exc).__name__}); falling back to PCA")
        return PCA(n_components=min(n_components, emb.shape[1]),
                   random_state=C.RANDOM_STATE).fit_transform(emb)


def project_2d(emb, cache=None):
    """2-D projection for the scatter plot. UMAP preferred, PCA as fallback."""
    cache = cache or C.P_UMAP
    if cache.exists():
        p = np.load(cache)
        if len(p) == len(emb):
            return p
    proj = _reduce(emb, 2)
    np.save(cache, proj)
    return proj


def representatives(emb, texts, labels, cluster_id, n=None, mode="central"):
    """Documents nearest the centroid ('central') or a uniform draw ('random').

    Central is what the LLM should see: those documents are the cluster's core.
    A random draw pulls in boundary cases and pushes the model toward vague
    labels. Random is for YOU, when checking whether a cluster holds together
    at its edges.
    """
    n = n or C.REPS_PER_CLUSTER
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    if mode == "central":
        centroid = emb[idx].mean(axis=0, keepdims=True)
        order = idx[np.argsort(-cosine_similarity(emb[idx], centroid).ravel())]
    else:
        order = np.random.RandomState(C.RANDOM_STATE).permutation(idx)
    return list(pd.Series(texts).iloc[order[:n]])


def all_representatives(emb, texts, labels, n=None, mode="central"):
    return {int(c): representatives(emb, texts, labels, c, n, mode)
            for c in sorted(set(labels)) if c != -1}


def distinctive_terms(texts, labels, top=10):
    """Terms over-represented in each cluster. A cross-check on your reading."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    tf = TfidfVectorizer(max_features=20000, min_df=5, stop_words="english")
    M = tf.fit_transform(texts)
    vocab = np.array(tf.get_feature_names_out())
    out = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        m = labels == c
        inside = np.asarray(M[m].mean(axis=0)).ravel()
        outside = np.asarray(M[~m].mean(axis=0)).ravel()
        out[int(c)] = list(vocab[np.argsort(-(inside - outside))[:top]])
    return out
