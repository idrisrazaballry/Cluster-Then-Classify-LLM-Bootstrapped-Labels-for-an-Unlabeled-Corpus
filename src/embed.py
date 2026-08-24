"""Phase 1 -- embeddings, cached to disk.

Two backends:
  minilm  -- sentence-transformers, the real one. Needs a download.
  tfidf   -- TF-IDF + TruncatedSVD, no network. A weaker stand-in that lets you
             exercise the whole pipeline offline before spending API credits.
             Expect materially worse clustering; it is a smoke test, not a
             substitute.
"""
import numpy as np

import config as C


def _normalise(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    # A handful of documents reduce to zero vectors under TF-IDF (all tokens
    # below min_df). Unguarded this produces NaN and KMeans dies with a message
    # that does not mention the cause.
    return X / np.clip(n, 1e-12, None)


def _tfidf_svd(texts, n_components=100):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(max_features=20000, min_df=3, ngram_range=(1, 2),
                        sublinear_tf=True, stop_words="english").fit_transform(texts)
    Z = TruncatedSVD(n_components=n_components, random_state=C.RANDOM_STATE).fit_transform(X)
    return _normalise(Z)


def _minilm(texts):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(C.EMBED_MODEL)
    return model.encode(list(texts), batch_size=C.EMBED_BATCH,
                        show_progress_bar=True, normalize_embeddings=True)


def get_embeddings(texts, backend="minilm", cache=None, force=False):
    cache = cache or C.P_EMB
    if cache.exists() and not force:
        emb = np.load(cache)
        if len(emb) == len(texts):
            print(f"[phase1] loaded cached embeddings {emb.shape}")
            return emb
        print("[phase1] cache row count mismatch, recomputing")

    emb = _minilm(texts) if backend == "minilm" else _tfidf_svd(texts)
    np.save(cache, emb)
    print(f"[phase1] computed {backend} embeddings {emb.shape} -> {cache.name}")
    return emb
