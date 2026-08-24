"""Phase 4 -- train the downstream classifier on bootstrapped labels.

A warning that belongs next to the numbers this produces: the held-out split
carries bootstrapped labels too, so the score here measures how well the
classifier reproduced the CLUSTERING, not how well it learned the task. On this
corpus that gap is enormous -- roughly 97% fidelity to clusters that were
themselves only ~66% aligned to the true classes. Phase 5 is what converts this
into a meaningful number.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import config as C


def make_split(n, y, test_size=None, seed=None):
    """Index-level split so text and embedding variants share the same rows."""
    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=test_size or C.TEST_SIZE,
        random_state=seed if seed is not None else C.RANDOM_STATE,
        stratify=y,
    )
    return tr, te


def tfidf_classifier():
    """The cheap deployable: no embedding model needed at inference."""
    return Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def embedding_classifier():
    """Usually stronger -- but note it predicts labels derived from clustering
    these same embeddings, so part of its edge is circular and tends to shrink
    once measured against ground truth."""
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def train_text(texts, y, tr):
    m = tfidf_classifier()
    m.fit(np.asarray(texts)[tr], np.asarray(y)[tr])
    return m


def train_emb(emb, y, tr):
    m = embedding_classifier()
    m.fit(emb[tr], np.asarray(y)[tr])
    return m
