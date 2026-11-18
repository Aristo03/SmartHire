from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from src import config
from src.models.recommender import split_skills


def reduce_dimensions(X, n_components: int = 100):
    """TF-IDF (50,000 dims) -> 100 dims with Truncated SVD (a.k.a. LSA), then re-normalize."""
    lsa = make_pipeline(TruncatedSVD(n_components, random_state=config.RANDOM_STATE),
                        Normalizer(copy=False))
    return lsa, lsa.fit_transform(X)


def evaluate_k(X_reduced, k_values, sample_size: int = 5000) -> pd.DataFrame:
    """Inertia (elbow method) + silhouette score for each candidate k."""
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X_reduced)
        sil = silhouette_score(X_reduced, km.labels_, sample_size=sample_size,
                               random_state=config.RANDOM_STATE)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def describe_clusters(km, lsa, vectorizer, jobs: pd.DataFrame, n_terms: int = 8) -> pd.DataFrame:
    """Top TF-IDF terms, top skills and common job titles for every cluster."""
    centers = lsa.named_steps["truncatedsvd"].inverse_transform(km.cluster_centers_)
    terms = vectorizer.get_feature_names_out()
    rows = []
    for c in range(km.n_clusters):
        mask = km.labels_ == c
        skills = Counter(s for v in jobs.loc[mask, "skills"] for s in split_skills(v))
        rows.append({
            "cluster": c,
            "size": int(mask.sum()),
            "top_terms": ", ".join(terms[i] for i in np.argsort(centers[c])[::-1][:n_terms]),
            "top_skills": ", ".join(s for s, _ in skills.most_common(5)),
            "example_titles": " | ".join(jobs.loc[mask, "jobtitle"].str.lower().value_counts().head(3).index),
        })
    return pd.DataFrame(rows)