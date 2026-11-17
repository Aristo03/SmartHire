import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

from src.features.text_features import build_tfidf
from src.features.match_features import extract_skills


def split_skills(value) -> list[str]:
    """'python|sql|excel' -> ['python', 'sql', 'excel'] (handles empty/NaN)."""
    return [s for s in str(value).split("|") if s and s != "nan"]


class JobRecommender:
    """Hybrid content-based job recommender.

    score(job) = cosine(resume, job) + role_weight * cosine(role_profile, job)
    role_profile = average TF-IDF vector of all training resumes in the
    candidate's predicted category (from the resume classifier).
    """

    def __init__(self, max_features: int = 50_000, role_weight: float = 1.0):
        self.vectorizer = build_tfidf(min_df=3, max_df=0.5, max_features=max_features)
        self.role_weight = role_weight
        self.role_profiles = {}

    def fit(self, jobs: pd.DataFrame, resumes: pd.DataFrame | None = None):
        self.jobs = jobs.reset_index(drop=True)
        # Job title repeated 3x: short, but the most informative part of a posting
        docs = (self.jobs["jobtitle"].fillna("") + " ") * 3 + self.jobs["description"].fillna("")
        self.job_matrix = self.vectorizer.fit_transform(docs)
        if resumes is not None:
            self.fit_role_profiles(resumes)
        return self

    def fit_role_profiles(self, resumes: pd.DataFrame):
        """One 'average resume' vector per category, in the job vector space."""
        resume_vecs = normalize(self.vectorizer.transform(resumes["Resume"]))
        for category, idx in resumes.groupby("Category").indices.items():
            profile = np.asarray(resume_vecs[idx].mean(axis=0))
            self.role_profiles[category] = normalize(profile)
        return self

    def score_jobs(self, resume_text: str, category: str | None = None) -> np.ndarray:
        resume_vec = self.vectorizer.transform([resume_text])
        scores = linear_kernel(resume_vec, self.job_matrix).ravel()      # cosine similarity
        if category in self.role_profiles:                               # hybrid boost
            role_vec = self.role_profiles[category]
            scores = scores + self.role_weight * linear_kernel(role_vec, self.job_matrix).ravel()
        return scores

    def recommend(self, resume_text: str, category: str | None = None,
                  top_n: int = 10, max_experience=None) -> pd.DataFrame:
        scores = self.score_jobs(resume_text, category)

        if max_experience is not None:   # hide jobs needing more experience than the candidate has
            too_senior = self.jobs["exp_min"].fillna(0).to_numpy() > max_experience
            scores[too_senior] = -1

                # Take extra candidates, then drop reposts of the same job (same title + company)
        candidates = np.argsort(scores)[::-1][: top_n * 3]
        top_idx = (self.jobs.loc[candidates, ["jobtitle", "company"]]
                   .drop_duplicates().head(top_n).index.to_numpy())
        result = self.jobs.loc[top_idx, ["jobtitle", "company", "location", "experience", "skills"]].copy()
        result["match_score"] = scores[top_idx].round(3)

        resume_skills = set(extract_skills(resume_text))
        result["matched_skills"] = result["skills"].apply(
            lambda s: sorted(resume_skills & set(split_skills(s))))
        return result.reset_index(drop=True)