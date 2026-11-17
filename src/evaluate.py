import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config
from src.models.classifier import build_classifier

# A job counts as "relevant" for a resume category if its TITLE contains one of these keywords.
# Weak (approximate) labels: good enough to compare versions of the recommender.
CATEGORY_TITLE_KEYWORDS = {
    "Advocate": ["advocate", "lawyer", "legal", "law", "litigation", "counsel"],
    "Arts": ["artist", "art", "animator", "illustrator", "graphic", "creative", "fine arts"],
    "Automation Testing": ["automation", "test", "qa", "selenium"],
    "Blockchain": ["blockchain", "ethereum", "solidity", "crypto"],
    "Business Analyst": ["business analyst", "analyst"],
    "Civil Engineer": ["civil", "site engineer", "structural", "construction"],
    "Data Science": ["data scien", "machine learning", "analytics", "data analyst", "ml", "ai"],
    "Database": ["database", "dba", "sql", "oracle", "mysql"],
    "DevOps Engineer": ["devops", "cloud", "aws", "sre", "build", "release"],
    "DotNet Developer": ["net", "dotnet", "c#", "asp"],
    "ETL Developer": ["etl", "informatica", "data warehouse", "datastage", "bi"],
    "Electrical Engineering": ["electrical", "electronics", "power", "maintenance"],
    "HR": ["hr", "human resource", "recruit", "talent", "payroll"],
    "Hadoop": ["hadoop", "big data", "spark", "data engineer"],
    "Health and fitness": ["fitness", "trainer", "gym", "nutrition", "dietitian", "health", "yoga"],
    "Java Developer": ["java", "j2ee"],
    "Mechanical Engineer": ["mechanical", "production", "design engineer", "maintenance"],
    "Network Security Engineer": ["network", "security", "firewall", "cyber"],
    "Operations Manager": ["operations", "operation"],
    "PMO": ["pmo", "project manag", "program manag", "project coordinator"],
    "Python Developer": ["python", "django"],
    "SAP Developer": ["sap", "abap"],
    "Sales": ["sales", "business development", "marketing"],
    "Testing": ["test", "qa", "quality"],
    "Web Designing": ["web design", "ui", "ux", "front end", "frontend", "html", "web developer"],
}


def relevance_masks(jobs: pd.DataFrame) -> dict:
    """category -> boolean array: is each job relevant to that category?"""
    titles = jobs["jobtitle"].fillna("").str.lower()
    return {
        cat: titles.str.contains(r"\b(?:" + "|".join(map(re.escape, kws)) + ")", regex=True).to_numpy()
        for cat, kws in CATEGORY_TITLE_KEYWORDS.items()
    }


def random_baseline(jobs: pd.DataFrame, resumes: pd.DataFrame) -> float:
    """Expected Precision@K if we recommended jobs at random."""
    relevant = relevance_masks(jobs)
    return float(np.mean([relevant[c].mean() for c in resumes["Category"]]))


def evaluate_recommender(rec, resumes: pd.DataFrame, k: int = 10,
                         role_weight: float = 1.0, n_splits: int = 3) -> pd.DataFrame:
    """Precision@K for every resume, without leakage:
    - the category used for the hybrid boost comes from out-of-fold classifier predictions
    - role profiles are built only from the training folds
    """
    X, y = resumes["Resume"], resumes["Category"].to_numpy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    predicted = cross_val_predict(build_classifier(), X, y, cv=cv)
    relevant = relevance_masks(rec.jobs)

    saved_profiles, saved_weight = rec.role_profiles, rec.role_weight
    rec.role_weight = role_weight
    rows = []
    for train_idx, test_idx in cv.split(X, y):
        rec.role_profiles = {}
        rec.fit_role_profiles(resumes.iloc[train_idx])
        for i in test_idx:
            category = predicted[i] if role_weight > 0 else None
            top = np.argsort(rec.score_jobs(X.iloc[i], category))[::-1][:k]
            rows.append({"category": y[i], "precision_at_k": relevant[y[i]][top].mean()})
    rec.role_profiles, rec.role_weight = saved_profiles, saved_weight   # restore
    return pd.DataFrame(rows)