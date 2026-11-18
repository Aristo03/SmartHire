from collections import Counter

import numpy as np
import pandas as pd

from src.features.match_features import extract_skills
from src.models.recommender import split_skills


def skill_gap_report(resume_text: str, rec, category: str | None = None,
                     top_jobs: int = 50, top_skills: int = 15,
                     min_demand_pct: int = 10) -> dict:
    """Compare the candidate's skills with what their best-matching jobs ask for.

    1. Find the candidate's top-N matching jobs (the "target market").
    2. Count how often each skill appears in those jobs (demand).
    3. Mark which in-demand skills the candidate has / is missing.
    """
    candidate = set(extract_skills(resume_text))

    scores = rec.score_jobs(resume_text, category)
    top_idx = np.argsort(scores)[::-1][:top_jobs]
    counts = Counter(skill for value in rec.jobs.loc[top_idx, "skills"]
                     for skill in split_skills(value))

    market = pd.DataFrame(counts.most_common(top_skills), columns=["skill", "jobs"])
    market["demand_pct"] = (100 * market["jobs"] / top_jobs).round().astype(int)
    market["have"] = market["skill"].isin(candidate)

    # Weighted coverage: having a skill that 80% of jobs want counts more than one 10% want
    total = market["jobs"].sum()
    coverage = float(market.loc[market["have"], "jobs"].sum() / total) if total else 0.0

    return {
        "candidate_skills": sorted(candidate),
        "market": market.drop(columns="jobs"),
        # Only flag gaps that at least min_demand_pct% of matching jobs ask for (ignore noise)
        "missing": market.loc[~market["have"] & (market["demand_pct"] >= min_demand_pct), "skill"].tolist(),
        "coverage": round(coverage, 3),
    }