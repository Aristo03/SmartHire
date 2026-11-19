"""End-to-end SmartHire engine: one object that the web app trains once and reuses."""
import pandas as pd

from src import config
from src.models.classifier import build_classifier, predict_top_k
from src.models.recommender import JobRecommender
from src.models.skill_gap import skill_gap_report


class SmartHire:
    def __init__(self, jobs: pd.DataFrame, resumes: pd.DataFrame):
        self.classifier = build_classifier().fit(resumes["Resume"], resumes["Category"])
        self.recommender = JobRecommender().fit(jobs, resumes)
        self.roles = sorted(resumes["Category"].unique())

    @classmethod
    def from_files(cls):
        return cls(pd.read_csv(config.JOBS_APP), pd.read_csv(config.RESUMES_CLEAN))

    def analyze(self, resume_text: str, target_role: str | None = None,
                max_experience: int | None = None, top_n: int = 10) -> dict:
        """Run every model on one resume and return all results in one dictionary."""
        predictions = predict_top_k(self.classifier, resume_text, k=3)
        predicted_role = predictions[0][0]
        role = target_role or predicted_role          # user can override the predicted role

        jobs = self.recommender.recommend(resume_text, category=role, top_n=top_n,
                                          max_experience=max_experience)
        gap = skill_gap_report(resume_text, self.recommender, category=role)
        family = jobs["job_family"].mode().iat[0] if "job_family" in jobs and len(jobs) else None

        return {
            "resume_text": resume_text,
            "predicted_role": predicted_role,
            "role_alternatives": [r for r, _ in predictions[1:]],
            "target_role": role,
            "job_family": family,
            "skills": gap["candidate_skills"],
            "jobs": jobs,
            "gap": gap,
        }

    def role_fit(self, resume_text: str, role: str) -> dict:
        """Skill-gap report against a specific target role (used by the mentor)."""
        return skill_gap_report(resume_text, self.recommender, category=role)