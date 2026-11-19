from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

FIGURES_DIR = ROOT / "reports" / "figures"

RESUME_RAW = RAW_DIR / "UpdatedResumeDataSet.csv"
JOBS_RAW = RAW_DIR / "naukri_com-job_sample.csv"
RESUMES_CLEAN = PROCESSED_DIR / "resumes_clean.csv"
JOBS_CLEAN = PROCESSED_DIR / "jobs_clean.csv"
JOBS_APP = PROCESSED_DIR / "jobs_app.csv.gz"   # compressed job corpus for the web app

RANDOM_STATE = 42