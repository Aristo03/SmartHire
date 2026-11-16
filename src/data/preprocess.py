import re
import pandas as pd

JOB_DROP_COLS = ["site_name", "numberofpositions", "uniq_id"]
JOB_KEEP_COLS = ["jobid", "jobtitle", "company", "location", "industry",
                 "functional_area", "education", "experience",
                 "exp_min", "exp_max", "payrate", "postdate", "description"]


def clean_description(text: str) -> str:
    """Remove site boilerplate, contact info and the trailing metadata block."""
    text = str(text)
    text = re.sub(r"^\s*Job Description\s*(Send me Jobs like this)?\s*", "", text, flags=re.I)
    text = re.split(r"Salary\s*:", text, maxsplit=1)[0]   # cut trailing metadata
    text = re.sub(r"\S+@\S+", " ", text)                   # emails
    text = re.sub(r"\+?\d[\d\s-]{8,}\d", " ", text)        # phone numbers
    text = re.sub(r"\s+", " ", text)                       # extra whitespace
    return text.strip()


def parse_experience(exp):
    """'3 - 8 yrs' -> (3, 8). Returns (None, None) if it can't be parsed."""
    match = re.search(r"(\d+)\s*-\s*(\d+)", str(exp))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=JOB_DROP_COLS, errors="ignore")
    df = df.dropna(subset=["jobtitle", "jobdescription"])
    df = df.rename(columns={"skills": "functional_area",
                            "joblocation_address": "location"})

    df["description"] = df["jobdescription"].apply(clean_description)
    df[["exp_min", "exp_max"]] = pd.DataFrame(
        df["experience"].apply(parse_experience).tolist(), index=df.index)

    df = df[df["description"].str.len() > 50]                      # drop near-empty posts
    df = df.drop_duplicates(subset=["jobtitle", "company", "description"])
    return df[JOB_KEEP_COLS].reset_index(drop=True)


def clean_resumes(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)