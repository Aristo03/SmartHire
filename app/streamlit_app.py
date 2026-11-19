"""SmartHire web portal: upload a resume -> role prediction, job matches, skill gaps, mentor."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # make `src` importable when run via `streamlit run`
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.mentor import EXAMPLE_QUESTIONS, answer
from src.parsing.resume_parser import extract_text
from src.pipeline import SmartHire

st.set_page_config(page_title="SmartHire", page_icon="🎯", layout="wide")


# ---------- Load once, then cache ----------
@st.cache_resource(show_spinner="Training models on 21,000+ jobs (first load only, about 30 s)...")
def load_engine() -> SmartHire:
    return SmartHire.from_files()


@st.cache_data
def load_samples() -> pd.Series:
    return pd.read_csv(config.RESUMES_CLEAN).groupby("Category")["Resume"].first()


@st.cache_data(show_spinner="Analyzing resume...")
def run_analysis(text: str, target_role, max_experience, top_n: int) -> dict:
    return engine.analyze(text, target_role=target_role, max_experience=max_experience, top_n=top_n)


engine = load_engine()
samples = load_samples()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Options")
    years = st.slider("Your years of experience", 0, 20, 2)
    hide_senior = st.checkbox("Hide jobs that need more experience", value=True)
    top_n = st.slider("Number of job matches", 5, 25, 10)
    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. **Classifier** (TF-IDF + Linear SVM) predicts your role, 88.6% CV accuracy\n"
        "2. **Hybrid recommender** (cosine similarity + role profile) ranks 21k jobs, Precision@10 = 0.75\n"
        "3. **Skill-gap report** compares your skills with your top-50 matching jobs\n"
        "4. **K-Means** (k=15) groups jobs into families\n\n"
        "Classical ML only, no LLMs."
    )

# ---------- Header + input ----------
st.title("🎯 SmartHire")
st.caption("Resume-to-job matching & career guidance engine")

col_upload, col_sample = st.columns([3, 2])
with col_upload:
    uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
with col_sample:
    sample_choice = st.selectbox("…or try a sample resume", ["—"] + list(samples.index))

resume_text = None
if uploaded is not None:
    try:
        resume_text = extract_text(uploaded.getvalue(), uploaded.name)
    except ValueError as err:
        st.error(str(err))
        st.stop()
elif sample_choice != "—":
    resume_text = samples[sample_choice]

if not resume_text:
    st.info("👆 Upload a resume (PDF, DOCX or TXT) or pick a sample to get started.")
    st.stop()

target = st.selectbox("Target role", ["Auto-detect from my resume"] + engine.roles)
target_role = None if target.startswith("Auto") else target

result = run_analysis(resume_text, target_role, years if hide_senior else None, top_n)
gap, jobs = result["gap"], result["jobs"]

# ---------- Low-information warning ----------
if len(result["skills"]) < 3:
    st.warning(
        f"⚠️ Only **{len(result['skills'])}** recognizable skill(s) were found in this resume. "
        "Results may be less accurate. Low coverage here may mean the resume doesn't **list** "
        "skills clearly, not that you lack them. Tip: add a dedicated *Skills* section "
        "(e.g. 'Python, SQL, Excel, Recruitment')."
    )

# ---------- Summary metrics ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted role", result["predicted_role"])
m2.metric("Job family", result["job_family"] or "—")
m3.metric("Skills detected", len(result["skills"]))
m4.metric(f"Skill coverage ({result['target_role']})", f"{gap['coverage']:.0%}")

tab_jobs, tab_gap, tab_mentor, tab_text = st.tabs(
    ["🎯 Matched jobs", "🧭 Skill gap", "💬 Mentor", "📄 Resume text"])

# ---------- Tab 1: Jobs ----------
with tab_jobs:
    if jobs.empty:
        st.info("No jobs found with these filters. Try increasing your years of experience.")
    else:
        table = jobs.assign(matched_skills=jobs["matched_skills"].str.join(", "))
        cols = ["jobtitle", "company", "location", "experience", "job_family", "match_score", "matched_skills"]
        st.dataframe(
            table[[c for c in cols if c in table.columns]],
            hide_index=True, width="stretch",
            column_config={
                "jobtitle": "Job title", "company": "Company", "location": "Location",
                "experience": "Experience", "job_family": "Job family",
                "matched_skills": "Your matching skills",
                "match_score": st.column_config.ProgressColumn(
                    "Match score", min_value=0.0, max_value=float(jobs["match_score"].max()), format="%.2f"),
            },
        )
        st.caption("Match score = cosine(resume, job) + cosine(role profile, job). Higher is better; compare jobs relative to each other.")
        st.subheader("Top 3 in detail")
        for job in jobs.head(3).itertuples():
            with st.expander(f"**{job.jobtitle}** · {job.company}"):
                st.write(str(job.description)[:1200] + ("…" if len(str(job.description)) > 1200 else ""))

# ---------- Tab 2: Skill gap ----------
with tab_gap:
    market = gap["market"]
    left, right = st.columns(2)
    with left:
        st.subheader("✅ In-demand skills you have")
        have = market.loc[market["have"], "skill"].tolist()
        st.write(", ".join(f"`{s}`" for s in have) or "None detected")
    with right:
        st.subheader("❌ Skills to add")
        st.write(", ".join(f"`{s}`" for s in gap["missing"]) or "No major gaps 🎉")

    if not market.empty:
        chart = (
            alt.Chart(market.assign(status=market["have"].map({True: "You have it", False: "Missing"})))
            .mark_bar()
            .encode(
                x=alt.X("demand_pct:Q", title="% of your top-50 matching jobs asking for this skill"),
                y=alt.Y("skill:N", sort="-x", title=None),
                color=alt.Color("status:N", title=None,
                                scale=alt.Scale(domain=["You have it", "Missing"], range=["#2E8B57", "#D9534F"])),
                tooltip=["skill", "demand_pct", "status"],
            )
            .properties(height=30 * len(market) + 40)
        )
        st.altair_chart(chart, width="stretch")
    st.caption(f"Target role: **{result['target_role']}**. Change it with the *Target role* selector above.")

# ---------- Tab 3: Rule-based mentor ----------
with tab_mentor:
    st.caption("A rule-based assistant that answers from the ML results above (not a chatbot / LLM).")
    chat_key = f"chat_{hash((resume_text, result['target_role']))}"
    history = st.session_state.setdefault(chat_key, [])

    quick = st.columns(3)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if quick[i % 3].button(q, key=f"quick_{i}", width="stretch"):
            history.append((q, answer(q, result, engine)))

    with st.form("ask", clear_on_submit=True):
        question = st.text_input("Ask your own question", placeholder="e.g. Am I a good fit for Java Developer?")
        if st.form_submit_button("Ask") and question.strip():
            history.append((question, answer(question, result, engine)))

    for q, a in reversed(history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.markdown(a)

# ---------- Tab 4: Raw text ----------
with tab_text:
    st.caption("Text extracted from your resume. If this looks empty or garbled, the results above will be less reliable.")
    st.text_area("Extracted text", resume_text, height=400, disabled=True, label_visibility="collapsed")