"""Rule-based career mentor: answers questions using the ML outputs (no LLM).

Each question is matched to an *intent* by keywords, and the answer is built
from the analysis dictionary produced by SmartHire.analyze().
"""

import re

EXAMPLE_QUESTIONS = [
    "What skills am I missing?",
    "What should I learn first?",
    "Which jobs fit me best?",
    "What role suits me?",
    "What are my strongest skills?",
    "Am I a good fit for Data Science?",
]


def _has(question: str, *words) -> bool:
    return any(w in question for w in words)


def answer(question: str, analysis: dict, engine=None) -> str:
    q = question.lower().strip()
    gap = analysis["gap"]
    market = gap["market"]

    # 1. "Am I a good fit for <role>?"  -> skill-gap against that specific role
    if engine is not None:
        for role in engine.roles:
            if re.search(rf"\b{re.escape(role.lower())}\b", q):   # whole words only
                report = engine.role_fit(analysis["resume_text"], role)
                missing = ", ".join(report["missing"][:5]) or "nothing major"
                verdict = ("a strong" if report["coverage"] >= 0.6 else
                           "a partial" if report["coverage"] >= 0.3 else "a weak")
                return (f"For **{role}**, your resume covers **{report['coverage']:.0%}** of the skills "
                        f"its best-matching jobs ask for. That's {verdict} fit. "
                        f"Top gaps to close: **{missing}**.")

    # 2. Salary: deliberately out of scope
    if _has(q, "salary", "pay", "ctc", "package"):
        return ("I can't estimate salaries: most postings in the dataset say *'Not Disclosed by Recruiter'*, "
                "so there wasn't enough labelled data to train a reliable salary model.")

    # 3. What to learn first -> highest-demand missing skills
    if _has(q, "first", "priority", "prioritize", "next", "start", "learn"):
        top = market[~market["have"]].head(3)
        if top.empty:
            return "You already have all the top in-demand skills for your target jobs. Focus on projects that prove them!"
        steps = "; ".join(f"**{r.skill}** (wanted by {r.demand_pct}% of your matching jobs)" for r in top.itertuples())
        return f"Learn these in order of market demand: {steps}."

    # 4. Missing skills / gaps
    if _has(q, "missing", "gap", "lack", "improve", "weak", "need"):
        if not gap["missing"]:
            return "No significant skill gaps found for your target jobs. 🎉"
        return (f"Compared with your top-50 matching **{analysis['target_role']}** jobs, you're missing: "
                f"**{', '.join(gap['missing'])}**. Your weighted skill coverage is **{gap['coverage']:.0%}**.")

    # 5. Strengths
    if _has(q, "strong", "strength", "best skill", "good at"):
        have = market[market["have"]]
        if have.empty:
            return ("I couldn't match any of your skills to in-demand skills. "
                    "Try adding a clear *Skills* section to your resume.")
        items = ", ".join(f"**{r.skill}** ({r.demand_pct}%)" for r in have.itertuples())
        return f"Your most valuable skills (and the % of matching jobs that want them): {items}."

    # 6. Jobs
    if _has(q, "job", "apply", "match", "company", "companies", "opening"):
        top = analysis["jobs"].head(3)
        lines = "; ".join(f"**{r.jobtitle}** at {r.company}" for r in top.itertuples())
        return f"Your top 3 matches are: {lines}. See the *Matched jobs* tab for the full list."

    # 7. Role / career direction
    if _has(q, "role", "category", "suit", "career", "who am i", "domain", "field", "family"):
        alts = ", ".join(analysis["role_alternatives"]) or "none with high confidence"
        return (f"Your resume reads most like **{analysis['predicted_role']}** "
                f"(other possible roles: {alts}). Your best matches fall in the "
                f"**{analysis['job_family']}** job family.")

    return ("I'm a rule-based mentor, so I understand questions like: "
            + " · ".join(f"*{e}*" for e in EXAMPLE_QUESTIONS))