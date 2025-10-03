#resume_ranker.py (JD + FRF + final score) first edit

import json
import re
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume / Profile Ranker (FRF + JD)", layout="wide")
st.title("🏅 Resume / Profile Ranker — FRF + JD blended scoring")

# ---------------- Sidebar: Inputs ----------------
st.sidebar.header("Inputs")

json_path = st.sidebar.text_input("Path to profiles.json", value="profiles.json")
load_btn = st.sidebar.button("Load Profiles")

st.sidebar.markdown("---")
st.sidebar.header("FRF (Requirements)")
frf_skills_text = st.sidebar.text_area(
    "Target Skills (one per line)",
    value="python\nkubernetes\nairflow",
    height=110,
    help="These feed the FRF similarity score."
)
must_have_text = st.sidebar.text_area(
    "Must-have Skills (strict; all must appear)",
    value="python",
    height=70
)
min_years = st.sidebar.number_input("Minimum total experience (years)", min_value=0, value=2)

st.sidebar.markdown("---")
st.sidebar.header("JD (Job Description)")
jd_text = st.sidebar.text_area(
    "Paste the full JD here",
    value="We are hiring a Data Engineer with strong Python, Airflow, and Kubernetes experience. "
          "Experience with Spark/Snowflake is a plus. 3+ years preferred.",
    height=160,
    help="This feeds the JD similarity score."
)

st.sidebar.markdown("---")
st.sidebar.header("Weights (Final Score)")
w_frf = st.sidebar.slider("Weight: FRF skills similarity", 0.0, 1.0, 0.50, 0.05)
w_jd  = st.sidebar.slider("Weight: JD similarity",           0.0, 1.0, 0.40, 0.05)
boost_all_must_have = st.sidebar.slider("Boost: all must-have skills present", 0.0, 0.5, 0.10, 0.01)
boost_min_years     = st.sidebar.slider("Boost: meets minimum years",          0.0, 0.5, 0.05, 0.01)

st.sidebar.markdown("---")
top_n = st.sidebar.number_input("Top N to show", min_value=1, value=20)
download_name = st.sidebar.text_input("Save ranked CSV as", value="ranked_profiles.csv")

# --------------- Helpers -----------------
def years_from_experiences(exps) -> float:
    """Crude extractor: sum numeric 'X years' mentions in experience text."""
    years = 0.0
    if not isinstance(exps, list):
        return years
    for e in exps:
        txt = " ".join(str(e.get(k, "")) for k in ("title", "subtitle", "caption", "metadata"))
        for val in re.findall(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", txt, flags=re.I):
            try:
                years += float(val)
            except:
                pass
    return years

def flatten_profile(p: dict) -> str:
    """Build a single text blob per profile for vectorization."""
    parts: List[str] = []
    for key in ("headline", "about", "summary"):
        if p.get(key):
            parts.append(str(p.get(key)))
    if p.get("experiences"):
        for e in p["experiences"]:
            parts.append(" ".join(str(e.get(k, "")) for k in ("title", "subtitle", "caption", "metadata")))
    # skills usually [{title: ...}]
    if p.get("skills"):
        try:
            sks = [s.get("title") for s in p["skills"] if isinstance(s, dict)]
            parts.append(" ".join([s for s in sks if s]))
        except Exception:
            parts.append(str(p["skills"]))
    return " ".join(parts)

def contains_all(haystack: str, needles: List[str]) -> bool:
    hs = haystack.lower()
    return all(n.lower() in hs for n in needles if n.strip())

# --------------- Main -----------------
if load_btn:
    # Load profiles.json
    try:
        raw = Path(json_path).read_text(encoding="utf-8")
        profiles = json.loads(raw)
    except Exception as e:
        st.error(f"Failed to load {json_path}: {e}")
        st.stop()

    if not profiles:
        st.warning("profiles.json is empty.")
        st.stop()

    # Normalize to a DataFrame
    rows = []
    for p in profiles:
        url = p.get("linkedinUrl") or p.get("profileUrl") or ""
        name = p.get("fullName") or p.get("name") or ""
        headline = p.get("headline") or ""
        text = flatten_profile(p)
        years = years_from_experiences(p.get("experiences"))
        rows.append({"name": name, "headline": headline, "url": url, "text": text, "years": years})

    df = pd.DataFrame(rows)
    st.success(f"Loaded {len(df)} profiles.")
    st.markdown("**Preview (first 5):**")
    st.dataframe(df[["name", "headline", "years", "url"]].head(5), use_container_width=True)

    # --------- Build queries for FRF and JD ----------
    frf_skills = [s.strip() for s in frf_skills_text.splitlines() if s.strip()]
    must_have  = [s.strip() for s in must_have_text.splitlines() if s.strip()]

    frf_query = " ".join(frf_skills).strip()
    jd_query  = (jd_text or "").strip()

    if not frf_query and not jd_query:
        st.error("Provide at least FRF skills and/or JD text.")
        st.stop()

    # --------- Vectorize and compute similarities ----------
    # We’ll vectorize FRF+JD along with all profile texts in ONE vocabulary for comparisons.
    # Corpus order: [FRF, JD, profile1, profile2, ...]
    corpus = []
    if frf_query:
        corpus.append(frf_query)
    else:
        corpus.append("")  # placeholder
    if jd_query:
        corpus.append(jd_query)
    else:
        corpus.append("")

    corpus += df["text"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
    X = vectorizer.fit_transform(corpus)

    frf_vec = X[0:1]    # shape (1, V)
    jd_vec  = X[1:2]    # shape (1, V)
    prof_vecs = X[2:]   # shape (N, V)

    frf_sims = cosine_similarity(frf_vec, prof_vecs).ravel()  # 0..1
    jd_sims  = cosine_similarity(jd_vec,  prof_vecs).ravel()  # 0..1

    # --------- Rule-based boosts ----------
    boosts = []
    for _, row in df.iterrows():
        text = row["text"]
        years = row["years"]
        boost = 0.0
        if must_have and contains_all(text, must_have):
            boost += float(boost_all_must_have)
        if years >= float(min_years):
            boost += float(boost_min_years)
        boosts.append(boost)

    df["frf_similarity"] = frf_sims
    df["jd_similarity"]  = jd_sims

    # Normalize and blend (weights)
    # Final score = w_frf * frf_similarity + w_jd * jd_similarity + boosts
    # You can tune weights from the sidebar.
    df["boost"] = boosts
    df["final_score"] = w_frf * df["frf_similarity"] + w_jd * df["jd_similarity"] + df["boost"]

    # --------- Sort and display ----------
    df_sorted = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)

    st.subheader("🏆 Ranked Candidates")
    show_cols = ["name","headline","years","frf_similarity","jd_similarity","boost","final_score","url"]
    st.dataframe(df_sorted[show_cols].head(int(top_n)), use_container_width=True)

    # --------- Best / Highest ranking ----------
    if len(df_sorted):
        best = df_sorted.iloc[0]
        st.markdown("### 🥇 Highest-Ranked Candidate")
        st.write(f"**Name:** {best['name'] or 'N/A'}")
        st.write(f"**Headline:** {best['headline'] or 'N/A'}")
        st.write(f"**Years:** {best['years']:.1f}")
        st.write(f"**FRF Similarity:** {best['frf_similarity']:.3f}")
        st.write(f"**JD Similarity:** {best['jd_similarity']:.3f}")
        st.write(f"**Boosts:** {best['boost']:.3f}")
        st.write(f"**Final Score:** {best['final_score']:.3f}")
        if best["url"]:
            st.write(f"**Profile:** {best['url']}")

    # --------- Save & download ----------
    try:
        df_sorted.to_csv(download_name, index=False)
        st.success(f"Saved ranking to `{download_name}`")
    except Exception as e:
        st.error(f"Failed to save `{download_name}`: {e}")

    st.download_button(
        "⬇️ Download Ranked CSV",
        data=df_sorted.to_csv(index=False).encode("utf-8"),
        file_name=download_name,
        mime="text/csv",
    )
