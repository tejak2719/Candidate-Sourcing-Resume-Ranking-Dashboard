# sourcing_to_ranking.py — FRF + Skills→URLs + Scrape + Rank + LLM Summaries + Chat
import os
import json
import re
import urllib.parse
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import date

from apify_client import ApifyClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NEW: OpenAI client for LLM features ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # handled below

# ================== CONFIG ==================
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "apify_api_MualSmXxsQykCnmHev5dr5ok8QGO6041MoSR")
GOOGLE_SEARCH_ACTOR_ID = "apify/google-search-scraper"
LINKEDIN_ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "dev_fusion/Linkedin-Profile-Scraper")

# LLM settings (you can change model if you like)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")   # small/cheap & good enough for summaries/parsing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")     # set this in your shell to enable LLM features
# ============================================

client = ApifyClient(APIFY_API_TOKEN)

st.set_page_config(page_title="Sourcing → Scraping → Ranking", layout="wide")
st.title("🔗 SkillGrub")

# ---------- Session state ----------
ss = st.session_state
ss.setdefault("urls", [])
ss.setdefault("profiles", [])
ss.setdefault("profiles_path", "profiles.json")
ss.setdefault("frf", {})
ss.setdefault("frf_min_years", 2)
ss.setdefault("rank_df", None)          # will hold ranked DataFrame for the chatbot tab
ss.setdefault("rank_context", {})       # will hold FRF/JD/must-have context for LLM

def exp_choice_to_min_years(choice: str) -> int:
    if not choice:
        return 0
    if choice.strip().startswith(">"):
        return 10
    try:
        return int(choice.split("-")[0])
    except Exception:
        return 0

# --- NEW: LLM client helper ---
def get_llm_client():
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None

tabs = st.tabs(["1️⃣ Skills → LinkedIn URLs", "2️⃣ Scrape Profiles", "3️⃣ Rank Profiles", "4️⃣ Chat with shortlist"])

# -------------------------------------------------------------------------
# TAB 1: Skills → URLs (Google actor) + FRF inputs
# -------------------------------------------------------------------------
with tabs[0]:
    st.header("1️⃣ Skills → Bulk LinkedIn Profile URLs")

    # Search settings
    c1, c2, c3 = st.columns(3)
    with c1:
        location = st.text_input("Location (optional)", value="")
    with c2:
        title_kw = st.text_input("Role/Title filter (optional)", value="")
    with c3:
        results_per_query = st.number_input("Max results / query", 50, 1000, 300, 50)

    c4, c5, c6 = st.columns(3)
    with c4:
        country_iso = st.selectbox("Search country (ISO-2)", ["", "us", "in", "gb", "au", "ca", "de", "fr", "es", "sg"], index=1)
    with c5:
        language_code = st.selectbox("Language", ["en", "hi", "fr", "de", "es"], index=0)
    with c6:
        chunk_size = st.slider("Skills per query (auto-split)", 3, 8, 5)

    skills_text = st.text_area(
        "Paste 10–15 skills (one per line)",
        "python\nkubernetes\nairflow\nspark\nsnowflake",
        height=150,
    )

    # FRF inputs
    st.subheader("📋 FRF Details (from Hiring Manager)")
    f1, f2, f3 = st.columns(3)
    with f1:
        band = st.text_input("Band", value="")
        urgency = st.selectbox("Urgency", ["Low", "Medium", "High"])
        billable = st.radio("Billable", ["Yes", "No"], horizontal=True)
        bill_rate = st.number_input("Bill Rate", min_value=0, step=10)
    with f2:
        experience_choice = st.selectbox(
            "Experience (years)",
            ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10", ">10"],
        )
        customer = st.text_input("For which customer", value="")
        vertical = st.text_input("Vertical", value="")
    with f3:
        start_date: date = st.date_input("Start Date")
        end_date: date = st.date_input("End Date")
        customer_location = st.text_input("Customer Location", value="")
        bench_open = st.radio("Bench (Open Requirement)", ["Yes", "No"], horizontal=True)

    def persist_frf():
        frf = {
            "band": band,
            "experience": experience_choice,
            "urgency": urgency,
            "start_date": str(start_date) if isinstance(start_date, date) else "",
            "end_date": str(end_date) if isinstance(end_date, date) else "",
            "billable": billable,
            "bill_rate": bill_rate,
            "customer": customer,
            "vertical": vertical,
            "customer_location": customer_location,
            "bench_open": bench_open,
            "skills": [s.strip() for s in skills_text.splitlines() if s.strip()],
            "role_title_filter": title_kw,
            "location_filter": location,
            "country_code": country_iso,
            "language_code": language_code,
        }
        ss["frf"] = frf
        ss["frf_min_years"] = exp_choice_to_min_years(experience_choice)
        try:
            Path("frf.json").write_text(json.dumps(frf, indent=2), encoding="utf-8")
        except Exception as e:
            st.warning(f"Could not save frf.json: {e}")

    # Build Google queries
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def build_boolean_query(skill_list: List[str], title: str = "", loc: str = "") -> str:
        if not skill_list:
            return ""
        base = (
            "(site:linkedin.com/in OR site:linkedin.com/pub) "
            "-site:linkedin.com/company -site:linkedin.com/school "
            "-site:linkedin.com/jobs -site:linkedin.com/learning"
        )
        if len(skill_list) == 1:
            clause = skill_list[0]
        else:
            clause = " AND ".join(f'"{s}"' for s in skill_list)
        q = f"{base} {clause}"
        if title.strip():
            q += f' "{title.strip()}"'
        if loc.strip():
            q += f' "{loc.strip()}"'
        return " ".join(q.split())

    skills = [s for s in skills_text.splitlines() if s.strip()]
    queries = [build_boolean_query(g, title_kw, location) for g in chunks(skills, chunk_size)]
    queries_str = "\n".join(queries)
    st.markdown("### 🔍 Generated queries")
    st.code(queries_str or "(empty)")

    if st.button("🚀 Run Skills → URLs"):
        if not queries_str:
            st.error("No queries built.")
        else:
            persist_frf()
            run_input = {
                "queries": queries_str,
                "maxResults": int(results_per_query),
                "languageCode": language_code,
                "includeUnfilteredResults": True,
            }
            if country_iso:
                run_input["countryCode"] = country_iso

            try:
                run = client.actor(GOOGLE_SEARCH_ACTOR_ID).call(run_input=run_input)
                items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            except Exception as e:
                st.error(f"Google actor run failed: {e}")
                items = []

            urls = []

            def maybe_add(u: Optional[str]):
                if not u:
                    return
                if "linkedin.com/in/" in u or "linkedin.com/pub/" in u:
                    p = urllib.parse.urlparse(u)
                    clean = urllib.parse.urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
                    urls.append(clean)

            for it in items:
                maybe_add(it.get("url"))
                for key in ("results", "organicResults", "searchResults", "items"):
                    if isinstance(it.get(key), list):
                        for r in it[key]:
                            maybe_add(r.get("url"))

            df_urls = pd.DataFrame({"linkedinUrl": urls}).drop_duplicates().reset_index(drop=True)
            ss["urls"] = df_urls["linkedinUrl"].tolist()

            st.success(f"Found {len(df_urls)} LinkedIn URLs (saved to session).")
            with st.expander("Preview URLs", expanded=True):
                st.dataframe(df_urls, use_container_width=True)

            st.download_button(
                "⬇️ Download URLs (TXT)",
                "\n".join(ss["urls"]).encode("utf-8"),
                file_name="linkedin_profile_urls.txt",
            )
            df_urls.to_csv("linkedin_profile_urls.csv", index=False)
            st.info("Saved as linkedin_profile_urls.csv · FRF saved as frf.json")

# -------------------------------------------------------------------------
# TAB 2: Scrape Profiles — ensure all URLs are kept (placeholders for failures)
# -------------------------------------------------------------------------
with tabs[1]:
    st.header("2️⃣ Scrape LinkedIn Profiles")

    if ss.get("frf"):
        with st.expander("FRF Summary (from Tab 1)", expanded=False):
            st.json(ss["frf"])

    default_urls_text = "\n".join(ss.get("urls", []))
    with st.expander("URLs detected from Tab 1", expanded=bool(default_urls_text)):
        if default_urls_text:
            st.code(default_urls_text, language="text")
        else:
            st.caption("No URLs yet. Generate in Tab 1 or upload/paste below.")

    uploaded = st.file_uploader("Upload linkedin_profile_urls.txt", type=["txt"])
    pasted = st.text_area("Or paste LinkedIn URLs (one per line)", default_urls_text, height=180)

    c7, c8 = st.columns(2)
    with c7:
        results_limit = st.number_input("Results Limit", 1, 20, 5)
    with c8:
        search_limit = st.number_input("Search Limit", 1, 50, 10)

    urls = []
    if uploaded:
        urls += [u.strip() for u in uploaded.read().decode("utf-8").splitlines() if u.strip()]
    if pasted.strip():
        urls += [u.strip() for u in pasted.splitlines() if u.strip()]
    urls = sorted(set([u for u in urls if "linkedin.com/in/" in u or "linkedin.com/pub/" in u]))

    st.write(f"Detected {len(urls)} URLs")

    if st.button("🚀 Scrape Profiles"):
        profiles: List[dict] = []
        if not urls:
            st.error("No URLs provided.")
        else:
            run_input = {"profileUrls": urls, "resultsLimit": int(results_limit), "searchLimit": int(search_limit)}
            try:
                run = client.actor(LINKEDIN_ACTOR_ID).call(run_input=run_input)
                profiles = list(client.dataset(run["defaultDatasetId"]).iterate_items())
            except Exception as e:
                st.error(f"LinkedIn actor run failed: {e}")
                profiles = []

            # --- Ensure every URL is represented (placeholder for failures) ---
            scraped_urls = {p.get("linkedinUrl") or p.get("profileUrl") for p in profiles}
            for u in urls:
                if u not in scraped_urls:
                    profiles.append(
                        {
                            "linkedinUrl": u,
                            "fullName": None,
                            "headline": None,
                            "skills": [],
                            "experiences": [],
                            "about": "",
                            "summary": "",
                        }
                    )

            # Persist to session + disk
            ss["profiles"] = profiles
            try:
                with open("profiles.json", "w", encoding="utf-8") as f:
                    json.dump(profiles, f, ensure_ascii=False, indent=2)
                ss["profiles_path"] = "profiles.json"
                st.success(f"Saved {len(profiles)} profiles to profiles.json (includes placeholders).")
            except Exception as e:
                st.error(f"Failed to save profiles.json: {e}")

            # Preview
            if profiles:
                st.markdown("### Preview (first profile)")
                st.json(profiles[0])

# -------------------------------------------------------------------------
# Utilities shared by Rank + Chat
# -------------------------------------------------------------------------
def years_from_exps(exps):
    yrs = 0.0
    if not isinstance(exps, list):
        return yrs
    for e in exps:
        txt = " ".join(str(e.get(k, "")) for k in ("title", "subtitle", "caption", "metadata"))
        for val in re.findall(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", txt, flags=re.I):
            try:
                yrs += float(val)
            except:
                pass
    return yrs

def flatten_profile(p: dict) -> str:
    parts = []
    for key in ("headline", "about", "summary"):
        if p.get(key):
            parts.append(str(p.get(key)))
    if p.get("experiences"):
        for e in p["experiences"]:
            parts.append(" ".join(str(e.get(k, "")) for k in ("title", "subtitle", "caption", "metadata")))
    if p.get("skills"):
        try:
            sks = [s.get("title") for s in p["skills"] if isinstance(s, dict)]
            parts.append(" ".join([s for s in sks if s]))
        except Exception:
            parts.append(str(p["skills"]))
    return " ".join(parts)

# -------------------------------------------------------------------------
# TAB 3: Rank Profiles — now with LLM summaries
# -------------------------------------------------------------------------
with tabs[2]:
    st.header("3️⃣ Rank Profiles")

    use_session_profiles = bool(ss.get("profiles"))
    st.toggle("Use profiles from session (Tab 2)", value=use_session_profiles, key="use_session")

    if ss["use_session"]:
        profiles = ss.get("profiles", [])
        if profiles:
            st.info(f"Using {len(profiles)} profiles from session.")
        else:
            st.warning("No profiles in session. Falling back to file input.")
    else:
        profiles = []

    if not profiles:
        json_path = st.text_input("Path to profiles.json", value=ss.get("profiles_path", "profiles.json"))
        if st.button("Load Profiles from File"):
            try:
                profiles = json.loads(Path(json_path).read_text(encoding="utf-8"))
                ss["profiles"] = profiles
                ss["profiles_path"] = json_path
                st.success(f"Loaded {len(profiles)} profiles from file.")
            except Exception as e:
                st.error(f"Failed to load {json_path}: {e}")
                profiles = []

    default_min_years = ss.get("frf_min_years", 2)

    r1, r2 = st.columns(2)
    with r1:
        frf_skills_text = st.text_area(
            "FRF Skills (one per line)",
            "\n".join(ss.get("frf", {}).get("skills", ["python", "kubernetes", "airflow"])),
            height=100,
        )
        must_have_text = st.text_area("Must-have Skills", "python", height=80)
        min_years = st.number_input("Minimum total experience (years)", 0, 20, default_min_years)
    with r2:
        jd_text = st.text_area(
            "Job Description",
            "We are hiring a Data Engineer with strong Python, Airflow, and Kubernetes experience.",
            height=180,
        )
        w_frf = st.slider("Weight: FRF similarity", 0.0, 1.0, 0.5, 0.05)
        w_jd = st.slider("Weight: JD similarity", 0.0, 1.0, 0.4, 0.05)
        boost_all = st.slider("Boost: all must-have present", 0.0, 0.5, 0.1, 0.01)
        boost_years = st.slider("Boost: meets min years", 0.0, 0.5, 0.05, 0.01)

    total_profiles = len(profiles)
    default_top_n = total_profiles if total_profiles > 0 else 10
    top_n = st.number_input("Top N candidates", 1, max(1, total_profiles if total_profiles > 0 else 1000), default_top_n)

    if st.button("🏁 Rank Now"):
        if not profiles:
            st.error("No profiles available. Complete Tab 2 or load from file.")
            st.stop()

        rows = []
        for p in profiles:
            rows.append(
                {
                    "name": p.get("fullName") or p.get("name") or "",
                    "headline": p.get("headline") or "",
                    "url": p.get("linkedinUrl") or p.get("profileUrl") or "",
                    "text": flatten_profile(p),
                    "years": years_from_exps(p.get("experiences")),
                }
            )
        df = pd.DataFrame(rows)

        frf_query = " ".join([s.strip() for s in frf_skills_text.splitlines() if s.strip()])
        jd_query = (jd_text or "").strip()
        corpus = [frf_query or "", jd_query or ""] + df["text"].tolist()

        vec = TfidfVectorizer(stop_words="english", max_features=8000)
        X = vec.fit_transform(corpus)

        frf_vec, jd_vec, prof_vecs = X[0:1], X[1:2], X[2:]
        frf_sims = cosine_similarity(frf_vec, prof_vecs).ravel()
        jd_sims = cosine_similarity(jd_vec, prof_vecs).ravel()

        boosts = []
        musts = [m for m in must_have_text.splitlines() if m.strip()]
        for _, row in df.iterrows():
            boost = 0.0
            if musts and all(m.lower() in row["text"].lower() for m in musts):
                boost += float(boost_all)
            if row["years"] >= float(min_years):
                boost += float(boost_years)
            boosts.append(boost)

        df["frf_similarity"] = frf_sims
        df["jd_similarity"] = jd_sims
        df["boost"] = boosts
        df["final_score"] = w_frf * df["frf_similarity"] + w_jd * df["jd_similarity"] + df["boost"]

        df_sorted = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
        ss["rank_df"] = df_sorted
        ss["rank_context"] = {
            "frf_skills": [s.strip() for s in frf_skills_text.splitlines() if s.strip()],
            "must_have": musts,
            "jd_text": jd_text,
            "min_years": float(min_years),
        }

        st.subheader("🏆 Ranked Candidates")
        cols = ["name", "headline", "years", "frf_similarity", "jd_similarity", "boost", "final_score", "url"]
        st.dataframe(df_sorted[cols].head(int(top_n)), use_container_width=True)

        if len(df_sorted):
            best = df_sorted.iloc[0]
            st.markdown("### 🥇 Best Candidate")
            st.write(f"**Name:** {best['name'] or 'N/A'}")
            st.write(f"**Headline:** {best['headline'] or 'N/A'}")
            st.write(f"**Years:** {best['years']:.1f}")
            st.write(f"**FRF Similarity:** {best['frf_similarity']:.3f}")
            st.write(f"**JD Similarity:** {best['jd_similarity']:.3f}")
            st.write(f"**Boosts:** {best['boost']:.3f}")
            st.write(f"**Final Score:** {best['final_score']:.3f}")
            if best["url"]:
                st.write(f"[LinkedIn Profile]({best['url']})")

        st.download_button(
            "⬇️ Download Ranked CSV",
            df_sorted.to_csv(index=False).encode("utf-8"),
            file_name="ranked_profiles.csv",
        )

        # --- NEW: LLM Summaries for Top N ---
        st.markdown("----")
        st.subheader("🧠 LLM Summaries & Justifications")

        llm_client = get_llm_client()
        if not llm_client:
            st.warning("Set OPENAI_API_KEY to enable LLM summaries.")
        else:
            def llm_summary_for_candidate(row: pd.Series, ctx: Dict[str, Any]) -> str:
                skills_snippet = ", ".join(ctx["frf_skills"][:12]) if ctx.get("frf_skills") else ""
                musts_snippet = ", ".join(ctx.get("must_have", []))
                prompt = f"""
You are a recruiting assistant. Summarize candidate fit in 4-6 bullet points.
Use the FRF skills, must-haves, and JD for context, and explain *why* the candidate ranks where they do.

Candidate:
- Name: {row.get('name') or 'N/A'}
- Headline: {row.get('headline') or 'N/A'}
- Years (heuristic): {row.get('years', 0)}
- Profile URL: {row.get('url') or ''}

Signals:
- FRF similarity: {row.get('frf_similarity', 0):.3f}
- JD similarity: {row.get('jd_similarity', 0):.3f}
- Boosts: {row.get('boost', 0):.3f}
- Final score: {row.get('final_score', 0):.3f}

FRF skills: {skills_snippet or 'N/A'}
Must-have skills: {musts_snippet or 'None'}
Minimum years required: {ctx.get('min_years', 0)}
JD (short): {ctx.get('jd_text', '')[:500]}

Rules:
- Be concise. 4–6 bullets.
- If must-haves are missing or years below threshold, call it out.
- Mention the top reasons for rank (e.g., “5+ yrs Python”, “mentions Airflow & K8s”, etc).
- If the profile text is sparse, say “Limited data available” and justify using what we have.
"""
                try:
                    resp = llm_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": "You create concise hiring summaries."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    return f"(LLM error: {e})"

            if st.button("🧠 Generate LLM Summaries (Top N)"):
                df_to_sum = ss["rank_df"].head(int(top_n)).copy()
                for i, (_, row) in enumerate(df_to_sum.iterrows(), 1):
                    with st.expander(f"{i}. {row['name'] or 'N/A'} — {row['headline'] or ''}", expanded=False):
                        st.write(llm_summary_for_candidate(row, ss["rank_context"]))

# -------------------------------------------------------------------------
# TAB 4: Chatbot — ask questions; LLM → filters → results
# -------------------------------------------------------------------------
with tabs[3]:
    st.header("4️⃣ Chat with your shortlist")

    llm_client = get_llm_client()
    if ss.get("rank_df") is None:
        st.info("Please run ranking in Tab 3 first.")
        st.stop()

    if not llm_client:
        st.warning("Set OPENAI_API_KEY to enable the chatbot.")
        st.stop()

    q = st.text_input("Ask about your candidates",
                      value="Show me top 5 candidates with >3 years experience and Kubernetes.")
    ask = st.button("💬 Ask")

    def llm_parse_query_to_filters(question: str) -> Dict[str, Any]:
        """
        Ask LLM to extract a simple filter spec:
        {
          "min_years": float|None,
          "skills_all": [list of strings that must appear],
          "skills_any": [list of strings that are nice to have],
          "top_k": int|None
        }
        """
        schema_hint = """
Return ONLY a JSON object with keys:
- min_years: number or null
- skills_all: array of strings (strict must-have skills)
- skills_any: array of strings (optional skills)
- top_k: integer or null
No extra text.
"""
        try:
            resp = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You convert hiring questions into JSON filters."},
                    {"role": "user", "content": question + "\n\n" + schema_hint},
                ],
                temperature=0.0,
            )
            txt = resp.choices[0].message.content.strip()
            # be defensive: content should be JSON; if not, try to fix minimal
            return json.loads(txt)
        except Exception:
            # fallback: crude heuristic
            out = {"min_years": None, "skills_all": [], "skills_any": [], "top_k": None}
            m = re.search(r">?\s*(\d+)\s*\+?\s*years?", question.lower())
            if m:
                out["min_years"] = float(m.group(1))
            # extract simple skills tokens (comma/space separated words)
            tokens = re.findall(r"[a-zA-Z+#\.]{2,}", question)
            common = {"years", "experience", "top", "candidates", "show", "with", "and", "or"}
            skills = [t.lower() for t in tokens if t.lower() not in common]
            out["skills_any"] = skills[:5]
            m2 = re.search(r"top\s+(\d+)", question.lower())
            if m2:
                out["top_k"] = int(m2.group(1))
            return out

    def apply_llm_filters_to_df(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
        filtered = df.copy()
        # min years
        if spec.get("min_years") is not None:
            filtered = filtered[filtered["years"] >= float(spec["min_years"])]
        # skills_all (strict): all tokens must appear in text
        for s in (spec.get("skills_all") or []):
            filtered = filtered[filtered["text"].str.lower().str.contains(str(s).lower(), na=False)]
        # skills_any (loose): at least one appears
        any_skills = spec.get("skills_any") or []
        if any_skills:
            mask = False
            for s in any_skills:
                mask = mask | filtered["text"].str.lower().str.contains(str(s).lower(), na=False)
            filtered = filtered[mask]
        # sort by final score desc
        filtered = filtered.sort_values(by="final_score", ascending=False)
        # top_k
        if spec.get("top_k"):
            filtered = filtered.head(int(spec["top_k"]))
        return filtered

    if ask:
        df_ranked = ss["rank_df"].copy()
        spec = llm_parse_query_to_filters(q)
        result = apply_llm_filters_to_df(df_ranked, spec)

        st.markdown("**Parsed filters (from LLM):**")
        st.json(spec)
        st.markdown("**Results:**")
        show_cols = ["name", "headline", "years", "final_score", "url"]
        st.dataframe(result[show_cols], use_container_width=True)

        # brief natural-language answer
        try:
            msg = f"Found {len(result)} candidates. Top names: " + ", ".join(result["name"].head(5).fillna("N/A"))
            st.info(msg)
        except Exception:
            pass

