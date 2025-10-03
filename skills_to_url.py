import os
import json
import math
import textwrap
import urllib.parse
import streamlit as st
import pandas as pd
from apify_client import ApifyClient

# ================== CONFIG ==================
# hardcoded as requested (normally use an env var)
APIFY_API_TOKEN = "apify_api_MualSmXxsQykCnmHev5dr5ok8QGO6041MoSR"

# official google search actor
GOOGLE_SEARCH_ACTOR_ID = "apify/google-search-scraper"
# ============================================


st.set_page_config(page_title="Skills → LinkedIn Profile URLs (via Google)", layout="wide")
st.title("🔎 Skills → Bulk LinkedIn Profile URLs (via Google)")

# ---------- Sidebar controls ----------
st.sidebar.header("Search Settings")

location = st.sidebar.text_input("Location (optional)", value="", help='e.g., "Bangalore", "India"')
title_kw = st.sidebar.text_input("Role/Title filter (optional)", value="", help='e.g., "Data Engineer"')

results_per_query = st.sidebar.number_input(
    "Max results per query",
    min_value=50, max_value=1000, value=300, step=50,
    help="Total results the actor should return per query."
)

country_iso = st.sidebar.selectbox(
    "Search country (ISO-2 code)",
    options=["", "us", "in", "gb", "au", "ca", "de", "fr", "es", "sg"],
    index=1,  # set to "in" by default; change if you want
    help="Leave blank to let Google decide automatically."
)

language_code = st.sidebar.selectbox(
    "Language",
    options=["en", "hi", "fr", "de", "es"],
    index=0,
)

chunk_size = st.sidebar.slider(
    "Skills per query (auto-splitting)",
    min_value=3, max_value=8, value=5, step=1,
    help="We’ll split your skills into groups of this size and run multiple queries."
)

show_debug = st.sidebar.checkbox("Show debug info", value=False)

# ---------- Main inputs ----------
st.subheader("🧩 Skills (from FRF)")
skills_text = st.text_area(
    "Paste 10–15 skills, one per line",
    value="python\nkubernetes\nairflow\nspark\nsnowflake",
    height=180
)

def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def build_boolean_query(skills: list[str], title: str = "", loc: str = "") -> str:
    """
    Build a broad Google query that targets LinkedIn people profiles.
    - Allow both /in and /pub (older profiles).
    - For 1 skill: keep unquoted; for many: use AND on quoted skills.
    """
    skills = [s.strip() for s in skills if s.strip()]
    if not skills:
        return ""

    if len(skills) == 1:
        skills_clause = skills[0]
    else:
        skills_clause = " AND ".join(f'"{s}"' for s in skills)

    q = f'(site:linkedin.com/in OR site:linkedin.com/pub) {skills_clause}'
    if title.strip():
        q += f' "{title.strip()}"'
    if loc.strip():
        q += f' "{loc.strip()}"'
    return " ".join(q.split())

skills = [s for s in skills_text.splitlines() if s.strip()]
skill_groups = list(chunks(skills, chunk_size)) if skills else []

queries = [build_boolean_query(g, title_kw, location) for g in skill_groups] or []
# The Apify Google actor accepts a single STRING; multiple queries can be separated by newlines.
queries_str = "\n".join(q for q in queries if q)

st.markdown("### 🔍 Generated Google query/queries")
if not queries_str:
    st.code("(empty)", language="text")
else:
    st.code(queries_str, language="text")

run_btn = st.button("🚀 Search & Collect LinkedIn URLs")


@st.cache_data(show_spinner=False)
def run_google_search(token: str, queries_string: str, max_results: int, country_iso: str, language: str):
    """
    Run Apify Google Search actor. 'queries' must be a STRING.
    If you include multiple queries, separate them with newlines.
    """
    client = ApifyClient(token)

    run_input = {
        "queries": queries_string,                # <--- string, not list
        "maxResults": int(max_results),
        "languageCode": language,
        "includeUnfilteredResults": True,
    }
    if country_iso:
        run_input["countryCode"] = country_iso   # ISO-2

    run = client.actor(GOOGLE_SEARCH_ACTOR_ID).call(run_input=run_input)
    dataset_client = client.dataset(run["defaultDatasetId"])
    items = list(dataset_client.iterate_items())
    return items

def extract_linkedin_profile_urls(items: list[dict]) -> pd.DataFrame:
    """
    Collect linkedin.com/in and linkedin.com/pub links from flat and nested shapes.
    Deduplicate and clean query params.
    """
    urls = []

    def maybe_add(u: str):
        if not u:
            return
        if "linkedin.com/in/" in u or "linkedin.com/pub/" in u:
            parsed = urllib.parse.urlparse(u)
            clean = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
            urls.append(clean)

    for it in items:
        # flat
        maybe_add(it.get("url"))

        # nested shapes the actor sometimes returns
        for key in ("results", "organicResults", "searchResults", "items"):
            nested = it.get(key)
            if isinstance(nested, list):
                for r in nested:
                    maybe_add(r.get("url"))

    df = pd.DataFrame({"linkedinUrl": urls}).drop_duplicates(subset=["linkedinUrl"]).reset_index(drop=True)
    return df


if run_btn:
    if not skills:
        st.error("Please enter at least one skill.")
        st.stop()
    if not queries_str:
        st.error("Couldn’t build queries. Try reducing chunk size or adding skills.")
        st.stop()
    if not APIFY_API_TOKEN:
        st.error("APIFY_API_TOKEN is missing.")
        st.stop()

    with st.spinner("Searching Google for LinkedIn profiles…"):
        try:
            items = run_google_search(APIFY_API_TOKEN, queries_str, results_per_query, country_iso, language_code)
        except Exception as e:
            st.error(f"Apify run failed: {e}")
            st.stop()

    if show_debug:
        st.info(f"Raw items returned: {len(items)}")
        if items:
            st.caption("First raw item (trimmed):")
            st.json(items[0] if isinstance(items[0], dict) else {"item": str(items[0])})

    df_urls = extract_linkedin_profile_urls(items)
    st.success(f"Found {len(df_urls)} LinkedIn profile URLs.")
    st.markdown("### 🔗 URLs (ready for your app.py)")
    st.dataframe(df_urls, use_container_width=True)

    st.download_button(
        "⬇️ Download URLs (TXT)",
        data="\n".join(df_urls["linkedinUrl"]).encode("utf-8"),
        file_name="linkedin_profile_urls.txt",
        mime="text/plain",
    )

    st.download_button(
        "⬇️ Download URLs (CSV)",
        data=df_urls.to_csv(index=False).encode("utf-8"),
        file_name="linkedin_profile_urls.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Download raw Google items (JSON)",
        data=json.dumps(items, indent=2).encode("utf-8"),
        file_name="google_results_raw.json",
        mime="application/json",
    )

    st.markdown("---")
    st.caption("Next: paste linkedin_profile_urls.txt into your LinkedIn profile scraper app (app.py).")


##XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
##XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# import os
# import json
# import urllib.parse
# import streamlit as st
# import pandas as pd
# from apify_client import ApifyClient

# # ====== CONFIG ======
# APIFY_API_TOKEN = "apify_api_MualSmXxsQykCnmHev5dr5ok8QGO6041MoSR"
# GOOGLE_SEARCH_ACTOR_ID = "apify/google-search-scraper"
# # ====================

# st.set_page_config(page_title="Skills → LinkedIn Profile URLs (via Google)", layout="wide")
# st.title("🔎 Skills → Bulk LinkedIn Profile URLs (via Google)")

# # ---- Sidebar inputs ----
# st.sidebar.header("Search Settings")
# location = st.sidebar.text_input("Location (optional)", value="")
# title_kw = st.sidebar.text_input("Role/Title filter (optional)", value="")
# results_per_query = st.sidebar.number_input("Max results per query", min_value=10, max_value=1000, value=200, step=10)

# # IMPORTANT: countryCode must be ISO-3166 alpha-2 (us, in, gb, au, etc.)
# country_iso = st.sidebar.selectbox(
#     "Search country (ISO code)",
#     options=["", "us", "in", "gb", "au", "ca", "de", "fr", "es", "sg"],
#     index=1,  # "us"=0 if you prefer; 1 puts "in" if you want
#     help="Use ISO country codes. Leave blank to let Google decide automatically."
# )

# language_code = st.sidebar.selectbox(
#     "Language",
#     options=["en", "hi", "fr", "de", "es"],
#     index=0
# )

# st.subheader("🧩 Skills (from FRF)")
# skills_text = st.text_area(
#     "Paste 10–15 skills, one per line",
#     value="python\nkubernetes\nairflow\nspark\nsnowflake",
#     height=160
# )

# def build_boolean_query(skills: list[str], title: str = "", loc: str = "") -> str:
#     # Force people profiles on LinkedIn and AND the skills
#     parts = [f'"{s.strip()}"' for s in skills if s.strip()]
#     skill_clause = " AND ".join(parts) if parts else ""
#     q = f'site:linkedin.com/in ({skill_clause})'.strip()
#     if title.strip():
#         q += f' "{title.strip()}"'
#     if loc.strip():
#         q += f' "{loc.strip()}"'
#     return " ".join(q.split())

# skills = [s for s in skills_text.splitlines() if s.strip()]
# query = build_boolean_query(skills, title_kw, location)

# st.markdown("### 🔍 Generated Google query")
# st.code(query or "(empty)", language="text")

# run_btn = st.button("🚀 Search & Collect LinkedIn URLs")

# @st.cache_data(show_spinner=False)
# def run_google_search(apify_token: str, q: str, max_results: int, country_iso: str, language: str):
#     client = ApifyClient(apify_token)

#     run_input = {
#         # NOTE: 'queries' must be a STRING for this actor.
#         "queries": q,
#         "maxResults": int(max_results),
#         "languageCode": language,
#         # Only include countryCode if provided; otherwise omit it
#         **({"countryCode": country_iso} if country_iso else {}),
#         "includeUnfilteredResults": False,
#     }

#     run = client.actor(GOOGLE_SEARCH_ACTOR_ID).call(run_input=run_input)
#     dataset_client = client.dataset(run["defaultDatasetId"])
#     items = list(dataset_client.iterate_items())
#     return items

# def extract_linkedin_profile_urls(items: list[dict]) -> pd.DataFrame:
#     urls = []
#     for it in items:
#         url = it.get("url") or ""
#         if "linkedin.com/in/" in url:
#             parsed = urllib.parse.urlparse(url)
#             clean = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
#             if "/company/" in clean or "/pub/" in clean:
#                 continue
#             urls.append(clean)
#     return pd.DataFrame({"linkedinUrl": urls}).drop_duplicates(subset=["linkedinUrl"]).reset_index(drop=True)

# if run_btn:
#     if not query:
#         st.error("Please enter at least one skill.")
#         st.stop()
#     if not APIFY_API_TOKEN:
#         st.error("APIFY_API_TOKEN is missing.")
#         st.stop()

#     with st.spinner("Searching Google for LinkedIn profiles…"):
#         try:
#             items = run_google_search(APIFY_API_TOKEN, query, results_per_query, country_iso, language_code)
#         except Exception as e:
#             st.error(f"Apify run failed: {e}")
#             st.stop()

#     if not items:
#         st.warning("No results. Try fewer skills or widen your filters.")
#         st.stop()

#     df_urls = extract_linkedin_profile_urls(items)
#     st.success(f"Found {len(df_urls)} LinkedIn profile URLs.")
#     st.markdown("### 🔗 URLs (ready for your app.py)")
#     st.dataframe(df_urls, use_container_width=True)

#     st.download_button(
#         "⬇️ Download URLs (TXT)",
#         data="\n".join(df_urls["linkedinUrl"]).encode("utf-8"),
#         file_name="linkedin_profile_urls.txt",
#         mime="text/plain",
#     )
#     st.download_button(
#         "⬇️ Download URLs (CSV)",
#         data=df_urls.to_csv(index=False).encode("utf-8"),
#         file_name="linkedin_profile_urls.csv",
#         mime="text/csv",
#     )
#     st.download_button(
#         "⬇️ Download raw Google items (JSON)",
#         data=json.dumps(items, indent=2).encode("utf-8"),
#         file_name="google_results_raw.json",
#         mime="application/json",
#     )

#     st.markdown("---")
#     st.caption("Next: paste linkedin_profile_urls.txt into your profile-scraper app.py.")
