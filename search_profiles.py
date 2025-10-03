import os
import json
import streamlit as st
import pandas as pd
from apify_client import ApifyClient

# ================== CONFIG ==================
# Hardcoded as requested (normally use an env var)
APIFY_API_TOKEN = "" #Add token 

# Default people-search actor. You can change this in the sidebar at runtime.
# Tip: Pick a LinkedIn "people search" actor from the Apify Store and paste its ID.
DEFAULT_SEARCH_ACTOR_ID = "curious_coder/linkedin-people-search-scraper"
# ============================================

st.set_page_config(page_title="LinkedIn People Search (Skills → URLs)", layout="wide")
st.title("🔎 LinkedIn People Search — skills → profile URLs")

# Sidebar controls
st.sidebar.header("Search Settings")
actor_id = st.sidebar.text_input(
    "Apify Actor ID (People Search)",
    value=DEFAULT_SEARCH_ACTOR_ID,
    help="Paste the actor ID of a LinkedIn people-search actor from the Apify Store."
)
location = st.sidebar.text_input("Location (optional)", value="")
title_filter = st.sidebar.text_input("Role/Title keywords (optional)", value="")
results_limit = st.sidebar.number_input("Max people to fetch", min_value=10, max_value=5000, value=200, step=10)
per_page_limit = st.sidebar.number_input("Results per page (actor-dependent)", min_value=10, max_value=100, value=25, step=5)

st.subheader("🧩 Skills (from FRF)")
skills_input = st.text_area(
    "Paste 10–15 skills, one per line",
    value="python\nkubernetes\nairflow\nspark\nsnowflake",
    height=150
)

def build_query(skills: list[str], title_kw: str = "", loc: str = "") -> str:
    skills_part = " ".join(s.strip() for s in skills if s.strip())
    title_part = f' "{title_kw.strip()}"' if title_kw.strip() else ""
    loc_part = f' "{loc.strip()}"' if loc.strip() else ""
    return (skills_part + title_part + loc_part).strip()

skills = [s for s in skills_input.splitlines() if s.strip()]
query_str = build_query(skills, title_filter, location)

st.markdown("### 🔍 Generated Query")
st.code(query_str or "(empty)", language="text")

run_btn = st.button("🚀 Search LinkedIn Profiles")

@st.cache_data(show_spinner=False)
def run_people_search(actor_id: str, token: str, query: str, loc: str, title_kw: str,
                      max_results: int, page_size: int):
    """
    Runs a LinkedIn people-search actor on Apify and returns dataset items.
    NOTE: Different store actors use different input field names.
    We send a broad payload; unknown keys are ignored by most actors.
    """
    client = ApifyClient(token)

    run_input = {
        # common keys across many actors:
        "query": query,
        "keywords": query,
        "location": loc or None,
        "title": title_kw or None,
        # limits (varies by actor, include a few common ones):
        "maxResults": max_results,
        "resultsLimit": max_results,
        "perPage": page_size,
        "searchLimit": page_size,
        "maxItems": max_results,
        "limit": max_results,
    }

    run = client.actor(actor_id).call(run_input=run_input)
    dataset_client = client.dataset(run["defaultDatasetId"])
    return list(dataset_client.iterate_items())

def extract_profile_urls(items: list[dict]) -> pd.DataFrame:
    """
    Normalize to DataFrame with columns: fullName, headline, location, linkedinUrl, source.
    Tries multiple common field names used by different actors.
    """
    rows = []
    for it in items:
        url = it.get("linkedinUrl") or it.get("profileUrl") or it.get("url")
        name = it.get("fullName") or it.get("name") or it.get("title")
        headline = it.get("headline") or it.get("subTitle") or it.get("summary")
        loc = it.get("location") or it.get("addressWithCountry") or it.get("city")

        # Sometimes actors return publicIdentifier instead of a full URL
        if not url:
            public_id = it.get("publicIdentifier")
            if public_id:
                url = f"https://www.linkedin.com/in/{public_id}/"

        if url:
            rows.append({
                "fullName": name,
                "headline": headline,
                "location": loc,
                "linkedinUrl": url,
                "source": it.get("source") or it.get("actor") or "",
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["linkedinUrl"])
    return df

if run_btn:
    if not APIFY_API_TOKEN:
        st.error("APIFY_API_TOKEN is empty.")
        st.stop()

    with st.spinner("Running people search on Apify…"):
        try:
            items = run_people_search(
                actor_id=actor_id.strip(),
                token=APIFY_API_TOKEN.strip(),
                query=query_str,
                loc=location,
                title_kw=title_filter,
                max_results=int(results_limit),
                page_size=int(per_page_limit),
            )
        except Exception as e:
            st.error(f"Apify run failed: {e}")
            st.stop()

    if not items:
        st.warning("No results. Try fewer skills, remove location, or try a different actor ID.")
        st.stop()

    df_urls = extract_profile_urls(items)
    st.success(f"Fetched {len(df_urls)} unique LinkedIn profile URLs.")

    st.markdown("### 🔗 Profile URLs (ready for your app.py)")
    st.dataframe(df_urls, use_container_width=True)

    st.download_button(
        "⬇️ Download URLs (TXT)",
        data="\n".join(df_urls["linkedinUrl"].tolist()).encode("utf-8"),
        file_name="linkedin_profile_urls.txt",
        mime="text/plain",
    )

    st.download_button(
        "⬇️ Download Profiles (CSV)",
        data=df_urls.to_csv(index=False).encode("utf-8"),
        file_name="linkedin_profiles.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Download Raw Items (JSON)",
        data=json.dumps(items, indent=2).encode("utf-8"),
        file_name="raw_search_results.json",
        mime="application/json",
    )

    st.markdown("---")
    st.caption("Tip: paste linkedin_profile_urls.txt into your other Streamlit app (app.py).")
