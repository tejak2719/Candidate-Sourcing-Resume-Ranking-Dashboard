import os
import json
import streamlit as st
import pandas as pd
from apify_client import ApifyClient

# --- CONFIG ---
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")  # keep yours or env var, add an apify token after APIFY_API_TOEN
ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "dev_fusion/Linkedin-Profile-Scraper")  # the profile-by-URL actor

client = ApifyClient(APIFY_API_TOKEN)

st.set_page_config(page_title="LinkedIn Profile Scraper", layout="wide")
st.title("🔎 LinkedIn Profile Scraper")

# ----------------- INPUTS -----------------
st.sidebar.header("Input")
st.sidebar.caption("Use either upload or paste URLs. One URL per line.")

# A) Upload the TXT exported from skills_to_url.py
uploaded = st.sidebar.file_uploader("Upload linkedin_profile_urls.txt", type=["txt"])

# B) Or paste manually
pasted = st.sidebar.text_area(
    "Or paste LinkedIn Profile URLs (one per line)",
    value="",
    height=160
)

# Limits
results_limit = st.sidebar.number_input("Results Limit (per profile)", min_value=1, value=5)
search_limit = st.sidebar.number_input("Search Limit (per profile)", min_value=1, value=10)

def read_urls():
    urls = []
    if uploaded is not None:
        try:
            data = uploaded.read().decode("utf-8", errors="ignore")
            urls.extend([u.strip() for u in data.splitlines() if u.strip()])
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
    if pasted.strip():
        urls.extend([u.strip() for u in pasted.splitlines() if u.strip()])
    # Dedup + keep only linkedin profiles
    urls = [u for u in urls if "linkedin.com/in/" in u or "linkedin.com/pub/" in u]
    dedup = sorted(set(urls))
    return dedup

urls = read_urls()
st.write(f"**Detected {len(urls)} profile URLs**")

if st.sidebar.button("Scrape Profiles"):
    if not urls:
        st.error("No valid LinkedIn profile URLs found. Upload or paste first.")
        st.stop()

    with st.spinner("Fetching profiles from LinkedIn via Apify…"):
        run_input = {
            "profileUrls": urls,
            "resultsLimit": int(results_limit),
            "searchLimit": int(search_limit),
        }

        try:
            run = client.actor(ACTOR_ID).call(run_input=run_input)
            dataset_client = client.dataset(run["defaultDatasetId"])
            profiles = list(dataset_client.iterate_items())
        except Exception as e:
            st.error(f"Apify actor run failed: {e}")
            st.stop()

    if not profiles:
        st.warning("No profiles returned. Try fewer URLs or verify the actor limits.")
        st.stop()

    # Save to disk for the ranker
    try:
        with open("profiles.json", "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
        st.success("Saved scraped profiles to `profiles.json` (for resume_ranker.py).")
    except Exception as e:
        st.error(f"Failed to save profiles.json: {e}")

    # UI: show structured summary
    tab1, tab2 = st.tabs(["📊 Structured View", "🗂 Raw JSON Data"])

    with tab1:
        for profile in profiles:
            st.subheader(f"👤 {profile.get('fullName', 'Unknown')}")
            st.write(profile.get("headline", ""))

            col1, col2 = st.columns([1,3])
            with col1:
                if profile.get("profilePic"):
                    st.image(profile["profilePic"], width=160)
            with col2:
                st.markdown(f"**Location:** {profile.get('addressWithCountry','N/A')}")
                st.markdown(f"**Connections:** {profile.get('connections','N/A')}")
                st.markdown(f"**Followers:** {profile.get('followers','N/A')}")
                st.markdown(f"**Current Role:** {profile.get('jobTitle','N/A')} at {profile.get('companyName','N/A')}")
                if profile.get("linkedinUrl"):
                    st.markdown(f"**Profile URL:** [LinkedIn]({profile.get('linkedinUrl')})")

            # Experience table
            if profile.get("experiences"):
                st.markdown("### 💼 Experience")
                exp_df = pd.DataFrame(profile["experiences"])
                show_cols = [c for c in ["title","subtitle","caption","metadata"] if c in exp_df.columns]
                st.dataframe(exp_df[show_cols], use_container_width=True)

            # Education table
            if profile.get("educations"):
                st.markdown("### 🎓 Education")
                edu_df = pd.DataFrame(profile["educations"])
                show_cols = [c for c in ["title","subtitle","caption"] if c in edu_df.columns]
                st.dataframe(edu_df[show_cols], use_container_width=True)

            # Skills list
            if profile.get("skills"):
                st.markdown("### 🛠 Skills")
                try:
                    skills = [s.get("title") for s in profile["skills"] if isinstance(s, dict)]
                    st.write(", ".join([s for s in skills if s]))
                except Exception:
                    st.write(profile["skills"])

            st.markdown("---")

    with tab2:
        for i, profile in enumerate(profiles, 1):
            st.markdown(f"### Raw Data for Profile {i}")
            st.json(profile)
