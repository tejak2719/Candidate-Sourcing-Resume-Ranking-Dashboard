# End-to-End Sourcing → Scraping → Ranking + Chatbot

This project is a **Streamlit-powered Proof of Concept (POC)** that demonstrates an automated pipeline for **talent sourcing, LinkedIn profile scraping, candidate ranking, and interactive querying**.  
It integrates **Apify Actors (Google Search & LinkedIn Scraper)** for data sourcing and scraping, and uses **TF-IDF similarity scoring** to rank candidates against a provided FRF (Functional Requirement Form) and JD (Job Description).

---

## Project Workflow (4 Tabs)

### 1️⃣ Skills → LinkedIn URLs
- Input **10–15 skills** (from FRF or hiring manager).  
- Optional filters: location, role/title, country, language.  
- The app queries Google via Apify’s Google Search Actor to fetch **LinkedIn profile URLs** matching the skills.  
- Results are displayed in a table and can be downloaded as TXT or CSV.  

### 2️⃣ Scrape Profiles
- Upload or use URLs generated in Tab 1.  
- Uses Apify’s **LinkedIn Profile Scraper Actor** to fetch profile details such as:  
  - Name, headline, location  
  - Skills, experience, education  
  - LinkedIn profile URL  
- Profiles are stored locally in `profiles.json` for ranking.  

### 3️⃣ Rank Profiles
- Load profiles from Tab 2 or from `profiles.json`.  
- Input:  
  - **FRF Skills**  
  - **Must-have Skills**  
  - **Minimum Years of Experience**  
  - **Job Description (JD)**  
- Calculates:  
  - **FRF Similarity** (skills vs profile)  
  - **JD Similarity** (JD vs profile)  
  - **Boosts** (if must-have skills are present and min years condition met)  
  - **Final Score** (weighted combination of above)  
- Displays a ranked table of candidates and highlights the **Best Candidate**.  
- Ranking results can be downloaded as CSV.  

### 4️⃣ Chat with Shortlist
- Natural language chatbot interface.  
- Ask questions like:  
  - “Show me top 5 candidates with >3 years experience and Kubernetes.”  
- The chatbot parses your query into filters (skills, years, top N) and applies them to the ranked profiles.  
- Displays filtered candidate results in real time.  

---

## How to Run

1. **Clone the repository**  
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

   (Make sure you have Python 3.9+ installed)

3. **Set environment variables**  
   - `APIFY_API_TOKEN` → Your Apify token  
   - `APIFY_ACTOR_ID` → (Optional) Custom LinkedIn Actor ID  
   - `OPENAI_API_KEY` → For chatbot (optional if you don’t want Tab 4)  

   Example (Linux/Mac):  
   ```bash
   export APIFY_API_TOKEN="your_apify_token"
   export OPENAI_API_KEY="your_openai_key"
   ```

   Example (Windows PowerShell):  
   ```powershell
   setx APIFY_API_TOKEN "your_apify_token"
   setx OPENAI_API_KEY "your_openai_key"
   ```

4. **Run the app**  
   ```bash
   streamlit run sourcing_to_ranking.py
   ```

5. **Access in browser**  
   - Local URL: `http://localhost:8501`  

---

## Project Structure

```
├── sourcing_to_ranking.py    # Main Streamlit app (4 tabs)
├── profiles.json              # Saved LinkedIn profile data
├── frf.json                   # Saved FRF details
├── linkedin_profile_urls.csv  # Exported LinkedIn URLs
├── ranked_profiles.csv        # Exported ranking results
└── requirements.txt           # Python dependencies
```

---

## About the Authors

1. **Teja Konda**  
   - Github: [tejak2719](https://github.com/tejak2719)  
   - Linkedin: [Teja Konda](https://www.linkedin.com/in/teja-konda-1927s/)  

2. **Chandra Bezawada**  
   - Github: [chandra-bezawada-2729](https://github.com/chandra-bezawada-2729)  
   - Linkedin: [Chandra K Bezawada](https://www.linkedin.com/in/chandra-k-bezawada2729/)  
