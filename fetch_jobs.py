import json
import os
import re
import time
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(".env")

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
INDIANAPI_KEY = os.environ.get("INDIANAPI_KEY", "")

with open("profile.json") as f:
    profile = json.load(f)

target_roles = profile["target_roles"]


def strip_html(html):
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()


def role_slug(role):
    return role.lower().replace(" ", "_").replace("/", "_")


TITLE_KEYWORDS = ["data", "engineer", "analyst", "scientist", "analytics", "intelligence", "bi ", "etl", "sql", "software"]

def is_relevant_title(title):
    if not title:
        return False
    t = title.lower()
    return any(k in t for k in TITLE_KEYWORDS)

rows = []


# --- Arbeitnow (EU: de, nl) ---
for role in target_roles:
    slug = role_slug(role)
    cache_file = f"cache/arbeitnow_{slug}.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            jobs = json.load(f)
        print(f"  cache: arbeitnow/{role} ({len(jobs)} jobs)")
    else:
        try:
            r = requests.get(
                "https://www.arbeitnow.com/api/job-board-api",
                params={"search": role, "page": 1},
                timeout=10
            )
            r.raise_for_status()
            jobs = r.json().get("data", [])
            with open(cache_file, "w") as f:
                json.dump(jobs, f)
            print(f"  fetched: arbeitnow/{role} ({len(jobs)} jobs)")
        except Exception as e:
            print(f"  SKIP arbeitnow/{role}: {e}")
            jobs = []
        time.sleep(1)

    for job in jobs:
        full_desc = strip_html(job.get("description", ""))
        if not is_relevant_title(job.get("title")):
            continue
        rows.append({
            "country": "eu",
            "title": job.get("title"),
            "company": job.get("company_name"),
            "location": job.get("location"),
            "description": full_desc[:300],
            "salary_min": None,
            "salary_max": None,
            "posted_date": job.get("created_at"),
            "url": job.get("url"),
            "source": "arbeitnow",
            "full_description": full_desc,
        })


# --- Remotive (Remote) ---
for role in target_roles:
    slug = role_slug(role)
    cache_file = f"cache/remotive_{slug}.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            jobs = json.load(f)
        print(f"  cache: remotive/{role} ({len(jobs)} jobs)")
    else:
        try:
            r = requests.get(
                "https://remotive.com/api/remote-jobs",
                params={"search": role, "limit": 50},
                timeout=10
            )
            r.raise_for_status()
            jobs = r.json().get("jobs", [])
            with open(cache_file, "w") as f:
                json.dump(jobs, f)
            print(f"  fetched: remotive/{role} ({len(jobs)} jobs)")
        except Exception as e:
            print(f"  SKIP remotive/{role}: {e}")
            jobs = []
        time.sleep(1)

    for job in jobs:
        full_desc = strip_html(job.get("description", ""))
        if not is_relevant_title(job.get("title")):
            continue
        rows.append({
            "country": "remote",
            "title": job.get("title"),
            "company": job.get("company_name"),
            "location": job.get("candidate_required_location"),
            "description": full_desc[:300],
            "salary_min": None,
            "salary_max": None,
            "posted_date": job.get("publication_date"),
            "url": job.get("url"),
            "source": "remotive",
            "full_description": full_desc,
        })


# --- JSearch via RapidAPI (US + India) ---
if not RAPIDAPI_KEY:
    print("  SKIP JSearch: RAPIDAPI_KEY not set in .env")
else:
    JSEARCH_COUNTRIES = {"us": "United States", "in": "India"}
    for country_code, country_name in JSEARCH_COUNTRIES.items():
        for role in target_roles:
            slug = role_slug(role)
            cache_file = f"cache/jsearch_{country_code}_{slug}.json"
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    jobs = json.load(f)
                print(f"  cache: jsearch/{country_code}/{role} ({len(jobs)} jobs)")
            else:
                try:
                    r = requests.get(
                        "https://jsearch.p.rapidapi.com/search-v2",
                        headers={
                            "x-rapidapi-key": RAPIDAPI_KEY,
                            "x-rapidapi-host": "jsearch.p.rapidapi.com",
                            "Content-Type": "application/json"
                        },
                        params={"query": f"{role} in {country_name}", "num_pages": "3", "country": country_code, "date_posted": "all"},
                        timeout=15
                    )
                    r.raise_for_status()
                    jobs = r.json().get("data", {}).get("jobs", [])
                    with open(cache_file, "w") as f:
                        json.dump(jobs, f)
                    print(f"  fetched: jsearch/{country_code}/{role} ({len(jobs)} jobs)")
                except Exception as e:
                    print(f"  SKIP jsearch/{country_code}/{role}: {e}")
                    jobs = []
                time.sleep(1)

            for job in jobs:
                full_desc = job.get("job_description", "")
                if not is_relevant_title(job.get("job_title")):
                    continue
                rows.append({
                    "country": country_code,
                    "title": job.get("job_title"),
                    "company": job.get("employer_name"),
                    "location": f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(", "),
                    "description": full_desc[:300],
                    "salary_min": job.get("job_min_salary"),
                    "salary_max": job.get("job_max_salary"),
                    "posted_date": job.get("job_posted_at_datetime_utc"),
                    "url": job.get("job_apply_link"),
                    "source": "jsearch",
                    "full_description": full_desc,
                })



# --- The Muse (US tech jobs) ---
MUSE_CATEGORIES = {
    "Data Engineer": "Data Science",
    "Software Engineer": "Software Engineer",
    "Analytics Engineer": "Data Science",
    "Data Analyst": "Data Science",
    "Business Intelligence Developer": "Data Science",
    "Data Scientist": "Data Science",
}

fetched_muse_categories = set()
for role in target_roles:
    category = MUSE_CATEGORIES.get(role, "Data Science")
    if category in fetched_muse_categories:
        continue
    fetched_muse_categories.add(category)

    slug = category.lower().replace(" ", "_")
    cache_file = f"cache/muse_{slug}.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            jobs = json.load(f)
        print(f"  cache: muse/{category} ({len(jobs)} jobs)")
    else:
        try:
            jobs = []
            for page in range(1, 4):
                r = requests.get(
                    "https://www.themuse.com/api/public/jobs",
                    params={"category": category, "page": page, "descending": "true"},
                    timeout=10
                )
                r.raise_for_status()
                results = r.json().get("results", [])
                if not results:
                    break
                jobs.extend(results)
                time.sleep(0.5)
            with open(cache_file, "w") as f:
                json.dump(jobs, f)
            print(f"  fetched: muse/{category} ({len(jobs)} jobs)")
        except Exception as e:
            print(f"  SKIP muse/{category}: {e}")
            jobs = []
        time.sleep(1)

    for job in jobs:
        full_desc = strip_html(job.get("contents", ""))
        if not is_relevant_title(job.get("name")):
            continue
        rows.append({
            "country": "us",
            "title": job.get("name"),
            "company": (job.get("company") or {}).get("name"),
            "location": ((job.get("locations") or [{}])[0]).get("name"),
            "description": full_desc[:300],
            "salary_min": None,
            "salary_max": None,
            "posted_date": job.get("publication_date"),
            "url": (job.get("refs") or {}).get("landing_page"),
            "source": "themuse",
            "full_description": full_desc,
        })


# --- IndianAPI (India jobs) ---
if not INDIANAPI_KEY:
    print("  SKIP IndianAPI: INDIANAPI_KEY not set in .env")
else:
    for role in target_roles:
        slug = role_slug(role)
        cache_file = f"cache/indianapi_{slug}.json"
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                jobs = json.load(f)
            print(f"  cache: indianapi/{role} ({len(jobs)} jobs)")
        else:
            try:
                r = requests.get(
                    "https://jobs.indianapi.in/jobs",
                    headers={"X-Api-Key": INDIANAPI_KEY},
                    params={"title": role, "limit": "50"},
                    timeout=15
                )
                r.raise_for_status()
                jobs = r.json() if isinstance(r.json(), list) else []
                with open(cache_file, "w") as f:
                    json.dump(jobs, f)
                print(f"  fetched: indianapi/{role} ({len(jobs)} jobs)")
            except Exception as e:
                print(f"  SKIP indianapi/{role}: {e}")
                jobs = []
            time.sleep(1)

        for job in jobs:
            full_desc = f"{job.get('job_description', '')} {job.get('role_and_responsibility', '')} {job.get('education_and_skills', '')}".strip()
            if not is_relevant_title(job.get("title")):
                continue
            rows.append({
                "country": "in",
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "description": full_desc[:300],
                "salary_min": None,
                "salary_max": None,
                "posted_date": job.get("posted_date"),
                "url": job.get("apply_link"),
                "source": "indianapi",
                "full_description": full_desc,
            })


df = pd.DataFrame(rows)
df.to_csv("jobs.csv", index=False)
print(f"\nSaved {len(df)} jobs to jobs.csv")
print("\nSummary:")
print(df.groupby(["source", "country"]).size().to_string())
