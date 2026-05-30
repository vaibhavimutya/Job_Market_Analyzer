import json
import os
import requests
from dotenv import load_dotenv

load_dotenv(".env")

APP_ID = os.environ["ADZUNA_APP_ID"]
APP_KEY = os.environ["ADZUNA_APP_KEY"]

# --- configure these ---
WHAT_PHRASE = "data engineer"
WHERE       = "California"       # city or region, e.g. "san francisco", "london"
SALARY_MIN  = 80000
SALARY_MAX  = 200000
SORT_BY     = "date"           # "date" or "relevance"
# -----------------------

CACHE_FILE = "cache/probe_us_data_engineer.json"

def fetch():
    url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": 10,
        "what_phrase": WHAT_PHRASE,
        "where": WHERE,
        "salary_min": SALARY_MIN,
        "salary_max": SALARY_MAX,
        "full_time": 1,
        "sort_by": SORT_BY,
        "max_days_old": 7,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        data = json.load(f)
    print("(loaded from cache)")
else:
    data = fetch()
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print("(fetched from API)")

print(f"\nTotal results: {data['count']}")

job = data["results"][0]
print(f"\nTitle:    {job.get('title')}")
print(f"Company:  {job.get('company', {}).get('display_name')}")
print(f"Location: {job.get('location', {}).get('display_name')}")
print(f"Salary min: {job.get('salary_min')}")
print(f"Salary max: {job.get('salary_max')}")
print(f"\nDescription (first 200 chars):\n{job.get('description', '')[:200]}")
