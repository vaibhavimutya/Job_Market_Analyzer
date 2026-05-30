import hashlib
import json
import os
import re
import time
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
TIMEOUT = 10
MAX_CHARS = 5000

df = pd.read_csv("jobs.csv")

already_have = "full_description" in df.columns
cached_count = 0
fetched_count = 0
failed_count = 0

full_descriptions = {}

for idx, row in df.iterrows():
    url = row.get("url")

    if already_have and not pd.isna(row.get("full_description")) and str(row.get("full_description")).strip():
        full_descriptions[idx] = row["full_description"]
        continue

    if not url or pd.isna(url):
        full_descriptions[idx] = ""
        continue

    key = hashlib.md5(str(url).encode()).hexdigest()[:12]
    cache_path = f"cache/scrape_{key}.json"

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            full_descriptions[idx] = json.load(f)["text"]
        cached_count += 1
        continue

    try:
        r = requests.get(url, timeout=TIMEOUT, headers=HEADERS, allow_redirects=True)
        # strip tags, collapse whitespace, trim
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]
    except Exception as e:
        print(f"  SKIP idx={idx}: {e}")
        text = str(row.get("description") or "")
        failed_count += 1

    with open(cache_path, "w") as f:
        json.dump({"url": url, "text": text}, f)

    full_descriptions[idx] = text
    fetched_count += 1
    print(f"  scraped idx={idx} ({len(text)} chars)")
    time.sleep(1)

df["full_description"] = df.index.map(full_descriptions)
df.to_csv("jobs.csv", index=False)

print(f"\nDone. cached={cached_count}, fetched={fetched_count}, failed={failed_count}")
print(f"Updated jobs.csv with full_description column ({len(df)} rows)")