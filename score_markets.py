import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def parse_skills(val):
    if pd.isna(val) or not str(val).strip():
        return []
    return [s.strip() for s in str(val).split(",") if s.strip()]


def country_skill_lists(df):
    result = {}
    for country, group in df.groupby("country"):
        skills = []
        for val in group["required_skills"]:
            skills.extend(parse_skills(val))
        result[country] = skills
    return result


def skill_match(group, profile_skills, model):
    profile_emb = model.encode(profile_skills)
    per_job_scores = []
    for _, row in group.iterrows():
        job_skills = parse_skills(row["required_skills"])
        if not job_skills:
            continue
        job_emb = model.encode(job_skills)
        sims = cosine_similarity(job_emb, profile_emb)
        covered = (sims.max(axis=1) >= 0.6).sum()
        per_job_scores.append(covered / len(job_skills))
    return float(np.mean(per_job_scores)) if per_job_scores else 0.0


def seniority_score(group, target_seniority):
    def sim(level):
        if pd.isna(level) or level in ("unclear", "notmentioned"):
            return 0.5
        if level in target_seniority:
            return 1.0
        if level == "senior":
            return 0.3
        if level == "lead":
            return 0.0
        return 0.5
    return group["seniority_level"].apply(sim).mean()


def visa_score(group, country, home_country):
    counts = group["visa_sponsorship_mentioned"].value_counts()
    yes = counts.get("yes", 0)
    no = counts.get("no", 0)
    unclear = counts.get("unclear", 0)
    total = yes + no + unclear
    if country == home_country:
        return yes, no, unclear, 1.0
    score = (yes * 1 + unclear * 0.5) / total if total else 0.0
    return yes, no, unclear, score


def volume_score(count, total):
    # proportion of total jobs — no scaling, reflects actual share
    return count / total if total else 0.0


_DISPLAY_NOISE = {"software", "environment", "learning", "analytics", "computer science", "automation"}

def top_skills(skill_list, n=10):
    from collections import Counter
    counts = Counter(s.lower() for s in skill_list if s.lower() not in _DISPLAY_NOISE)
    canonical = {s.lower(): s for s in reversed(skill_list)}
    seen, result = set(), []
    for lower, _ in counts.most_common():
        if lower not in seen:
            seen.add(lower)
            result.append(canonical[lower])
        if len(result) == n:
            break
    return result


def main():
    df = pd.read_csv("jobs_enriched.csv")
    with open("profile.json") as f:
        profile = json.load(f)

    profile_skills = profile["skills"]
    target_seniority = profile.get("target_seniority", ["entry", "junior", "mid"])
    citizenship = profile.get("citizenship", "")
    home_country = "in" if "indian" in citizenship.lower() else citizenship.lower()[:2]

    # merge remote jobs into every country's group for scoring
    remote = df[df["country"] == "remote"]
    non_remote = df[df["country"] != "remote"]
    total_jobs = len(non_remote)

    skill_lists = country_skill_lists(non_remote)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    rows = []
    verification = []

    for country, group in non_remote.groupby("country"):
        combined = pd.concat([group, remote], ignore_index=True)

        sm = skill_match(combined, profile_skills, model)
        ss = seniority_score(combined, target_seniority)
        yes, no, unclear, vs = visa_score(group, country, home_country)
        vol = volume_score(len(group), total_jobs)

        dist = group["seniority_level"].value_counts().to_dict()

        rows.append({
            "country": country,
            "job_count": len(group),
            "skill_match": round(sm, 3),
            "seniority_match": round(ss, 3),
            "visa_score": round(vs, 3),
            "volume": round(vol, 3),
            "final_score": round((sm + ss + vs + vol) / 4, 3)
        })

        verification.append({
            "country": country,
            "yes": yes,
            "no": no,
            "unclear": unclear,
            "visa_score": round(vs, 3),
            "seniority_distribution": dist,
            "seniority_score": round(ss, 3)
        })

    scores = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    scores.to_csv("scores.csv", index=False)

    def parse_date(val):
        try:
            v = float(val)
            return pd.Timestamp(v, unit="s", tz="UTC")
        except (ValueError, TypeError):
            return pd.to_datetime(val, utc=True, errors="coerce")

    df["posted_date"] = df["posted_date"].apply(parse_date)
    valid_dates = df["posted_date"].dropna()
    if not valid_dates.empty:
        date_min = valid_dates.min().strftime("%b %d, %Y")
        date_max = valid_dates.max().strftime("%b %d, %Y")
        print(f"\nScores based on {len(df)} job postings collected between {date_min} and {date_max}.")
    else:
        print(f"\nScores based on {len(df)} job postings.")

    print("\n=== Ranked Markets ===")
    print(scores.to_string(index=False))

    print("\n=== Top 10 Skills per Country (by frequency) ===")
    for country, skills in skill_lists.items():
        print(f"  {country}: {top_skills(skills)}")

    print("\n=== Visa & Seniority Verification ===")
    vdf = pd.DataFrame(verification)
    print(vdf.to_string(index=False))


if __name__ == "__main__":
    main()
