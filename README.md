# Job Market Analyzer

A data pipeline that scores global job markets based on how well they match your resume. It fetches thousands of live job postings, extracts structured information using NLP, and ranks each country/market across skill fit, seniority, visa sponsorship, and job volume — so you know exactly where to focus your job search.

## Why This Exists

Job searching across multiple countries (US, EU, India, Remote) is chaotic. Every market has different demands, different visa policies, and different seniority expectations. This tool answers: **"Which market is actually the best fit for me right now?"** — with data, not guesswork.

---

## Pipeline Overview

The pipeline runs in 4 sequential steps:

```
resume PDF
    │
    ▼
parse_resume.py     → profile.json        (your skills, roles, visa status)
    │
    ▼
fetch_jobs.py       → jobs.csv            (raw job postings across all markets)
    │
    ▼
extract_skills.py   → jobs_enriched.csv   (each job annotated with skills, seniority, visa info)
    │
    ▼
score_markets.py    → scores.csv          (ranked market scores + top skills per region)
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Copy `.env.example` to `.env` and fill in your real API keys:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=...    # console.anthropic.com
RAPIDAPI_KEY=...         # jsearch.p.rapidapi.com
INDIANAPI_KEY=...        # jobs.indianapi.in
ADZUNA_APP_ID=...        # developer.adzuna.com
ADZUNA_APP_KEY=...       # developer.adzuna.com
```

---

## Running the Pipeline

```bash
# Step 1 — parse your resume
python parse_resume.py resume/your_resume.pdf

# Step 2 — fetch jobs from all data sources
python fetch_jobs.py

# Step 3 — extract skills, seniority, and visa info from each job
python extract_skills.py

# Step 4 — score and rank each market
python score_markets.py
```

---

## Data Sources

| Source | Markets | API Key Required | Notes |
|--------|---------|-----------------|-------|
| [Arbeitnow](https://www.arbeitnow.com/api) | EU (DE, NL) | No | Full job descriptions included |
| [Remotive](https://remotive.com/api/remote-jobs) | Remote worldwide | No | Remote-only roles |
| [JSearch (RapidAPI)](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) | US, India | Yes | Aggregates LinkedIn, Indeed, Glassdoor |
| [The Muse](https://www.themuse.com/developers/api/v2) | US | No | Tech-focused roles |
| [IndianAPI](https://indianapi.in) | India | Yes | India-specific job board |

---

## Step 1 — Resume Parsing with Claude (Anthropic LLM)

`parse_resume.py` uses **Claude Sonnet** (via the Anthropic API) to parse the raw resume PDF into structured JSON.

**Why Claude here?** Resume parsing is a one-time operation on a single document. The input is unstructured prose — degrees, job titles, skill lists, dates — all formatted differently across candidates. Claude handles this with a single prompt, returning a clean JSON object with fields like `skills`, `seniority_level`, `visa_status_us`, and `target_roles`. A rule-based parser would need dozens of regex patterns and still miss edge cases.

Output (`profile.json`):
```json
{
  "name": "...",
  "skills": ["Python", "SQL", "dbt", "Spark", ...],
  "seniority_level": "mid",
  "target_seniority": ["entry", "junior", "mid"],
  "visa_status_us": "requires sponsorship",
  "target_roles": ["Data Engineer", "Analytics Engineer", ...]
}
```

---

## Step 2 — Job Fetching

`fetch_jobs.py` queries each data source for all target roles and stores raw results in `cache/` as JSON. On subsequent runs, cached files are used instead of making new API calls.

Jobs are filtered by title relevance before saving to avoid noise from unrelated postings.

---

## Step 3 — Skill & Info Extraction with Local NLP Models

`extract_skills.py` annotates each job posting with:

- **Required skills** — extracted from the job description
- **Seniority level** — entry / junior / mid / senior
- **Years of experience** required
- **Visa sponsorship** — yes / no / notmentioned

### Why NLP models instead of Claude?

The pipeline processes **thousands of job descriptions**. Using Claude for each one would cost ~$0.003–$0.01 per call — that's $10–$50+ per full pipeline run, plus significant latency. Local transformer models run free on CPU/GPU, process each job in milliseconds, and produce consistent structured output. Claude is ideal for one-shot unstructured parsing (the resume); NLP models are better for batch processing at scale.

### How each field is extracted:

**Skills — three-layer approach:**

1. **NER model** (`algiraldohe/lm-ner-linkedin-skills-recognition`, HuggingFace): a BERT-based encoder-only model fine-tuned on LinkedIn job descriptions. It runs in 500-token overlapping chunks (BERT's max is 512 tokens) and merges results across chunks.

2. **Semantic validation** (`is_tech_skill()`): every entity the NER model extracts is checked against a set of known tech seed terms (Python, SQL, AWS, Docker, etc.) using `all-MiniLM-L6-v2` cosine similarity. Only entities with similarity ≥ 0.45 pass through. This filters out false positives like "team player" or "communication" being tagged as skills.

3. **Keyword scan**: a compiled regex taxonomy of 150+ explicit tech skills (AWS services, databases, frameworks, ML tools, BI tools) catches anything the NER model misses. The keyword list covers ambiguous abbreviations like `SQL`, `ETL`, `BI`, `CI/CD` that NER sometimes skips. The two NER + keyword results are unioned together.

**Seniority & Years of Experience — SpaCy Matcher:**

SpaCy's rule-based `Matcher` looks for explicit patterns — "senior", "junior", "entry-level", "2+ years", "3-5 years". This is faster and more reliable than a model for straightforward keyword detection. No API tokens needed.

**Visa Sponsorship — Zero-shot Classification:**

`facebook/bart-large-mnli` (encoder-decoder) classifies visa-relevant sentences against two candidate labels: `"visa sponsored"` vs `"no visa sponsorship"`. Zero-shot means no training data required — and unlike regex, it correctly handles negation (e.g. "we do not offer visa sponsorship" → `no`).

---

## Step 4 — Market Scoring

`score_markets.py` aggregates enriched job data by country and produces a `final_score` for each market, plus a breakdown of the top in-demand skills per region.

### Signals (averaged equally into `final_score`):

| Signal | How It's Calculated |
|--------|---------------------|
| `skill_match` | For each job, compute % of required skills covered by your profile using **cosine similarity ≥ 0.6** (`all-MiniLM-L6-v2`). Average across all jobs in the market. Cosine similarity handles synonyms — "ML" matches "machine learning", "Postgres" matches "PostgreSQL". |
| `seniority_match` | Fraction of jobs matching your target seniority (entry/junior/mid). Senior roles score 0.3 (partial credit — still signals demand). Unclear seniority scores 0.5. |
| `visa_score` | `(yes_count + 0.5 × unclear_count) / total_jobs`. Unclear gets half-credit since no mention often means it depends. Your home country always scores 1.0. Remote jobs are excluded (visa doesn't apply). |
| `volume` | Your market's share of total non-remote job postings. A market with 40% of all jobs scores 0.4. |

Remote jobs are merged into every country's skill/seniority scoring (since remote roles are accessible from any country) but excluded from volume.

### Top Skills Per Region

After scoring, the pipeline prints the **top 10 most in-demand skills per country**, derived from frequency counts across all extracted job skills.

Generic buzzwords that appear everywhere but carry no signal are filtered out before this ranking. Terms like `"software"`, `"environment"`, `"learning"`, `"analytics"`, `"computer science"`, and `"automation"` appear in nearly every job description — including non-technical ones — so they inflate skill counts without telling you anything useful. These are excluded via a noise filter, leaving only meaningful, actionable skills like `Python`, `Spark`, `dbt`, `Tableau`.

Example output:
```
=== Top 10 Skills per Country (by frequency) ===
  us:     ['Python', 'SQL', 'AWS', 'Spark', 'dbt', 'Airflow', 'Snowflake', ...]
  eu:     ['Python', 'SQL', 'Azure', 'Kafka', 'Docker', ...]
  in:     ['Python', 'SQL', 'Java', 'AWS', 'Spark', ...]
```

This answers: *"What skills are EU employers specifically looking for?"* — useful for knowing what to upskill in before targeting a particular market.

---

## Caching

All API responses and per-job NLP extractions are cached to `cache/` as JSON. This means:

- `fetch_jobs.py` won't re-hit APIs on re-runs
- `extract_skills.py` skips already-processed job indices
- Delete specific cache files (e.g. `cache/jsearch_us_data_engineer.json`) to force a refresh for that source/role

---

## Output

`scores.csv` columns:

| Column | Description |
|--------|-------------|
| `country` | Market code (`us`, `eu`, `in`, `remote`) |
| `job_count` | Number of job postings fetched |
| `skill_match` | 0–1, how well your skills cover job requirements |
| `seniority_match` | 0–1, fraction of jobs at your target level |
| `visa_score` | 0–1, fraction of jobs mentioning sponsorship |
| `volume` | 0–1, normalized job count share |
| `final_score` | Average of all 4 signals |

---

## Models Used

| Model | Architecture | Task |
|-------|-------------|------|
| `claude-sonnet-4-6` (Anthropic) | LLM | Resume parsing — one-shot unstructured text → JSON |
| `algiraldohe/lm-ner-linkedin-skills-recognition` | BERT (encoder-only) | Named entity recognition on job descriptions |
| `facebook/bart-large-mnli` | BART (encoder-decoder) | Zero-shot visa sponsorship classification |
| `spacy en_core_web_sm` | Rule-based Matcher | Seniority and years-of-experience extraction |
| `all-MiniLM-L6-v2` | Sentence Transformer | Skill semantic similarity (profile match + tech validation) |
