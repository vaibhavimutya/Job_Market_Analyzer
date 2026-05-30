import json
import os
import re
import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv(".env")

# requires: python -m spacy download en_core_web_sm
print("Loading models...")
ner = pipeline("ner", model="algiraldohe/lm-ner-linkedin-skills-recognition", aggregation_strategy="simple")
tokenizer = AutoTokenizer.from_pretrained("algiraldohe/lm-ner-linkedin-skills-recognition")
zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
nlp = spacy.load("en_core_web_sm")

_st_model = SentenceTransformer("all-MiniLM-L6-v2")
_TECH_SEEDS = ["Python", "SQL", "AWS", "machine learning", "data pipeline",
               "cloud computing", "API", "database", "ETL", "Apache Spark",
               "Kubernetes", "Docker", "Terraform", "Java", "Scala"]
_seed_embs = _st_model.encode(_TECH_SEEDS)

def is_tech_skill(word):
    emb = _st_model.encode([word])
    return float(cosine_similarity(emb, _seed_embs).max()) >= 0.45

matcher = Matcher(nlp.vocab)

matcher.add("SENIORITY_SENIOR", [[{"LOWER": {"IN": ["senior", "sr", "lead", "principal", "staff"]}}]])
matcher.add("SENIORITY_MID", [
    [{"LOWER": "mid"}, {"TEXT": "-"}, {"LOWER": "level"}],
    [{"LOWER": {"IN": ["intermediate", "medior"]}}],
])
matcher.add("SENIORITY_JUNIOR", [[{"LOWER": {"IN": ["junior", "jr", "associate"]}}]])
matcher.add("SENIORITY_ENTRY", [
    [{"LOWER": "entry"}, {"TEXT": "-"}, {"LOWER": "level"}],
    [{"LOWER": "new"}, {"TEXT": "-"}, {"LOWER": "grad"}],
    [{"LOWER": "new"}, {"LOWER": "grad"}],
    [{"LOWER": {"IN": ["entry", "graduate"]}}],
])
matcher.add("YEARS_EXP", [
    [{"LIKE_NUM": True}, {"TEXT": "+"}, {"LOWER": {"IN": ["year", "years"]}}],
    [{"LIKE_NUM": True}, {"TEXT": {"IN": ["-", "to"]}}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}],
    [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}],
])

SENIORITY_PRIORITY = ["SENIORITY_SENIOR", "SENIORITY_MID", "SENIORITY_JUNIOR", "SENIORITY_ENTRY"]
SENIORITY_MAP = {"SENIORITY_SENIOR": "senior", "SENIORITY_MID": "mid", "SENIORITY_JUNIOR": "junior", "SENIORITY_ENTRY": "entry"}
VISA_KEYWORDS = ["visa", "sponsor", "authorization", "work permit", "h1b", "h-1b", "ead", "green card"]

SKILL_TAXONOMY = [
    # --- Cloud: AWS ---
    "AWS", "S3", "EC2", "Lambda", "Redshift", "Glue", "EMR", "Athena", "RDS",
    "DynamoDB", "SageMaker", "CloudFormation", "ECS", "EKS", "CloudWatch", "Kinesis",
    "Step Functions", "API Gateway", "IAM", "VPC", "SNS", "SQS",
    # --- Cloud: Azure ---
    "Azure", "Azure Data Factory", "Azure Databricks", "Azure Synapse", "Azure Blob Storage",
    "Azure Functions", "Azure DevOps", "Azure SQL", "CosmosDB", "Azure ML",
    "Azure Event Hubs", "Azure Stream Analytics", "Azure Logic Apps", "ADLS",
    # --- Cloud: GCP ---
    "GCP", "BigQuery", "Dataflow", "Dataproc", "Cloud Composer", "Vertex AI",
    "Cloud Run", "Cloud Functions", "Pub/Sub", "Spanner", "Cloud Storage",
    # --- Programming Languages ---
    "Python", "SQL", "Java", "Scala", "R", "JavaScript", "TypeScript",
    "Bash", "Shell", "Go", "Rust", "C++", "C#", "Kotlin", "Swift",
    # --- Data Engineering ---
    "Apache Spark", "PySpark", "Kafka", "Apache Kafka", "Apache Airflow", "Airflow",
    "Apache Flink", "Flink", "Apache Beam", "dbt", "dbt Cloud", "Hadoop", "Hive",
    "Presto", "Trino", "Delta Lake", "Apache Iceberg", "Iceberg", "Apache Hudi", "Hudi",
    "Databricks", "Snowflake", "Fivetran", "Stitch", "Airbyte", "Informatica",
    "SSIS", "Talend", "DataStage", "NiFi", "Luigi",
    # --- Databases ---
    "PostgreSQL", "MySQL", "Oracle", "SQL Server", "MongoDB", "Cassandra", "Redis",
    "Elasticsearch", "Neo4j", "Teradata", "HBase", "CockroachDB", "Redshift",
    "Pinecone", "ChromaDB", "Weaviate", "ClickHouse",
    # --- BI & Visualization ---
    "Power BI", "Tableau", "Looker", "Grafana", "Metabase", "Apache Superset",
    "QlikView", "Qlik Sense", "Matplotlib", "Seaborn", "Plotly", "D3.js",
    "Looker Studio", "Excel", "Google Sheets",
    # --- Machine Learning & AI ---
    "TensorFlow", "PyTorch", "scikit-learn", "XGBoost", "LightGBM", "CatBoost",
    "Keras", "Hugging Face", "BERT", "GPT", "LangChain", "LlamaIndex",
    "MLflow", "Kubeflow", "Weights & Biases", "Optuna",
    "regression", "classification", "clustering", "NLP", "computer vision",
    "reinforcement learning", "feature engineering", "model deployment",
    "ARIMA", "Prophet", "time series", "forecasting", "A/B testing",
    "hypothesis testing", "statistical analysis", "Bayesian",
    # --- Data Science Libraries ---
    "pandas", "NumPy", "SciPy", "statsmodels", "NLTK", "spaCy",
    # --- DevOps & Infrastructure ---
    "Docker", "Kubernetes", "Terraform", "Ansible", "Helm", "ArgoCD",
    "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI",
    "Prometheus", "Grafana", "Datadog", "New Relic", "Splunk",
    # --- Data Formats & Protocols ---
    "Parquet", "Avro", "ORC", "JSON", "YAML", "Protobuf", "REST API", "GraphQL", "gRPC",
    # --- Data Quality & Governance ---
    "Great Expectations", "Monte Carlo", "data lineage", "data catalog",
    "data quality", "data governance", "Apache Atlas", "Collibra",
    "ETL", "ELT", "data modeling", "dimensional modeling", "star schema",
    "data warehousing", "data lakehouse", "data lake", "data mesh",
    # --- Software Engineering ---
    "Spring Boot", "FastAPI", "Flask", "Django", "React", "Node.js", "Express",
    "microservices", "JPA", "Hibernate", "Maven", "Gradle", "gRPC",
    "OOP", "design patterns", "SOLID", "TDD", "unit testing",
    # --- Version Control & Collaboration ---
    "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence",
    "Agile", "Scrum", "Kanban",
    # --- Security & Compliance ---
    "HIPAA", "GDPR", "SOC 2", "data encryption", "IAM", "OAuth", "JWT",
]

# precompile patterns once at load time (case-insensitive word-boundary match)
_taxonomy_patterns = [
    (skill, re.compile(r'(?<![a-zA-Z0-9])' + re.escape(skill) + r'(?![a-zA-Z0-9])', re.IGNORECASE))
    for skill in SKILL_TAXONOMY
]

def keyword_scan(text):
    found = set()
    for skill, pattern in _taxonomy_patterns:
        if pattern.search(text):
            found.add(skill)
    return found


def extract_skills_chunked(text, max_tokens=500, overlap=50):
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return []
    skills = set()
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        entities = ner(chunk_text)
        for e in entities:
            word = e["word"].strip()
            if e["score"] >= 0.75 and len(word) > 2 and not word.startswith("##") and is_tech_skill(word):
                skills.add(word)
    return list(skills)


def extract_seniority(doc):
    matches = matcher(doc)
    found = {nlp.vocab.strings[mid] for mid, _, _ in matches}
    for label in SENIORITY_PRIORITY:
        if label in found:
            return SENIORITY_MAP[label]
    return "notmentioned"


def extract_years_exp(doc):
    matches = matcher(doc)
    for match_id, start, end in matches:
        if nlp.vocab.strings[match_id] == "YEARS_EXP":
            for token in doc[start:end]:
                if token.like_num:
                    try:
                        return int(token.text)
                    except ValueError:
                        pass
    return None


def extract_visa(text):
    sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if any(k in s.lower() for k in VISA_KEYWORDS)]
    if not sentences:
        return "notmentioned"
    result = zsc(" ".join(sentences[:5]), candidate_labels=["visa sponsored", "no visa sponsorship"])
    return "yes" if result["labels"][0] == "visa sponsored" else "no"


df = pd.read_csv("jobs.csv")
desc_col = "full_description" if "full_description" in df.columns else "description"
print(f"Using column: {desc_col}")

cached_indices = set()
for fname in os.listdir("cache"):
    if fname.startswith("skills_v2_"):
        with open(f"cache/{fname}") as f:
            cached_indices.add(json.load(f)["_idx"])

to_process = df[~df.index.isin(cached_indices)].copy()
print(f"Total: {len(df)}, cached: {len(cached_indices)}, to process: {len(to_process)}")

all_results = {}
for fname in os.listdir("cache"):
    if fname.startswith("skills_v2_"):
        with open(f"cache/{fname}") as f:
            item = json.load(f)
            all_results[item["_idx"]] = item

for idx, row in to_process.iterrows():
    text = str(row.get(desc_col) or "")

    if not text.strip():
        item = {"_idx": idx, "required_skills": [], "seniority_level": "notmentioned",
                "visa_sponsorship_mentioned": "notmentioned", "years_experience_required": None}
    else:
        doc = nlp(text[:50000])
        skills = list(set(extract_skills_chunked(text)) | keyword_scan(text))
        seniority = extract_seniority(doc)
        years_exp = extract_years_exp(doc)
        visa = extract_visa(text)
        item = {"_idx": idx, "required_skills": skills, "seniority_level": seniority,
                "visa_sponsorship_mentioned": visa, "years_experience_required": years_exp}

    with open(f"cache/skills_v2_{idx}.json", "w") as f:
        json.dump(item, f)

    all_results[idx] = item

    if idx % 10 == 0:
        print(f"  idx={idx} | skills={len(item['required_skills'])} | seniority={item['seniority_level']} | visa={item['visa_sponsorship_mentioned']} | exp={item['years_experience_required']}")

df["required_skills"] = df.index.map(lambda i: ", ".join(all_results.get(i, {}).get("required_skills", [])))
df["seniority_level"] = df.index.map(lambda i: all_results.get(i, {}).get("seniority_level", ""))
df["visa_sponsorship_mentioned"] = df.index.map(lambda i: all_results.get(i, {}).get("visa_sponsorship_mentioned", ""))
df["years_experience_required"] = df.index.map(lambda i: all_results.get(i, {}).get("years_experience_required"))

df.to_csv("jobs_enriched.csv", index=False)
print(f"\nSaved {len(df)} rows to jobs_enriched.csv")
