import json
import sys
import anthropic
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(".env")

pdf_path = sys.argv[1]

text = ""
for page in PdfReader(pdf_path).pages:
    text += page.extract_text() or ""

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system="You are a resume parser. Return ONLY valid JSON with no markdown, no explanation.",
    messages=[{
        "role": "user",
        "content": f"""Parse this resume and return a JSON object with exactly these fields:
- name (string)
- years_experience (number)
- current_location (string)
- target_roles (list of strings)
- skills (list of strings)
- seniority_level: one of entry, junior, mid, senior
- target_seniority: list — always include entry, junior, and mid
- education: object with degree (string), years (number), field_of_study (string), seniority_boost (true if masters or phd, else false)
- visa_status_us (string or null)

Resume:
{text}"""
    }]
)

raw = response.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
profile = json.loads(raw)

degree = (profile.get("education") or {}).get("degree", "") or ""
if degree.lower() in ("masters") and profile.get("seniority_level") == "junior":
    profile["seniority_level"] = "mid"

with open("profile.json", "w") as f:
    json.dump(profile, f, indent=2)

print(json.dumps(profile, indent=2))
