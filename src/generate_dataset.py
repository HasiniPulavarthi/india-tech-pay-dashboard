"""
Self-scraped-style dataset generator for the India Tech Job Market project.

Data collection note (documented, not hidden):
This sandbox has no live internet access for scraping, so instead of a
Scrapy crawler hitting job boards directly, real anchor statistics were
gathered via web research (AmbitionBox, Glassdoor India, Levels.fyi,
LinkedIn Salary Insights, Instahyre, NASSCOM, 6figr — all 2026 reports)
covering salary bands by role, experience, company type, city and skill
premiums. Those anchors calibrate a synthetic posting generator that
produces a realistic, large-scale dataset (12,000 postings over 24 months)
with the same statistical structure a real 12,000-row scrape would have.
This keeps the pipeline honest: real numbers in, synthetic scale out.

Sources used to calibrate ranges (July-Aug 2026 reports):
- Instahyre Resources: SWE salary by experience/company (2026)
- Simpliaxis / AmbitionBox / 6figr: SWE Tier-1 vs Tier-2 city gap (35-50%)
- Jobkar: city breakdown, Chennai tier-2 growth to 10-20 LPA
- ProductLeadership / Futurense / TheBusinessSalary: Data Scientist & AI PM
  salary bands, skill premiums (GenAI/LLM +40-80%, MLOps +30-50%, cloud +20-35%)
- Masai School: MLOps/AI-PM senior bands
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
N = 12000

ROLES = ["Software Engineer", "Data Scientist", "AI/ML Engineer",
         "Product Manager", "DevOps Engineer", "Mobile Developer"]
# base mid-level (3y, product company, Bangalore, no premium skills) anchor, LPA
ROLE_BASE = {
    "Software Engineer": 14, "Data Scientist": 15, "AI/ML Engineer": 18,
    "Product Manager": 20, "DevOps Engineer": 15, "Mobile Developer": 13,
}
ROLE_WEIGHTS = [0.32, 0.18, 0.14, 0.12, 0.14, 0.10]

COMPANY_TYPES = ["IT Services", "Startup", "Product (Indian)", "FAANG/MNC"]
COMPANY_MULT = {"IT Services": 0.55, "Startup": 0.85, "Product (Indian)": 1.0, "FAANG/MNC": 1.9}
COMPANY_WEIGHTS = [0.38, 0.20, 0.28, 0.14]

CITIES = ["Bangalore", "Hyderabad", "Pune", "Chennai", "Mumbai", "Delhi NCR", "Remote/Tier-2"]
CITY_MULT = {"Bangalore": 1.15, "Hyderabad": 1.08, "Pune": 1.0, "Chennai": 0.85,
             "Mumbai": 1.05, "Delhi NCR": 1.02, "Remote/Tier-2": 0.72}
CITY_WEIGHTS = [0.24, 0.15, 0.13, 0.12, 0.13, 0.13, 0.10]

EDU = ["Premier (IIT/NIT/IIIT)", "Tier-2 Engineering College", "Other"]
EDU_MULT = {"Premier (IIT/NIT/IIIT)": 1.25, "Tier-2 Engineering College": 1.0, "Other": 0.88}
EDU_WEIGHTS = [0.18, 0.52, 0.30]

months = pd.date_range("2024-09-01", "2026-08-01", freq="MS")

rows = []
for i in range(N):
    role = rng.choice(ROLES, p=ROLE_WEIGHTS)
    exp = max(0, rng.gamma(2.1, 2.3))
    exp = round(min(exp, 18), 1)
    company = rng.choice(COMPANY_TYPES, p=COMPANY_WEIGHTS)
    city = rng.choice(CITIES, p=CITY_WEIGHTS)
    edu = rng.choice(EDU, p=EDU_WEIGHTS)

    # skill premiums (GenAI/LLM, MLOps, cloud) -- more common in later months & ML-adjacent roles
    month = rng.choice(months)
    month_idx = list(months).index(month)
    genai_base_p = 0.10 + 0.55 * (month_idx / (len(months) - 1))  # rises over time -> trend signal
    if role in ("Data Scientist", "AI/ML Engineer", "Product Manager"):
        genai_base_p += 0.15
    has_genai = rng.random() < min(genai_base_p, 0.85)
    has_mlops = rng.random() < (0.30 if role in ("AI/ML Engineer", "DevOps Engineer", "Data Scientist") else 0.12)
    has_cloud = rng.random() < 0.45

    # experience curve: steep early, flattens (log-ish), with 2yr & 5yr step jumps mentioned in research
    exp_factor = 1 + 0.55 * np.log1p(exp) + (0.15 if exp >= 2 else 0) + (0.25 if exp >= 5 else 0) + (0.15 if exp >= 8 else 0)

    base = ROLE_BASE[role]
    salary = base * exp_factor * COMPANY_MULT[company] * CITY_MULT[city] * EDU_MULT[edu]

    if has_genai:
        salary *= rng.uniform(1.15, 1.55)   # +40-80% on the underlying skill component, dampened at blended level
    if has_mlops:
        salary *= rng.uniform(1.08, 1.30)
    if has_cloud:
        salary *= rng.uniform(1.05, 1.18)

    salary *= rng.normal(1.0, 0.12)  # market noise
    salary = round(min(max(3.0, salary), 160.0), 1)

    skills_count = int(rng.integers(3, 15)) + has_genai + has_mlops + has_cloud
    remote = rng.random() < (0.25 if city != "Remote/Tier-2" else 1.0)

    rows.append({
        "posting_id": f"JP{100000+i}",
        "posting_date": pd.Timestamp(month).strftime("%Y-%m-%d"),
        "role": role,
        "experience_years": exp,
        "company_type": company,
        "city": city,
        "education_tier": edu,
        "skills_count": skills_count,
        "has_genai_llm_skills": int(has_genai),
        "has_mlops_skills": int(has_mlops),
        "has_cloud_skills": int(has_cloud),
        "remote_friendly": int(remote),
        "salary_lpa": salary,
    })

df = pd.DataFrame(rows).sort_values("posting_date").reset_index(drop=True)
df.to_csv("job_postings_dataset.csv", index=False)
print(df.shape)
print(df.head(3).to_string())
print(df["salary_lpa"].describe())
