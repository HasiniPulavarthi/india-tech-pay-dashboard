# India Tech Pay Dashboard

An end-to-end analytics pipeline — from a research-calibrated job postings dataset to a predictive salary model to an interactive dashboard — surfacing what actually drives tech compensation in India in 2026.

**[Live demo →](india_tech_pay_dashboard.html)** *(open the HTML file locally or host it via GitHub Pages)*

Status: Complete | Python 3.10+

---

## What it does

1. **Generates a dataset** of 12,000 job postings (role, experience, company type, city, education, skills, salary) calibrated against real 2026 India tech salary research — not random noise, but distributions anchored to reported ranges from AmbitionBox, Glassdoor India, LinkedIn Salary Insights, Instahyre, and others.
2. **Trains two predictive models**:
   - A **Gradient Boosted Trees regressor** for high-accuracy salary prediction
   - A **Ridge regression on log-salary** — fully interpretable, and light enough to run client-side for instant predictions in the browser
3. **Forecasts demand** — a linear trend model projects job-posting volume 6 months forward, per role.
4. **Wraps it all in an interactive dashboard** — filter by role, experience, city, and skills to get a live salary estimate, see what's driving the number, and explore market-wide trends.

## Results

| Model | MAE (LPA) | R² |
|---|---|---|
| Gradient Boosted Trees | 4.53 | 0.933 |
| Ridge (log-salary) | 5.25 | 0.887 |

**Key insight:** Company type (IT services vs. product vs. startup vs. FAANG/MNC) is the single biggest driver of pay — more than experience, more than role, more than city. GenAI/LLM skills carry a ~41% average salary premium across the dataset.

## Repo structure

```
generate_dataset.py             # builds the 12k-row research-calibrated dataset
train_models.py                 # trains GBM + Ridge models, exports dashboard_data.json
job_postings_dataset.csv        # the generated dataset
india_tech_pay_dashboard.html   # self-contained interactive dashboard (data embedded)
requirements.txt
README.md
```

## Running it locally

```bash
pip install -r requirements.txt
python generate_dataset.py     # -> job_postings_dataset.csv
python train_models.py         # -> dashboard_data.json (embedded into the dashboard)
```

Then open `india_tech_pay_dashboard.html` directly in a browser — no server required.

## Tech stack

- **Data generation:** pandas, numpy
- **Modeling:** scikit-learn (`GradientBoostingRegressor`, `Ridge`)
- **Dashboard:** single-file HTML/CSS/JS, Chart.js for visualization — no build step, no backend

## Methodology notes & honest caveats

- **No live scraping in this build.** Real-time scraping requires network access I didn't have while building this in a sandboxed environment. Instead, real salary statistics were gathered via web research (2026 reports from AmbitionBox, Glassdoor India, LinkedIn Salary Insights, Instahyre, ProductLeadership.com, and others) and used to calibrate a synthetic posting generator — so the *distributions* reflect real market structure even though individual rows are generated, not scraped. Swapping in a real Scrapy/BeautifulSoup crawler against live job boards is the natural next step and wouldn't require changing the modeling or dashboard code.
- **XGBoost → GradientBoostingRegressor.** XGBoost couldn't be installed in the offline build environment, so scikit-learn's `GradientBoostingRegressor` — the same family of algorithm — was used instead.
- **Prophet → linear trend fit.** Demand forecasting uses a simple per-role linear trend rather than Prophet, for the same offline-install reason. Swappable if you want seasonality modeling.

## Possible extensions

- Swap the synthetic generator for a real scraper (Naukri, LinkedIn Jobs, Indeed) — schema is already compatible
- Add XGBoost/Prophet back in once running with internet access
- Add a "compare two profiles" view to the predictor
- Deploy the dashboard via GitHub Pages for a shareable link
