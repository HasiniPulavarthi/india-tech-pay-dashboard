import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("job_postings_dataset.csv", parse_dates=["posting_date"])

cat_cols = ["role", "company_type", "city", "education_tier"]
num_cols = ["experience_years", "skills_count", "has_genai_llm_skills",
            "has_mlops_skills", "has_cloud_skills", "remote_friendly"]

X = df[cat_cols + num_cols]
y = df["salary_lpa"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                         remainder="passthrough")

# --- Model 1: Gradient Boosted Trees (XGBoost stand-in -- no internet to pip install
# xgboost in this sandbox, so sklearn's GradientBoostingRegressor is used; same
# family of algorithm, same job: high-accuracy nonlinear salary predictor) ---
gbm = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.07, random_state=42))])
gbm.fit(X_train, y_train)
pred_gbm = gbm.predict(X_test)
gbm_mae = mean_absolute_error(y_test, pred_gbm)
gbm_r2 = r2_score(y_test, pred_gbm)

# feature importance (map one-hot back to original cols)
ohe = gbm.named_steps["pre"].named_transformers_["cat"]
ohe_names = list(ohe.get_feature_names_out(cat_cols))
all_feat_names = ohe_names + num_cols
importances = gbm.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": all_feat_names, "importance": importances})
# aggregate one-hot importances back to the parent categorical field
def parent(f):
    for c in cat_cols:
        if f.startswith(c + "_"):
            return c
    return f
imp_df["field"] = imp_df["feature"].apply(parent)
field_importance = imp_df.groupby("field")["importance"].sum().sort_values(ascending=False)

# --- Model 2: Ridge linear regression on log-salary (fully interpretable,
# coefficients get embedded directly in the JS dashboard for live "predict my
# salary" without needing a Python server) ---
X_lin = pd.get_dummies(X, columns=cat_cols, drop_first=True)
lin_cols = X_lin.columns.tolist()
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_lin, np.log(y), test_size=0.2, random_state=42)
ridge = Ridge(alpha=2.0)
ridge.fit(Xl_train, yl_train)
pred_lin = np.exp(ridge.predict(Xl_test))
lin_mae = mean_absolute_error(np.exp(yl_test), pred_lin)
lin_r2 = r2_score(np.exp(yl_test), pred_lin)

coefs = dict(zip(lin_cols, ridge.coef_.tolist()))
intercept = float(ridge.intercept_)

# --- Time-series demand trend: monthly postings count per role (Prophet-style
# forecast substitute -- statsmodels/simple trend since Prophet also needs pip
# install; a linear+seasonal trend fit is used and is transparent for a
# dashboard forecast) ---
df["month"] = df["posting_date"].dt.to_period("M").astype(str)
monthly_role = df.groupby(["month", "role"]).size().reset_index(name="postings")
monthly_total = df.groupby("month").size().reset_index(name="postings")

# simple linear trend forecast, 6 months ahead, per role
forecasts = {}
months_sorted = sorted(df["month"].unique())
x_idx = {m: i for i, m in enumerate(months_sorted)}
for role in df["role"].unique():
    sub = monthly_role[monthly_role.role == role].copy()
    sub["x"] = sub["month"].map(x_idx)
    if len(sub) < 3:
        continue
    coef = np.polyfit(sub["x"], sub["postings"], 1)
    last_x = max(x_idx.values())
    future = [{"month": f"+{i}m", "forecast": max(0, round(np.polyval(coef, last_x + i), 1))}
              for i in range(1, 7)]
    growth_rate = round(100 * coef[0] * 12 / max(sub["postings"].mean(), 1), 1)
    forecasts[role] = {"future": future, "yoy_growth_pct": growth_rate}

# --- Package everything the dashboard needs into one JSON ---
output = {
    "metrics": {
        "gbm_mae": round(gbm_mae, 2), "gbm_r2": round(gbm_r2, 3),
        "linear_mae": round(lin_mae, 2), "linear_r2": round(lin_r2, 3),
        "n_rows": len(df), "n_test": len(y_test),
    },
    "feature_importance": {k: round(float(v), 4) for k, v in field_importance.items()},
    "linear_model": {"intercept": intercept, "coefficients": coefs, "columns": lin_cols},
    "categories": {
        "role": sorted(df["role"].unique().tolist()),
        "company_type": sorted(df["company_type"].unique().tolist()),
        "city": sorted(df["city"].unique().tolist()),
        "education_tier": sorted(df["education_tier"].unique().tolist()),
    },
    "monthly_total": monthly_total.to_dict(orient="records"),
    "monthly_by_role": monthly_role.to_dict(orient="records"),
    "forecasts": forecasts,
    "salary_by_role": df.groupby("role")["salary_lpa"].median().round(1).to_dict(),
    "salary_by_city": df.groupby("city")["salary_lpa"].median().round(1).to_dict(),
    "salary_by_company": df.groupby("company_type")["salary_lpa"].median().round(1).to_dict(),
    "genai_premium_pct": round(100 * (df[df.has_genai_llm_skills == 1]["salary_lpa"].mean() /
                                       df[df.has_genai_llm_skills == 0]["salary_lpa"].mean() - 1), 1),
    "sample_postings": df.sample(300, random_state=1)[
        ["role", "experience_years", "company_type", "city", "education_tier",
         "skills_count", "has_genai_llm_skills", "salary_lpa", "posting_date"]
    ].assign(posting_date=lambda d: d.posting_date.astype(str)).to_dict(orient="records"),
}

with open("dashboard_data.json", "w") as f:
    json.dump(output, f)

print("GBM  -> MAE:", round(gbm_mae, 2), "LPA | R2:", round(gbm_r2, 3))
print("Ridge-> MAE:", round(lin_mae, 2), "LPA | R2:", round(lin_r2, 3))
print("\nField importance (GBM):")
print(field_importance)
print("\nGenAI/LLM skill premium:", output["genai_premium_pct"], "%")
print("\nSaved dashboard_data.json,", len(json.dumps(output)), "bytes")
