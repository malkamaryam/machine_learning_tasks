"""
Stage 7 - Explainability - STEP 2: Scale up to ALL hospitals
=================================================================
Same idea as the single-hospital test, but now we calculate a SHAP
explanation for every hospital that has a valid prediction, and save
the results so the dashboard can show ANY hospital's personalized
"why did you score this way" breakdown.

WHAT WE SAVE FOR EACH HOSPITAL:
- shap_<feature> columns: how many points each feature added/subtracted
  for THIS specific hospital (same idea as the single-hospital test)
- shap_top_positive: which feature helped this hospital's score the MOST
- shap_top_negative: which feature hurt this hospital's score the MOST
  (this directly answers "what's the biggest improvement opportunity?")
"""

import pandas as pd
import joblib
import shap

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"

model = joblib.load(folder + r"\baseline_model_v3.pkl")
df = pd.read_csv(
    folder + r"\hospital_features_with_model_v3.csv", dtype={"Facility ID": str}
)

input_features = [
    "quality_patient_exp",
    "readmission_rate_avg",
    "cost_value",
    "peer_group",
    "MORT_net",
    "Safety_net",
    "READM_net",
]

# Only hospitals with a real prediction can get a real explanation
scorable = df.dropna(subset=input_features + ["model_predicted_rating"]).copy()
print(f"Calculating SHAP explanations for {len(scorable)} hospitals...")

# ---------------------------------------------------------------
# Build the explainer ONCE, then apply it to ALL hospitals at once.
# This is much faster than calling it one hospital at a time -- SHAP
# is built to handle a whole table of hospitals in a single call.
# ---------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(scorable[input_features])

# shap_values.values is a table: one row per hospital, one column per
# feature, same shape as scorable[input_features]. We turn it into a
# proper dataframe with clear column names.
shap_df = pd.DataFrame(
    shap_values.values,
    columns=[f"shap_{f}" for f in input_features],
    index=scorable.index,
)

scorable = pd.concat([scorable, shap_df], axis=1)

# ---------------------------------------------------------------
# For each hospital, find which feature helped MOST and which hurt MOST.
# This directly answers the project's core question: "which metric
# represents the largest improvement opportunity?"
# ---------------------------------------------------------------
shap_cols = [f"shap_{f}" for f in input_features]


def get_top_positive(row):
    values = row[shap_cols]
    best_col = values.idxmax()
    return best_col.replace("shap_", ""), round(values[best_col], 3)


def get_top_negative(row):
    values = row[shap_cols]
    worst_col = values.idxmin()
    return worst_col.replace("shap_", ""), round(values[worst_col], 3)


scorable[["shap_top_positive_feature", "shap_top_positive_value"]] = scorable.apply(
    lambda row: pd.Series(get_top_positive(row)), axis=1
)
scorable[["shap_top_negative_feature", "shap_top_negative_value"]] = scorable.apply(
    lambda row: pd.Series(get_top_negative(row)), axis=1
)

# ---------------------------------------------------------------
# Merge these new columns back onto the FULL hospital list (so
# hospitals without a prediction still appear in the file, just with
# blank SHAP columns -- consistent with how we've handled missing
# data throughout this whole project).
# ---------------------------------------------------------------
new_cols = shap_cols + [
    "shap_top_positive_feature",
    "shap_top_positive_value",
    "shap_top_negative_feature",
    "shap_top_negative_value",
]
df_final = df.merge(scorable[["Facility ID"] + new_cols], on="Facility ID", how="left")

df_final.to_csv(folder + r"\hospital_features_with_shap.csv", index=False)

print("Saved hospital_features_with_shap.csv\n")
print("How often each feature is the TOP POSITIVE driver, across all hospitals:")
print(df_final["shap_top_positive_feature"].value_counts())
print(
    "\nHow often each feature is the TOP NEGATIVE driver (biggest improvement opportunity), across all hospitals:"
)
print(df_final["shap_top_negative_feature"].value_counts())
