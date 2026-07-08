"""
Stage 6 - Baseline Model + Scoring (v3)
=====================================
Same 7 input features as v2 (patient experience, readmission rate, cost,
peer group, MORT_net, Safety_net, READM_net) -- but swaps the model type
from Random Forest to GRADIENT BOOSTING to see if a different algorithm
performs better on the same data.

NOTE: We originally planned to also add a "Timeliness net" feature here,
matching MORT_net/Safety_net/READM_net. On closer inspection, CMS's data
does NOT provide a Better/Worse breakdown for the Timeliness or Patient
Experience groups (only Mortality, Safety, and Readmission have that
breakdown) -- so that addition isn't possible with a simple net-score
approach. This is documented here rather than silently skipped. A real
Timeliness signal would require pulling in Timely_and_Effective_Care-
Hospital.csv's actual measure values directly, which is a bigger future
addition (see project notes).

WHAT IS GRADIENT BOOSTING (plain words)?
Random Forest builds many trees INDEPENDENTLY and averages their
guesses. Gradient Boosting builds trees ONE AFTER ANOTHER, where each
new tree specifically tries to fix the mistakes the previous trees
made. This sequential "learn from your errors" approach often performs
better on structured/tabular data like ours, though it can also overfit
more easily if not careful -- which is exactly why we still test it
the same fair way, on hidden test data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"
df = pd.read_csv(folder + r"\hospital_features_final.csv", dtype={"Facility ID": str})
raw = pd.read_csv(folder + r"\combined_hospitals.csv", dtype={"Facility ID": str})

category_pairs = {
    "MORT_net": ("Count of MORT Measures Better", "Count of MORT Measures Worse"),
    "Safety_net": ("Count of Safety Measures Better", "Count of Safety Measures Worse"),
    "READM_net": ("Count of READM Measures Better", "Count of READM Measures Worse"),
}
for new_col, (better_col, worse_col) in category_pairs.items():
    raw[better_col] = pd.to_numeric(raw[better_col], errors="coerce")
    raw[worse_col] = pd.to_numeric(raw[worse_col], errors="coerce")
    raw[new_col] = raw[better_col] - raw[worse_col]

df = df.merge(
    raw[["Facility ID"] + list(category_pairs.keys())], on="Facility ID", how="left"
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
target = "quality_overall"

model_df = df.dropna(subset=[target]).copy()

fill_cols = [
    "quality_patient_exp",
    "readmission_rate_avg",
    "cost_value",
    "MORT_net",
    "Safety_net",
    "READM_net",
]
for col in fill_cols:
    model_df[col] = model_df.groupby("peer_group")[col].transform(
        lambda x: x.fillna(x.mean())
    )
for col in fill_cols:
    model_df[col] = model_df[col].fillna(model_df[col].mean())

X = model_df[input_features]
y = model_df[target]

print(f"Training on {len(model_df)} hospitals with a known official rating.\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# Gradient Boosting settings:
# n_estimators=200    -> build up to 200 trees, each correcting past errors
# learning_rate=0.05   -> how much each new tree is allowed to correct --
#                         smaller = more cautious, less likely to overfit
# max_depth=3          -> keep each individual tree simple/shallow, which
#                         is standard practice for boosting (many small
#                         simple trees, rather than a few big ones)
# ---------------------------------------------------------------
model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.3f} stars")
print(f"R-squared: {r2:.3f}")

importance = pd.Series(model.feature_importances_, index=input_features).sort_values(
    ascending=False
)
print("\nWhich features mattered most to the model's predictions:")
print(importance)

joblib.dump(model, folder + r"\baseline_model_v3.pkl")

df_scored = df.copy()
for col in fill_cols:
    df_scored[col] = df_scored.groupby("peer_group")[col].transform(
        lambda x: x.fillna(x.mean())
    )
    df_scored[col] = df_scored[col].fillna(df_scored[col].mean())

scorable = df_scored.dropna(subset=["peer_group"]).copy()
scorable["model_predicted_rating"] = model.predict(scorable[input_features])

df_scored = df_scored.merge(
    scorable[["Facility ID", "model_predicted_rating"]], on="Facility ID", how="left"
)
df_scored.to_csv(folder + r"\hospital_features_with_model_v3.csv", index=False)
print("\nSaved baseline_model_v3.pkl and hospital_features_with_model_v3.csv")
