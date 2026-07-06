"""
Stage 6 - Baseline Model + Scoring (v1)
=====================================
Trains a Random Forest model to predict a hospital's OFFICIAL CMS star
rating (Hospital overall rating) using our engineered features:
  - patient experience rating
  - average readmission rate
  - Medicare spending (cost)
  - peer group (hospital type/ownership category)

WHY THIS MATTERS:
Our value_score so far was a hand-built formula (50% quality, 50% cost --
a rule WE chose). This model instead learns the relationship between
metrics and quality from the data itself, and we can check: does it
actually predict CMS's own official rating well? If yes, that's strong
evidence our chosen metrics genuinely capture hospital quality -- not
just according to us, but according to CMS's independent judgment.

HOW WE CHECK IF IT WORKS (plain words):
We hide 20% of hospitals from the model during training (the "test set").
After training, we ask the model to guess ratings for those HIDDEN
hospitals and compare its guesses to the real answers. This tells us
how the model would perform on hospitals it's never seen -- a fair test,
not just "did it memorize the ones it studied."
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"
df = pd.read_csv(folder + r"\hospital_features_final.csv", dtype={"Facility ID": str})

# ---------------------------------------------------------------
# STEP 1: Pick our INPUT features (what the model looks at) and our
# TARGET (what it's trying to predict).
# We deliberately EXCLUDE quality_overall from the inputs, since that
# IS the target -- including it would be "cheating" (the model would
# just copy the answer instead of learning a real relationship).
# ---------------------------------------------------------------
input_features = [
    "quality_patient_exp",
    "readmission_rate_avg",
    "cost_value",
    "peer_group",
]
target = "quality_overall"

# Drop hospitals missing the target -- can't train/test without a real answer
model_df = df.dropna(subset=[target]).copy()

# For the inputs, missing values are filled with the peer-group average.
# This is different from earlier stages (where we deliberately left gaps
# as NaN) -- most ML models CANNOT handle missing values directly, so we
# fill them here specifically for the model to work. The original
# hospital_features_final.csv is untouched.
for col in ["quality_patient_exp", "readmission_rate_avg", "cost_value"]:
    model_df[col] = model_df.groupby("peer_group")[col].transform(
        lambda x: x.fillna(x.mean())
    )
# if a whole peer group has no data for a column, fall back to overall mean
for col in ["quality_patient_exp", "readmission_rate_avg", "cost_value"]:
    model_df[col] = model_df[col].fillna(model_df[col].mean())

X = model_df[input_features]
y = model_df[target]

print(f"Training on {len(model_df)} hospitals with a known official rating.\n")

# ---------------------------------------------------------------
# STEP 2: Split into training data (model learns from this) and test
# data (hidden during training, used only to check performance after).
# random_state=42 just means "always split the same way" so results
# are reproducible if we run this again.
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# STEP 3: Train the Random Forest.
# n_estimators=200 means "build 200 different decision trees and
# average their guesses" -- more trees generally = more stable
# predictions, with diminishing returns past a few hundred.
# ---------------------------------------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# STEP 4: Check performance on the HIDDEN test hospitals.
# MAE (Mean Absolute Error): on average, how many stars off was the
#   model's guess? (e.g. 0.4 means guesses were off by less than half
#   a star on average -- pretty good, since ratings are 1-5 whole stars)
# R² (R-squared): what % of the variation in ratings does the model
#   explain? 1.0 = perfect, 0.0 = no better than just guessing the
#   average every time. Real-world models often land between 0.3-0.7.
# ---------------------------------------------------------------
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.3f} stars")
print(f"R-squared: {r2:.3f}")

# ---------------------------------------------------------------
# STEP 5: Which features mattered most? This is a first, simple look
# at explainability -- Stage 7 will go deeper with SHAP values, but
# this gives us an immediate, easy-to-read ranking.
# ---------------------------------------------------------------
importance = pd.Series(model.feature_importances_, index=input_features).sort_values(ascending=False)
print("\nWhich features mattered most to the model's predictions:")
print(importance)

# ---------------------------------------------------------------
# STEP 6: Save the trained model AND the predictions for every
# hospital (not just the test set) -- this becomes our new,
# ML-driven "model_score" column, alongside the original value_score.
# ---------------------------------------------------------------
joblib.dump(model, folder + r"\baseline_model_v1.pkl")

df_scored = df.copy()
# fill missing inputs the same way for the FULL dataset before scoring everyone
for col in ["quality_patient_exp", "readmission_rate_avg", "cost_value"]:
    df_scored[col] = df_scored.groupby("peer_group")[col].transform(
        lambda x: x.fillna(x.mean())
    )
    df_scored[col] = df_scored[col].fillna(df_scored[col].mean())

# only score hospitals that have a peer_group assigned
scorable = df_scored.dropna(subset=["peer_group"]).copy()
scorable["model_predicted_rating"] = model.predict(scorable[input_features])

df_scored = df_scored.merge(
    scorable[["Facility ID", "model_predicted_rating"]], on="Facility ID", how="left"
)

df_scored.to_csv(folder + r"\hospital_features_with_model_v1.csv", index=False)
print("\nSaved baseline_model_v1.pkl and hospital_features_with_model_v1.csv")
