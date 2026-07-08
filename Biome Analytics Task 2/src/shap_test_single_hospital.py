"""
Stage 7 - Explainability - STEP 1: Test on ONE hospital first
=================================================================
Before running SHAP on all 5,432 hospitals, we test it on a single
example hospital, so we can read the output carefully and confirm it
makes sense before scaling up.

WHAT THIS DOES:
Loads our v3 model (Gradient Boosting) and picks one hospital. For that
one hospital, SHAP tells us exactly how many "points" each feature
added or subtracted from its predicted rating, compared to a typical
hospital.
"""

import pandas as pd
import joblib
import shap

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"

# Load the v3 model we already trained
model = joblib.load(folder + r"\baseline_model_v3.pkl")

# Load the data with predictions already in it (from v3's script)
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

# Pick ONE hospital that has a real prediction, just to test with
example = df.dropna(subset=input_features + ["model_predicted_rating"]).iloc[[0]]
print("Example hospital:", example["Facility Name"].values[0])
print("Its predicted rating:", round(example["model_predicted_rating"].values[0], 2))
print("\nIts raw feature values:")
print(example[input_features])

# ---------------------------------------------------------------
# Build a SHAP "explainer" for our model. TreeExplainer is a version
# specifically optimized for tree-based models (Random Forest,
# Gradient Boosting) -- fast and exact for these model types.
# ---------------------------------------------------------------
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for just this one hospital's feature values
shap_values = explainer(example[input_features])

print("\n--- SHAP explanation for this hospital ---")
print(
    "Starting point (average prediction across all hospitals):",
    round(shap_values.base_values[0], 3),
)

for i, feature in enumerate(input_features):
    contribution = shap_values.values[0][i]
    print(f"  {feature:25s}: {contribution:+.3f}")

print("\nSum of starting point + all contributions should equal the prediction:")
print(
    "Check:",
    round(shap_values.base_values[0] + shap_values.values[0].sum(), 3),
    "vs actual prediction:",
    round(example["model_predicted_rating"].values[0], 3),
)
