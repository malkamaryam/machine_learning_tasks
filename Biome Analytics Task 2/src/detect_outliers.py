"""
Stage 5 - Outlier Detection
=============================
Finds hospitals that are UNUSUALLY good or bad compared to their peer
group -- not just "below average", but statistically far enough from
average that it's worth a closer look.

HOW IT WORKS (plain words):
Within each peer group, we already have value_score_peer (Stage 4),
which tells us how far above/below the peer-group average a hospital
is, measured in "standard deviations" (a standard unit of spread).

A hospital sitting near 0 is typical for its peer group.
A hospital at +2.0 or beyond is unusually GOOD for its peer group.
A hospital at -2.0 or beyond is unusually BAD for its peer group.

DOCUMENTED ASSUMPTION:
Threshold = 2.0 standard deviations. This is the standard statistical
convention (roughly the top/bottom 5% under a normal distribution).
Change OUTLIER_THRESHOLD below to make flagging stricter or looser --
no other code needs to change.
"""

import pandas as pd

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"
merged = pd.read_csv(folder + r"\hospital_features_with_peers.csv", dtype={"Facility ID": str})

OUTLIER_THRESHOLD = 2.0  # standard deviations from peer-group average

# ---------------------------------------------------------------
# STEP 1: Classify each hospital based on value_score_peer.
# ---------------------------------------------------------------
def classify(score):
    if pd.isna(score):
        return "Insufficient data"
    elif score >= OUTLIER_THRESHOLD:
        return "Strong positive outlier"
    elif score <= -OUTLIER_THRESHOLD:
        return "Strong negative outlier"
    else:
        return "Typical for peer group"

merged["outlier_status"] = merged["value_score_peer"].apply(classify)

# ---------------------------------------------------------------
# STEP 2: For each hospital, identify its single BIGGEST driver --
# the one metric dragging it down (or lifting it up) the most.
# This directly answers the project's core question: "which metric
# represents the largest improvement opportunity?"
# ---------------------------------------------------------------
driver_cols = {
    "z_quality_overall_peer": "Overall hospital rating",
    "z_quality_patient_exp_peer": "Patient experience rating",
    "z_readmission_peer": "Readmission rate",
    "z_cost_peer": "Medicare spending (cost)",
}

def find_biggest_driver(row):
    values = {label: row[col] for col, label in driver_cols.items() if pd.notna(row[col])}
    if not values:
        return "Insufficient data", None
    # the driver with the LOWEST (most negative) z-score is the biggest drag
    worst_metric = min(values, key=values.get)
    worst_value = values[worst_metric]
    return worst_metric, round(worst_value, 2)

merged[["biggest_drag_metric", "biggest_drag_value"]] = merged.apply(
    lambda row: pd.Series(find_biggest_driver(row)), axis=1
)

def find_biggest_strength(row):
    values = {label: row[col] for col, label in driver_cols.items() if pd.notna(row[col])}
    if not values:
        return "Insufficient data", None
    best_metric = max(values, key=values.get)
    best_value = values[best_metric]
    return best_metric, round(best_value, 2)

merged[["biggest_strength_metric", "biggest_strength_value"]] = merged.apply(
    lambda row: pd.Series(find_biggest_strength(row)), axis=1
)

# ---------------------------------------------------------------
# STEP 3: Save and summarize
# ---------------------------------------------------------------
merged.to_csv(folder + r"\hospital_features_final.csv", index=False)

print("Saved hospital_features_final.csv\n")
print("Outlier status counts (across all hospitals):")
print(merged["outlier_status"].value_counts())

print("\nOutlier status counts PER PEER GROUP:")
print(merged.groupby("peer_group_label")["outlier_status"].value_counts())

print("\nExample strong negative outliers (biggest improvement opportunities):")
negative = merged[merged["outlier_status"] == "Strong negative outlier"]
print(negative[["Facility Name", "State", "peer_group_label", "value_score_peer",
                 "biggest_drag_metric", "biggest_drag_value"]].head(10))
