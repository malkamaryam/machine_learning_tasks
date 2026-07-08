"""
Stage 3 - Feature Engineering
==============================
Turns the raw combined_hospitals.csv into a clean feature table with:
  - readmission_rate_avg   (average of 6 readmission measures, lower = better)
  - cost_value             (MSPB spending, lower = better)
  - quality_overall        (Hospital overall rating, 1-5, higher = better)
  - quality_patient_exp    (H_STAR_RATING, 1-5, higher = better)
  - z-scores for each of the above (so they're all on the same scale)
  - value_score            (composite quality-vs-cost score)

DOCUMENTED ASSUMPTIONS (read this before trusting the score):
1. Missing values are left as NaN and skipped per-hospital, per-calculation.
   We do NOT fill with dataset-wide averages, because that would blur the
   real signal that "missing" often means "hospital too small to report".
   This is temporary -- Stage 4 will upgrade to peer-group imputation.
2. All 6 readmission measures (AMI, CABG, COPD, HF, HIP_KNEE, PN) are
   weighted EQUALLY. No adjustment for how common/rare each condition is.
3. Quality and cost are combined with equal weight (50/50) in the final
   value_score. This is a starting assumption you can change later.
4. z-scores are computed using the FULL national dataset (not peer groups)
   at this stage. Peer-group-relative z-scores come in Stage 4.
"""

import pandas as pd

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"
combined = pd.read_csv(folder + r"\combined_hospitals.csv", dtype={"Facility ID": str})

# Treat CMS's text placeholders as real missing values everywhere
missing_labels = ["Not Applicable", "Not Available", "N/A", "", " "]
combined = combined.replace(missing_labels, pd.NA)

# ---------------------------------------------------------------
# STEP 1: Make sure the numeric columns are actually numeric
# (CSV import sometimes leaves numbers as text if a column had
# any "Not Applicable" text mixed in)
# ---------------------------------------------------------------
readmission_cols = [
    "READM_30_AMI",
    "READM_30_CABG",
    "READM_30_COPD",
    "READM_30_HF",
    "READM_30_HIP_KNEE",
    "READM_30_PN",
]
numeric_cols = readmission_cols + ["Value", "Hospital overall rating", "H_STAR_RATING"]

for col in numeric_cols:
    combined[col] = pd.to_numeric(combined[col], errors="coerce")

# ---------------------------------------------------------------
# STEP 2: Build the 4 core feature columns
# ---------------------------------------------------------------

# Quality signal 1: overall CMS star rating (already 1-5, higher = better)
combined["quality_overall"] = combined["Hospital overall rating"]

# Quality signal 2: patient experience star rating (already 1-5, higher = better)
combined["quality_patient_exp"] = combined["H_STAR_RATING"]

# Quality signal 3: average readmission rate across 6 conditions.
# skipna=True means: if a hospital is missing 2 of the 6, average the
# other 4 instead of throwing the hospital out entirely.
combined["readmission_rate_avg"] = combined[readmission_cols].mean(axis=1, skipna=True)

# Cost signal: MSPB spending value (already lower = better, no change needed)
combined["cost_value"] = combined["Value"]

# ---------------------------------------------------------------
# STEP 3: Convert everything to z-scores so they're comparable.
# z-score = (value - national_average) / national_std_dev
# This turns "% readmitted" and "$ spent" and "1-5 stars" into the
# same unit: "how many standard deviations from average is this hospital".
# ---------------------------------------------------------------


def zscore(series):
    return (series - series.mean()) / series.std()


combined["z_quality_overall"] = zscore(combined["quality_overall"])
combined["z_quality_patient_exp"] = zscore(combined["quality_patient_exp"])

# Readmission and cost are "lower = better", so we FLIP the sign after
# z-scoring. That way, positive always means "good" across every feature,
# and we can add them together consistently.
combined["z_readmission"] = -zscore(combined["readmission_rate_avg"])
combined["z_cost"] = -zscore(combined["cost_value"])

# ---------------------------------------------------------------
# STEP 4: Composite value score
# quality_component = average of the 3 quality z-scores
# cost_component     = the cost z-score
# value_score        = 50% quality, 50% cost (documented assumption #3)
#
# .mean(axis=1, skipna=True) again means a hospital missing ONE quality
# signal still gets scored using whichever ones it has, instead of
# becoming blank entirely.
# ---------------------------------------------------------------
quality_z_cols = ["z_quality_overall", "z_quality_patient_exp", "z_readmission"]
combined["quality_component"] = combined[quality_z_cols].mean(axis=1, skipna=True)
combined["cost_component"] = combined["z_cost"]


# BUG FIX: a plain 0.5*quality + 0.5*cost breaks to NaN if EITHER side
# is missing, even when the other side has good data. Fixed with a
# weighted average that skips missing components and renormalizes the
# remaining weight -- see build_peer_groups.py for the same fix and a
# fuller explanation of why this matters.
def weighted_score(row):
    weights = {"quality_component": 0.5, "cost_component": 0.5}
    available = {k: row[k] for k, w in weights.items() if pd.notna(row[k])}
    if not available:
        return pd.NA
    total_weight = sum(weights[k] for k in available)
    return sum(row[k] * weights[k] for k in available) / total_weight


combined["value_score"] = combined.apply(weighted_score, axis=1)
combined["value_score"] = pd.to_numeric(combined["value_score"], errors="coerce")

# ---------------------------------------------------------------
# STEP 5: Save a clean feature table -- just the columns we need
# for scoring, clustering, and the dashboard later.
# ---------------------------------------------------------------
feature_cols = [
    "Facility ID",
    "Facility Name",
    "State",
    "Hospital Type",
    "Hospital Ownership",
    "quality_overall",
    "quality_patient_exp",
    "readmission_rate_avg",
    "cost_value",
    "z_quality_overall",
    "z_quality_patient_exp",
    "z_readmission",
    "z_cost",
    "quality_component",
    "cost_component",
    "value_score",
]
features = combined[feature_cols]
features.to_csv(folder + r"\hospital_features.csv", index=False)

print("Saved hospital_features.csv with", len(features), "rows.")
print("\nvalue_score summary:")
print(features["value_score"].describe())
print("\nHow many hospitals have a usable value_score (not NaN)?")
print(features["value_score"].notna().sum(), "out of", len(features))
