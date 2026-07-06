"""
Stage 4 - Peer Grouping (Unsupervised ML)
===========================================
Groups hospitals into "peer groups" -- similar hospitals -- so later
comparisons are apples-to-apples (a tiny rural clinic isn't compared
against a huge urban teaching hospital).

DOCUMENTED ASSUMPTIONS:
1. We do NOT have bed count or teaching-hospital status in this dataset
   (CMS's Provider Data Catalog doesn't include facility size -- that
   lives in a separate system called HCRIS / Medicare Cost Reports).
   This is a known limitation. If bed size data is added later, re-run
   this script with it included for finer-grained peer groups.
2. Clustering uses 4 characteristics we DO have:
     - Hospital Type       (Acute Care, Critical Access, Psychiatric, etc.)
     - Hospital Ownership  (Government, Nonprofit, For-profit, etc.)
     - Emergency Services  (Yes/No)
     - Birthing friendly designation (Yes/No)
3. State is NOT used in clustering (see explanation below) -- it's kept
   as a separate filter column instead, so clustering finds hospitals
   that are structurally similar regardless of location, and State can
   be layered on top in the dashboard as an independent filter.
4. Number of peer groups (k) is chosen using the "elbow method" -- we
   test k=2 through k=10 and pick the point where adding more groups
   stops meaningfully improving the grouping. A chart is saved so this
   choice is visible and reviewable, not a black-box pick.

WHY NOT USE STATE IN THE CLUSTERING MATH:
Clustering measures "distance" between hospitals. If State were included
as one-hot columns (50 separate yes/no columns), two hospitals that are
IDENTICAL in type/ownership/ER access but sit in different states would
be treated as very "far apart" -- which defeats the purpose of finding
hospitals that are actually similar in how they operate. So State is
kept separate and used as a plain filter in the dashboard instead.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")  # save charts to file instead of popping up a window
import matplotlib.pyplot as plt

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"

# ---------------------------------------------------------------
# STEP 1: Load the combined hospital data (has the raw characteristic
# columns) and the feature table (has value_score) so we can merge
# peer groups back onto it.
# ---------------------------------------------------------------
combined = pd.read_csv(folder + r"\combined_hospitals.csv", dtype={"Facility ID": str})
features = pd.read_csv(folder + r"\hospital_features.csv", dtype={"Facility ID": str})

missing_labels = ["Not Applicable", "Not Available", "N/A", "", " "]
combined = combined.replace(missing_labels, pd.NA)

cluster_cols = [
    "Hospital Type",
    "Hospital Ownership",
    "Emergency Services",
]
# NOTE: "Meets criteria for birthing friendly designation" was tested and
# REMOVED from clustering. It's a sparse voluntary certification (mostly
# blank, not a meaningful "No") and was artificially splitting otherwise
# identical hospitals into separate clusters based only on whether they
# applied for this specific maternity-care badge -- not on how the
# hospital actually operates. Confirmed via manual inspection: clusters
# 0/1 and 2/3 had identical Type+Ownership+ER but differed only on this
# column being NaN vs "Y".

cluster_data = combined[["Facility ID"] + cluster_cols].copy()

# Hospitals with missing characteristic data can't be clustered reliably.
# We label these as their own "Unclassified" group rather than dropping
# them or guessing -- transparency over guessing.
cluster_data[cluster_cols] = cluster_data[cluster_cols].fillna("Unknown")

# ---------------------------------------------------------------
# STEP 2: One-hot encode. This turns each category into a set of
# 0/1 columns the algorithm can do math on.
# Example: "Hospital Type" = "Acute Care Hospitals" becomes a column
# "Hospital Type_Acute Care Hospitals" = 1, and all other hospital
# type columns = 0 for that row.
# ---------------------------------------------------------------
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(cluster_data[cluster_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cluster_cols))

# ---------------------------------------------------------------
# STEP 3: Elbow method -- try k = 2 through 10, record how "tight"
# the clusters are each time (inertia = total distance of hospitals
# from their assigned cluster center; lower is tighter, but it
# ALWAYS goes down as k increases, so we look for the point of
# diminishing returns, not the lowest number).
# ---------------------------------------------------------------
inertias = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(encoded_df)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of peer groups (k)")
plt.ylabel("Inertia (lower = tighter groups)")
plt.title("Elbow Method - Choosing Number of Peer Groups")
plt.grid(True)
plt.savefig(folder + r"\elbow_chart.png")
print("Saved elbow_chart.png -- open this to see the elbow point.")
print("\nInertia by k:")
for k, inertia in zip(k_range, inertias):
    print(f"  k={k}: {inertia:.1f}")

# ---------------------------------------------------------------
# STEP 4: Fit final model with chosen k.
# Default set to 6 as a reasonable starting point based on 4 binary/
# categorical features -- REVIEW elbow_chart.png and adjust FINAL_K
# below if the elbow point looks different once you see the chart.
# ---------------------------------------------------------------
FINAL_K = 4  # re-check elbow_chart.png after each run and adjust if the bend point looks different

final_model = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
cluster_data["peer_group"] = final_model.fit_predict(encoded_df)

# Give clusters human-readable labels based on their most common
# Hospital Type + Ownership combo, so "peer_group 3" isn't a mystery.
cluster_data["peer_group_label"] = cluster_data["peer_group"].astype(str)
label_lookup = (
    cluster_data.groupby("peer_group")[["Hospital Type", "Hospital Ownership"]]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Mixed")
)
label_lookup["peer_group_label"] = (
    "Cluster " + label_lookup.index.astype(str) + ": "
    + label_lookup["Hospital Type"].astype(str) + " / "
    + label_lookup["Hospital Ownership"].astype(str)
)
cluster_data = cluster_data.merge(
    label_lookup[["peer_group_label"]], on="peer_group", suffixes=("", "_named")
)
cluster_data["peer_group_label"] = cluster_data["peer_group_label_named"]
cluster_data = cluster_data.drop(columns=["peer_group_label_named"])

# ---------------------------------------------------------------
# STEP 5: Merge peer_group back onto the feature table, and recompute
# value_score WITHIN each peer group (so a hospital is compared to
# similar hospitals, not the whole country). This also fixes most of
# the missing-score problem, since peer-group z-scores use whichever
# hospitals in that small group actually have data.
# ---------------------------------------------------------------
merged = features.merge(
    cluster_data[["Facility ID", "peer_group", "peer_group_label"]],
    on="Facility ID", how="left"
)

def zscore(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * 0  # avoid divide-by-zero if a group has 1 hospital or no variation
    return (series - series.mean()) / std

# recompute peer-group-relative z-scores
merged["z_quality_overall_peer"] = merged.groupby("peer_group")["quality_overall"].transform(zscore)
merged["z_quality_patient_exp_peer"] = merged.groupby("peer_group")["quality_patient_exp"].transform(zscore)
merged["z_readmission_peer"] = -merged.groupby("peer_group")["readmission_rate_avg"].transform(zscore)
merged["z_cost_peer"] = -merged.groupby("peer_group")["cost_value"].transform(zscore)

quality_peer_cols = ["z_quality_overall_peer", "z_quality_patient_exp_peer", "z_readmission_peer"]
merged["quality_component_peer"] = merged[quality_peer_cols].mean(axis=1, skipna=True)
merged["cost_component_peer"] = merged["z_cost_peer"]

# BUG FIX: previously we did 0.5*quality + 0.5*cost directly, which
# breaks (returns NaN) if EITHER side is missing -- even if the other
# side has perfectly good data. This silently zeroed out ~1,600
# Critical Access hospitals whose cost data legitimately doesn't exist
# (CMS doesn't calculate MSPB for them) but who DO have real quality data.
#
# FIX: weighted average that skips missing components and renormalizes
# the remaining weight, instead of failing outright. A hospital missing
# cost data is now scored 100% on quality, rather than "unknown".
def weighted_score(row):
    weights = {"quality_component_peer": 0.5, "cost_component_peer": 0.5}
    available = {k: row[k] for k, w in weights.items() if pd.notna(row[k])}
    if not available:
        return pd.NA
    total_weight = sum(weights[k] for k in available)
    return sum(row[k] * weights[k] for k in available) / total_weight

merged["value_score_peer"] = merged.apply(weighted_score, axis=1)
merged["value_score_peer"] = pd.to_numeric(merged["value_score_peer"], errors="coerce")

merged.to_csv(folder + r"\hospital_features_with_peers.csv", index=False)

print("\nSaved hospital_features_with_peers.csv")
print("\nPeer group sizes:")
print(cluster_data["peer_group_label"].value_counts())
print("\nHospitals with a usable peer-group value_score (not NaN):")
print(merged["value_score_peer"].notna().sum(), "out of", len(merged))