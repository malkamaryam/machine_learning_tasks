"""
Diagnostic - Check data availability PER PEER GROUP
=======================================================
Before deciding how to fix the "Insufficient data" problem, we need to
see WHICH of our 4 metrics (if any) are actually reported for Critical
Access and Psychiatric hospitals. This tells us whether we can rescue
these hospitals with a partial score, or whether they truly have zero
usable data under CMS's current reporting rules.
"""

import pandas as pd

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"
merged = pd.read_csv(folder + r"\hospital_features_final.csv", dtype={"Facility ID": str})

metric_cols = ["quality_overall", "quality_patient_exp", "readmission_rate_avg", "cost_value"]

print("Data availability (% of hospitals WITH a real value) by peer group:\n")
for pg_label in merged["peer_group_label"].dropna().unique():
    subset = merged[merged["peer_group_label"] == pg_label]
    print(f"--- {pg_label} (n={len(subset)}) ---")
    for col in metric_cols:
        available_pct = round(subset[col].notna().mean() * 100, 1)
        print(f"  {col:25s}: {available_pct}% have data")
    print()
