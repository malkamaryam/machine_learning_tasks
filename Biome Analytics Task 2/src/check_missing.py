"""
Check how much data is missing in each important column.
This tells us WHERE we have gaps before we decide how to handle them.
"""

import pandas as pd

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"
combined = pd.read_csv(folder + r"\combined_hospitals.csv", dtype={"Facility ID": str})

# CMS uses several different text labels for "no real number here".
# We tell pandas to treat all of these as missing (NaN), so it can
# count them properly instead of treating them as valid text values.
missing_labels = ["Not Applicable", "Not Available", "N/A", "", " "]
combined = combined.replace(missing_labels, pd.NA)

# Pick out just the columns we actually care about scoring later.
# (You can add/remove column names here as you learn more about the file.)
key_columns = [
    "Hospital overall rating",
    "H_STAR_RATING",
    "READM_30_HF",
    "READM_30_PN",
    "READM_30_AMI",
    "Value",  # MSPB spending value
]

print("Missing data check (out of", len(combined), "hospitals):\n")
for col in key_columns:
    if col in combined.columns:
        missing_count = combined[col].isna().sum()
        missing_percent = round(missing_count / len(combined) * 100, 1)
        print(f"{col:30s} missing: {missing_count:5d}  ({missing_percent}%)")
    else:
        print(f"{col:30s} -- column not found, check exact name")
