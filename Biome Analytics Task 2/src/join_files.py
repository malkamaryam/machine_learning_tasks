"""
Join the 4 CMS hospital files into one combined file.
Matches rows using "Facility ID" — the shared ID every file has.
"""

import pandas as pd

# STEP 1: Tell Python where your files live.
# (This is your folder path — already filled in for you.)
folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"

# STEP 2: Open each of the 4 files.
# pd.read_csv() just means "open this spreadsheet file".
# dtype={"Facility ID": str} forces Python to treat Facility ID as TEXT
# in every file, so "010001" never gets silently turned into the number
# 10001. This keeps the ID consistent across all 4 files.
id_as_text = {"Facility ID": str}

general_info = pd.read_csv(folder + r"\Hospital_General_Information.csv", dtype=id_as_text)
hcahps       = pd.read_csv(folder + r"\HCAHPS-Hospital.csv", dtype=id_as_text, low_memory=False)
readmissions = pd.read_csv(folder + r"\Unplanned_Hospital_Visits-Hospital.csv", dtype=id_as_text, low_memory=False)
spending     = pd.read_csv(folder + r"\HOSPITAL_QUARTERLY_MSPB_6_DECIMALS.csv", dtype=id_as_text)

# STEP 3: Print how many rows each file has, just so we can see
# everything loaded correctly before we do anything else.
print("General info rows:", len(general_info))
print("HCAHPS rows:", len(hcahps))
print("Readmissions rows:", len(readmissions))
print("Spending rows:", len(spending))

# STEP 3b: RESHAPE (pivot) the HCAHPS and readmissions files.
#
# Right now, these files have MANY rows per hospital -- one row per
# question/measure. We need ONE row per hospital, with each question
# turned into its own column, so it can be safely stapled to the
# other files without duplicating rows.
#
# pivot_table does exactly this "flip sideways" operation:
#   index   = what should stay as one row per hospital (Facility ID)
#   columns = what should become new column headers (the question names)
#   values  = what number should fill in those new columns (the score)

# HCAHPS stores its numbers in 3 DIFFERENT columns depending on measure type:
#   - most measures  -> "HCAHPS Answer Percent"
#   - a few measures  -> "HCAHPS Linear Mean Value"
#   - star ratings    -> "Patient Survey Star Rating"
# Each row only ever fills ONE of these three - the rest are blank.
# We combine them into a single "hcahps_value" column before pivoting,
# so every measure type (including star ratings) gets captured.

value_cols = ["HCAHPS Answer Percent", "HCAHPS Linear Mean Value", "Patient Survey Star Rating"]
for col in value_cols:
    hcahps[col] = hcahps[col].replace(["Not Applicable", "Not Available"], pd.NA)
    hcahps[col] = pd.to_numeric(hcahps[col], errors="coerce")

hcahps["hcahps_value"] = (
    hcahps["HCAHPS Answer Percent"]
    .combine_first(hcahps["HCAHPS Linear Mean Value"])
    .combine_first(hcahps["Patient Survey Star Rating"])
)

hcahps_wide = hcahps.pivot_table(
    index="Facility ID",
    columns="HCAHPS Measure ID",
    values="hcahps_value",
    aggfunc="first"
).reset_index()

readmissions_wide = readmissions.pivot_table(
    index="Facility ID",
    columns="Measure ID",
    values="Score",
    aggfunc="first"
).reset_index()

print("HCAHPS rows after pivot (should be ~5432):", len(hcahps_wide))
print("Readmissions rows after pivot (should be ~5432):", len(readmissions_wide))

# STEP 4: Staple the files together, one at a time, matching on Facility ID.
# "how='left'" means: keep every hospital from general_info,
# even if it's missing from one of the other files.
combined = general_info.merge(hcahps_wide, on="Facility ID", how="left")
combined = combined.merge(readmissions_wide, on="Facility ID", how="left")
combined = combined.merge(spending, on="Facility ID", how="left")

print("Combined rows:", len(combined))

# STEP 5: Save the result as one new file.
combined.to_csv(folder + r"\combined_hospitals.csv", index=False)
print("Saved combined_hospitals.csv")