import pandas as pd

df = pd.read_csv("HCAHPS-Hospital.csv")

# Step 1: replace "Not Applicable" / "Not Available" with real NaN in all 3 value columns
value_cols = ["HCAHPS Answer Percent", "HCAHPS Linear Mean Value", "Patient Survey Star Rating"]

for col in value_cols:
    df[col] = df[col].replace(["Not Applicable", "Not Available"], pd.NA)
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Step 2: coalesce into a single unified value — combine_first fills gaps in order
df["hcahps_value"] = (
    df["HCAHPS Answer Percent"]
    .combine_first(df["HCAHPS Linear Mean Value"])
    .combine_first(df["Patient Survey Star Rating"])
)

# Step 3: pivot using the unified column
pivot = df.pivot_table(
    index="Facility ID",              # confirm this matches your CCN join key
    columns="HCAHPS Measure ID",
    values="hcahps_value",
    aggfunc="first"
).reset_index()

pivot.to_csv("hcahps_pivoted_fixed.csv", index=False)
print(pivot["H_STAR_RATING"].describe())
print(pivot["H_STAR_RATING"].isna().mean())