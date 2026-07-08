import pandas as pd

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"
combined = pd.read_csv(folder + r"\combined_hospitals.csv", dtype={"Facility ID": str})
merged = pd.read_csv(
    folder + r"\hospital_features_with_peers.csv", dtype={"Facility ID": str}
)

check = merged[["Facility ID", "peer_group"]].merge(
    combined[
        [
            "Facility ID",
            "Hospital Type",
            "Hospital Ownership",
            "Emergency Services",
            "Meets criteria for birthing friendly designation",
        ]
    ],
    on="Facility ID",
    how="left",
)

for pg in sorted(check["peer_group"].dropna().unique()):
    subset = check[check["peer_group"] == pg]
    print(f"\n--- Cluster {int(pg)} (n={len(subset)}) ---")
    print(
        subset[
            [
                "Hospital Type",
                "Hospital Ownership",
                "Emergency Services",
                "Meets criteria for birthing friendly designation",
            ]
        ]
        .value_counts(dropna=False)
        .head(3)
    )
