"""
Stage 8 - Full Interactive Dashboard
=======================================
Lets a user pick a State, Hospital Type, or Peer Group, then select a
specific hospital and see:
  1. Its value score and whether it's flagged as an outlier
  2. A SHAP breakdown -- which specific metrics pushed its score up/down
  3. A radar chart comparing it to its peer group average
  4. A table of the biggest outliers in the current filtered view

HOW TO RUN THIS FILE:
    streamlit run dashboard.py
(NOT "python dashboard.py" -- Streamlit apps use their own launch command)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

folder = r"C:\Users\malka\OneDrive\Desktop\Hospital project"

st.set_page_config(page_title="Hospital Value Dashboard", layout="wide")


# ---------------------------------------------------------------
# STEP 1: Load the data. @st.cache_data means "only reload this file
# from disk if it actually changes" -- makes the app much faster,
# since otherwise Streamlit reloads everything on every click.
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        folder + r"\hospital_features_with_shap.csv", dtype={"Facility ID": str}
    )
    return df


df = load_data()

st.title("Hospital Value & Performance Dashboard")
st.caption(
    "Explore hospital quality-vs-cost value scores, peer comparisons, and improvement opportunities."
)

# ---------------------------------------------------------------
# STEP 2: Sidebar filters. st.sidebar puts these controls in a
# left-hand panel instead of the main page, which is the standard
# layout for filter controls in dashboards.
# ---------------------------------------------------------------
st.sidebar.header("Filters")

states = sorted(df["State"].dropna().unique())
selected_state = st.sidebar.selectbox("State", ["All"] + states)

peer_groups = sorted(df["peer_group_label"].dropna().unique())
selected_peer_group = st.sidebar.selectbox("Peer Group", ["All"] + peer_groups)

hospital_types = sorted(df["Hospital Type"].dropna().unique())
selected_hospital_type = st.sidebar.selectbox("Hospital Type", ["All"] + hospital_types)

# Apply filters -- start with everything, narrow down based on selections
filtered_df = df.copy()
if selected_state != "All":
    filtered_df = filtered_df[filtered_df["State"] == selected_state]
if selected_peer_group != "All":
    filtered_df = filtered_df[filtered_df["peer_group_label"] == selected_peer_group]
if selected_hospital_type != "All":
    filtered_df = filtered_df[filtered_df["Hospital Type"] == selected_hospital_type]

st.sidebar.write(f"{len(filtered_df)} hospitals match your filters.")

# ---------------------------------------------------------------
# STEP 3: Hospital picker -- choose ONE specific hospital from the
# filtered list to look at in detail.
# ---------------------------------------------------------------
hospital_names = sorted(filtered_df["Facility Name"].dropna().unique())

if len(hospital_names) == 0:
    st.warning("No hospitals match your current filters. Try loosening them.")
    st.stop()

selected_hospital = st.selectbox("Choose a hospital to inspect:", hospital_names)

hospital_row = filtered_df[filtered_df["Facility Name"] == selected_hospital].iloc[0]

# ---------------------------------------------------------------
# STEP 4: Show this hospital's headline numbers.
# st.columns lets us place several stat boxes side by side.
# ---------------------------------------------------------------
st.subheader(f"{selected_hospital}")
st.write(
    f"**Peer group:** {hospital_row['peer_group_label']}  |  **State:** {hospital_row['State']}"
)

col1, col2, col3 = st.columns(3)
col1.metric(
    "Value Score (vs peers)",
    (
        f"{hospital_row['value_score_peer']:.2f}"
        if pd.notna(hospital_row["value_score_peer"])
        else "N/A"
    ),
)
col2.metric(
    "Outlier Status",
    (
        hospital_row["outlier_status"]
        if pd.notna(hospital_row["outlier_status"])
        else "N/A"
    ),
)
col3.metric(
    "CMS Overall Rating",
    (
        f"{hospital_row['quality_overall']:.0f} stars"
        if pd.notna(hospital_row["quality_overall"])
        else "N/A"
    ),
)

# ---------------------------------------------------------------
# STEP 5: SHAP breakdown chart -- shows exactly which metrics pushed
# this hospital's score up or down, using the SHAP values we
# calculated in Stage 7. A horizontal bar chart works well here:
# bars pointing right = helped the score, left = hurt the score.
# ---------------------------------------------------------------
st.subheader("Why did this hospital score this way?")

shap_features = [
    "quality_patient_exp",
    "readmission_rate_avg",
    "cost_value",
    "peer_group",
    "MORT_net",
    "Safety_net",
    "READM_net",
]
feature_labels = {
    "quality_patient_exp": "Patient Experience",
    "readmission_rate_avg": "Readmission Rate",
    "cost_value": "Medicare Spending (Cost)",
    "peer_group": "Peer Group",
    "MORT_net": "Mortality (Better vs Worse)",
    "Safety_net": "Safety (Better vs Worse)",
    "READM_net": "Readmission (Better vs Worse)",
}

shap_cols = [f"shap_{f}" for f in shap_features]
has_shap = all(pd.notna(hospital_row[c]) for c in shap_cols)

if has_shap:
    shap_vals = [hospital_row[c] for c in shap_cols]
    labels = [feature_labels[f] for f in shap_features]
    colors = [
        "#2E7D32" if v >= 0 else "#C62828" for v in shap_vals
    ]  # green = helped, red = hurt

    # sort so the biggest effects (positive or negative) appear at the top
    order = sorted(range(len(shap_vals)), key=lambda i: abs(shap_vals[i]))
    shap_vals_sorted = [shap_vals[i] for i in order]
    labels_sorted = [labels[i] for i in order]
    colors_sorted = [colors[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=shap_vals_sorted,
            y=labels_sorted,
            orientation="h",
            marker_color=colors_sorted,
        )
    )
    fig.update_layout(
        xaxis_title="Impact on predicted rating (+ helps, - hurts)",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Green bars pushed this hospital's score UP. Red bars pulled it DOWN. "
        "Longer bars = bigger impact for this specific hospital."
    )
else:
    st.info(
        "Not enough data available to generate a SHAP explanation for this hospital."
    )

# ---------------------------------------------------------------
# STEP 6: Radar chart -- compares this hospital's raw metrics
# against its peer group's average, so the user can see at a glance
# where it's ahead or behind.
# ---------------------------------------------------------------
st.subheader("This hospital vs. its peer group average")

radar_metrics = {
    "Overall Rating": "quality_overall",
    "Patient Experience": "quality_patient_exp",
    "Readmission Rate (inverted)": "z_readmission_peer",  # already flipped so higher = better
    "Cost (inverted)": "z_cost_peer",  # already flipped so higher = better
}

peer_group_id = hospital_row["peer_group"]
peer_subset = df[df["peer_group"] == peer_group_id]

hospital_values = []
peer_avg_values = []
for label, col in radar_metrics.items():
    hospital_values.append(hospital_row[col] if pd.notna(hospital_row[col]) else 0)
    peer_avg_values.append(peer_subset[col].mean())

categories = list(radar_metrics.keys())

fig2 = go.Figure()
fig2.add_trace(
    go.Scatterpolar(
        r=hospital_values + [hospital_values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=selected_hospital,
    )
)
fig2.add_trace(
    go.Scatterpolar(
        r=peer_avg_values + [peer_avg_values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="Peer Group Average",
        opacity=0.5,
    )
)
fig2.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=True,
    height=450,
)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "Note: Overall Rating and Patient Experience are on a 1-5 star scale. "
    "Readmission Rate and Cost are shown as z-scores (standard deviations from "
    "peer group average, already flipped so higher always means better)."
)

# ---------------------------------------------------------------
# STEP 7: Table of the biggest outliers in the current filtered view
# -- lets the user explore beyond just the one selected hospital.
# ---------------------------------------------------------------
st.subheader("Biggest outliers in your current filter")

outlier_table = filtered_df[
    filtered_df["outlier_status"].isin(
        ["Strong positive outlier", "Strong negative outlier"]
    )
].copy()

if len(outlier_table) > 0:
    outlier_table = outlier_table.sort_values("value_score_peer")
    display_cols = [
        "Facility Name",
        "State",
        "peer_group_label",
        "value_score_peer",
        "outlier_status",
        "shap_top_negative_feature",
        "shap_top_positive_feature",
    ]
    st.dataframe(
        outlier_table[display_cols].rename(
            columns={
                "Facility Name": "Hospital",
                "peer_group_label": "Peer Group",
                "value_score_peer": "Value Score",
                "outlier_status": "Status",
                "shap_top_negative_feature": "Biggest Weakness",
                "shap_top_positive_feature": "Biggest Strength",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write("No strong outliers found in the current filter selection.")
