# Reusable helper functions for the diabetes readmission pipeline.

import pandas as pd
import numpy as np


def map_diag(code):
    if pd.isna(code):
        return "Missing"
    try:
        code = float(code)
    except ValueError:
        return "Other"  # V- and E- codes
    if 390 <= code <= 459 or code == 785:
        return "Circulatory"
    if 460 <= code <= 519 or code == 786:
        return "Respiratory"
    if 520 <= code <= 579 or code == 787:
        return "Digestive"
    if 250 <= code < 251:
        return "Diabetes"
    if 800 <= code <= 999:
        return "Injury"
    if 710 <= code <= 739:
        return "Musculoskeletal"
    if 580 <= code <= 629 or code == 788:
        return "Genitourinary"
    if 140 <= code <= 239:
        return "Neoplasms"
    if 240 <= code <= 279:
        return "Endocrine_other"
    return "Other"


def map_discharge(code):
    if code == 1:
        return "Home"
    if code in [6, 8]:
        return "Home_with_care"
    if code in [2, 3, 4, 5, 9, 10, 15, 22, 23, 24, 27, 28, 29, 30]:
        return "Transferred_facility"
    if code == 7:
        return "Left_AMA"
    return "Other"


def map_admission_source(code):
    if code in [1, 2, 3]:
        return "Referral"
    if code in [4, 5, 6, 10, 18, 22, 25, 26]:
        return "Transfer"
    if code == 7:
        return "Emergency"
    if code in [11, 12, 13, 14, 23, 24]:
        return "Delivery_Birth"
    return "Other"


def top_shap_features(shap_vals, feature_names, n=10):
    if shap_vals.ndim == 3 and shap_vals.shape[2] > 1:
        shap_vals = shap_vals[:, :, 1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:n]
    return pd.Series(mean_abs[idx], index=np.array(feature_names)[idx])
