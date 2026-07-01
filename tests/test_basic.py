#Unit tests for map_diag and map_admission_source helper functions

import pytest
from biome_analytics.helpers import map_diag, map_admission_source

# map_diag tests

# Missing / non-numeric inputs
def test_map_diag_nan_returns_missing():
    assert map_diag(None) == "Missing"


def test_map_diag_vcode_returns_other():
    assert map_diag("V45") == "Other"


def test_map_diag_ecode_returns_other():
    assert map_diag("E890") == "Other"


# Circulatory: 390-459, 785
def test_map_diag_circulatory_lower_bound():
    assert map_diag(390) == "Circulatory"


def test_map_diag_circulatory_upper_bound():
    assert map_diag(459) == "Circulatory"


def test_map_diag_circulatory_midrange():
    assert map_diag(410) == "Circulatory"


def test_map_diag_circulatory_symptom_code():
    assert map_diag(785) == "Circulatory"


# Respiratory: 460-519, 786
def test_map_diag_respiratory_lower_bound():
    assert map_diag(460) == "Respiratory"


def test_map_diag_respiratory_upper_bound():
    assert map_diag(519) == "Respiratory"


def test_map_diag_respiratory_symptom_code():
    assert map_diag(786) == "Respiratory"


# Digestive: 520-579, 787
def test_map_diag_digestive_lower_bound():
    assert map_diag(520) == "Digestive"


def test_map_diag_digestive_upper_bound():
    assert map_diag(579) == "Digestive"


def test_map_diag_digestive_symptom_code():
    assert map_diag(787) == "Digestive"


# Diabetes: 250 <= code < 251
def test_map_diag_diabetes_exact():
    assert map_diag(250) == "Diabetes"


def test_map_diag_diabetes_decimal():
    assert map_diag(250.83) == "Diabetes"


def test_map_diag_diabetes_string_input():
    assert map_diag("250.01") == "Diabetes"


# 251 should NOT be diabetes, falls into Endocrine_other
def test_map_diag_diabetes_upper_boundary_excluded():
    assert map_diag(251) == "Endocrine_other"


# Injury: 800-999
def test_map_diag_injury_lower_bound():
    assert map_diag(800) == "Injury"


def test_map_diag_injury_upper_bound():
    assert map_diag(999) == "Injury"


# Musculoskeletal: 710-739
def test_map_diag_musculoskeletal_lower_bound():
    assert map_diag(710) == "Musculoskeletal"


def test_map_diag_musculoskeletal_upper_bound():
    assert map_diag(739) == "Musculoskeletal"


# Genitourinary: 580-629, 788
def test_map_diag_genitourinary_lower_bound():
    assert map_diag(580) == "Genitourinary"


def test_map_diag_genitourinary_upper_bound():
    assert map_diag(629) == "Genitourinary"


def test_map_diag_genitourinary_symptom_code():
    assert map_diag(788) == "Genitourinary"


# Neoplasms: 140-239
def test_map_diag_neoplasms_lower_bound():
    assert map_diag(140) == "Neoplasms"


def test_map_diag_neoplasms_upper_bound():
    assert map_diag(239) == "Neoplasms"


# Endocrine_other: 240-279
def test_map_diag_endocrine_lower_bound():
    assert map_diag(240) == "Endocrine_other"


def test_map_diag_endocrine_upper_bound():
    assert map_diag(279) == "Endocrine_other"


# Other (unclassified numeric code)
def test_map_diag_other_unclassified():
    assert map_diag(100) == "Other"


# map_admission_source tests

# Referral: 1, 2, 3
def test_map_admission_source_referral_1():
    assert map_admission_source(1) == "Referral"


def test_map_admission_source_referral_2():
    assert map_admission_source(2) == "Referral"


def test_map_admission_source_referral_3():
    assert map_admission_source(3) == "Referral"


# Transfer: 4, 5, 6, 10, 18, 22, 25, 26
def test_map_admission_source_transfer_4():
    assert map_admission_source(4) == "Transfer"


def test_map_admission_source_transfer_10():
    assert map_admission_source(10) == "Transfer"


def test_map_admission_source_transfer_18():
    assert map_admission_source(18) == "Transfer"


def test_map_admission_source_transfer_26():
    assert map_admission_source(26) == "Transfer"


# Emergency: 7
def test_map_admission_source_emergency():
    assert map_admission_source(7) == "Emergency"


# Delivery_Birth: 11, 12, 13, 14, 23, 24
def test_map_admission_source_delivery_11():
    assert map_admission_source(11) == "Delivery_Birth"


def test_map_admission_source_delivery_14():
    assert map_admission_source(14) == "Delivery_Birth"


def test_map_admission_source_delivery_23():
    assert map_admission_source(23) == "Delivery_Birth"


def test_map_admission_source_delivery_24():
    assert map_admission_source(24) == "Delivery_Birth"


# Other: any code not covered above
def test_map_admission_source_other():
    assert map_admission_source(99) == "Other"
