"""Unit tests for src.evaluation."""

import numpy as np

from src.evaluation import compute_far_frr, compute_full_metrics, find_eer


def test_compute_far_frr_shapes():
    genuine = [0.9, 0.85, 0.95, 0.8]
    impostor = [0.2, 0.3, 0.1, 0.25]

    thresholds, far_list, frr_list = compute_far_frr(genuine, impostor)

    assert len(thresholds) == len(far_list) == len(frr_list)
    # At threshold 0, everyone is accepted -> FAR should be 1.0, FRR 0.0
    assert np.isclose(far_list[0], 1.0)
    assert np.isclose(frr_list[0], 0.0)


def test_find_eer_returns_reasonable_threshold():
    genuine = np.random.normal(0.9, 0.05, 200)
    impostor = np.random.normal(0.3, 0.05, 200)

    thresholds, far_list, frr_list = compute_far_frr(genuine, impostor)
    eer_threshold, eer_value = find_eer(thresholds, far_list, frr_list)

    assert 0.0 <= eer_threshold <= 1.0
    assert 0.0 <= eer_value <= 1.0


def test_compute_full_metrics_perfect_separation():
    genuine = [0.9, 0.95, 0.99]
    impostor = [0.1, 0.05, 0.01]

    metrics = compute_full_metrics(
        genuine, impostor, threshold=0.5, model_name="TestModel"
    )

    assert metrics["model_name"] == "TestModel"
    assert metrics["accuracy"] == 1.0
    assert metrics["far"] == 0.0
    assert metrics["frr"] == 0.0
