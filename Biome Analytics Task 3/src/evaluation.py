"""Evaluation metrics for speaker verification systems.

Computes False Acceptance Rate (FAR), False Rejection Rate (FRR), and
the Equal Error Rate (EER) threshold from genuine/impostor similarity
scores.
"""

import numpy as np


def compute_far_frr(genuine_scores, impostor_scores, thresholds=None):
    """Compute FAR and FRR curves across a range of thresholds.

    Parameters
    ----------
    genuine_scores : array-like
        Similarity scores for genuine (matching) attempts.
    impostor_scores : array-like
        Similarity scores for impostor (non-matching) attempts.
    thresholds : array-like, optional
        Thresholds to evaluate. Defaults to 0.0 to 1.0 in steps of 0.01.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(thresholds, far_list, frr_list)``
    """
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)

    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    far_list = []
    frr_list = []
    for t in thresholds:
        far = np.mean(impostor_scores >= t)
        frr = np.mean(genuine_scores < t)
        far_list.append(far)
        frr_list.append(frr)

    return thresholds, np.array(far_list), np.array(frr_list)


def find_eer(thresholds, far_list, frr_list):
    """Find the threshold where FAR and FRR are closest (the EER point).

    Returns
    -------
    tuple[float, float]
        ``(eer_threshold, eer_value)``
    """
    diffs = np.abs(np.array(far_list) - np.array(frr_list))
    idx = np.argmin(diffs)
    eer_threshold = thresholds[idx]
    eer_value = (far_list[idx] + frr_list[idx]) / 2
    return eer_threshold, eer_value


def compute_full_metrics(
    genuine_scores, impostor_scores, threshold, model_name="Model"
):
    """Compute accuracy, FAR, and FRR at a fixed decision threshold.

    Returns
    -------
    dict
        Dictionary with keys ``model_name``, ``accuracy``, ``far``, ``frr``.
    """
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)

    far = np.mean(impostor_scores >= threshold)
    frr = np.mean(genuine_scores < threshold)

    correct_genuine = np.sum(genuine_scores >= threshold)
    correct_impostor = np.sum(impostor_scores < threshold)
    total = len(genuine_scores) + len(impostor_scores)
    accuracy = (correct_genuine + correct_impostor) / total

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
    }
