"""Small dependency-free metrics needed by both MVE lanes."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Sequence


class MetricError(ValueError):
    """Raised when a metric is undefined for the supplied observations."""


def _finite(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise MetricError(f"{name} must contain finite observations")
    return result


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average_rank
        cursor = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_values = _finite(left, "left")
    right_values = _finite(right, "right")
    if len(left_values) != len(right_values) or len(left_values) < 2:
        raise MetricError("Spearman correlation requires equal lengths >= 2")
    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    left_mean = fmean(left_ranks)
    right_mean = fmean(right_ranks)
    numerator = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks, strict=True)
    )
    denominator = math.sqrt(
        sum((rank - left_mean) ** 2 for rank in left_ranks)
        * sum((rank - right_mean) ** 2 for rank in right_ranks)
    )
    if denominator == 0:
        raise MetricError("Spearman correlation is undefined for a constant input")
    return numerator / denominator


def binary_auroc(labels: Sequence[bool | int], scores: Sequence[float]) -> float:
    score_values = _finite(scores, "scores")
    if len(labels) != len(score_values):
        raise MetricError("labels and scores must have equal lengths")
    normalized_labels: list[bool] = []
    for label in labels:
        if isinstance(label, bool):
            normalized_labels.append(label)
        elif isinstance(label, int) and label in (0, 1):
            normalized_labels.append(bool(label))
        else:
            raise MetricError('labels must contain only bool or integer 0/1 values')
    positives = sum(normalized_labels)
    negatives = len(normalized_labels) - positives
    if positives == 0 or negatives == 0:
        raise MetricError("AUROC requires both positive and negative labels")
    ranks = _average_ranks(score_values)
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, normalized_labels, strict=True) if label
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)
