from __future__ import annotations

import pytest

from georeliab_mve.metrics import MetricError, binary_auroc, spearman_correlation
from georeliab_mve.splits import LeakageError, validate_scene_disjoint
from georeliab_mve.statistics import (
    StatisticsError,
    holm_adjust,
    paired_scene_bootstrap,
    tost_equivalence,
)


def clean_splits():
    return {
        'dev': ['d1', 'd2'],
        'reference-token': ['r1'],
        'calibration': ['c1'],
        'test': ['t1', 't2'],
    }


def test_scene_disjoint_protocol_accepts_clean_partition():
    result = validate_scene_disjoint(clean_splits())
    assert result.total_scenes == 6


def test_scene_disjoint_protocol_rejects_leakage():
    splits = clean_splits()
    splits['test'].append('d1')
    with pytest.raises(LeakageError, match='scene leakage'):
        validate_scene_disjoint(splits)


def test_scene_block_bootstrap_uses_scene_aggregates():
    baseline = {'s1': [1.0, 3.0], 's2': [2.0, 4.0], 's3': [3.0, 5.0]}
    treatment = {'s1': [2.0, 4.0], 's2': [3.0, 5.0], 's3': [4.0, 6.0]}
    result = paired_scene_bootstrap(
        baseline, treatment, n_resamples=10_000, seed=7
    )
    assert result.effect == pytest.approx(1.0)
    assert result.ci_lower == pytest.approx(1.0)
    assert result.ci_upper == pytest.approx(1.0)
    assert result.n_scenes == 3


def test_scene_block_bootstrap_rejects_unpaired_scenes():
    with pytest.raises(StatisticsError, match='identical scene ids'):
        paired_scene_bootstrap({'s1': 1.0}, {'s2': 1.0})


def test_tost_equivalence_accepts_inside_margin_and_rejects_outside():
    inside = tost_equivalence({'s1': 0.0, 's2': 0.0, 's3': 0.0})
    outside = tost_equivalence({'s1': 0.03, 's2': 0.03, 's3': 0.03})
    assert inside.equivalent is True
    assert outside.equivalent is False


def test_holm_step_down_adjustment():
    result = holm_adjust({'a': 0.01, 'b': 0.03, 'c': 0.20})
    assert result['a'].adjusted_p == pytest.approx(0.03)
    assert result['a'].rejected is True
    assert result['b'].adjusted_p == pytest.approx(0.06)
    assert result['b'].rejected is False


def test_metrics_handle_ties_and_perfect_order():
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert binary_auroc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.0


@pytest.mark.parametrize('labels', ([0, -1], [0, 2], [0, 0.5], ['0', '1']))
def test_auroc_rejects_non_binary_labels(labels):
    with pytest.raises(MetricError, match='only bool or integer 0/1'):
        binary_auroc(labels, [0.1, 0.9])



@pytest.mark.parametrize("bad_observations", ([True], [False], ["1.0"], "1.0"))
def test_scene_statistics_reject_non_numeric_sequences(bad_observations):
    with pytest.raises(StatisticsError, match="numeric observations"):
        paired_scene_bootstrap(
            {"s1": bad_observations},
            {"s1": [1.0]},
            n_resamples=10,
        )
