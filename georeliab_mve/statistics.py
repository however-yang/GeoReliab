"""Dependency-free scene-block inference utilities.

All resampling operates on scene aggregates. Pixel-level pseudo-replication is
not accepted by these APIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from statistics import fmean, stdev
from typing import Mapping, Sequence


class StatisticsError(ValueError):
    """Raised when a statistical comparison is not valid."""


SceneValues = Mapping[str, float | Sequence[float]]


def _scene_means(values: SceneValues, name: str) -> dict[str, float]:
    if not values:
        raise StatisticsError(f"{name} must contain at least one scene")
    result: dict[str, float] = {}
    for scene, raw in values.items():
        if not isinstance(scene, str) or not scene:
            raise StatisticsError(f"{name} contains an invalid scene id")
        if isinstance(raw, bool):
            raise StatisticsError(f"{name}[{scene!r}] must be numeric")
        if isinstance(raw, (int, float)):
            scene_values = [float(raw)]
        else:
            if isinstance(raw, (str, bytes)):
                raise StatisticsError(
                    f"{name}[{scene!r}] must contain numeric observations"
                )
            try:
                observations = list(raw)
            except TypeError as exc:
                raise StatisticsError(
                    f"{name}[{scene!r}] must contain numeric observations"
                ) from exc
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in observations
            ):
                raise StatisticsError(
                    f"{name}[{scene!r}] must contain numeric observations"
                )
            scene_values = [float(value) for value in observations]
        if not scene_values or any(not math.isfinite(value) for value in scene_values):
            raise StatisticsError(
                f"{name}[{scene!r}] must contain finite observations"
            )
        result[scene] = fmean(scene_values)
    return result


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    effect: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_scenes: int
    n_resamples: int
    seed: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def paired_scene_bootstrap(
    baseline: SceneValues,
    treatment: SceneValues,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Estimate ``treatment - baseline`` using paired scene-block bootstrap."""

    if n_resamples < 1:
        raise StatisticsError("n_resamples must be positive")
    if not 0 < confidence < 1:
        raise StatisticsError("confidence must be in (0, 1)")
    baseline_means = _scene_means(baseline, "baseline")
    treatment_means = _scene_means(treatment, "treatment")
    if set(baseline_means) != set(treatment_means):
        missing_treatment = sorted(set(baseline_means) - set(treatment_means))
        missing_baseline = sorted(set(treatment_means) - set(baseline_means))
        raise StatisticsError(
            "paired comparisons require identical scene ids; "
            f"missing treatment={missing_treatment}, "
            f"missing baseline={missing_baseline}"
        )
    scenes = sorted(baseline_means)
    differences = [
        treatment_means[scene] - baseline_means[scene] for scene in scenes
    ]
    generator = random.Random(seed)
    sample_size = len(differences)
    estimates = [
        fmean(differences[generator.randrange(sample_size)] for _ in scenes)
        for _ in range(n_resamples)
    ]
    estimates.sort()
    tail = (1 - confidence) / 2
    lower_index = max(0, math.floor(tail * (n_resamples - 1)))
    upper_index = min(
        n_resamples - 1, math.ceil((1 - tail) * (n_resamples - 1))
    )
    return BootstrapResult(
        effect=fmean(differences),
        ci_lower=estimates[lower_index],
        ci_upper=estimates[upper_index],
        confidence=confidence,
        n_scenes=sample_size,
        n_resamples=n_resamples,
        seed=seed,
    )


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 200
    epsilon = 3e-14
    fp_min = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fp_min:
        d = fp_min
    d = 1.0 / d
    h = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        numerator = (
            iteration
            * (b - iteration)
            * x
            / ((qam + m2) * (a + m2))
        )
        d = 1.0 + numerator * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + numerator / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        h *= d * c
        numerator = (
            -(a + iteration)
            * (qab + iteration)
            * x
            / ((a + m2) * (qap + m2))
        )
        d = 1.0 + numerator * d
        if abs(d) < fp_min:
            d = fp_min
        c = 1.0 + numerator / c
        if abs(c) < fp_min:
            c = fp_min
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    raise StatisticsError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0 <= x <= 1:
        raise StatisticsError("incomplete-beta x must be in [0, 1]")
    if x in (0.0, 1.0):
        return x
    log_prefix = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    prefix = math.exp(log_prefix)
    if x < (a + 1) / (a + b + 2):
        return prefix * _beta_continued_fraction(a, b, x) / a
    return 1 - prefix * _beta_continued_fraction(b, a, 1 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise StatisticsError("degrees_of_freedom must be positive")
    if value == 0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    beta = _regularized_incomplete_beta(
        degrees_of_freedom / 2.0, 0.5, x
    )
    return 1 - 0.5 * beta if value > 0 else 0.5 * beta


@dataclass(frozen=True, slots=True)
class TOSTResult:
    mean_difference: float
    lower_bound: float
    upper_bound: float
    p_lower: float
    p_upper: float
    alpha: float
    equivalent: bool
    n_scenes: int

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def tost_equivalence(
    differences: SceneValues,
    *,
    margin: float = 0.02,
    alpha: float = 0.05,
) -> TOSTResult:
    """Run a paired scene-level two one-sided tests equivalence test."""

    if margin <= 0 or not math.isfinite(margin):
        raise StatisticsError("margin must be finite and positive")
    if not 0 < alpha < 0.5:
        raise StatisticsError("alpha must be in (0, 0.5)")
    means = _scene_means(differences, "differences")
    values = list(means.values())
    if len(values) < 2:
        raise StatisticsError("TOST requires at least two scenes")
    mean_difference = fmean(values)
    standard_error = stdev(values) / math.sqrt(len(values))
    lower_bound = -margin
    upper_bound = margin
    if standard_error == 0:
        strictly_inside = lower_bound < mean_difference < upper_bound
        p_lower = 0.0 if strictly_inside else 1.0
        p_upper = 0.0 if strictly_inside else 1.0
    else:
        degrees_of_freedom = len(values) - 1
        lower_t = (mean_difference - lower_bound) / standard_error
        upper_t = (mean_difference - upper_bound) / standard_error
        p_lower = 1 - _student_t_cdf(lower_t, degrees_of_freedom)
        p_upper = _student_t_cdf(upper_t, degrees_of_freedom)
    return TOSTResult(
        mean_difference=mean_difference,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        p_lower=p_lower,
        p_upper=p_upper,
        alpha=alpha,
        equivalent=p_lower < alpha and p_upper < alpha,
        n_scenes=len(values),
    )


@dataclass(frozen=True, slots=True)
class HolmResult:
    raw_p: float
    adjusted_p: float
    rejected: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def holm_adjust(
    p_values: Mapping[str, float], *, alpha: float = 0.05
) -> dict[str, HolmResult]:
    """Apply Holm's step-down family-wise error correction."""

    if not p_values:
        raise StatisticsError("p_values must not be empty")
    if not 0 < alpha < 1:
        raise StatisticsError("alpha must be in (0, 1)")
    for name, p_value in p_values.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(p_value, bool)
            or not isinstance(p_value, (int, float))
            or not math.isfinite(float(p_value))
            or not 0 <= float(p_value) <= 1
        ):
            raise StatisticsError(f"invalid p-value for {name!r}: {p_value!r}")
    ordered = sorted(
        ((name, float(p_value)) for name, p_value in p_values.items()),
        key=lambda pair: (pair[1], pair[0]),
    )
    total = len(ordered)
    adjusted: dict[str, HolmResult] = {}
    running_max = 0.0
    for rank, (name, raw_p) in enumerate(ordered):
        adjusted_p = min(1.0, max(running_max, (total - rank) * raw_p))
        running_max = adjusted_p
        adjusted[name] = HolmResult(
            raw_p=raw_p,
            adjusted_p=adjusted_p,
            rejected=adjusted_p <= alpha,
        )
    return adjusted

