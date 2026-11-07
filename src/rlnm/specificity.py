"""Selective specificity with deterministic, streaming permutation counts."""

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .stats import bh_fdr, groups, matrix, shared_residuals, specific_effect, welch_t


@dataclass
class SpecificityResult:
    symptoms: list[str]
    counts: np.ndarray
    regressors: np.ndarray
    residuals: np.ndarray
    t: np.ndarray
    effect: np.ndarray
    p: np.ndarray
    q: np.ndarray
    significant: np.ndarray
    specific_maps: np.ndarray


def permutation_order(n, seed, draw):
    """A draw's RNG does not depend on worker count, batching or completion order."""
    return np.random.default_rng(np.random.SeedSequence([seed, draw])).permutation(n)


def _count_chunk(residuals, n_target, observed, draw_ids, seed, sided, orders):
    counts = np.zeros(residuals.shape[1], dtype=np.int64)
    for draw in draw_ids:
        order = permutation_order(len(residuals), seed, draw) if orders is None else orders[draw]
        t = welch_t(residuals[order[:n_target]], residuals[order[n_target:]])
        if sided == "two":
            counts += np.abs(t) >= np.abs(observed)
        else:
            counts += t >= observed
    return counts


def permutation_p(
    residuals,
    n_target,
    observed,
    *,
    n_permutations=10000,
    seed=0,
    jobs=1,
    batch_size=128,
    sided="one",
    orders=None,
):
    """Permute fixed residual maps, matching upstream's inferential procedure.

    This does NOT refit the label-dependent LOLO regression per permutation. It is
    a reproduction option, not a claim that this null is calibrated for every design.
    Optional explicit zero-based permutation rows enable cross-language validation.
    """
    residuals = matrix(residuals)
    observed = np.asarray(observed, dtype=float)
    if observed.shape != (residuals.shape[1],) or not np.isfinite(observed).all():
        raise ValueError("Observed t map must be finite and match the voxel count")
    if not isinstance(n_target, (int, np.integer)) or not 2 <= n_target <= len(residuals) - 2:
        raise ValueError("Both target and contrast need at least two lesions")
    for name, value in [
        ("n_permutations", n_permutations),
        ("jobs", jobs),
        ("batch_size", batch_size),
    ]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0 or sided not in ("one", "two"):
        raise ValueError("Use a nonnegative integer seed and sided='one' or 'two'")
    if orders is not None:
        orders = np.asarray(orders)
        if (
            orders.shape != (n_permutations, len(residuals))
            or not np.issubdtype(orders.dtype, np.integer)
            or not np.all(np.sort(orders, axis=1) == np.arange(len(residuals)))
        ):
            raise ValueError("Every permutation row must contain each lesion index exactly once")
    chunks = [
        range(start, min(start + batch_size, n_permutations))
        for start in range(0, n_permutations, batch_size)
    ]
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=jobs, prefer="threads", return_as="generator")(
            delayed(_count_chunk)(residuals, n_target, observed, chunk, seed, sided, orders)
            for chunk in chunks
        )
        total = sum(results, start=np.zeros(residuals.shape[1], dtype=np.int64))
    return (total + 1) / (n_permutations + 1)


def specificity(
    maps,
    labels,
    *,
    n_permutations=10000,
    seed=0,
    jobs=1,
    batch_size=128,
    sided="one",
    fdr=0.05,
    orders=None,
):
    """One-vs-rest analysis; BH is applied separately to each symptom's domain."""
    maps = matrix(maps)
    names, codes, counts = groups(labels, len(maps))
    if not np.isfinite(fdr) or not 0 < fdr <= 1:
        raise ValueError("fdr must be in (0, 1]")
    regressors, residuals = shared_residuals(maps, labels)
    t, effect, p = [], [], []
    for code, count in enumerate(counts):
        target = codes == code
        statistic = welch_t(residuals[target], residuals[~target])
        t.append(statistic)
        effect.append(specific_effect(residuals[target], residuals[~target]))
        p.append(
            permutation_p(
                residuals,
                count,
                statistic,
                n_permutations=n_permutations,
                seed=seed,
                jobs=jobs,
                batch_size=batch_size,
                sided=sided,
                orders=orders,
            )
        )
    t, effect, p = np.asarray(t), np.asarray(effect), np.asarray(p)
    q = np.stack([bh_fdr(row) for row in p])
    significant = q < fdr
    return SpecificityResult(
        names.tolist(),
        counts,
        regressors,
        residuals,
        t,
        effect,
        p,
        q,
        significant,
        np.where(significant, effect, 0),
    )
