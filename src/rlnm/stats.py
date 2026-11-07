"""Array calculations ported from pBFSLab/rLNM (see NOTICE).

Columns are voxels in an explicitly chosen analysis domain, rows are lesions.
The zero-variance convention follows upstream: zero t/effect, even if means differ.
"""

import numpy as np


def matrix(values, name="maps"):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty, finite 2-D array")
    return values


def groups(labels, n):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("One symptom label is required per lesion")
    if any(not str(v).strip() for v in labels):
        raise ValueError("Symptom labels must be nonempty")
    names, codes = np.unique(labels.astype(str), return_inverse=True)
    counts = np.bincount(codes)
    if len(names) < 2 or np.any(counts < 2):
        raise ValueError("At least two symptoms with at least two lesions each are required")
    return names, codes, counts


def bh_fdr(p_values):
    """BH correction over all supplied finite p values; invalid inputs are errors."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p values must be finite and between zero and one")
    flat = p.ravel()
    order = np.argsort(flat, kind="stable")
    adjusted = flat[order] * len(flat) / np.arange(1, len(flat) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty_like(flat)
    result[order] = adjusted
    return result.reshape(p.shape)


def _group_moments(a, b):
    a, b = matrix(a, "group_a"), matrix(b, "group_b")
    if min(len(a), len(b)) < 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Groups need >=2 rows and the same voxel count")
    return a, b, a.mean(0) - b.mean(0), a.var(0, ddof=1), b.var(0, ddof=1)


def welch_t(a, b):
    """Voxelwise Welch statistic; target minus contrast."""
    a, b, difference, va, vb = _group_moments(a, b)
    denominator = np.sqrt(va / len(a) + vb / len(b))
    return np.divide(difference, denominator, out=np.zeros_like(difference), where=denominator > 0)


def specific_effect(a, b):
    """Signed standardized mean difference, using the df-weighted pooled SD."""
    a, b, difference, va, vb = _group_moments(a, b)
    denominator = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return np.divide(difference, denominator, out=np.zeros_like(difference), where=denominator > 0)


def shared_residuals(maps, labels):
    """Return signed LOLO regressors and OLS residuals, with an intercept.

    Each symptom receives equal weight in the shared regressor. Within the lesion's
    own symptom the lesion itself is excluded. Regression is across domain voxels.
    Sensitivity masks do not enter this regression (matches the pinned upstream).
    """
    maps = matrix(maps)
    _, codes, counts = groups(labels, len(maps))
    sums = np.stack([maps[codes == g].sum(0) for g in range(len(counts))])
    means = sums / counts[:, None]
    shared = means.sum(0)
    regressors = np.empty_like(maps)
    for i, code in enumerate(codes):
        regressors[i] = (shared - means[code] + (sums[code] - maps[i]) / (counts[code] - 1)) / len(
            counts
        )
    x = regressors - regressors.mean(1, keepdims=True)
    y = maps - maps.mean(1, keepdims=True)
    xx = np.sum(x * x, axis=1)
    slopes = np.divide(np.sum(x * y, axis=1), xx, out=np.zeros(len(xx)), where=xx > 0)
    return regressors, y - slopes[:, None] * x
