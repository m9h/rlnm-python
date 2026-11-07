"""Functional projection, signed overlap and streamed null testing."""

from dataclasses import dataclass

import numpy as np

from .stats import bh_fdr, matrix


def mean_connectivity(connectome, lesion_indices):
    """Average selected FC rows. This is not correlation of an averaged timeseries.

    Indices are zero-based ROW indices, not whole-volume voxel indices. Geometry
    and index conversion are the responsibility of the spatial I/O layer.
    """
    if connectome.ndim != 2 or min(connectome.shape) < 1:
        raise ValueError("Connectome must be a nonempty 2-D matrix")
    maps = []
    for indices in lesion_indices:
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any((indices < 0) | (indices >= len(connectome)))
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("Lesion indices must be unique, nonempty, valid integer row indices")
        # Sum in bounded blocks to avoid copying a whole large lesion x voxel matrix.
        total = np.zeros(connectome.shape[1], dtype=np.float64)
        count = np.zeros(connectome.shape[1], dtype=np.int64)
        for start in range(0, len(indices), 256):
            block = np.asarray(connectome[indices[start : start + 256]], dtype=float)
            if np.isinf(block).any():
                raise ValueError("Connectome contains infinity")
            valid = ~np.isnan(block)
            total += np.where(valid, block, 0).sum(0)
            count += valid.sum(0)
        maps.append(np.divide(total, count, out=np.zeros_like(total), where=count > 0))
    if not maps:
        raise ValueError("At least one lesion is required")
    return np.stack(maps)


def signed_overlap(maps, tail_fraction=0.25):
    """Upstream percentile-based signed overlap (MATLAB prctile / Hazen).

    Each lesion labels its upper/lower tails; the more frequent sign wins at each
    voxel, ties yield zero. This is not thresholding correlation at +/-0.25.
    """
    maps = matrix(maps)
    if not 0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must lie between zero and 0.5")
    low, high = np.quantile(maps, [tail_fraction, 1 - tail_fraction], axis=1, method="hazen")
    # Upstream assigns +1 first, then -1: the lower tail wins when cutoffs tie.
    negative_mask = maps <= low[:, None]
    positive = ((maps >= high[:, None]) & ~negative_mask).sum(0)
    negative = negative_mask.sum(0)
    return np.where(
        positive > negative,
        positive / len(maps),
        np.where(negative > positive, -negative / len(maps), 0),
    )


@dataclass
class SensitivityResult:
    observed: np.ndarray
    p: np.ndarray
    q: np.ndarray
    significant: np.ndarray
    n_nulls: int
    tail: str


def sensitivity_from_nulls(observed, null_maps, *, fdr=0.005, tail="upstream-sign"):
    """Stream an iterable of null overlap maps; use actual draw count and +1.

    upstream-sign selects a directional tail from the observed sign, as upstream;
    two-sided compares absolute values. Neither is silently substituted for the other.
    """
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 1 or not observed.size or not np.isfinite(observed).all():
        raise ValueError("Observed overlap must be a finite nonempty vector")
    if not np.isfinite(fdr) or not 0 < fdr <= 1 or tail not in ("upstream-sign", "two-sided"):
        raise ValueError("Invalid FDR or sensitivity tail")
    count = np.zeros(observed.shape, dtype=np.int64)
    n_nulls = 0
    for null in null_maps:
        null = np.asarray(null, dtype=float)
        if null.shape != observed.shape or not np.isfinite(null).all():
            raise ValueError("Every null map must be finite and match the observed shape")
        count += (
            (np.abs(null) >= np.abs(observed))
            if tail == "two-sided"
            else np.where(observed >= 0, null >= observed, null <= observed)
        )
        n_nulls += 1
    if n_nulls == 0:
        raise ValueError("At least one null map is required")
    p = (count + 1) / (n_nulls + 1)
    q = bh_fdr(p)
    return SensitivityResult(observed, p, q, (q < fdr) & (observed != 0), n_nulls, tail)
