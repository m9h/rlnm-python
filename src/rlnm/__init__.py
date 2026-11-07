"""Robust lesion network mapping, with explicit geometry and provenance."""

from .sensitivity import mean_connectivity, sensitivity_from_nulls, signed_overlap
from .specificity import specificity
from .stats import bh_fdr, shared_residuals, specific_effect, welch_t

__version__ = "0.1.0"
UPSTREAM_COMMIT = "059ade5fc2a332bb2a246ef7db0bd7c4217446e4"

__all__ = [
    "bh_fdr",
    "mean_connectivity",
    "sensitivity_from_nulls",
    "shared_residuals",
    "signed_overlap",
    "specific_effect",
    "specificity",
    "welch_t",
]
