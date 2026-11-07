"""Integer-translation spatial nulls, with optional coverage-frequency matching.

Annealing ports upstream's objective and move rule. SciPy's asymptotic KS test
is not guaranteed identical to MATLAB kstest2. See docs/methods.md.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class NullConfig:
    method: str = "annealing"
    max_placement_trials: int = 1000
    max_candidates: int = 100
    optimization_steps: int = 1000
    target_rmse: float = 0.005
    temperature: float = 1e-4
    cooling: float = 0.997
    ks_alpha: float = 0.05

    def __post_init__(self):
        if self.method not in ("annealing", "reposition"):
            raise ValueError("Null method must be annealing or reposition")
        for value in (self.max_placement_trials, self.max_candidates, self.optimization_steps):
            if not isinstance(value, int) or value < 1:
                raise ValueError("Spatial null iteration limits must be positive integers")
        if (
            not np.isfinite([self.target_rmse, self.temperature, self.cooling, self.ks_alpha]).all()
            or self.target_rmse < 0
            or self.temperature <= 0
            or not 0 < self.cooling <= 1
            or not 0 < self.ks_alpha < 1
        ):
            raise ValueError("Invalid annealing settings")


class SpatialNulls:
    """Shape/orientation preserving lesion translations within a binary mask.

    All indices use zero-based NIfTI array order, flattened with order='F'. The
    frequency criterion matches the distribution of voxel coverage, not specific
    voxel locations and not each pair of lesions' overlap.
    """

    def __init__(self, lesions, brain_mask, *, seed=0, config=None):
        mask = np.asarray(brain_mask)
        if mask.ndim != 3 or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError("Placement mask must be a nonempty binary 3-D array")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.mask = mask.astype(bool).ravel(order="F")
        self.shape = mask.shape
        self.centers = np.flatnonzero(self.mask)
        self.config = config or NullConfig()
        self.seed = seed
        self.lesions, self.offsets = [], []
        frequency = np.zeros(mask.size, dtype=int)
        for lesion in lesions:
            lesion = np.asarray(lesion)
            if (
                lesion.ndim != 1
                or lesion.size == 0
                or not np.issubdtype(lesion.dtype, np.integer)
                or np.any((lesion < 0) | (lesion >= mask.size))
                or len(np.unique(lesion)) != len(lesion)
            ):
                raise ValueError("Lesions must contain unique valid integer voxel indices")
            if not self.mask[lesion].all():
                raise ValueError("Observed lesions extend outside the placement mask")
            coordinates = np.array(np.unravel_index(lesion, self.shape, order="F")).T
            self.offsets.append(coordinates - coordinates[0])
            self.lesions.append(lesion.copy())
            frequency[lesion] += 1
        if not self.lesions:
            raise ValueError("At least one lesion is required")
        self.target = np.sort(frequency[self.centers] / len(self.lesions))

    def _place(self, offsets, rng):
        for _ in range(self.config.max_placement_trials):
            anchor = np.array(np.unravel_index(rng.choice(self.centers), self.shape, order="F"))
            coordinates = offsets + anchor
            if np.any(coordinates < 0) or np.any(coordinates >= np.asarray(self.shape)):
                continue
            indices = np.ravel_multi_index(coordinates.T, self.shape, order="F")
            if self.mask[indices].all():
                return indices
        raise RuntimeError("No valid lesion placement within max_placement_trials")

    def _loss(self, frequency):
        values = np.sort(frequency[self.centers] / len(self.lesions))
        return float(np.sqrt(np.mean((values - self.target) ** 2)))

    def draw(self, draw_id):
        if not isinstance(draw_id, int) or draw_id < 0:
            raise ValueError("draw_id must be a nonnegative integer")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, draw_id]))
        cfg = self.config
        for candidate in range(cfg.max_candidates):
            try:
                placed = [self._place(offsets, rng) for offsets in self.offsets]
            except RuntimeError:
                continue
            frequency = np.zeros(len(self.mask), dtype=int)
            for indices in placed:
                frequency[indices] += 1
            loss = self._loss(frequency)
            initial_loss, temperature, steps = loss, cfg.temperature, 0
            if cfg.method == "annealing":
                for step in range(cfg.optimization_steps):
                    if loss <= cfg.target_rmse:
                        break
                    steps = step + 1
                    index = rng.integers(len(placed))
                    try:
                        proposal = self._place(self.offsets[index], rng)
                    except RuntimeError:
                        temperature *= cfg.cooling
                        continue
                    new_frequency = frequency.copy()
                    new_frequency[placed[index]] -= 1
                    new_frequency[proposal] += 1
                    proposed_loss = self._loss(new_frequency)
                    if proposed_loss < loss or rng.random() < np.exp(
                        -(proposed_loss - loss) / max(temperature, np.finfo(float).eps)
                    ):
                        frequency, loss, placed[index] = new_frequency, proposed_loss, proposal
                    temperature *= cfg.cooling
                ks_p = float(
                    ks_2samp(
                        self.target, frequency[self.centers] / len(placed), method="asymp"
                    ).pvalue
                )
                if not np.isfinite(ks_p) or ks_p <= cfg.ks_alpha:
                    continue
            else:
                ks_p = None
            return placed, {
                "draw": draw_id,
                "candidate": candidate + 1,
                "initial_rmse": initial_loss,
                "rmse": loss,
                "optimization_steps": steps,
                "ks_p": ks_p,
            }
        raise RuntimeError(
            f"Null draw {draw_id} failed after {cfg.max_candidates} candidates; "
            "no failed draw was silently accepted. Review mask and settings."
        )

    def metadata(self):
        return {"seed": self.seed, "rng": "PCG64 SeedSequence([seed, draw])", **asdict(self.config)}
