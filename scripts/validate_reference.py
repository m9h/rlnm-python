"""Compare Python with original pinned MATLAB code, using MATLAB or Octave.

The engine is external. This script never calls a Python reimplementation its
MATLAB reference. Inputs include explicit permutations, independent of RNGs.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from rlnm import UPSTREAM_COMMIT
from rlnm.io import sha256
from rlnm.sensitivity import mean_connectivity, signed_overlap
from rlnm.specificity import permutation_order, specificity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--engine", choices=["matlab", "octave"], default="octave")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--upstream-example",
        action="store_true",
        help="Also use the full supplied C2 domain maps instead of toy maps",
    )
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Reference checkout does not match the pinned commit")
    engine = shutil.which(args.engine)
    if engine is None:
        raise RuntimeError(f"{args.engine} is required to execute reference code")
    rng = np.random.default_rng(42)
    maps = rng.normal(size=(18, 29))
    labels = np.repeat(np.arange(3), 6)
    fc = rng.normal(size=(14, 14))
    # Constant target column exercises threshold ties when paired with other fixtures.
    lesion_rows = np.array([[0, 2, 3], [1, 5, 6], [4, 8, 13], [2, 9, 11]])
    oracle_dir = Path(__file__).resolve().parents[1] / "validation"

    def quote(path):
        return "'" + str(path).replace("'", "''") + "'"

    with tempfile.TemporaryDirectory(prefix="rlnm-oracle-") as tmp:
        if args.upstream_example:
            from rlnm.examples import prepare_upstream
            from rlnm.io import load_maps, load_mask, read_manifest

            prepared = prepare_upstream(upstream, Path(tmp) / "prepared")
            rows = read_manifest(prepared / "participants.tsv")
            reference_image, mask = load_mask(prepared / "brain_mask.nii.gz")
            maps = load_maps(rows, reference_image, mask)
            _, labels = np.unique([row["symptom"] for row in rows], return_inverse=True)
        orders = np.stack([permutation_order(len(maps), 51, i) for i in range(83)])
        input_file, output_file = Path(tmp) / "input.mat", Path(tmp) / "output.mat"
        savemat(
            input_file,
            {
                "maps": maps,
                "labels": labels + 1,
                "orders": orders + 1,
                "fc": fc,
                "lesion_rows": lesion_rows + 1,
            },
        )
        expression = (
            f"addpath({quote(oracle_dir)}); rlnm_oracle({quote(input_file)},"
            f"{quote(output_file)},{quote(upstream)});"
        )
        cmd = (
            [engine, "-batch", expression]
            if args.engine == "matlab"
            else [engine, "--no-gui", "--quiet", "--eval", expression]
        )
        subprocess.run(cmd, check=True)
        reference = loadmat(output_file)
        one = specificity(maps, labels, n_permutations=len(orders), orders=orders, sided="one")
        two = specificity(maps, labels, n_permutations=len(orders), orders=orders, sided="two")
        projected = mean_connectivity(fc, lesion_rows)
        results = {
            "regressors": one.regressors,
            "residuals": one.residuals,
            "t": one.t,
            "effect": one.effect,
            "p_one": one.p,
            "q_one": one.q,
            "p_two": two.p,
            "q_two": two.q,
            "projected": projected,
            "overlap": signed_overlap(projected)[None, :],
        }
        errors = {}
        for key, actual in results.items():
            np.testing.assert_allclose(
                actual,
                reference[key],
                rtol=1e-11,
                atol=1e-12,
                err_msg=f"Cross-language mismatch: {key}",
            )
            errors[key] = float(np.max(np.abs(actual - reference[key])))
        report = {
            "status": "passed",
            "engine": args.engine,
            "upstream_commit": commit,
            "fixture": {
                "maps": len(maps),
                "voxels": maps.shape[1],
                "symptoms": len(np.unique(labels)),
                "permutations": 83,
                "upstream_example": args.upstream_example,
            },
            "input_sha256": sha256(input_file),
            "max_absolute_errors": errors,
            "limitations": [
                "No NIfTI I/O comparison in this oracle",
                "No annealing or KS acceptance equivalence claimed",
                "Shared permutation rows; RNG sequences are intentionally different",
                "Not an end-to-end reproduction of the paper",
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
