import json

import numpy as np
import pytest

from rlnm.benchmark import run_benchmark
from rlnm.eigenbrain import connectivity_gradients, nested_benchmark
from rlnm.research_examples import create_research_demo


def _fixture():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(35, 5))
    reference -= reference.mean(0)
    reference, _ = np.linalg.qr(reference)
    subjects = np.stack([reference + 0.03 * rng.normal(size=reference.shape) for _ in range(7)])
    pet = np.stack([s[:, 0] + 0.05 * rng.normal(size=35) for s in subjects])
    return reference, subjects, pet


def test_heldout_target_cannot_change_prediction_baseline_or_selection():
    reference, subjects, pet = _fixture()
    a = nested_benchmark(reference, subjects, pet, components=[1, 3, 5])
    changed = pet.copy()
    changed[2] = np.random.default_rng(5).normal(size=35) * 100
    b = nested_benchmark(reference, subjects, changed, components=[1, 3, 5])
    np.testing.assert_array_equal(a["predictions"][2], b["predictions"][2])
    np.testing.assert_array_equal(a["baselines"][2], b["baselines"][2])
    np.testing.assert_array_equal(a["inner_mse"][2], b["inner_mse"][2])
    assert a["rows"][2]["components"] == b["rows"][2]["components"]
    assert a["rows"][2]["model_mse"] != b["rows"][2]["model_mse"]


def test_orthogonal_alignment_and_analytic_common_map():
    reference, _, _ = _fixture()
    rotation, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(5, 5)))
    subjects = np.stack([reference, -reference, reference @ rotation, reference @ rotation.T])
    target = reference @ np.arange(1, 6)
    result = nested_benchmark(reference, subjects, np.tile(target, (4, 1)), components=[5])
    np.testing.assert_allclose(result["predictions"], result["observed"], atol=1e-13)
    assert all(r["model_r"] == pytest.approx(1) for r in result["rows"])
    assert all(r["deviation_r"] is None for r in result["rows"])


def test_baseline_really_excludes_test_subject():
    reference, subjects, pet = _fixture()
    result = nested_benchmark(reference, subjects, pet, components=[1])
    standardized = (pet - pet.mean(1, keepdims=True)) / pet.std(1, keepdims=True)
    for i, baseline in enumerate(result["baselines"]):
        np.testing.assert_allclose(baseline, standardized[np.arange(len(pet)) != i].mean(0))


def test_invalid_basis_and_outcomes():
    reference, subjects, pet = _fixture()
    with pytest.raises(ValueError, match="Component counts"):
        nested_benchmark(reference, subjects, pet, components=[9])
    with pytest.raises(ValueError, match="constant"):
        nested_benchmark(reference, subjects, np.ones_like(pet), components=[1])
    reference[:, 1] = reference[:, 0]
    with pytest.raises(ValueError, match="rank deficient"):
        nested_benchmark(reference, subjects, pet, components=[3])


def test_brainspace_embedding_and_benchmark(tmp_path):
    pytest.importorskip("brainspace")
    directory = create_research_demo(tmp_path / "demo")
    reference = np.load(directory / "reference.npz")["fc"]
    a = connectivity_gradients(reference, n_components=3, sparsity=0.5, seed=1)
    b = connectivity_gradients(reference, n_components=3, sparsity=0.5, seed=1)
    np.testing.assert_array_equal(a, b)
    output = tmp_path / "out"
    result = run_benchmark(directory / "benchmark.json", output)
    assert result["n_subjects"] == 8 and result["experimental"]
    folds = json.loads((output / "folds.json").read_text())
    for fold in folds:
        assert fold["test_subject_id"] not in fold["training_subject_ids"]
    assert json.loads((output / "provenance.json").read_text())["status"] == "complete"
    assert "data:image/png;base64," in (output / "report.html").read_text()


def test_reference_overlap_and_parcel_order_rejected(tmp_path):
    directory = create_research_demo(tmp_path / "demo")
    path = directory / "reference.npz"
    ref = dict(np.load(path))
    ref["subject_ids"] = np.array(["synthetic:paired-0"])
    np.savez(path, **ref)
    with pytest.raises(ValueError, match="disjoint"):
        run_benchmark(directory / "benchmark.json", tmp_path / "bad-reference")
    ref["subject_ids"] = np.array(["external:1"])
    ref["parcel_ids"] = ref["parcel_ids"][::-1]
    np.savez(path, **ref)
    with pytest.raises(ValueError, match="order"):
        run_benchmark(directory / "benchmark.json", tmp_path / "bad-order")
