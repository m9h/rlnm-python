import json

import nibabel as nib
import numpy as np
import pytest
from scipy.stats import pearsonr

from rlnm.annotate import _annotation_image, _config, parcellate_common, run_annotation
from rlnm.annotation import association, spatial_surrogates, validate_distances
from rlnm.research_examples import create_research_demo


def test_annotation_null_side_tails_and_ties():
    x = np.array([1.0, 2, 4, 9, -1])
    y = np.array([5.0, 0, 2, 1, 3])
    draws = np.stack([y, -y, y[::-1], np.roll(y, 1)])
    r, p, null_r = association(x, y, draws)
    expected = np.array([pearsonr(x, row).statistic for row in draws])
    assert r == pytest.approx(pearsonr(x, y).statistic)
    np.testing.assert_allclose(null_r, expected)
    assert p == (1 + np.count_nonzero(np.abs(expected) >= abs(r) - 1e-12)) / 5
    assert p >= 3 / 5  # Both signed versions of the observed map count as ties.
    assert association(x, x, np.stack([x, -x]))[1] == 1


def test_constant_and_malformed_nulls_fail():
    with pytest.raises(ValueError, match="constant"):
        association(np.arange(5), np.ones(5), np.ones((3, 5)))
    with pytest.raises(ValueError, match="draws, parcels"):
        association(np.arange(5), np.arange(5), np.ones((5, 3)))
    with pytest.raises(ValueError, match="constant"):
        association(np.arange(5), np.arange(5), np.ones((3, 5)))


def test_common_coverage_preserves_zero_and_rejects_missing():
    atlas = np.array([1, 1, 2, 2, 3, 3])
    target = np.arange(6.0)
    a = np.array([0.0, 2, np.nan, 4, np.nan, np.nan])
    b = np.array([1.0, 3, 4, 6, 7, 8])
    values, counts, coverage, keep = parcellate_common(
        target, atlas, np.ones(6, bool), [a, b], [1, 2, 3], 0.75
    )
    np.testing.assert_array_equal(keep, [True, False, False])
    np.testing.assert_array_equal(counts, [2, 1, 0])
    np.testing.assert_allclose(coverage, [1, 0.5, 0])
    assert values[1, 0] == 1  # A genuine zero is included in the parcel average.
    assert np.isnan(values[:, 2]).all()


def test_bad_distance_geometry_fails():
    x = np.arange(20.0)
    d = np.abs(x[:, None] - x)
    validate_distances(d, np.zeros(20, int), 20)
    d[0, 1] = 0
    with pytest.raises(ValueError, match="distances"):
        validate_distances(d, np.zeros(20, int), 20)


def test_resampling_is_explicit_and_outside_fov_is_missing(tmp_path):
    source = nib.Nifti1Image(np.arange(27.0).reshape(3, 3, 3), np.eye(4))
    source.header.set_xyzt_units("mm")
    path = tmp_path / "source.nii.gz"
    nib.save(source, path)
    reference = nib.Nifti1Image(np.ones((5, 3, 3)), np.eye(4))
    reference.header.set_xyzt_units("mm")
    spec = {"id": "a", "map": path.name, "resample": False, "zero_is_missing": False}
    with pytest.raises(ValueError, match="Grid mismatch"):
        _annotation_image(spec, tmp_path, tmp_path, reference, [], [])
    spec["resample"] = True
    transforms = []
    result = _annotation_image(spec, tmp_path, tmp_path, reference, [], transforms)
    np.testing.assert_allclose(result[:3], source.get_fdata())
    assert np.isnan(result[3:]).all()
    assert result[0, 0, 0] == 0
    assert len(transforms) == 1
    spec["zero_is_missing"] = True
    result = _annotation_image(spec, tmp_path, tmp_path, reference, [], [])
    assert np.isnan(result[0, 0, 0])


def test_annotation_config_rejects_ambiguous_or_surface_query():
    cfg = {
        "map": "x",
        "mask": "x",
        "atlas": "x",
        "distances": "x",
        "space": "MNI152",
        "annotations": [{"id": "a", "citation": "x", "map": "x", "neuromaps": {}}],
    }
    with pytest.raises(ValueError, match="exactly one"):
        _config(cfg)
    cfg["annotations"] = [
        {
            "id": "a",
            "citation": "x",
            "neuromaps": {"source": "a", "desc": "b", "space": "fsaverage", "res": "164k"},
        }
    ]
    with pytest.raises(ValueError, match="MNI152"):
        _config(cfg)


def test_spatial_backend_reproducibility_and_diagnostics(tmp_path):
    pytest.importorskip("brainsmash")
    directory = create_research_demo(tmp_path / "demo")
    data = np.load(directory / "distances.npz")
    y = np.sin(np.arange(60) / 10)
    a, qa = spatial_surrogates(y, data["distances"], data["blocks"], n_nulls=9, seed=3)
    b, qb = spatial_surrogates(y, data["distances"], data["blocks"], n_nulls=9, seed=3)
    np.testing.assert_array_equal(a, b)
    for block in np.unique(data["blocks"]):
        selected = data["blocks"] == block
        np.testing.assert_allclose(a[:, selected].mean(1), y[selected].mean(), atol=1e-12)
    assert qa == qb
    assert len(qa) == 2 and qa[0]["seed"] != qa[1]["seed"]
    assert len(qa[0]["distance"]) == len(qa[0]["empirical"])


def test_annotation_pipeline_and_failure_provenance(tmp_path):
    pytest.importorskip("brainsmash")
    directory = create_research_demo(tmp_path / "demo")
    config_path = directory / "annotate.json"
    config = json.loads(config_path.read_text())
    config["n_nulls"] = 19
    config_path.write_text(json.dumps(config))
    output = tmp_path / "out"
    result = run_annotation(config_path, output)
    assert result["n_parcels"] == 60
    assert result["results"][0]["r"] > 0.9
    assert all(0 < row["p_spatial"] <= row["q_bh"] <= 1 for row in result["results"])
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["status"] == "complete"
    assert len(provenance["inputs"]) == 7
    assert "data:image/png;base64," in (output / "report.html").read_text()
    with pytest.raises(ValueError, match="empty"):
        run_annotation(config_path, output)
    # Failed spatial domain validation must not leave a successful-looking run.
    data = dict(np.load(directory / "distances.npz"))
    data["parcel_ids"] = data["parcel_ids"] + 100
    np.savez(directory / "distances.npz", **data)
    with pytest.raises(ValueError, match="exactly match"):
        run_annotation(config_path, tmp_path / "failed")
    assert json.loads((tmp_path / "failed/provenance.json").read_text())["status"] == "failed"
