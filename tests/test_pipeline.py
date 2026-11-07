import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from rlnm.examples import create_demo
from rlnm.io import load_image, read_manifest, same_grid, write_map
from rlnm.pipeline import run


def test_roundtrip_geometry_and_voxel_order(tmp_path):
    affine = np.array([[-2.0, 0, 0, 9], [0, 3, 0, -12], [0, 0, 4, 5], [0, 0, 0, 1]])
    reference = nib.Nifti1Image(np.zeros((3, 4, 5)), affine)
    indices = np.array([0, 5, 27, 59])
    values = np.array([3.0, -2, 7, 11])
    path = tmp_path / "out.nii.gz"
    write_map(path, values, reference, indices)
    result = load_image(path)
    np.testing.assert_array_equal(result.get_fdata().ravel(order="F")[indices], values)
    np.testing.assert_allclose(result.affine, affine)


@pytest.mark.parametrize("analysis", ["specificity", "sensitivity"])
def test_end_to_end(tmp_path, analysis):
    demo = create_demo(tmp_path / "input")
    path = demo / f"{analysis}.json"
    config = json.loads(path.read_text())
    config["n_permutations"] = 19
    path.write_text(json.dumps(config))
    out = tmp_path / "result"
    summary = run(path, out)
    provenance = json.loads((out / "provenance.json").read_text())
    assert summary["n_permutations"] == 19
    assert provenance["status"] == "complete"
    assert all(len(item["sha256"]) == 64 for item in provenance["inputs"])
    assert "data:image/png;base64," in (out / "report.html").read_text()
    for image in out.glob("*_q.nii.gz"):
        values = nib.load(image).get_fdata()
        assert np.all((values >= 0) & (values <= 1))
    with pytest.raises(ValueError, match="empty"):
        run(path, out)


def test_affine_of_every_subject_is_checked(tmp_path):
    demo = create_demo(tmp_path / "input")
    path = demo / "group-0_sample-3.nii.gz"
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 3] += 2
    replacement = nib.Nifti1Image(image.get_fdata(), affine)
    replacement.header.set_xyzt_units("mm")
    nib.save(replacement, path)
    out = tmp_path / "result"
    with pytest.raises(ValueError, match="Grid mismatch"):
        run(demo / "specificity.json", out)
    assert json.loads((out / "provenance.json").read_text())["status"] == "failed"


def test_manifest_duplicate_and_spaces(tmp_path):
    path = tmp_path / "image with spaces.nii.gz"
    path.touch()
    manifest = tmp_path / "subjects.tsv"
    manifest.write_text("participant_id\tsymptom\tmap\np1\ta\timage with spaces.nii.gz\n")
    assert Path(read_manifest(manifest)[0]["map"]) == path
    with manifest.open("a") as handle:
        handle.write("p2\tb\timage with spaces.nii.gz\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_unknown_config_rejected_before_creating_output(tmp_path):
    demo = create_demo(tmp_path / "input")
    path = demo / "specificity.json"
    config = json.loads(path.read_text())
    config["num_permutations"] = 1000
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unknown"):
        run(path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_conflicting_forms_and_units(tmp_path):
    image = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 1
    image.set_qform(shifted, 1)
    path = tmp_path / "bad.nii.gz"
    nib.save(image, path)
    with pytest.raises(ValueError, match="Conflicting"):
        load_image(path)
    a = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
    b = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
    a.header.set_xyzt_units("mm")
    b.header.set_xyzt_units("meter")
    with pytest.raises(ValueError, match="unit"):
        same_grid(a, b)
