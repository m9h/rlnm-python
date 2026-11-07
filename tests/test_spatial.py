import numpy as np
import pytest

from rlnm.spatial import NullConfig, SpatialNulls


@pytest.mark.parametrize("method", ["reposition", "annealing"])
def test_shape_volume_and_domain_preserved(method):
    mask = np.zeros((7, 8, 9), bool)
    mask[1:-1, 1:-1, 1:-1] = True
    points = [np.array([[2, 2, 3], [2, 3, 3], [3, 3, 3]]), np.array([[4, 5, 6], [4, 5, 5]])]
    lesions = [np.ravel_multi_index(p.T, mask.shape, order="F") for p in points]
    generator = SpatialNulls(lesions, mask, seed=31, config=NullConfig(method=method))
    a, diagnostics = generator.draw(9)
    b, _ = generator.draw(9)
    for original, moved, repeat in zip(points, a, b, strict=True):
        coordinates = np.array(np.unravel_index(moved, mask.shape, order="F")).T
        np.testing.assert_array_equal(coordinates - coordinates[0], original - original[0])
        np.testing.assert_array_equal(moved, repeat)
        assert mask.ravel(order="F")[moved].all()
    if method == "annealing":
        assert diagnostics["ks_p"] > 0.05


def test_lesions_outside_mask_rejected():
    mask = np.zeros((4, 4, 4), bool)
    mask[1:3, 1:3, 1:3] = True
    with pytest.raises(ValueError, match="outside"):
        SpatialNulls([np.array([0])], mask)
