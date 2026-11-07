import itertools

import numpy as np
import pytest
from scipy.stats import false_discovery_control, ttest_ind

from rlnm.sensitivity import mean_connectivity, sensitivity_from_nulls, signed_overlap
from rlnm.specificity import permutation_p, specificity
from rlnm.stats import bh_fdr, shared_residuals, specific_effect, welch_t


def test_welch_and_fdr_against_scipy():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(7, 19)), rng.normal(size=(11, 19))
    reference = ttest_ind(a, b, axis=0, equal_var=False)
    np.testing.assert_allclose(welch_t(a, b), reference.statistic, atol=1e-12)
    np.testing.assert_allclose(bh_fdr(reference.pvalue), false_discovery_control(reference.pvalue))
    expected = (a.mean(0) - b.mean(0)) / np.sqrt(
        (6 * a.var(0, ddof=1) + 10 * b.var(0, ddof=1)) / 16
    )
    np.testing.assert_allclose(specific_effect(a, b), expected)


def test_residuals_against_design_matrix_lstsq():
    rng = np.random.default_rng(9)
    maps = rng.normal(size=(11, 27))
    labels = np.array(["a"] * 3 + ["b"] * 8)
    regressors, residuals = shared_residuals(maps, labels)
    for i in range(len(maps)):
        others = np.arange(len(maps)) != i
        x = np.stack([maps[(labels == name) & others].mean(0) for name in ("a", "b")]).mean(0)
        design = np.column_stack([np.ones(len(x)), x])
        expected = maps[i] - design @ np.linalg.lstsq(design, maps[i], rcond=None)[0]
        np.testing.assert_allclose(regressors[i], x, atol=1e-14)
        np.testing.assert_allclose(residuals[i], expected, atol=1e-14)
    np.testing.assert_allclose(residuals.mean(1), 0, atol=1e-14)


def test_exact_enumerated_permutation_tail():
    maps = np.array([[0.0, 1], [2, 4], [3, 2], [5, 0], [8, 3], [9, 2]])
    orders = []
    for indices in itertools.combinations(range(6), 3):
        orders.append(list(indices) + [i for i in range(6) if i not in indices])
    orders = np.asarray(orders)
    observed = ttest_ind(maps[:3], maps[3:], equal_var=False).statistic
    null = np.stack(
        [ttest_ind(maps[o[:3]], maps[o[3:]], equal_var=False).statistic for o in orders]
    )
    expected = (1 + (np.abs(null) >= np.abs(observed)).sum(0)) / (len(orders) + 1)
    actual = permutation_p(
        maps, 3, observed, orders=orders, n_permutations=len(orders), sided="two"
    )
    np.testing.assert_array_equal(actual, expected)


def test_worker_and_batch_invariance():
    maps = np.random.default_rng(12).normal(size=(9, 17))
    labels = ["a"] * 4 + ["b"] * 5
    a = specificity(maps, labels, n_permutations=37, seed=18, jobs=1, batch_size=7)
    b = specificity(maps, labels, n_permutations=37, seed=18, jobs=3, batch_size=13)
    np.testing.assert_array_equal(a.p, b.p)
    np.testing.assert_array_equal(a.q, b.q)
    assert np.all(a.p >= 1 / 38)


def test_mean_fc_row_projection_and_nan_semantics():
    fc = np.array([[1.0, 2, np.nan], [3, 4, np.nan], [5, 6, 8]])
    result = mean_connectivity(fc, [np.array([0, 1]), np.array([2])])
    np.testing.assert_array_equal(result, [[2, 3, 0], [5, 6, 8]])
    with pytest.raises(ValueError):
        mean_connectivity(fc, [np.array([0, 0])])


def test_matlab_percentiles_and_lower_tail_tie_precedence():
    # With four samples MATLAB quartiles are 0.5 and 2.5 (Hazen), not 0.75/2.25.
    np.testing.assert_array_equal(signed_overlap([[0, 1, 2, 3]]), [-1, 0, 0, 1])
    np.testing.assert_array_equal(signed_overlap([[0, 1, 2, 3, 4, 5]]), [-1, -1, 0, 0, 1, 1])
    # Upstream assigns -1 after +1; tied cutoffs must not cancel.
    np.testing.assert_array_equal(signed_overlap([[4, 4, 4]]), [-1, -1, -1])
    np.testing.assert_array_equal(signed_overlap([[0, 3], [3, 0]]), [0, 0])


def test_streaming_sensitivity_actual_count_and_ties():
    observed = np.array([0.5, -0.5, 0])
    nulls = [[0.5, -0.1, 0], [0.2, -0.7, 0.1], [-0.9, 0.9, -0.2]]
    result = sensitivity_from_nulls(observed, iter(nulls))
    np.testing.assert_array_equal(result.p, [0.5, 0.5, 0.75])
    assert result.n_nulls == 3
    assert not result.significant[2]
    two = sensitivity_from_nulls(observed, nulls, tail="two-sided")
    np.testing.assert_array_equal(two.p, [0.75, 0.75, 1])


def test_reject_invalid_data():
    for p in ([np.nan], [-0.01], [1.1], []):
        with pytest.raises(ValueError):
            bh_fdr(p)
    with pytest.raises(ValueError, match="two lesions"):
        shared_residuals(np.ones((3, 5)), ["a", "a", "b"])
    with pytest.raises(ValueError):
        specificity([[1, np.nan], [2, 3], [2, 4], [3, 5]], ["a", "a", "b", "b"])
    with pytest.raises(ValueError):
        sensitivity_from_nulls([0.2], [])
