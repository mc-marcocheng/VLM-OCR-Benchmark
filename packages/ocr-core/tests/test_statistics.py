"""Tests for ocr_core.statistics."""

from __future__ import annotations

import pytest

from ocr_core.statistics import bootstrap_ci, paired_bootstrap_test, summarise


class TestSummarise:
    def test_empty(self):
        ss = summarise([])
        assert ss.n == 0
        assert ss.mean == 0
        assert ss.std == 0

    def test_single_value(self):
        ss = summarise([42.0])
        assert ss.n == 1
        assert ss.mean == 42.0
        assert ss.std == 0.0
        assert ss.ci_lower == 42.0
        assert ss.ci_upper == 42.0
        assert ss.min == 42.0
        assert ss.max == 42.0

    def test_two_values(self):
        ss = summarise([10.0, 20.0])
        assert ss.n == 2
        assert ss.mean == pytest.approx(15.0)
        assert ss.median == pytest.approx(15.0)
        assert ss.min == 10.0
        assert ss.max == 20.0
        assert ss.std > 0

    def test_identical_values(self):
        ss = summarise([5.0, 5.0, 5.0, 5.0])
        assert ss.mean == 5.0
        assert ss.std == 0.0
        assert ss.ci_lower == pytest.approx(5.0)
        assert ss.ci_upper == pytest.approx(5.0)

    def test_large_sample(self):
        import numpy as np

        rng = np.random.default_rng(123)
        vals = rng.normal(100, 10, size=1000).tolist()
        ss = summarise(vals)
        assert ss.n == 1000
        assert ss.mean == pytest.approx(100, abs=1)
        assert ss.ci_lower < ss.mean
        assert ss.ci_upper > ss.mean


class TestBootstrapCI:
    def test_constant_values(self):
        import numpy as np

        arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        lo, hi = bootstrap_ci(arr)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_ci_contains_mean(self):
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = bootstrap_ci(arr, confidence=0.95)
        mean = arr.mean()
        assert lo <= mean <= hi

    def test_wider_ci_at_higher_confidence(self):
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        lo_90, hi_90 = bootstrap_ci(arr, confidence=0.90)
        lo_99, hi_99 = bootstrap_ci(arr, confidence=0.99)
        assert (hi_99 - lo_99) >= (hi_90 - lo_90) - 0.01  # small tolerance

    def test_median_statistic(self):
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0, 100.0])
        lo, hi = bootstrap_ci(arr, statistic="median")
        assert lo <= float(np.median(arr))


class TestPairedBootstrapTest:
    def test_identical_scores(self):
        a = [0.9, 0.8, 0.7, 0.85, 0.95]
        p = paired_bootstrap_test(a, a)
        assert p == pytest.approx(1.0)

    def test_very_different_scores(self):
        a = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        b = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        p = paired_bootstrap_test(a, b)
        assert p < 0.05

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            paired_bootstrap_test([1, 2, 3], [1, 2])

    def test_p_value_between_0_and_1(self):
        a = [0.5, 0.6, 0.7, 0.8]
        b = [0.55, 0.65, 0.72, 0.78]
        p = paired_bootstrap_test(a, b)
        assert 0.0 <= p <= 1.0
