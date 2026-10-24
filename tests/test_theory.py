import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from theory import (
    theoretical_bias,
    theoretical_bias_squared,
    theoretical_variance,
    theoretical_mse,
    theoretical_h_opt,
)


class TestTheory(unittest.TestCase):
    def test_bias_at_h1(self):
        # Dla H = 1 obciążenie powinno wynosić dokładnie 0
        self.assertEqual(theoretical_bias(alpha=0.02, H=1), 0.0)
        self.assertEqual(theoretical_bias_squared(alpha=0.02, H=1), 0.0)

    def test_variance_at_h1(self):
        # Dla H = 1 wariancja to po prostu wariancja szumu
        var_z = 25.0
        self.assertEqual(theoretical_variance(var_z=var_z, H=1), var_z)

    def test_mse_decomposition_consistency(self):
        alpha = 0.02
        var_z = 25.0
        H = np.array([1, 10, 50, 100])

        b2 = theoretical_bias_squared(alpha, H)
        v = theoretical_variance(var_z, H)
        mse = theoretical_mse(alpha, var_z, H)

        np.testing.assert_allclose(mse, b2 + v)

    def test_optimal_h_known_case(self):
        # Dla alpha = 0.02, var_z = 25.0:
        # H_opt ≈ (2 * 25 / 0.02^2)^(1/3) = (50 / 0.0004)^(1/3) = 125000^(1/3) = 50
        h_opt = theoretical_h_opt(alpha=0.02, var_z=25.0)
        self.assertEqual(h_opt, 50)

        # MSE(50) powinno być mniejsze niż MSE(49) i MSE(51)
        mse_50 = theoretical_mse(0.02, 25.0, 50)
        mse_49 = theoretical_mse(0.02, 25.0, 49)
        mse_51 = theoretical_mse(0.02, 25.0, 51)

        self.assertLess(mse_50, mse_49)
        self.assertLess(mse_50, mse_51)


if __name__ == "__main__":
    unittest.main()
