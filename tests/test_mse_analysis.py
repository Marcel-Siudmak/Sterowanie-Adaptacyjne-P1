import unittest
import numpy as np
from pathlib import Path
import tempfile
import sys

# Dodanie katalogu src do sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from mse_analysis import compute_empirical_mse, plot_mse_curve


class TestMSEAnalysis(unittest.TestCase):
    def test_compute_empirical_mse_shape_and_values(self):
        alpha = 0.02
        var_z = 25.0
        N = 200
        H_max = 20
        M = 5

        mse = compute_empirical_mse(alpha, var_z, N, H_max, M)

        self.assertEqual(mse.shape, (H_max,))
        self.assertFalse(np.isnan(mse).any())
        self.assertTrue((mse > 0).all())

    def test_plot_mse_curve_runs(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        H_range = np.arange(1, 21)
        mse = np.linspace(10, 1, 10).tolist() + np.linspace(1, 10, 10).tolist()
        mse = np.array(mse)

        plot_mse_curve(ax, H_range, mse, alpha=0.02, var_z=25.0)
        self.assertGreater(len(ax.lines), 0)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
