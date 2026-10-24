import unittest
import numpy as np
from pathlib import Path
import tempfile
import sys

# Dodanie katalogu src do sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from time_domain import simulate_time_domain, plot_time_domain


class TestTimeDomain(unittest.TestCase):
    def test_simulation_output_shapes_and_keys(self):
        N = 100
        alpha = 0.02
        var_z = 25.0
        h_values = [2, 10, 30]

        signal_star, measurements, estimates = simulate_time_domain(
            alpha=alpha,
            var_z=var_z,
            N=N,
            h_values=h_values,
            seed=42,
        )

        self.assertEqual(signal_star.shape, (N,))
        self.assertEqual(measurements.shape, (N,))
        self.assertEqual(set(estimates.keys()), set(h_values))

        for H in h_values:
            self.assertEqual(estimates[H].shape, (N,))
            # Pierwsze H-1 próbek to NaN
            self.assertTrue(np.isnan(estimates[H][: H - 1]).all())
            # Od H-1 do końca nie ma NaN
            self.assertFalse(np.isnan(estimates[H][H - 1 :]).any())

    def test_reproducibility_with_seed(self):
        _, m1, _ = simulate_time_domain(0.02, 25.0, 50, [2], seed=123)
        _, m2, _ = simulate_time_domain(0.02, 25.0, 50, [2], seed=123)
        np.testing.assert_array_equal(m1, m2)

    def test_plot_generation_and_saving(self):
        import matplotlib

        matplotlib.use("Agg")

        signal_star, measurements, estimates = simulate_time_domain(
            alpha=0.02,
            var_z=25.0,
            N=50,
            h_values=[2, 10],
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_time_domain.png"
            plot_time_domain(
                signal_star=signal_star,
                measurements=measurements,
                estimates=estimates,
                alpha=0.02,
                var_z=25.0,
                n_view=40,
                save_path=save_path,
            )
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
