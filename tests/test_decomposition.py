import unittest
from pathlib import Path
import tempfile
import sys
import matplotlib

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from decomposition import plot_decomposition


class TestDecomposition(unittest.TestCase):
    def test_plot_decomposition_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_decomposition.png"
            plot_decomposition(
                alpha=0.02,
                var_z=25.0,
                H_max=50,
                save_path=save_path,
                show_plot=False,
            )
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
