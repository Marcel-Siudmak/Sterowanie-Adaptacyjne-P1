import matplotlib.pyplot as plt
from pathlib import Path

from time_domain import main as run_time_domain
from mse_analysis import main as run_mse_analysis

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def main():
    print("=" * 70)
    print(" PROJEKT: Estymacja sygnału z trendem za pomocą estymatora SMA (Grupa G)")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Generowanie przebiegów czasowych w dziedzinie czasu (§5.1)...")
    run_time_domain(show_plot=False)

    print("\n[2/2] Obliczanie charakterystyk MSE(H) metodą Monte Carlo (§5.2)...")
    run_mse_analysis(show_plot=False)

    print("\n" + "=" * 70)
    print(f"[SUKCES] Wszystkie wykresy zapisano w folderze: {PLOTS_DIR.name}/")
    print("Otwieranie okien z wykresami...")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
