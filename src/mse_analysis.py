import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from signal_generator import generate_signal, generate_measurements
from estimator import estimate_sma

# STAŁE KONFIGURACYJNE
N = 1000       # ilość punktów sygnałów (zgodnie z README N >= 1000)
H_max = 100    # maksymalna wielkość okna H
M = 100        # liczba powtórzeń dla uśrednienia wyniku
ALPHA = [0.02, 0.15]  # wartości alpha dla sygnału
VAR_Z = [1.0, 25.0]   # wariancja dla szumu
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def compute_empirical_mse(
    alpha: float,
    var_z: float,
    N: int,
    H_max: int,
    M: int,
) -> np.ndarray:
    """
    Oblicza empiryczny błąd średniokwadratowy (MSE) dla horyzontów H in [1, H_max]
    przy użyciu metody Monte Carlo (M niezależnych realizacji).
    """
    mse_accumulated = np.zeros(H_max, dtype=float)

    # 1. Stały sygnał wzorcowy theta_k* dla wszystkich prób
    signal = generate_signal(alpha, N)

    # 2. Pętla uśredniająca Monte Carlo (M niezależnych realizacji)
    for _ in range(M):
        measurements = generate_measurements(signal, var_z)

        # 3. Badanie estymatora dla każdego H ze zbioru {1, 2, ..., H_max}
        for H in range(1, H_max + 1):
            theta_hat = estimate_sma(measurements, H)

            # Odcięcie stanu nieustalonego: porównujemy wyłącznie próbki dla k >= H_max
            squared_errors = (theta_hat[H_max:] - signal[H_max:]) ** 2
            run_mse = np.mean(squared_errors)

            # Akumulacja błędu pod indeksem H - 1
            mse_accumulated[H - 1] += run_mse

    # 4. Średnia z M realizacji
    return mse_accumulated / M


def plot_mse_curve(
    ax: plt.Axes,
    H_range: np.ndarray,
    mse: np.ndarray,
    alpha: float,
    var_z: float,
) -> None:
    """
    Rysuje pojedynczą krzywą MSE(H) z zaznaczonym punktem minimum.
    """
    ax.plot(H_range, mse, label="Empiryczne MSE", color="tab:blue")

    # Wyznaczenie i zaznaczenie punktu minimum (optymalnego H)
    min_idx = np.argmin(mse)
    H_opt = H_range[min_idx]
    min_mse = mse[min_idx]

    ax.plot(H_opt, min_mse, marker="o", markersize=6, color="red")
    ax.annotate(
        f"$H_{{opt}}={H_opt}$\nMSE={min_mse:.2f}",
        xy=(H_opt, min_mse),
        xytext=(10, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        fontsize=9,
    )

    ax.set_title(f"α = {alpha}, var_z = {var_z}")
    ax.set_xlabel("Horyzont H")
    ax.set_ylabel("MSE")
    ax.grid(True)
    ax.legend()


def main(show_plot: bool = True):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(ALPHA), len(VAR_Z), figsize=(12, 8), squeeze=False)
    H_range = np.arange(1, H_max + 1)

    for i, alpha in enumerate(ALPHA):
        for j, var_z in enumerate(VAR_Z):
            ax = axes[i, j]
            mse = compute_empirical_mse(alpha, var_z, N, H_max, M)

            # 1. Rysowanie na wspólnym oknie siatki podwykresów
            plot_mse_curve(ax, H_range, mse, alpha, var_z)

            # 2. Zapis pojedynczego wykresu do osobnego pliku graficznego
            fig_single, ax_single = plt.subplots(figsize=(7, 5))
            plot_mse_curve(ax_single, H_range, mse, alpha, var_z)
            fig_single.tight_layout()

            single_path = PLOTS_DIR / f"mse_alpha_{alpha}_var_{var_z}.png"
            fig_single.savefig(single_path, dpi=300)
            plt.close(fig_single)
            print(f"Zapisano oddzielny wykres: {single_path.name}")

    fig.tight_layout()
    grid_path = PLOTS_DIR / "mse_all_variants.png"
    fig.savefig(grid_path, dpi=300)
    print(f"Zapisano wykres zbiorczy: {grid_path.name}")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    main()
