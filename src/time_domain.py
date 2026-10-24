import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from signal_generator import generate_signal, generate_measurements
from estimator import estimate_sma

# STAŁE KONFIGURACYJNE
N = 1000             # całkowita liczba kroków symulacji
N_VIEW = 300         # liczba punktów wyświetlana na wykresie dla czytelności
ALPHA = 0.02         # nachylenie trendu (Wariant 2 z README)
VAR_Z = 25.0         # wariancja szumu (Wariant 2 z README)
H_VALUES = [2, 50, 150]  # okna: H << H_opt, H_opt, H >> H_opt
RANDOM_SEED = 42     # ziarno losowości dla powtarzalności wykresu do raportu
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def simulate_time_domain(
    alpha: float,
    var_z: float,
    N: int,
    h_values: list[int],
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """
    Generuje sygnał wzorcowy, zaszumione pomiary oraz estymaty SMA dla zadanych okien H.
    """
    if seed is not None:
        np.random.seed(seed)

    signal_star = generate_signal(alpha, N)
    measurements = generate_measurements(signal_star, var_z)

    estimates = {}
    for H in h_values:
        estimates[H] = estimate_sma(measurements, H)

    return signal_star, measurements, estimates


def plot_time_domain(
    signal_star: np.ndarray,
    measurements: np.ndarray,
    estimates: dict[int, np.ndarray],
    alpha: float,
    var_z: float,
    n_view: int | None = None,
    k_start: int | None = None,
    save_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """
    Rysuje i zapisuje przebieg czasowy sygnału wzorcowego, pomiarów i estymat.
    Domyślnie wykres zaczyna się od punktu k_start odpowiadającego największemu
    testowanemu oknu H (max(H) - 1), dzięki czemu wszystkie estymaty są obecne
    od samego początku osi czasu (brak stanu nieustalonego z NaN).
    """
    if k_start is None:
        # Początek w chwili, gdy największe okno H ma już pierwszą wyznaczoną próbkę
        k_start = max(estimates.keys()) - 1

    if n_view is None:
        n_view = len(signal_star) - k_start

    k_end = min(len(signal_star), k_start + n_view)
    k = np.arange(k_start, k_end)

    plt.figure(figsize=(12, 6))

    # 1. Zaszumione pomiary (punkty w tle)
    plt.scatter(
        k,
        measurements[k_start:k_end],
        color="gray",
        alpha=0.35,
        s=12,
        label=r"Pomiary $\theta_k$ (szum)",
        zorder=1,
    )

    # 2. Sygnał wzorcowy (deterministyczny trend)
    plt.plot(
        k,
        signal_star[k_start:k_end],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=r"Sygnał wzorcowy $\theta_k^* = \alpha k$",
        zorder=4,
    )

    # 3. Odpowiedzi estymatora dla poszczególnych okien H
    styles = {
        H_VALUES[0]: {
            "color": "tab:orange",
            "lw": 1.2,
            "label": f"H = {H_VALUES[0]} (niedofiltracja – dominacja szumu)",
            "zorder": 2,
        },
        H_VALUES[1]: {
            "color": "tab:green",
            "lw": 2.2,
            "label": f"H = {H_VALUES[1]} (kompromis – optymalne $H_{{opt}}$)",
            "zorder": 5,
        },
        H_VALUES[2]: {
            "color": "tab:purple",
            "lw": 2.0,
            "label": f"H = {H_VALUES[2]} (nadfiltracja – dominacja opóźnienia)",
            "zorder": 3,
        },
    }

    for H, theta_hat in estimates.items():
        style = styles.get(
            H,
            {"color": None, "lw": 1.5, "label": f"H = {H}", "zorder": 3},
        )
        plt.plot(
            k,
            theta_hat[k_start:k_end],
            color=style["color"],
            linewidth=style["lw"],
            label=style["label"],
            zorder=style["zorder"],
        )

    plt.title(
        f"Przebiegi czasowe estymatora SMA dla wariantu (α = {alpha}, var_z = {var_z})",
        fontsize=13,
    )
    plt.xlabel("Krok dyskretny k", fontsize=11)
    plt.ylabel(r"Wartość sygnału $\theta$", fontsize=11)
    plt.xlim(k_start, k_end - 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Zapisano wykres: {save_path.name}")

    if show_plot:
        plt.show()


def main(show_plot: bool = True):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    signal_star, measurements, estimates = simulate_time_domain(
        alpha=ALPHA,
        var_z=VAR_Z,
        N=N,
        h_values=H_VALUES,
        seed=RANDOM_SEED,
    )

    save_path = PLOTS_DIR / "time_domain_response.png"
    plot_time_domain(
        signal_star=signal_star,
        measurements=measurements,
        estimates=estimates,
        alpha=ALPHA,
        var_z=VAR_Z,
        n_view=N_VIEW,
        save_path=save_path,
        show_plot=show_plot,
    )


if __name__ == "__main__":
    main()
