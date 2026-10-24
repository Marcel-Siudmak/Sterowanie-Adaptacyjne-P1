import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from signal_generator import generate_signal, generate_measurements
from estimator import estimate_sma
from theory import theoretical_mse, theoretical_h_opt

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


def plot_empirical_mse_curve(
    ax: plt.Axes,
    H_range: np.ndarray,
    mse: np.ndarray,
    alpha: float,
    var_z: float,
) -> tuple[int, float]:
    """
    Rysuje czystą krzywą empiryczną MSE(H) z zaznaczonym punktem minimum (po staremu).
    """
    ax.plot(H_range, mse, label="Empiryczne MSE", color="tab:blue")

    min_idx = np.argmin(mse)
    H_opt = int(H_range[min_idx])
    min_mse = float(mse[min_idx])

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

    return H_opt, min_mse


def plot_theory_comparison_curve(
    ax: plt.Axes,
    H_range: np.ndarray,
    mse: np.ndarray,
    alpha: float,
    var_z: float,
) -> tuple[int, float, int, float]:
    """
    Rysuje wykres porównawczy: empiryczne MSE vs analityczna krzywa teoretyczna z oboma minimami.
    """
    # 1. Empiryczne MSE (Monte Carlo)
    ax.plot(H_range, mse, label="Empiryczne MSE (Monte Carlo)", color="tab:blue", linewidth=1.8)

    min_idx = np.argmin(mse)
    H_opt_emp = int(H_range[min_idx])
    min_mse_emp = float(mse[min_idx])

    ax.plot(H_opt_emp, min_mse_emp, marker="o", markersize=6, color="red", label=f"Min emp.: $H={H_opt_emp}$")

    # 2. Teoretyczne MSE (wzór analityczny)
    mse_theo = theoretical_mse(alpha, var_z, H_range)
    H_opt_theo = theoretical_h_opt(alpha, var_z)
    min_mse_theo = float(theoretical_mse(alpha, var_z, H_opt_theo))

    ax.plot(
        H_range,
        mse_theo,
        label="Teoretyczne MSE (analityczne)",
        color="tab:orange",
        linestyle="--",
        linewidth=1.6,
    )
    ax.plot(
        H_opt_theo,
        min_mse_theo,
        marker="x",
        markersize=8,
        markeredgewidth=2,
        color="green",
        label=f"Min teor.: $H={H_opt_theo}$",
    )

    ax.annotate(
        f"$H_{{opt}}^{{emp}}={H_opt_emp}$ ({min_mse_emp:.2f})\n$H_{{opt}}^{{teor}}={H_opt_theo}$ ({min_mse_theo:.2f})",
        xy=(H_opt_emp, min_mse_emp),
        xytext=(10, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        fontsize=8.5,
    )

    ax.set_title(f"Porównanie MSE: α = {alpha}, var_z = {var_z}")
    ax.set_xlabel("Horyzont H")
    ax.set_ylabel("MSE")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    return H_opt_emp, min_mse_emp, H_opt_theo, min_mse_theo


# Zachowanie wstecznej kompatybilności nazwy
plot_mse_curve = plot_empirical_mse_curve


def save_summary_table(summary_rows: list[dict], output_path: Path) -> None:
    """Zapisuje tabelę podsumowującą wyniki w formacie Markdown."""
    lines = [
        "# Podsumowanie Wyników: Porównanie Horyzontu Optymalnego i MSE\n",
        "| Wariant | $\\alpha$ | $\\sigma_z^2$ | $H_{\\text{opt}}^{\\text{teor}}$ | $\\text{MSE}_{\\min}^{\\text{teor}}$ | $H_{\\text{opt}}^{\\text{emp}}$ | $\\text{MSE}_{\\min}^{\\text{emp}}$ | Błąd $\\Delta H$ |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['wariant']} | {row['alpha']} | {row['var_z']} | "
            f"{row['h_opt_theo']} | {row['mse_theo']:.3f} | "
            f"{row['h_opt_emp']} | {row['mse_emp']:.3f} | "
            f"{row['delta_h']} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nZapisano tabelę podsumowującą: {output_path.name}")


def print_summary_table(summary_rows: list[dict]) -> None:
    """Wypisuje sformatowaną tabelę podsumowującą w konsoli."""
    header = f"{'Wariant':<8} | {'alpha':<6} | {'var_z':<6} | {'H_opt(teor)':<11} | {'MSE(teor)':<10} | {'H_opt(emp)':<10} | {'MSE(emp)':<9} | {'Delta H':<7}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(" PORÓWNANIE WYNIKÓW EMPIRYCZNYCH (MONTE CARLO) Z TEORIĄ:")
    print(sep)
    print(header)
    print(sep)
    for row in summary_rows:
        print(
            f"{row['wariant']:<8} | {row['alpha']:<6.2f} | {row['var_z']:<6.1f} | "
            f"{row['h_opt_theo']:<11d} | {row['mse_theo']:<10.3f} | "
            f"{row['h_opt_emp']:<10d} | {row['mse_emp']:<9.3f} | "
            f"{row['delta_h']:<7d}"
        )
    print(sep)


def main(show_plot: bool = True):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    H_range = np.arange(1, H_max + 1)

    # 1. Wyznaczenie empirycznego MSE dla wszystkich wariantów
    computed_mse: dict[tuple[float, float], np.ndarray] = {}
    for alpha in ALPHA:
        for var_z in VAR_Z:
            computed_mse[(alpha, var_z)] = compute_empirical_mse(alpha, var_z, N, H_max, M)

    # =========================================================================
    # CZĘŚĆ A: Wykresy empiryczne (tak jak po staremu)
    # =========================================================================
    print("\n--- Generowanie wykresów empirycznych MSE (po staremu) ---")
    fig_emp, axes_emp = plt.subplots(len(ALPHA), len(VAR_Z), figsize=(12, 8), squeeze=False)

    for i, alpha in enumerate(ALPHA):
        for j, var_z in enumerate(VAR_Z):
            ax = axes_emp[i, j]
            mse = computed_mse[(alpha, var_z)]

            # Rysowanie na siatce zbiorczej
            plot_empirical_mse_curve(ax, H_range, mse, alpha, var_z)

            # Rysowanie i zapis pojedynczego pliku
            fig_single, ax_single = plt.subplots(figsize=(7, 5))
            plot_empirical_mse_curve(ax_single, H_range, mse, alpha, var_z)
            fig_single.tight_layout()

            single_path = PLOTS_DIR / f"mse_alpha_{alpha}_var_{var_z:g}.png"
            fig_single.savefig(single_path, dpi=300)
            plt.close(fig_single)
            print(f"Zapisano wykres empiryczny: {single_path.name}")

    fig_emp.tight_layout()
    grid_emp_path = PLOTS_DIR / "mse_all_variants.png"
    fig_emp.savefig(grid_emp_path, dpi=300)
    print(f"Zapisano zbiorczy wykres empiryczny: {grid_emp_path.name}")

    # =========================================================================
    # CZĘŚĆ B: Wykresy zestawienia z teorią (nowe, oddzielne pliki)
    # =========================================================================
    print("\n--- Generowanie wykresów porównania MSE: Empiria vs Teoria ---")
    fig_comp, axes_comp = plt.subplots(len(ALPHA), len(VAR_Z), figsize=(12, 8), squeeze=False)
    summary_rows = []
    variant_idx = 1

    for i, alpha in enumerate(ALPHA):
        for j, var_z in enumerate(VAR_Z):
            ax = axes_comp[i, j]
            mse = computed_mse[(alpha, var_z)]

            # Rysowanie na siatce zbiorczej porównania
            h_opt_emp, min_mse_emp, h_opt_theo, min_mse_theo = plot_theory_comparison_curve(
                ax, H_range, mse, alpha, var_z
            )

            summary_rows.append({
                "wariant": f"W{variant_idx}",
                "alpha": alpha,
                "var_z": var_z,
                "h_opt_theo": h_opt_theo,
                "mse_theo": min_mse_theo,
                "h_opt_emp": h_opt_emp,
                "mse_emp": min_mse_emp,
                "delta_h": abs(h_opt_emp - h_opt_theo),
            })
            variant_idx += 1

            # Rysowanie i zapis pojedynczego pliku porównania
            fig_single_comp, ax_single_comp = plt.subplots(figsize=(7, 5))
            plot_theory_comparison_curve(ax_single_comp, H_range, mse, alpha, var_z)
            fig_single_comp.tight_layout()

            single_comp_path = PLOTS_DIR / f"mse_vs_theory_alpha_{alpha}_var_{var_z:g}.png"
            fig_single_comp.savefig(single_comp_path, dpi=300)
            plt.close(fig_single_comp)
            print(f"Zapisano wykres porównania z teorią: {single_comp_path.name}")

    fig_comp.tight_layout()
    grid_comp_path = PLOTS_DIR / "mse_vs_theory_all_variants.png"
    fig_comp.savefig(grid_comp_path, dpi=300)
    print(f"Zapisano zbiorczy wykres porównania z teorią: {grid_comp_path.name}")

    # =========================================================================
    # CZĘŚĆ C: Tabela podsumowująca
    # =========================================================================
    print_summary_table(summary_rows)
    save_summary_table(summary_rows, PLOTS_DIR / "summary_table.md")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    main()
