"""Wizualizacja dekompozycji błędu średniokwadratowego (Bias-Variance Tradeoff).

Zgodnie z sekcją 5.2.4 specyfikacji README.md przedstawia na jednym wykresie:
1. Kwadrat obciążenia: Bias^2(H)
2. Wariancję: Var(H)
3. Całkowity błąd: MSE(H) = Bias^2(H) + Var(H)
oraz mechanizm powstawania minimum optymalnego okna H_opt.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from theory import (
    theoretical_bias_squared,
    theoretical_h_opt,
    theoretical_mse,
    theoretical_variance,
)

# STAŁE KONFIGURACYJNE
ALPHA = 0.02        # nachylenie trendu (Wariant 2 z README)
VAR_Z = 25.0        # wariancja szumu (Wariant 2 z README)
H_MAX = 100         # maksymalny horyzont
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def plot_decomposition(
    alpha: float,
    var_z: float,
    H_max: int,
    save_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """
    Rysuje wykres dekompozycji MSE na składową obciążenia (Bias^2) oraz wariancji (Var).
    """
    H_range = np.arange(1, H_max + 1)

    bias2 = theoretical_bias_squared(alpha, H_range)
    var = theoretical_variance(var_z, H_range)
    mse = theoretical_mse(alpha, var_z, H_range)

    h_opt = theoretical_h_opt(alpha, var_z)
    min_mse = theoretical_mse(alpha, var_z, h_opt)

    plt.figure(figsize=(10, 6))

    # 1. Krzywe składowe
    plt.plot(
        H_range,
        bias2,
        label=r"Kwadrat obciążenia $\mathrm{Bias}^2(H) = \frac{\alpha^2(H-1)^2}{4}$",
        color="tab:red",
        linewidth=2.0,
        linestyle="-.",
    )
    plt.plot(
        H_range,
        var,
        label=r"Wariancja $\mathrm{Var}(H) = \frac{\sigma_z^2}{H}$",
        color="tab:green",
        linewidth=2.0,
        linestyle="--",
    )
    plt.plot(
        H_range,
        mse,
        label=r"Całkowity błąd $\mathrm{MSE}(H) = \mathrm{Bias}^2(H) + \mathrm{Var}(H)$",
        color="tab:blue",
        linewidth=2.5,
    )

    # 2. Punkt minimum i linia optymalnego horyzontu
    plt.axvline(
        x=h_opt,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"Optymalny horyzont $H_{{opt}} = {h_opt}$",
    )
    plt.plot(h_opt, min_mse, marker="o", markersize=7, color="tab:blue")

    plt.annotate(
        f"Minimum $\\mathrm{{MSE}}$\n$H_{{opt}} = {h_opt}$\n$\\mathrm{{MSE}}_{{min}} = {min_mse:.3f}$",
        xy=(h_opt, min_mse),
        xytext=(h_opt + 10, min_mse + 2.0),
        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="tab:blue", alpha=0.9),
    )

    plt.title(
        f"Dekompozycja błędu MSE (Bias–Variance Tradeoff) dla α = {alpha}, var_z = {var_z}",
        fontsize=13,
    )
    plt.xlabel("Horyzont okna H", fontsize=11)
    plt.ylabel("Wartość błędu", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.xlim(1, H_max)
    plt.ylim(0, max(25.0, min_mse * 4))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Zapisano wykres dekompozycji: {save_path.name}")

    if show_plot:
        plt.show()


def main(show_plot: bool = True):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "mse_decomposition.png"
    plot_decomposition(
        alpha=ALPHA,
        var_z=VAR_Z,
        H_max=H_MAX,
        save_path=save_path,
        show_plot=show_plot,
    )


if __name__ == "__main__":
    main()
