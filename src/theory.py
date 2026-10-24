"""Moduł analityczny estymatora SMA.

Zawiera teoretyczne wzory na obciążenie, wariancję oraz błąd średniokwadratowy (MSE)
dla sygnału z trendem liniowym i zakłóceniem losowym zgodnie z sekcją 3 README.md.
"""

import numpy as np


def theoretical_bias(alpha: float, H: np.ndarray | int | float) -> np.ndarray | float:
    """
    Oblicza obciążenie estymatora SMA:
    Bias(H) = E[theta_hat] - theta* = -alpha * (H - 1) / 2
    """
    return -alpha * (H - 1) / 2.0


def theoretical_bias_squared(alpha: float, H: np.ndarray | int | float) -> np.ndarray | float:
    """
    Oblicza kwadrat obciążenia estymatora SMA:
    Bias^2(H) = (alpha^2 * (H - 1)^2) / 4
    """
    return (alpha**2 * (H - 1)**2) / 4.0


def theoretical_variance(var_z: float, H: np.ndarray | int | float) -> np.ndarray | float:
    """
    Oblicza wariancję estymatora SMA:
    Var(H) = sigma_z^2 / H
    """
    return var_z / H


def theoretical_mse(alpha: float, var_z: float, H: np.ndarray | int | float) -> np.ndarray | float:
    """
    Oblicza teoretyczny błąd średniokwadratowy:
    MSE(H) = Bias^2(H) + Var(H) = (alpha^2 * (H - 1)^2) / 4 + sigma_z^2 / H
    """
    return theoretical_bias_squared(alpha, H) + theoretical_variance(var_z, H)


def theoretical_h_opt(alpha: float, var_z: float) -> int:
    """
    Wyznacza analitycznie optymalną całkowitą szerokość okna H minimalizującą MSE:
    H_opt ≈ round((2 * sigma_z^2 / alpha^2)^(1/3))
    """
    continuous_opt = (2.0 * var_z / (alpha**2)) ** (1.0 / 3.0)
    h_candidate = int(np.round(continuous_opt))

    # Sprawdzenie całkowitego sąsiedztwa kandydata dla znalezienia ścisłego minimum dyskretnego
    candidates = [max(1, h_candidate - 1), max(1, h_candidate), h_candidate + 1]
    best_h = min(candidates, key=lambda h: theoretical_mse(alpha, var_z, h))
    return int(best_h)
