import numpy as np

def estimate_sma(measurements: np.ndarray, H: int) -> np.ndarray:
    theta_hat = np.full_like(measurements, fill_value=np.nan, dtype=float)
    kernel = np.ones(H) / H
    theta_hat[H - 1:] = np.convolve(measurements, kernel, mode="valid")
    
    return theta_hat




if __name__ == "__main__":
    # Test na prostym wektorze pomiarów
    pomiary = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    H = 3

    estymata = estimate_sma(pomiary, H)
    print("Pomiary: ", pomiary)
    print("Estymata:", estymata)
    # Dla k=0, 1: nan
    # Dla k=2: (1 + 2 + 3) / 3 = 2.0
    # Dla k=3: (2 + 3 + 4) / 3 = 3.0
    # Dla k=4: (3 + 4 + 5) / 3 = 4.0
