import numpy as np

def generate_signal(alpha: float, N: int) -> np.ndarray:
    k = np.arange(N)
    return alpha * k


def generate_noise(var_z: float, N: int) -> np.ndarray:
    a = np.sqrt(6.0 * var_z)
    return np.random.triangular(left=-a, mode=0.0, right=a, size=N)


def generate_measurements(signal: np.ndarray, var_z) -> np.ndarray:
    return signal + generate_noise(var_z, signal.size)


if __name__ == "__main__":
    alpha = 2
    N = 10
    var_z = 2
    theta_star = generate_signal(alpha, N)
    theta = generate_measurements(theta_star, var_z)
    print("THETA_STAR:", theta_star, "\nTHETA:", theta)
