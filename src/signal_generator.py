import numpy as np

def generate_signal(alpha: float, N: int) -> np.ndarray:
    k = np.arange(N)
    return alpha * k


def generate_noise(var_z: float, N: int) -> np.ndarray:
    a = np.sqrt(6.0 * var_z)
    return np.random.triangular(left=-a, mode=0.0, right=a, size=N)


def generate_measurements(N, alpha, var_z) -> tuple[np.ndarray, np.ndarray]:
    signal = generate_signal(alpha, N)
    measurements = signal + generate_noise(var_z, N)
    
    return signal, measurements

if __name__ == "__main__":
    alpha = 2
    N = 10
    var_z = 2
    theta_star, theta = generate_measurements(N, alpha, var_z)
    print("THETA_STAR:", theta_star, "\nTHETA:", theta)
