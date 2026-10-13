import numpy as np
import matplotlib.pyplot as plt

# VARIBLES
N = 100
H = 5
alpha = 2
var_z = 33

def main():
    t, signal, measure = generate_measurments(N, alpha, var_z)
    estimation = moving_average(measure, H)
    plt.figure(figsize=(10,6))
    plt.plot(t, signal, label='Sygnał prawdziwy', color='black', linewidth=2)
    plt.plot(t, measure, '.', label='Pomiary', color='gray', alpha=0.5, markersize=4)
    plt.plot(t[H - 1:], estimation, label=f'SMA (H={H})', color='red')

    plt.xlabel('Dyskretny czas k')
    plt.ylabel('Wartość sygnału')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()




def generate_measurments(N, alpha, var_z):
    t = np.arange(N)
    signal = t*alpha
    a = np.sqrt(6 * var_z)
    noise = np.random.triangular(-a, 0, a, size=N) 
    measure = signal + noise
    return t, signal, measure

def moving_average(measure, H):
    weights = np.ones(H) / H
    return np.convolve(measure, weights, mode='valid')

if __name__=='__main__':
    main()
