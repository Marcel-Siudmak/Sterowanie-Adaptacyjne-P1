# Importy i zmienne globalne 
import numpy as np
import matplotlib.pyplot as plt

# VARIBLES
N = 100 # liczba krokow czasowych symulacji 
H = 5 # szerokosc okna ruchomego
alpha = 2 # wspołczynnik kierunkowy trendu a 
var_z = 33 # wariancja szumu pomiarowego 

def main():
    # funkcja najpierw generuje dane a potem przepuszcza zaszumione pomiary przez filtr redniej ruchomej
    t, signal, measure = generate_measurments(N, alpha, var_z)
    estimation = moving_average(measure, H)
    plt.figure(figsize=(10,6)) # okno wykresu o wymiarach 10x6 cali
    plt.plot(t, signal, label='Sygnał prawdziwy', color='black', linewidth=2) # czarna ciagla linia sygnalu prawdziwego
    plt.plot(t, measure, '.', label='Pomiary', color='gray', alpha=0.5, markersize=4) # szare pojedyncze kropki dla zaszumionych pomiarow
    plt.plot(t[H - 1:], estimation, label=f'SMA (H={H})', color='red') # czerwona linia estymatora

    # kosmetyka wykresu
    plt.xlabel('Dyskretny czas k') # podpis osi X
    plt.ylabel('Wartość sygnału') # podpis osi Y
    plt.grid(True, linestyle='--', alpha=0.6) # przerywana siatka
    plt.legend()
    plt.show() # okienko z wykresem na ekranie



# funkcja generujaca dane 
def generate_measurments(N, alpha, var_z):
    t = np.arange(N) # tworzy jednowymiarowa tablice
    signal = t*alpha # realizacja wzoru na sygnal prawdziwy
    a = np.sqrt(6 * var_z)
    noise = np.random.triangular(-a, 0, a, size=N) # losuje N niezaleznych liczb 
    measure = signal + noise
    return t, signal, measure

# funkcja liczaca srednia ruchowa
def moving_average(measure, H):
    weights = np.ones(H) / H
    return np.convolve(measure, weights, mode='valid')

# funkcja symulacji Monte Carlo
def monte_carlo_mse(N, M, H_max, alpha, var_z):
    H_values = np.arange(2, H_max + 1)       # Badane okna H od 2 do 100
    mse_emp = np.zeros(len(H_values))        # Pusta tablica na sumę błędów dla każdego H
    k_steady = N - H_max                     # Liczba próbek dla k >= H_max (po odcięciu rozbiegówki)

    for _ in range(M):                       # Pętla Monte Carlo (M powtórzeń)
        t, signal, measure = generate_measurments(N, alpha, var_z)
        
        for i, H in enumerate(H_values):     # Pętla po wszystkich oknach H = 2..100
            est = moving_average(measure, H)
            
            # Bierzemy ostatnie k_steady próbek z obu tablic (chwile k >= H_max)
            blad = est[-k_steady:] - signal[-k_steady:]
            mse_emp[i] += np.mean(blad ** 2) # Dodajemy średni kwadrat błędu z tego przebiegu

    return H_values, mse_emp / M             # Zwracamy H oraz błąd uśredniony przez M powtórzeń

if __name__=='__main__':
    main()


