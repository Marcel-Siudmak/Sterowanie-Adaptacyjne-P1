# Projekt: Estymacja sygnału z trendem liniowym za pomocą estymatora okna ruchomego (Grupa G)

## 1. Cel i zakres projektu
Celem projektu jest implementacja, badanie symulacyjne oraz analityczna weryfikacja estymatora ze skończonym oknem ruchomym (*rolling window* / *Simple Moving Average* – SMA) dla sygnału deterministycznego z zakłóceniem losowym.

Projekt koncentruje się na wyznaczeniu optymalnej szerokości okna (horyzontu) $H_{\text{opt}} \in [2, 100]$ minimalizującej błąd średniokwadratowy (MSE) oraz zbadaniu zjawiska kompromisu między obciążeniem a wariancją (*bias–variance tradeoff*) w zależności od:
1. Dynamiki zmian sygnału rzeczywistego (nachylenie trendu $\alpha$).
2. Poziomu zakłóceń pomiarowych (wariancja szumu $$\text{Var}(z_k)$$).

---

## 2. Model matematyczny

### 2.1. Sygnał rzeczywisty (stan)
Dla grupy **G** sygnał rzeczywisty jest deterministycznym trendem liniowym w dyskretnym czasie $k$:
$$\theta_k^* = \alpha \cdot k, \quad k \in \{0, 1, \dots, N\}$$
gdzie $\alpha > 0$ jest współczynnikiem kierunkowym trendu.

### 2.2. Model pomiaru i zakłócenia
Wartość zmierzona w chwili $k$ stanowi sumę sygnału rzeczywistego oraz błędu pomiarowego:

$$
\theta_k = \theta_k^* + z_k
$$

Błąd pomiarowy $z_k$ jest ciągiem niezależnych zmiennych losowych (i.i.d.) o symetrycznym rozkładzie trójkątnym na przedziale $[-a, a]$:
- Wartość oczekiwana: $\mathbb{E}[z_k] = 0$
- Funkcja gęstości prawdopodobieństwa (PDF):

  $$
  f(z) = \begin{cases} 
  \frac{a - |z|}{a^2} & \text{dla } |z| \le a \\ 
  0 & \text{dla } |z| > a 
  \end{cases}
  $$

- Wariancja szumu: $\text{Var}(z_k) = \sigma_z^2 = \frac{a^2}{6} \implies a = \sqrt{6\sigma_z^2}$

### 2.3. Estymator okna ruchomego (SMA)
Estymata wartości sygnału w chwili $k$ wyznaczana jest jako średnia arytmetyczna z ostatnich $H$ pomiarów:

$$
\hat{\theta}_k = \frac{1}{H} \sum_{i=0}^{H-1} \theta_{k-i}, \quad H \in [2, 100]
$$

---

## 3. Analiza teoretyczna: Dekompozycja MSE (Bias-Variance)

Błąd średniokwadratowy estymatora dekomponuje się na kwadrat obciążenia oraz wariancję:

$$
\text{MSE}(H) = \mathbb{E}\left[(\hat{\theta}_k - \theta_k^*)^2\right] = \text{Bias}^2(\hat{\theta}_k) + \text{Var}(\hat{\theta}_k)
$$

### 3.1. Obciążenie (Bias)
Wartość oczekiwana estymatora wynosi:

$$
\mathbb{E}[\hat{\theta}_k] = \frac{1}{H} \sum_{i=0}^{H-1} \mathbb{E}[\theta_{k-i}] = \alpha k - \frac{\alpha (H - 1)}{2}
$$

Obciążenie: $\text{Bias}(\hat{\theta}_k) = \mathbb{E}[\hat{\theta}_k] - \theta_k^* = -\frac{\alpha (H - 1)}{2}$

Kwadrat obciążenia (rośnie monotonicznie z kwadratem długości okna $H$):

$$
\text{Bias}^2(H) = \frac{\alpha^2 (H - 1)^2}{4}
$$

### 3.2. Wariancja (Variance)
Ponieważ pomiary są sumą wartości deterministycznej i nieskorelowanych zakłóceń $z_k$:

$$
\text{Var}(\hat{\theta}_k) = \text{Var}\left( \frac{1}{H} \sum_{i=0}^{H-1} z_{k-i} \right) = \frac{\sigma_z^2}{H}
$$

Wariancja maleje wraz ze wzrostem horyzontu $H$.


### 3.3. Teoretyczna funkcja MSE i minimum analityczne
$$\text{MSE}(H) = \frac{\alpha^2 (H - 1)^2}{4} + \frac{\sigma_z^2}{H}$$

Przybliżone analityczne horyzont optymalny wyznacza się z warunku $\frac{\partial \text{MSE}}{\partial H} = 0$:
$$\frac{\alpha^2 (H - 1)}{2} - \frac{\sigma_z^2}{H^2} \approx 0 \implies H_{\text{opt}} \approx \left( \frac{2\sigma_z^2}{\alpha^2} \right)^{1/3}$$

**Wnioski teoretyczne:**
* Większa wariancja zakłóceń $\sigma_z^2$ przesuwa $H_{\text{opt}}$ w prawo (konieczność dłuższego uśredniania w celu redukcji szumu).
* Szybsza dynamika sygnału (większe $\alpha$) przesuwa $H_{\text{opt}}$ w lewo (duże okno wprowadza niedopuszczalne opóźnienie filtracji).

---

## 4. Plan eksperymentów symulacyjnych

### 4.1. Przypadki testowe (Macierz wariantów)
Zgodnie z wytycznymi z tablicy należy przeprowadzić badania dla czterech charakterystycznych kombinacji parametrów:
1. **Wariant 1:** Małe $\alpha$, mała wariancja $\text{Var}(z_k)$
2. **Wariant 2:** Małe $\alpha$, duża wariancja $\text{Var}(z_k)$
3. **Wariant 3:** Duże $\alpha$, mała wariancja $\text{Var}(z_k)$
4. **Wariant 4:** Duże $\alpha$, duża wariancja $\text{Var}(z_k)$

*Przykładowy dobór parametrów numerycznych:*
- $\alpha_{\text{małe}} = 0.02$, $\alpha_{\text{duże}} = 0.15$
- $\sigma_{z,\text{małe}}^2 = 1.0$ ($a = \sqrt{6} \approx 2.45$), $\sigma_{z,\text{duże}}^2 = 25.0$ ($a = \sqrt{150} \approx 12.25$)

### 4.2. Metodologia obliczania empirycznego MSE (Monte Carlo)
Aby uzyskać gładkie charakterystyki pozbawione przypadkowych fluktuacji:
1. Długość pojedynczego przebiegu: $N \ge 1000$ kroków czasowych.
2. Odcięcie stanu nieustalonego: do obliczania błędu uwzględniać wyłącznie próbki $k \ge H_{\max}$ ($k \ge 100$).
3. Liczba powtórzeń Monte Carlo: $M \ge 100$ niezależnych realizacji szumu dla każdej konfiguracji:
   $$\widehat{\text{MSE}}(H) = \frac{1}{M} \sum_{m=1}^{M} \left( \frac{1}{N - H_{\max} + 1} \sum_{k=H_{\max}}^{N} \left(\hat{\theta}_k^{(m)}(H) - \theta_k^*\right)^2 \right)$$

---

## 5. Wymagane wizualizacje i analiza wyników

### 5.1. Przebiegi czasowe (Time-domain plots)
Dla wybranego wariantu zestawić na jednym wykresie:
- Sygnał wzorcowy $\theta_k^*$,
- Zaszumione punkty pomiarowe $\theta_k$,
- Odpowiedź estymatora $\hat{\theta}_k$ dla trzech charakterystycznych wartości $H$:
  - $H \ll H_{\text{opt}}$ (np. $H = 2$) – dominacja szumu (niedofiltracja),
  - $H = H_{\text{opt}}$ – kompromis i najwierniejsza estymacja,
  - $H \gg H_{\text{opt}}$ (np. $H = 90$) – dominacja opóźnienia/obciążenia (nadfiltracja).

### 5.2. Charakterystyki jakości $\text{MSE}(H)$
1. Wykres krzywych empirycznych $\widehat{\text{MSE}}(H)$ w funkcji $H \in [2, 100]$ dla badanych 4 wariantów.
2. Zestawienie wyników eksperymentalnych z krzywymi analitycznymi wyliczonymi z wzoru teoretycznego.
3. Wyraźne oznaczenie punktów minimum ($H_{\text{opt}}$) dla każdej z krzywych.
4. Wykres dekompozycji: na jednym rysunku przedstawić składowe $\text{Bias}^2(H)$, $\text{Var}(H)$ oraz $\text{MSE}(H)$, ilustrujący powstawanie minimum.

---

## 6. Struktura raportu / projektu
1. **Wstęp teoretyczny**: Wyprowadzenie modelu, estymatora, rozkładu trójkątnego oraz dekompozycji MSE.
2. **Opis implementacji**: Narzędzia (np. Python / NumPy / Matplotlib), struktura kodu, generator liczb losowych o rozkładzie trójkątnym.
3. **Wyniki badań**: Zestawienie wykresów czasowych i krzywych MSE.
4. **Wnioski i dyskusja**: Porównanie $H_{\text{opt}}$ wyznaczonego empirycznie i teoretycznie, interpretacja wpływu parametrów $\alpha$ oraz $\sigma_z^2$.

