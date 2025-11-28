import numpy as np
from scipy.signal import get_window

def calcular_vector_energias_temporal(senal, fs, N, K, window="hamming"):
    x = senal - np.mean(senal)
    
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    
    if len(x) < N:
        xN = np.pad(x, (0, N - len(x)), mode='constant')
    elif len(x) > N:
        start = (len(x) - N) // 2
        xN = x[start:start + N]
    else:
        xN = x
    
    if window.lower() == "none" or window == "rect":
        w = np.ones(N)
    else:
        w = get_window(window, N, fftbins=True)
    
    xN_windowed = xN * w
    
    X = np.fft.fft(xN_windowed, n=N)
    
    N_half = N // 2
    X_positivas = X[:N_half]
    
    puntos_por_subbanda = N_half // K
    
    energias = np.zeros(K, dtype=np.float32)
    
    for i in range(K):
        inicio = i * puntos_por_subbanda
        
        if i == K - 1:
            fin = N_half
        else:
            fin = (i + 1) * puntos_por_subbanda
        
        Xi = X_positivas[inicio:fin]
        
        Ei = (1.0 / N) * np.sum(np.abs(Xi) ** 2)
        energias[i] = Ei
    
    return energias

def calcular_vector_energias(espectro_magnitud, numero_subbandas):
    return np.zeros(numero_subbandas, dtype=np.float32)

def calcular_estadisticos_energias(lista_vectores_energias):
    matriz = np.vstack(lista_vectores_energias)
    medias = np.mean(matriz, axis=0)
    desviaciones = np.std(matriz, axis=0, ddof=0)
    return medias, desviaciones

def normalizar_vector_energia(vector):
    return vector.astype(np.float32)
