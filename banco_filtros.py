import numpy as np
from scipy.signal import get_window


def calcular_vector_energias_temporal(senal, fs, N, K, window="hamming"):
    """Calcula energías según TEORÍA ADJUNTA EXACTA.
    
    PASOS SEGÚN TEORÍA:
    1. Remover DC (componente continua)
    2. Pre-énfasis para balancear espectro
    3. Ajustar/rellenar a N muestras (potencia de 2)
    4. Aplicar ventana temporal (Hamming)
    5. Calcular FFT de N puntos: y = fft(x), z = |y|
    6. Determinar ancho de banda común BW
    7. Dividir BW en K partes iguales (4 sub-bandas)
    8. Calcular energía por sub-banda: E = (1/N) * Σ|X(k)|²
    
    Args:
        senal: Señal de audio de entrada
        fs: Frecuencia de muestreo (16000 Hz)
        N: Tamaño FFT (4096 - potencia de 2)
        K: Número de sub-bandas (4 según teoría)
        window: Tipo de ventana ("hamming" recomendado)
    
    Returns:
        Vector de K energías [E1, E2, E3, E4]
    """
    # PASO 1: Eliminar componente DC
    x = senal - np.mean(senal)
    
    # PASO 2: Pre-énfasis (realzar altas frecuencias)
    # Fórmula: y[n] = x[n] - α*x[n-1], donde α = 0.97
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    
    # PASO 3: Ajustar a N muestras exactas
    if len(x) < N:
        # Rellenar con ceros si es más corta
        xN = np.pad(x, (0, N - len(x)), mode='constant')
    elif len(x) > N:
        # Tomar ventana centrada si es más larga
        start = (len(x) - N) // 2
        xN = x[start:start + N]
    else:
        xN = x
    
    # PASO 4: Aplicar ventana temporal
    if window.lower() == "none" or window == "rect":
        w = np.ones(N)
    else:
        w = get_window(window, N, fftbins=True)
    
    xN_windowed = xN * w
    
    # PASO 5: Calcular FFT completa
    # y = fft(x)
    X = np.fft.fft(xN_windowed, n=N)
    
    # PASO 6: Determinar BW (ancho de banda común)
    # Según teoría: usar solo frecuencias positivas (0 a fs/2)
    # Esto corresponde a la mitad de los puntos FFT
    N_half = N // 2
    X_positivas = X[:N_half]
    
    # PASO 7: Dividir BW en K partes iguales
    # Ejemplo con K=4: X1(k)=[X(0) X(1)], X2(k)=[X(2) X(3)], ...
    puntos_por_subbanda = N_half // K
    
    # PASO 8: Calcular energía por cada sub-banda
    # E = (1/N) * Σ|X(k)|²
    energias = np.zeros(K, dtype=np.float32)
    
    for i in range(K):
        inicio = i * puntos_por_subbanda
        
        if i == K - 1:
            # Última sub-banda toma puntos restantes
            fin = N_half
        else:
            fin = (i + 1) * puntos_por_subbanda
        
        # Extraer sub-banda Xi(k)
        Xi = X_positivas[inicio:fin]
        
        # Calcular energía: E = (1/N) * Σ|X(k)|²
        Ei = (1.0 / N) * np.sum(np.abs(Xi) ** 2)
        energias[i] = Ei
    
    return energias


def calcular_vector_energias(espectro_magnitud, numero_subbandas):
    """DEPRECATED: Método antiguo."""
    return np.zeros(numero_subbandas, dtype=np.float32)


def calcular_estadisticos_energias(lista_vectores_energias):
    """Calcula medias y desviaciones estandar por subbanda."""
    matriz = np.vstack(lista_vectores_energias)
    medias = np.mean(matriz, axis=0)
    desviaciones = np.std(matriz, axis=0, ddof=0)
    return medias, desviaciones


def normalizar_vector_energia(vector):
    """DEPRECATED: Ya no normalizamos suma=1 (teoría usa energía absoluta)."""
    return vector.astype(np.float32)
