import numpy as np
from scipy.signal import get_window


def calcular_vector_energias_temporal(senal, fs, N, K, window="hamming"):
    """Calcula energías por SUBBANDAS DE FRECUENCIA (método EXACTO del lab5 según imagen).
    
    MÉTODO CORRECTO (según imagen compartida):
    1. Preprocesamiento: DC removal, RMS normalization, pre-énfasis
    2. Ajustar señal a tamaño N
    3. Aplicar ventana a TODA la señal
    4. Calcular FFT de TODA la señal
    5. Dividir el ESPECTRO en K partes iguales (bandas de frecuencia)
    6. Calcular energía de cada banda: E = (1/N) * Σ|X(k)|²
    7. Normalización
    
    Args:
        senal: Señal de audio RAW (sin preprocesar)
        fs: Frecuencia de muestreo
        N: Tamaño fijo de ventana
        K: Número de bandas de frecuencia
        window: Tipo de ventana
    
    Returns:
        Vector de K energías normalizadas
    """
    # ========== PASO 1: PREPROCESAMIENTO ==========
    
    # Eliminar componente DC
    x = senal - np.mean(senal)
    
    # Normalizar RMS
    rms_val = np.sqrt(np.mean(x ** 2))
    if rms_val > 1e-8:
        x = x / rms_val
    
    # Pre-énfasis
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    
    # ========== PASO 2: AJUSTAR A TAMAÑO N ==========
    
    if len(x) < N:
        # Rellenar con ceros
        xN = np.pad(x, (0, N - len(x)), mode='constant')
    elif len(x) > N:
        # Tomar centro (donde suele estar la voz)
        start = (len(x) - N) // 2
        xN = x[start:start + N]
    else:
        xN = x
    
    # ========== PASO 3: APLICAR VENTANA A TODA LA SEÑAL ==========
    
    if window.lower() == "none" or window == "rect":
        w = np.ones(N)
    else:
        w = get_window(window, N, fftbins=True)
    
    xN_windowed = xN * w
    
    # ========== PASO 4: FFT DE TODA LA SEÑAL ==========
    
    X = np.fft.rfft(xN_windowed, n=N)  # Solo frecuencias positivas
    X_mag = np.abs(X)
    
    # ========== PASO 5: DIVIDIR ESPECTRO EN K BANDAS ==========
    
    n_bins = len(X_mag)
    band_size = n_bins // K
    Es = np.zeros(K, dtype=float)
    
    # ========== PASO 6: CALCULAR ENERGÍA POR BANDA ==========
    
    for i in range(K):
        # Definir rango de la banda i
        start_bin = i * band_size
        if i == K - 1:
            # Última banda toma todos los bins restantes
            end_bin = n_bins
        else:
            end_bin = (i + 1) * band_size
        
        # Extraer magnitudes de la banda
        band_mag = X_mag[start_bin:end_bin]
        
        # Energía = (1/N) * Σ|X(k)|²
        energy = np.sum(band_mag ** 2) / N
        Es[i] = float(energy)
    
    # ========== PASO 7: NORMALIZACIÓN ==========
    
    # Logaritmo para robustez
    Es = np.log10(Es + 1e-10)
    
    # Normalizar distribución relativa (suma = 1)
    E_sum = np.sum(Es)
    if E_sum != 0:
        Es = Es / E_sum
    
    return Es.astype(np.float32)


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
    """Normaliza el vector de energias para eliminar dependencia del volumen.
    Según la teoría, se divide cada componente por la suma total."""
    energia_total = np.sum(vector)
    # Umbral mínimo para evitar división por cero
    if energia_total <= 1e-12:
        # Si no hay energía, retornar vector uniforme normalizado
        return np.ones_like(vector, dtype=np.float32) / len(vector)
    return (vector / energia_total).astype(np.float32)
