import numpy as np


def dividir_espectro_en_subbandas(espectro_magnitud, numero_subbandas):
    """Implementa el banco de filtros como particion del vector FFT en subbandas."""
    N = len(espectro_magnitud)
    longitud_subbanda = N // numero_subbandas
    subbandas = []

    for i in range(numero_subbandas):
        inicio = i * longitud_subbanda
        fin = N if i == numero_subbandas - 1 else (i + 1) * longitud_subbanda
        subbandas.append(espectro_magnitud[inicio:fin])

    return subbandas


def calcular_energia_subbanda(subbanda, N_fft):
    """Calcula la energia de una subbanda según la teoría: E = (1/N) * Σ|X(k)|²
    donde N es el número total de puntos de la FFT completa."""
    if len(subbanda) == 0:
        return 0.0
    # Suma de las magnitudes al cuadrado dividido por N total (no por longitud de subbanda)
    energia = np.sum(np.abs(subbanda) ** 2) / N_fft
    return float(energia)


def calcular_vector_energias(espectro_magnitud, numero_subbandas):
    """Devuelve el vector de energias de cada subbanda."""
    N_fft = len(espectro_magnitud) * 2  # Espectro completo (mitad positiva)
    subbandas = dividir_espectro_en_subbandas(espectro_magnitud, numero_subbandas)
    energias = [calcular_energia_subbanda(sb, N_fft) for sb in subbandas]
    return np.array(energias, dtype=np.float32)


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
