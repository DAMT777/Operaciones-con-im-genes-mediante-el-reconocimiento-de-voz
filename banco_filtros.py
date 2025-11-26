import numpy as np


def dividir_espectro_en_subbandas(espectro_magnitud, numero_subbandas):
    """Implementa el banco de filtros como partición del vector FFT en subbandas."""
    N = len(espectro_magnitud)
    longitud_subbanda = N // numero_subbandas
    subbandas = []

    for i in range(numero_subbandas):
        inicio = i * longitud_subbanda
        if i == numero_subbandas - 1:
            fin = N
        else:
            fin = (i + 1) * longitud_subbanda
        subbandas.append(espectro_magnitud[inicio:fin])

    return subbandas


def calcular_energia_subbanda(subbanda):
    """Calcula la energía promedio de una subbanda."""
    if len(subbanda) == 0:
        return 0.0
    energia = np.mean(np.abs(subbanda) ** 2)
    return float(energia)


def calcular_vector_energias(espectro_magnitud, numero_subbandas):
    """Devuelve el vector de energías de cada subbanda."""
    subbandas = dividir_espectro_en_subbandas(espectro_magnitud, numero_subbandas)
    energias = [calcular_energia_subbanda(sb) for sb in subbandas]
    return np.array(energias, dtype=np.float32)


def calcular_estadisticos_energias(lista_vectores_energias):
    """Calcula medias y desviaciones estándar por subbanda."""
    matriz = np.vstack(lista_vectores_energias)
    medias = np.mean(matriz, axis=0)
    desviaciones = np.std(matriz, axis=0, ddof=0)
    return medias, desviaciones
