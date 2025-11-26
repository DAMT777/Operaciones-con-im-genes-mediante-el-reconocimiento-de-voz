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


def calcular_energia_subbanda(subbanda):
    """Calcula la energia promedio de una subbanda."""
    if len(subbanda) == 0:
        return 0.0
    energia = np.mean(np.abs(subbanda) ** 2)
    return float(energia)


def calcular_vector_energias(espectro_magnitud, numero_subbandas):
    """Devuelve el vector de energias de cada subbanda."""
    subbandas = dividir_espectro_en_subbandas(espectro_magnitud, numero_subbandas)
    energias = [calcular_energia_subbanda(sb) for sb in subbandas]
    return np.array(energias, dtype=np.float32)


def calcular_estadisticos_energias(lista_vectores_energias):
    """Calcula medias y desviaciones estandar por subbanda."""
    matriz = np.vstack(lista_vectores_energias)
    medias = np.mean(matriz, axis=0)
    desviaciones = np.std(matriz, axis=0, ddof=0)
    return medias, desviaciones


def normalizar_vector_energia(vector):
    """Normaliza el vector de energias para eliminar dependencia del volumen."""
    energia_total = np.sum(vector)
    if energia_total <= 0:
        return np.array(vector, dtype=np.float32)
    return (vector / energia_total).astype(np.float32)
