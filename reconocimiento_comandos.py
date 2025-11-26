import json
from pathlib import Path

import numpy as np

from configuracion import (
    ARCHIVO_UMBRALES,
    FRECUENCIA_MUESTREO_OBJETIVO,
    NUMERO_SUBBANDAS,
    MARGEN_ERROR_RELATIVO,
)
from procesamiento_audio import (
    filtrar_ruido_pasabajos,
    ajustar_longitud_potencia_de_dos,
    calcular_fft_magnitud,
)
from banco_filtros import calcular_vector_energias, normalizar_vector_energia

# Z-score maximo permitido por subbanda para aceptar un comando
UMBRAL_Z = 3.0
EPSILON_DESVIACION = 1e-6


def cargar_umbrales_desde_archivo():
    """Carga el archivo JSON con los umbrales de cada comando."""
    if not Path(ARCHIVO_UMBRALES).exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de umbrales: {ARCHIVO_UMBRALES}. Ejecute primero el entrenamiento."
        )
    with open(ARCHIVO_UMBRALES, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return datos


def procesar_senal_para_reconocimiento(senal):
    """Reproduce el mismo flujo de procesamiento usado en el entrenamiento."""
    senal_filtrada = filtrar_ruido_pasabajos(senal, FRECUENCIA_MUESTREO_OBJETIVO)
    senal_ajustada = ajustar_longitud_potencia_de_dos(senal_filtrada)
    espectro_magnitud = calcular_fft_magnitud(senal_ajustada)
    vector_energias = calcular_vector_energias(espectro_magnitud, NUMERO_SUBBANDAS)
    vector_energias = normalizar_vector_energia(vector_energias)
    return vector_energias


def _desviaciones_seguras(desviaciones, medias):
    """Piso minimo de desviacion para evitar divisiones por cero o casi cero."""
    tolerancia_relativa = MARGEN_ERROR_RELATIVO * np.abs(medias)
    return np.maximum(desviaciones, tolerancia_relativa + EPSILON_DESVIACION)


def reconocer_comando_por_energia(vector_energias, umbrales):
    """Devuelve el comando reconocido y el puntaje z-score total, o (None, inf)."""
    vector_normalizado = normalizar_vector_energia(vector_energias)
    mejor_comando = None
    mejor_puntaje = float("inf")

    for nombre_comando, datos_comando in umbrales.items():
        medias = normalizar_vector_energia(
            np.array(datos_comando["medias"], dtype=np.float32)
        )
        desviaciones = np.array(datos_comando["desviaciones"], dtype=np.float32)
        desv_seguras = _desviaciones_seguras(desviaciones, medias)

        zscores = np.abs(vector_normalizado - medias) / desv_seguras
        max_z = float(np.max(zscores))
        if max_z > UMBRAL_Z:
            continue

        puntaje = float(np.sum(zscores))

        if puntaje < mejor_puntaje:
            mejor_puntaje = puntaje
            mejor_comando = nombre_comando

    return mejor_comando, mejor_puntaje


def ejecutar_operacion_imagen(comando, ruta_imagen):
    """Aplica una operacion de ejemplo sobre una imagen de acuerdo al comando reconocido."""
    import cv2

    imagen = cv2.imread(str(ruta_imagen))
    if imagen is None:
        print(f"No se pudo cargar la imagen: {ruta_imagen}")
        return

    if comando == "COMANDO_1":
        imagen_salida = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)
    elif comando == "COMANDO_2":
        imagen_salida = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    elif comando == "COMANDO_3":
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen_salida = cv2.Canny(gris, 80, 160)
    else:
        print("Comando no reconocido para operacion de imagen.")
        return

    cv2.imshow(f"Resultado para {comando}", imagen_salida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
