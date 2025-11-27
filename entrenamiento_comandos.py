import json
from pathlib import Path

from configuracion import (
    DIRECTORIOS_COMANDOS,
    FRECUENCIA_MUESTREO_OBJETIVO,
    NUMERO_SUBBANDAS,
    ARCHIVO_UMBRALES,
)
from procesamiento_audio import (
    cargar_senal_desde_wav,
    re_muestrear_senal,
    eliminar_silencio_voz,
    aplicar_preenfasis,
    filtrar_ruido_pasabajos,
    ajustar_longitud_potencia_de_dos,
    calcular_fft_magnitud,
)
from banco_filtros import (
    calcular_vector_energias,
    calcular_estadisticos_energias,
    normalizar_vector_energia,
)


def obtener_rutas_wav_directorio(directorio):
    return sorted(Path(directorio).glob("*.wav"))


def _seleccionar_directorio_existente(rutas_candidatas):
    """Devuelve la primera ruta que exista; si ninguna existe, usa la primera para informar error."""
    for ruta in rutas_candidatas:
        if Path(ruta).exists():
            return Path(ruta)
    return Path(rutas_candidatas[0])


def procesar_senal_entrenamiento(ruta_archivo):
    """Aplica todos los pasos del documento a una sola grabación de entrenamiento.
    Sigue la teoría: acondicionamiento -> FFT -> banco de filtros -> energías."""
    fs_original, senal = cargar_senal_desde_wav(ruta_archivo)
    senal = re_muestrear_senal(fs_original, senal)
    senal = eliminar_silencio_voz(senal, FRECUENCIA_MUESTREO_OBJETIVO)  # Eliminar silencios
    senal = aplicar_preenfasis(senal)  # Pre-énfasis para voz
    senal = filtrar_ruido_pasabajos(senal, FRECUENCIA_MUESTREO_OBJETIVO)
    senal = ajustar_longitud_potencia_de_dos(senal)
    espectro_magnitud = calcular_fft_magnitud(senal)
    vector_energias = calcular_vector_energias(espectro_magnitud, NUMERO_SUBBANDAS)
    vector_energias = normalizar_vector_energia(vector_energias)
    return vector_energias


def entrenar_modelo_comandos():
    """Fase de entrenamiento: genera los vectores de umbrales para cada comando."""
    resultados_umbrales = {}

    for nombre_comando, rutas_candidatas in DIRECTORIOS_COMANDOS.items():
        ruta_directorio = _seleccionar_directorio_existente(rutas_candidatas)
        print(f"Entrenando comando: {nombre_comando} en {ruta_directorio}")
        rutas_wav = obtener_rutas_wav_directorio(ruta_directorio)

        if not rutas_wav:
            print(f"  [ADVERTENCIA] No se encontraron archivos .wav en {ruta_directorio}")
            continue

        lista_vectores_energias = []

        for ruta_wav in rutas_wav:
            print(f"    Procesando: {ruta_wav.name}")
            vector_energias = procesar_senal_entrenamiento(ruta_wav)
            lista_vectores_energias.append(vector_energias)

        medias, desviaciones = calcular_estadisticos_energias(lista_vectores_energias)

        resultados_umbrales[nombre_comando] = {
            "medias": medias.tolist(),
            "desviaciones": desviaciones.tolist(),
        }

    with open(ARCHIVO_UMBRALES, "w", encoding="utf-8") as f:
        json.dump(resultados_umbrales, f, indent=4, ensure_ascii=False)

    print(f"\nEntrenamiento finalizado. Umbrales guardados en: {ARCHIVO_UMBRALES}")


if __name__ == "__main__":
    entrenar_modelo_comandos()
