import json
from pathlib import Path
import numpy as np

from configuracion import (
    DIRECTORIOS_COMANDOS,
    FRECUENCIA_MUESTREO_OBJETIVO,
    N_FFT,
    NUMERO_SUBBANDAS,
    VENTANA,
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
    calcular_estadisticos_energias,
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
    """Procesa una grabación de entrenamiento (método EXACTO del lab5)."""
    import soundfile as sf
    from banco_filtros import calcular_vector_energias_temporal
    
    # Cargar archivo WAV (como lab5)
    x, fs_file = sf.read(str(ruta_archivo), dtype='float32')
    
    # Si es estereo, convertir a mono
    if x.ndim > 1:
        x = x.mean(axis=1)
    
    # Ajustar a N muestras (como lab5: truncar o rellenar)
    x = x[:N_FFT] if len(x) >= N_FFT else np.pad(x, (0, N_FFT - len(x)))
    
    # Calcular energías (incluye TODO el preprocesamiento)
    vector_energias = calcular_vector_energias_temporal(
        x,
        fs=fs_file,  # Usar fs del archivo (lab5 no remuestrea en entrenamiento)
        N=N_FFT,
        K=NUMERO_SUBBANDAS,
        window=VENTANA
    )
    
    return vector_energias


def entrenar_modelo_comandos():
    """Fase de entrenamiento: guarda PATRONES DE REFERENCIA (método EXACTO lab5)."""
    
    # Estructura como lab5
    resultados = {
        "fs": FRECUENCIA_MUESTREO_OBJETIVO,
        "N": N_FFT,
        "K": NUMERO_SUBBANDAS,
        "window": VENTANA,
        "commands": {}
    }

    for nombre_comando, rutas_candidatas in DIRECTORIOS_COMANDOS.items():
        ruta_directorio = _seleccionar_directorio_existente(rutas_candidatas)
        print(f"Entrenando comando: {nombre_comando} en {ruta_directorio}")
        rutas_wav = obtener_rutas_wav_directorio(ruta_directorio)

        if not rutas_wav:
            print(f"  [ADVERTENCIA] No se encontraron archivos .wav en {ruta_directorio}")
            continue

        # GUARDAR TODOS LOS PATRONES (como lab5)
        patrones = []

        for ruta_wav in rutas_wav:
            print(f"    Procesando: {ruta_wav.name}")
            vector_energias = procesar_senal_entrenamiento(ruta_wav)
            patrones.append(vector_energias.tolist())

        # Calcular estadísticas para información
        matriz = np.array(patrones)
        medias = np.mean(matriz, axis=0)
        desviaciones = np.std(matriz, axis=0, ddof=0)

        # Guardar TODO (patrones + estadísticas)
        resultados["commands"][nombre_comando] = {
            "mean": medias.tolist(),
            "std": desviaciones.tolist(),
            "count": len(patrones),
            "_patterns": patrones  # TODOS los patrones de referencia
        }
        
        print(f"    ✓ {len(patrones)} patrones guardados")

    with open(ARCHIVO_UMBRALES, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    print(f"\n✓ Entrenamiento finalizado. Patrones guardados en: {ARCHIVO_UMBRALES}")


if __name__ == "__main__":
    entrenar_modelo_comandos()
