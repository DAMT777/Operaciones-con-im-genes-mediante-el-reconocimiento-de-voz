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
    extraer_ventana_maxima_energia,
)
from banco_filtros import (
    calcular_estadisticos_energias,
)

def obtener_rutas_wav_directorio(directorio):
    return sorted(Path(directorio).glob("*.wav"))

def _seleccionar_directorio_existente(rutas_candidatas):
    for ruta in rutas_candidatas:
        if Path(ruta).exists():
            return Path(ruta)
    return Path(rutas_candidatas[0])

def procesar_senal_entrenamiento(ruta_archivo):
    from banco_filtros import calcular_vector_energias_temporal
    
    fs_original, senal = cargar_senal_desde_wav(ruta_archivo)
    senal = re_muestrear_senal(fs_original, senal)
    senal = filtrar_ruido_pasabajos(senal, FRECUENCIA_MUESTREO_OBJETIVO)
    senal = eliminar_silencio_voz(senal, FRECUENCIA_MUESTREO_OBJETIVO)
    senal = aplicar_preenfasis(senal)
    senal = extraer_ventana_maxima_energia(senal, N_FFT)
    
    vector_energias = calcular_vector_energias_temporal(
        senal,
        fs=FRECUENCIA_MUESTREO_OBJETIVO,
        N=N_FFT,
        K=NUMERO_SUBBANDAS,
        window=VENTANA
    )
    
    return vector_energias

def entrenar_modelo_comandos(directorios_comandos):
    resultados = {}
    
    for nombre_comando, rutas_candidatas in directorios_comandos.items():
        directorio = _seleccionar_directorio_existente(rutas_candidatas)
        archivos_wav = obtener_rutas_wav_directorio(directorio)
        
        print(f"\n{'='*60}")
        print(f"COMANDO: {nombre_comando}")
        print(f"Directorio: {directorio}")
        print(f"Archivos encontrados: {len(archivos_wav)}")
        print(f"{'='*60}")
        
        if len(archivos_wav) == 0:
            print(f"⚠ No se encontraron archivos .wav")
            continue
        
        vectores_energia = []
        for i, ruta in enumerate(archivos_wav, 1):
            try:
                vector = procesar_senal_entrenamiento(ruta)
                vectores_energia.append(vector)
                print(f"  {i}/{len(archivos_wav)} - {ruta.name}: {vector}")
            except Exception as e:
                print(f"  ✗ Error en {ruta.name}: {e}")
        
        if len(vectores_energia) == 0:
            print(f"⚠ No se procesó ningún archivo correctamente")
            continue
        
        medias, desviaciones = calcular_estadisticos_energias(vectores_energia)
        
        resultados[nombre_comando] = {
            "mean": medias.tolist(),
            "std": desviaciones.tolist(),
            "count": len(vectores_energia)
        }
        
        print(f"\nRESULTADO {nombre_comando}:")
        print(f"  Muestras procesadas: {len(vectores_energia)}")
        print(f"  Media: {medias}")
        print(f"  Desviación: {desviaciones}")
    
    datos_salida = {
        "config": {
            "fs": FRECUENCIA_MUESTREO_OBJETIVO,
            "N": N_FFT,
            "K": NUMERO_SUBBANDAS,
            "window": VENTANA
        },
        "commands": resultados
    }
    
    with open(ARCHIVO_UMBRALES, "w", encoding="utf-8") as f:
        json.dump(datos_salida, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Entrenamiento completado")
    print(f"✓ Umbrales guardados en: {ARCHIVO_UMBRALES}")
    print(f"{'='*60}\n")
    
    return datos_salida

if __name__ == "__main__":
    print("Iniciando entrenamiento del modelo de comandos...")
    entrenar_modelo_comandos(DIRECTORIOS_COMANDOS)
