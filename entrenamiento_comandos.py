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
    """Devuelve la primera ruta que exista; si ninguna existe, usa la primera para informar error."""
    for ruta in rutas_candidatas:
        if Path(ruta).exists():
            return Path(ruta)
    return Path(rutas_candidatas[0])


def procesar_senal_entrenamiento(ruta_archivo):
    """Procesa una grabación de entrenamiento CON PREPROCESADO COMPLETO (teoría).
    MISMO procedimiento que captura de micrófono para CONSISTENCIA."""
    from banco_filtros import calcular_vector_energias_temporal
    
    # 1. CARGAR WAV (con su fs original)
    fs_original, x = cargar_senal_desde_wav(ruta_archivo)
    
    # 2. REMUESTREAR A 16 kHz (CRÍTICO: alinear con micrófono)
    x = re_muestrear_senal(fs_original, x)
    fs = FRECUENCIA_MUESTREO_OBJETIVO
    
    # 3. FILTRO PASA-BAJAS (eliminar ruido > 3.5 kHz)
    x = filtrar_ruido_pasabajos(x, fs)
    
    # 4. ELIMINAR SILENCIO (recortar inicio/fin sin voz)
    x = eliminar_silencio_voz(x, fs)
    
    # 5. EXTRAER VENTANA DE MÁXIMA ENERGÍA (igual que micrófono)
    x = extraer_ventana_maxima_energia(x, N_FFT)
    
    # 6. Calcular energías (ya incluye DC removal, pre-énfasis, ventana)
    vector_energias = calcular_vector_energias_temporal(
        x,
        fs=fs,  # Ahora TODOS a 16 kHz
        N=N_FFT,
        K=NUMERO_SUBBANDAS,
        window=VENTANA
    )
    
    return vector_energias


def entrenar_modelo_comandos():
    """Entrenamiento según TEORÍA EXACTA del documento.
    
    PROCESO:
    1. Para cada comando, procesar TODAS las grabaciones M
    2. Calcular vector de energías [E1, E2, E3, E4] para cada grabación
    3. PROMEDIAR cada componente: EC1 = ΣEC1/M, EC2 = ΣEC2/M, ...
    4. Guardar vector promedio [EC1, EC2, EC3, EC4] como UMBRAL del comando
    
    IMPORTANTE: Según teoría, el umbral es simplemente el PROMEDIO de energías,
    NO un percentil ni distancia calculada.
    """
    
    resultados = {
        "fs": FRECUENCIA_MUESTREO_OBJETIVO,
        "N": N_FFT,
        "K": NUMERO_SUBBANDAS,
        "window": VENTANA,
        "commands": {}
    }

    for nombre_comando, rutas_candidatas in DIRECTORIOS_COMANDOS.items():
        ruta_directorio = _seleccionar_directorio_existente(rutas_candidatas)
        print(f"\n{'='*60}")
        print(f"Entrenando comando: {nombre_comando}")
        print(f"Directorio: {ruta_directorio}")
        print(f"{'='*60}")
        
        rutas_wav = obtener_rutas_wav_directorio(ruta_directorio)

        if not rutas_wav:
            print(f"  ⚠ ADVERTENCIA: No se encontraron archivos .wav")
            continue

        # Lista para almacenar vectores de energía de todas las grabaciones
        vectores_energia = []

        for idx, ruta_wav in enumerate(rutas_wav, 1):
            try:
                vector_energias = procesar_senal_entrenamiento(ruta_wav)
                vectores_energia.append(vector_energias)
                
                if idx % 20 == 0:
                    print(f"  Procesados: {idx}/{len(rutas_wav)} archivos...")
            except Exception as e:
                print(f"  ✗ Error en {ruta_wav.name}: {e}")
                continue

        if len(vectores_energia) == 0:
            print(f"  ✗ No se pudo procesar ningún archivo")
            continue
        
        # Convertir a matriz para cálculos
        matriz_energias = np.array(vectores_energia)
        M = len(vectores_energia)  # Número de grabaciones
        
        # PASO CLAVE SEGÚN TEORÍA: Promediar cada componente
        # EC1 = ΣEC1/M, EC2 = ΣEC2/M, EC3 = ΣEC3/M, EC4 = ΣEC4/M
        vector_promedio = np.mean(matriz_energias, axis=0)
        
        # Calcular desviación estándar (solo para información)
        desviacion = np.std(matriz_energias, axis=0, ddof=0)
        
        # Guardar resultado según teoría
        resultados["commands"][nombre_comando] = {
            "mean": vector_promedio.tolist(),  # Vector de umbrales [EC1, EC2, EC3, EC4]
            "std": desviacion.tolist(),
            "count": M
        }
        
        print(f"\n  ✓ Entrenamiento completado:")
        print(f"    - Grabaciones procesadas: {M}")
        print(f"    - Vector de umbrales: {vector_promedio}")
        print(f"    - Desviación estándar: {desviacion}")

    # Guardar archivo JSON
    with open(ARCHIVO_UMBRALES, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✓ Entrenamiento finalizado exitosamente")
    print(f"✓ Umbrales guardados en: {ARCHIVO_UMBRALES}")
    print(f"{'='*60}")


if __name__ == "__main__":
    entrenar_modelo_comandos()
