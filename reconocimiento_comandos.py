import json
from pathlib import Path

import numpy as np
import cv2

from configuracion import (
    ARCHIVO_UMBRALES,
    FRECUENCIA_MUESTREO_OBJETIVO,
    N_FFT,
    NUMERO_SUBBANDAS,
    VENTANA,
)
from procesamiento_audio import (
    aplicar_preenfasis,
    filtrar_ruido_pasabajos,
    ajustar_longitud_potencia_de_dos,
    calcular_fft_magnitud,
)
from banco_filtros import calcular_vector_energias, normalizar_vector_energia

# Constantes para el reconocimiento (método lab5)
EPSILON_DESVIACION = 1e-6  # Para evitar división por cero en normalización


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
    """Procesa señal para reconocimiento (método EXACTO del lab5).
    
    La señal ya viene con N muestras exactas de grabar_audio_microfono().
    """
    from banco_filtros import calcular_vector_energias_temporal
    
    # Calcular energías usando método lab5 (incluye todo el preprocesamiento)
    vector_energias = calcular_vector_energias_temporal(
        senal, 
        fs=FRECUENCIA_MUESTREO_OBJETIVO,
        N=N_FFT,
        K=NUMERO_SUBBANDAS,
        window=VENTANA
    )
    
    return vector_energias


def reconocer_comando_por_energia(vector_energias, umbrales):
    """Devuelve el comando reconocido usando DISTANCIA MÍNIMA A PATRONES (método lab5).
    
    MÉTODO CORRECTO LAB5:
    - Compara con TODOS los patrones de referencia de cada comando
    - Para cada comando, calcula distancia mínima entre todos sus patrones
    - El comando con menor distancia mínima gana
    """
    E = vector_energias
    
    print(f"\n  [ANÁLISIS] Vector entrada: {E}")
    print(f"  [ANÁLISIS] Suma: {np.sum(E):.6f}")
    print(f"  [ANÁLISIS] Min: {np.min(E):.6f}, Max: {np.max(E):.6f}")
    
    distancias_minimas = {}
    
    # Obtener comandos del formato correcto
    commands = umbrales.get("commands", umbrales)
    
    for nombre_comando, datos_comando in commands.items():
        # Obtener patrones de referencia
        patrones = datos_comando.get("_patterns", [])
        
        if not patrones:
            # Fallback: usar media si no hay patrones
            mean = np.array(datos_comando.get("mean", datos_comando.get("medias", [])), dtype=float)
            d_min = np.linalg.norm(E - mean)
            print(f"  {nombre_comando}: usando MEDIA (dist={d_min:.4f})")
        else:
            # MÉTODO LAB5: Distancia mínima a TODOS los patrones
            distancias = []
            for patron in patrones:
                patron_array = np.array(patron, dtype=float)
                dist = np.linalg.norm(E - patron_array)
                distancias.append(dist)
            
            d_min = min(distancias)
            d_mean = np.mean(distancias)
            print(f"  {nombre_comando}: min={d_min:.4f}, mean={d_mean:.4f} ({len(patrones)} patrones)")
        
        distancias_minimas[nombre_comando] = float(d_min)
    
    # El comando con menor distancia mínima gana
    mejor_comando = min(distancias_minimas.items(), key=lambda kv: kv[1])[0]
    mejor_dist = distancias_minimas[mejor_comando]
    
    print(f"\n  ✓ GANADOR: {mejor_comando} (dist_min={mejor_dist:.4f})")
    
    # Mostrar ranking completo
    sorted_dists = sorted(distancias_minimas.items(), key=lambda x: x[1])
    print(f"\n  Ranking:")
    for i, (label, dist) in enumerate(sorted_dists, 1):
        marca = "★" if i == 1 else " "
        print(f"    {marca} {i}° {label}: {dist:.4f}")
    
    return mejor_comando, mejor_dist


def cargar_imagen_opencv_unicode(ruta):
    """Carga imagen con OpenCV soportando rutas Unicode."""
    import numpy as np
    try:
        with open(ruta, 'rb') as f:
            datos = f.read()
        arr = np.frombuffer(datos, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error al cargar imagen: {e}")
        return None


def ejecutar_operacion_imagen(comando, ruta_imagen):
    """Aplica una operacion de ejemplo sobre una imagen de acuerdo al comando reconocido."""
    import cv2
    import tkinter as tk
    
    # Crear ventana temporal si no existe
    try:
        root = tk._default_root
        if root is None:
            root = tk.Tk()
            root.withdraw()
    except:
        root = tk.Tk()
        root.withdraw()
    
    # COMANDO_1: Segmentación con K-means
    if comando == "COMANDO_1":  # segmentar
        from ventana_segmentacion import VentanaSegmentacionKMeans
        VentanaSegmentacionKMeans(root, ruta_imagen)
        return
    
    # COMANDO_2: Compresión con DCT-2D
    elif comando == "COMANDO_2":  # comprimir
        from ventana_compresion import VentanaCompresionDCT
        VentanaCompresionDCT(root, ruta_imagen)
        return
    
    # COMANDO_3: Cifrado (por implementar)
    elif comando == "COMANDO_3":  # cifrar
        from ventana_cifrado import VentanaCifradoFrDCT
        VentanaCifradoFrDCT(root, ruta_imagen)
        return
    
    else:
        print("Comando no reconocido para operacion de imagen.")
