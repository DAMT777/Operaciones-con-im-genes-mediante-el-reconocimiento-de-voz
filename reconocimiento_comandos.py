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

EPSILON_DESVIACION = 1e-6

def cargar_umbrales_desde_archivo():
    if not Path(ARCHIVO_UMBRALES).exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de umbrales: {ARCHIVO_UMBRALES}. Ejecute primero el entrenamiento."
        )
    with open(ARCHIVO_UMBRALES, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return datos

def procesar_senal_para_reconocimiento(senal):
    from banco_filtros import calcular_vector_energias_temporal
    
    vector_energias = calcular_vector_energias_temporal(
        senal, 
        fs=FRECUENCIA_MUESTREO_OBJETIVO,
        N=N_FFT,
        K=NUMERO_SUBBANDAS,
        window=VENTANA
    )
    
    return vector_energias

def reconocer_comando_por_energia(vector_energias, umbrales):
    E = np.array(vector_energias, dtype=float)
    
    norma_E = np.linalg.norm(E)
    if norma_E > 1e-10:
        E_norm = E / norma_E
    else:
        E_norm = E
    
    print(f"\n{'='*60}")
    print(f"RECONOCIMIENTO DE COMANDO")
    print(f"{'='*60}")
    print(f"Vector entrada: {E}")
    print(f"Energía total: {np.sum(E):.6f}")
    print(f"Vector normalizado: {E_norm}")
    print(f"{'-'*60}")
    
    distancias = {}
    commands = umbrales.get("commands", umbrales)
    
    for nombre_comando, datos_comando in commands.items():
        umbral_vector = np.array(datos_comando.get("mean", []), dtype=float)
        
        if len(umbral_vector) == 0:
            print(f"⚠ {nombre_comando}: sin vector de umbrales")
            continue
        
        norma_umbral = np.linalg.norm(umbral_vector)
        if norma_umbral > 1e-10:
            umbral_norm = umbral_vector / norma_umbral
        else:
            umbral_norm = umbral_vector
        
        distancia = np.linalg.norm(E_norm - umbral_norm)
        distancias[nombre_comando] = distancia
        
        print(f"{nombre_comando}:")
        print(f"  Umbral normalizado: {umbral_norm}")
        print(f"  Distancia: {distancia:.6f}")
    
    if len(distancias) == 0:
        print(f"✗ No hay comandos para comparar")
        return None, float('inf')
    
    mejor_comando = min(distancias.items(), key=lambda x: x[1])
    comando_ganador = mejor_comando[0]
    distancia_minima = mejor_comando[1]
    
    print(f"{'-'*60}")
    print(f"RANKING DE COMANDOS (por cercanía):")
    sorted_distancias = sorted(distancias.items(), key=lambda x: x[1])
    for i, (cmd, dist) in enumerate(sorted_distancias, 1):
        marca = "★" if cmd == comando_ganador else " "
        print(f"  {marca} {i}° {cmd}: {dist:.6f}")
    
    print(f"{'='*60}")
    print(f"✓ COMANDO RECONOCIDO: {comando_ganador}")
    print(f"  Distancia: {distancia_minima:.6f}")
    print(f"{'='*60}\n")
    
    return comando_ganador, distancia_minima

def cargar_imagen_opencv_unicode(ruta):
    import numpy as np
    try:
        with open(ruta, 'rb') as f:
            datos = f.read()
        arr = np.frombuffer(datos, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error al cargar imagen: {e}")
        return None

def ejecutar_operacion_imagen(comando, ruta_imagen, pausar_callback=None, reanudar_callback=None):
    import cv2
    import tkinter as tk
    
    try:
        root = tk._default_root
        if root is None:
            root = tk.Tk()
            root.withdraw()
    except:
        root = tk.Tk()
        root.withdraw()
    
    if comando == "COMANDO_1":
        from ventana_segmentacion import VentanaSegmentacionKMeans
        VentanaSegmentacionKMeans(root, ruta_imagen, pausar_callback, reanudar_callback)
        return
    
    elif comando == "COMANDO_2":
        from ventana_compresion import VentanaCompresionDCT
        VentanaCompresionDCT(root, ruta_imagen, pausar_callback, reanudar_callback)
        return
    
    elif comando == "COMANDO_3":
        from ventana_cifrado import VentanaCifradoFrDCT
        VentanaCifradoFrDCT(root, ruta_imagen, pausar_callback, reanudar_callback)
        return
    
    else:
        print("Comando no reconocido para operacion de imagen.")
