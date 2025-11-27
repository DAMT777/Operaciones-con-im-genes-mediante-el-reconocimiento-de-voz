import json
from pathlib import Path

import numpy as np
import cv2

from configuracion import (
    ARCHIVO_UMBRALES,
    FRECUENCIA_MUESTREO_OBJETIVO,
    NUMERO_SUBBANDAS,
    MARGEN_ERROR_RELATIVO,
    MARGEN_PUNTUACION,
)
from procesamiento_audio import (
    aplicar_preenfasis,
    filtrar_ruido_pasabajos,
    ajustar_longitud_potencia_de_dos,
    calcular_fft_magnitud,
)
from banco_filtros import calcular_vector_energias, normalizar_vector_energia

# Z-score maximo permitido por subbanda para aceptar un comando
UMBRAL_Z_MAX = 4.0  # Z-score máximo permitido en cualquier subbanda (relajado)
UMBRAL_Z_PROMEDIO = 2.5  # Z-score promedio máximo permitido (relajado)
EPSILON_DESVIACION = 1e-9
MIN_CONFIANZA_RELATIVA = 0.15  # 15% mínimo de diferencia entre mejor y segundo (relajado)


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
    """Reproduce el mismo flujo de procesamiento usado en el entrenamiento.
    CRÍTICO: debe seguir EXACTAMENTE los mismos pasos que en entrenamiento."""
    senal_preenfasis = aplicar_preenfasis(senal)
    senal_filtrada = filtrar_ruido_pasabajos(senal_preenfasis, FRECUENCIA_MUESTREO_OBJETIVO)
    senal_ajustada = ajustar_longitud_potencia_de_dos(senal_filtrada)
    espectro_magnitud = calcular_fft_magnitud(senal_ajustada)
    vector_energias = calcular_vector_energias(espectro_magnitud, NUMERO_SUBBANDAS)
    vector_energias = normalizar_vector_energia(vector_energias)
    return vector_energias


def _desviaciones_seguras(desviaciones, medias):
    """Piso minimo de desviacion para evitar divisiones por cero.
    Usa un mínimo absoluto más conservador."""
    tolerancia_relativa = MARGEN_ERROR_RELATIVO * np.abs(medias)
    # Piso mínimo absoluto más conservador
    piso_minimo = np.maximum(tolerancia_relativa, 1e-8)
    return np.maximum(desviaciones, piso_minimo + EPSILON_DESVIACION)


def reconocer_comando_por_energia(vector_energias, umbrales):
    """Devuelve el comando reconocido usando múltiples métricas de similitud.
    Implementa validación estricta para evitar confusiones entre comandos similares."""
    vector_normalizado = normalizar_vector_energia(vector_energias)
    
    print(f"\n  [DEBUG] Vector de entrada normalizado: {vector_normalizado}")
    
    resultados = []
    rechazados = []
    
    for nombre_comando, datos_comando in umbrales.items():
        medias = np.array(datos_comando["medias"], dtype=np.float32)
        desviaciones = np.array(datos_comando["desviaciones"], dtype=np.float32)
        desv_seguras = _desviaciones_seguras(desviaciones, medias)
        
        # Calcular Z-scores
        zscores = np.abs(vector_normalizado - medias) / desv_seguras
        max_z = float(np.max(zscores))
        promedio_z = float(np.mean(zscores))
        
        # Filtro 1: Z-score máximo no debe exceder umbral
        if max_z > UMBRAL_Z_MAX:
            rechazados.append((nombre_comando, f"Z-max={max_z:.2f} > {UMBRAL_Z_MAX}"))
            continue
        
        # Filtro 2: Z-score promedio debe ser razonable
        if promedio_z > UMBRAL_Z_PROMEDIO:
            rechazados.append((nombre_comando, f"Z-prom={promedio_z:.2f} > {UMBRAL_Z_PROMEDIO}"))
            continue
        
        # Métrica 1: Suma de Z-scores (menor es mejor)
        puntaje_z = float(np.sum(zscores))
        
        # Métrica 2: Distancia euclidiana normalizada
        distancia_euclidiana = float(np.sqrt(np.sum((vector_normalizado - medias) ** 2)))
        
        # Métrica 3: Distancia de Manhattan
        distancia_manhattan = float(np.sum(np.abs(vector_normalizado - medias)))
        
        # Métrica 4: Correlación (mayor es mejor, así que invertimos)
        correlacion = float(np.corrcoef(vector_normalizado, medias)[0, 1])
        puntaje_correlacion = 1.0 - correlacion if not np.isnan(correlacion) else 1.0
        
        # Puntaje combinado (menor es mejor)
        puntaje_final = (
            0.40 * puntaje_z +           # 40% peso en Z-scores
            0.25 * distancia_euclidiana * 100 +  # 25% peso en distancia euclidiana
            0.20 * distancia_manhattan * 100 +   # 20% peso en distancia Manhattan
            0.15 * puntaje_correlacion * 10      # 15% peso en correlación
        )
        
        resultados.append({
            'nombre': nombre_comando,
            'puntaje_final': puntaje_final,
            'puntaje_z': puntaje_z,
            'max_z': max_z,
            'promedio_z': promedio_z,
            'distancia_euclidiana': distancia_euclidiana,
            'correlacion': correlacion
        })
    
    if not resultados:
        print("\n  [RECHAZADO] Ningún comando pasó los umbrales")
        for cmd, razon in rechazados:
            print(f"    • {cmd}: {razon}")
        return None, float("inf")
    
    # Ordenar por puntaje final (menor es mejor)
    resultados.sort(key=lambda x: x['puntaje_final'])
    
    mejor = resultados[0]
    mejor_comando = mejor['nombre']
    mejor_puntaje = mejor['puntaje_final']
    
    # Validación de confianza: debe haber diferencia clara con el segundo
    if len(resultados) > 1:
        segundo = resultados[1]
        segundo_puntaje = segundo['puntaje_final']
        
        # Calcular diferencia relativa
        if mejor_puntaje > 0:
            diferencia_relativa = (segundo_puntaje - mejor_puntaje) / mejor_puntaje
        else:
            diferencia_relativa = float('inf')
        
        # Rechazar si no hay suficiente confianza
        if diferencia_relativa < MIN_CONFIANZA_RELATIVA:
            print(f"\n  [RECHAZADO] Ambigüedad detectada:")
            print(f"    1° {mejor_comando}: {mejor_puntaje:.4f}")
            print(f"    2° {segundo['nombre']}: {segundo_puntaje:.4f}")
            print(f"    Diferencia relativa: {diferencia_relativa:.2%} < {MIN_CONFIANZA_RELATIVA:.2%}")
            print(f"    Sugerencia: Hablar más claro o re-entrenar")
            return None, mejor_puntaje
    
    # Mostrar información de diagnóstico
    print(f"\n  [RECONOCIDO] {mejor_comando}")
    print(f"    Puntaje final: {mejor_puntaje:.4f}")
    print(f"    Z-score max: {mejor['max_z']:.3f} (umbral: {UMBRAL_Z_MAX})")
    print(f"    Z-score prom: {mejor['promedio_z']:.3f} (umbral: {UMBRAL_Z_PROMEDIO})")
    print(f"    Correlación: {mejor['correlacion']:.4f}")
    
    if len(resultados) > 1:
        print(f"\n  Otros candidatos:")
        for i, r in enumerate(resultados[1:], 2):
            print(f"    {i}° {r['nombre']}: puntaje={r['puntaje_final']:.4f}, Z-max={r['max_z']:.3f}")
    
    return mejor_comando, mejor_puntaje


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
    
    # Importar la ventana de compresión
    if comando == "COMANDO_2":  # comprimir
        from ventana_compresion import VentanaCompresionDCT
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
        
        # Abrir ventana de compresión DCT
        VentanaCompresionDCT(root, ruta_imagen)
        return

    # Cargar imagen con soporte Unicode
    imagen = cargar_imagen_opencv_unicode(str(ruta_imagen))
    if imagen is None:
        print(f"No se pudo cargar la imagen: {ruta_imagen}")
        print("Sugerencia: Evite rutas con tildes, ñ u otros caracteres especiales.")
        return

    if comando == "COMANDO_1":
        imagen_salida = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)
    elif comando == "COMANDO_3":
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen_salida = cv2.Canny(gris, 80, 160)
    else:
        print("Comando no reconocido para operacion de imagen.")
        return

    cv2.imshow(f"Resultado para {comando}", imagen_salida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
