from pathlib import Path

# Directorio base donde se almacenan las grabaciones de entrenamiento.
# Dentro de esta carpeta se recomiendan tres subcarpetas: comando_1, comando_2, comando_3
RUTA_BASE_DATOS = Path("datos_entrenamiento")

# Directorios de entrenamiento: cada carpeta contiene muchos .wav de ese comando
# Se aceptan alias (A, B, C) para compatibilidad con datasets existentes.
DIRECTORIOS_COMANDOS = {
    "COMANDO_1": [RUTA_BASE_DATOS / "comando_1", RUTA_BASE_DATOS / "A"],
    "COMANDO_2": [RUTA_BASE_DATOS / "comando_2", RUTA_BASE_DATOS / "B"],
    "COMANDO_3": [RUTA_BASE_DATOS / "comando_3", RUTA_BASE_DATOS / "C"],
}

# Etiquetas legibles asociadas a cada comando
ETIQUETAS_COMANDOS = {
    "COMANDO_1": "segmentar",
    "COMANDO_2": "comprimir",
    "COMANDO_3": "cifrar",
}


# Parámetros de audio y análisis (según teoría adjunta)
FRECUENCIA_MUESTREO_OBJETIVO = 16000  # Hz - TODAS las señales se remuestrean a esta fs
N_FFT = 4096  # Tamaño fijo de ventana FFT
NUMERO_SUBBANDAS = 16  # Número de filtros/particiones (TEORÍA permite más de 4 para mejor discriminación)
VENTANA = "hamming"  # Tipo de ventana temporal

# Parámetros de preprocesado (teoría)
FRECUENCIA_CORTE_PB = 3500  # Hz - Filtro pasa-bajas (eliminar ruido > 3.5 kHz)
ORDEN_FILTRO = 4  # Orden del filtro Butterworth
PREENFASIS_ALPHA = 0.97  # Coeficiente de pre-énfasis

# Umbrales de silencio
UMBRAL_ENERGIA_SILENCIO = 0.01  # Para detectar/eliminar silencio
MARGEN_SILENCIO_MS = 100  # ms de margen al recortar silencio

# Archivo donde se guardan los umbrales de energía (resultado del entrenamiento)
ARCHIVO_UMBRALES = Path("umbrales_comandos.json")

# Duración de la grabación desde el micrófono (segundos)
# 1.0s permite buscar ventana de 0.256s (~4096 muestras) con máxima energía
DURACION_GRABACION_SEGUNDOS = 1.0
