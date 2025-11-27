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


# Parámetros de audio y análisis
FRECUENCIA_MUESTREO_OBJETIVO = 16000  # Hz
NUMERO_SUBBANDAS = 6  # Punto medio para balance entre discriminación y robustez
MARGEN_ERROR_RELATIVO = 0.05  # 5% - Más tolerante
MARGEN_PUNTUACION = 0.15  # 15% - Diferencia mínima entre mejor y segundo puntaje

# Archivo donde se guardan los umbrales de energía (resultado del entrenamiento)
ARCHIVO_UMBRALES = Path("umbrales_comandos.json")

# Duración de la grabación desde el micrófono (segundos)
DURACION_GRABACION_SEGUNDOS = 1.2  # Reducido para capturar mejor la palabra sin silencios
