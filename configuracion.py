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


# Parámetros de audio y análisis (según lab5)
FRECUENCIA_MUESTREO_OBJETIVO = 16000  # Hz (igual que lab5 usa 44100, pero 16kHz es suficiente para voz)
N_FFT = 4096  # Tamaño fijo de ventana FFT (como lab5)
NUMERO_SUBBANDAS = 6  # Número de segmentos temporales (lab5 usa 3, usamos 6)
VENTANA = "hamming"  # Tipo de ventana (igual que lab5)

# Archivo donde se guardan los umbrales de energía (resultado del entrenamiento)
ARCHIVO_UMBRALES = Path("umbrales_comandos.json")

# Duración de la grabación desde el micrófono (segundos)
# Aumentado para capturar palabras completas en español
DURACION_GRABACION_SEGUNDOS = 1.5  # 1.5 segundos para capturar "segmentar", "comprimir", "cifrar"
