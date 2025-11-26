from pathlib import Path

# Directorio base donde se almacenan las grabaciones de entrenamiento.
# Dentro de esta carpeta debe crear tres subcarpetas: comando_1, comando_2, comando_3
RUTA_BASE_DATOS = Path("datos_entrenamiento")

# Directorios de entrenamiento: cada carpeta contiene muchos .wav de ese comando
DIRECTORIOS_COMANDOS = {
    "COMANDO_1": RUTA_BASE_DATOS / "comando_1",  # por ejemplo: "abrir"
    "COMANDO_2": RUTA_BASE_DATOS / "comando_2",  # por ejemplo: "cerrar"
    "COMANDO_3": RUTA_BASE_DATOS / "comando_3",  # por ejemplo: "guardar"
}

# Parámetros de audio y análisis
FRECUENCIA_MUESTREO_OBJETIVO = 16000  # Hz
NUMERO_SUBBANDAS = 3
MARGEN_ERROR_RELATIVO = 0.05  # 5 %

# Archivo donde se guardan los umbrales de energía (resultado del entrenamiento)
ARCHIVO_UMBRALES = Path("umbrales_comandos.json")

# Duración de la grabación desde el micrófono (segundos)
DURACION_GRABACION_SEGUNDOS = 1.5
