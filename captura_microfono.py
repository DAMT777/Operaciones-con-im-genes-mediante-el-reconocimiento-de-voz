import sounddevice as sd
import numpy as np

from configuracion import FRECUENCIA_MUESTREO_OBJETIVO, DURACION_GRABACION_SEGUNDOS


def eliminar_silencio(senal, umbral_energia=0.015, margen_muestras=2400):
    """Elimina silencios al inicio y final de la señal.
    Esto mejora el reconocimiento al enfocarse en la parte con voz."""
    if len(senal) == 0:
        return senal
    
    # Calcular energía por ventana
    ventana = 320  # 20ms a 16kHz
    paso = ventana // 4  # Solapamiento del 75%
    
    energia = []
    for i in range(0, len(senal) - ventana, paso):
        e = np.sum(senal[i:i+ventana]**2)
        energia.append(e)
    
    energia = np.array(energia)
    
    if len(energia) == 0 or np.max(energia) == 0:
        return senal
    
    # Normalizar energía
    energia_norm = energia / np.max(energia)
    
    # Encontrar índices donde hay voz (umbral más bajo)
    indices_voz = np.where(energia_norm > umbral_energia)[0]
    
    if len(indices_voz) == 0:
        # Si no se detecta voz, devolver señal completa
        return senal
    
    # Convertir a índices de muestras con márgenes generosos
    inicio_voz = max(0, indices_voz[0] * paso - margen_muestras)
    fin_voz = min(len(senal), indices_voz[-1] * paso + ventana + margen_muestras)
    
    return senal[inicio_voz:fin_voz]


def grabar_audio_microfono(duracion_segundos=DURACION_GRABACION_SEGUNDOS):
    """Graba audio desde el micrófono y devuelve la señal procesada."""
    print("🎤 Grabando audio... Hable AHORA.")
    grabacion = sd.rec(
        int(duracion_segundos * FRECUENCIA_MUESTREO_OBJETIVO),
        samplerate=FRECUENCIA_MUESTREO_OBJETIVO,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    senal = grabacion[:, 0]
    
    # Eliminar silencios para mejorar reconocimiento
    senal = eliminar_silencio(senal)
    
    print(f"✓ Grabación finalizada ({len(senal)} muestras, {len(senal)/FRECUENCIA_MUESTREO_OBJETIVO:.2f}s)\n")
    return senal
