import sounddevice as sd
import numpy as np

from configuracion import FRECUENCIA_MUESTREO_OBJETIVO, DURACION_GRABACION_SEGUNDOS


def grabar_audio_microfono(duracion_segundos=DURACION_GRABACION_SEGUNDOS):
    """Graba audio desde el micrófono y devuelve la señal como vector float32."""
    print("Grabando audio... Hable ahora.")
    grabacion = sd.rec(
        int(duracion_segundos * FRECUENCIA_MUESTREO_OBJETIVO),
        samplerate=FRECUENCIA_MUESTREO_OBJETIVO,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    senal = grabacion[:, 0]
    print("Grabación finalizada.\n")
    return senal
