import sounddevice as sd
import numpy as np

from configuracion import FRECUENCIA_MUESTREO_OBJETIVO, N_FFT


def grabar_audio_microfono():
    """Graba audio desde el micrófono con duración más larga para capturar voz.
    Graba 1 segundo y extrae la ventana de N muestras con mayor energía."""
    
    # Grabar 1 segundo completo
    duracion_grabacion = 1.0  # segundos
    
    data = sd.rec(
        int(duracion_grabacion * FRECUENCIA_MUESTREO_OBJETIVO),
        samplerate=FRECUENCIA_MUESTREO_OBJETIVO,
        channels=1,
        dtype='float32',
        blocking=True
    )
    
    x_completo = data.flatten()
    
    # Buscar la ventana de N muestras con MAYOR energía (donde está la voz)
    mejor_energia = -1
    mejor_inicio = 0
    
    # Deslizar ventana de N muestras
    for i in range(0, len(x_completo) - N_FFT, N_FFT // 4):
        ventana = x_completo[i:i + N_FFT]
        energia = np.sum(ventana ** 2)
        
        if energia > mejor_energia:
            mejor_energia = energia
            mejor_inicio = i
    
    # Extraer la ventana con mayor energía
    x = x_completo[mejor_inicio:mejor_inicio + N_FFT]
    
    # Ajustar a N exactamente
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    
    return x
