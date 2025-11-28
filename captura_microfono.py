import sounddevice as sd
import numpy as np

from configuracion import FRECUENCIA_MUESTREO_OBJETIVO, N_FFT
from procesamiento_audio import filtrar_ruido_pasabajos, eliminar_silencio_voz

def grabar_audio_microfono():
    duracion_grabacion = 1.0
    
    try:
        device_info = sd.query_devices(kind='input')
        print(f"[MIC] Usando: {device_info['name']}")
    except:
        pass
    
    data = sd.rec(
        int(duracion_grabacion * FRECUENCIA_MUESTREO_OBJETIVO),
        samplerate=FRECUENCIA_MUESTREO_OBJETIVO,
        channels=1,
        dtype='float32',
        blocking=True
    )
    
    x_completo = data.flatten()
    
    x_completo = filtrar_ruido_pasabajos(x_completo, FRECUENCIA_MUESTREO_OBJETIVO)
    
    x_completo = eliminar_silencio_voz(x_completo, FRECUENCIA_MUESTREO_OBJETIVO)
    
    mejor_energia = -1
    mejor_inicio = 0
    
    for i in range(0, max(1, len(x_completo) - N_FFT), max(1, N_FFT // 4)):
        if i + N_FFT > len(x_completo):
            break
        ventana = x_completo[i:i + N_FFT]
        energia = np.sum(ventana ** 2)
        
        if energia > mejor_energia:
            mejor_energia = energia
            mejor_inicio = i
    
    if mejor_inicio + N_FFT <= len(x_completo):
        x = x_completo[mejor_inicio:mejor_inicio + N_FFT]
    else:
        x = x_completo[:N_FFT] if len(x_completo) >= N_FFT else x_completo
    
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    
    rms = np.sqrt(np.mean(x ** 2))
    if rms > 1e-6:
        x = x * (0.1 / rms)
        print(f"[MIC] RMS normalizado: {rms:.6f} → 0.1")
    
    return x
