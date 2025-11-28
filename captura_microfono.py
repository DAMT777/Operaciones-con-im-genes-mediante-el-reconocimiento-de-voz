import sounddevice as sd
import numpy as np

from configuracion import FRECUENCIA_MUESTREO_OBJETIVO, N_FFT
from procesamiento_audio import filtrar_ruido_pasabajos, eliminar_silencio_voz


def grabar_audio_microfono():
    """Graba audio desde el micrófono con PREPROCESADO COMPLETO (igual que entrenamiento).
    Graba 1 segundo, aplica filtros, y extrae ventana de N muestras con mayor energía."""
    
    # Grabar 1 segundo completo
    duracion_grabacion = 1.0  # segundos
    
    # Intentar obtener el dispositivo de entrada predeterminado
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
    
    # AMPLIFICAR la señal capturada (ganancia automática AGRESIVA)
    max_val = np.max(np.abs(x_completo))
    if max_val > 0.001:  # Umbral muy bajo para detectar cualquier señal
        # Normalizar a nivel alto (0.8 en lugar de 0.5)
        ganancia = min(0.8 / max_val, 15.0)  # Ganancia hasta x15
        x_completo = x_completo * ganancia
        print(f"[MIC] Ganancia aplicada: x{ganancia:.2f}")
    else:
        print(f"[MIC] Señal muy débil, aplicando ganancia máxima")
        x_completo = x_completo * 15.0
    
    # PREPROCESADO (igual que entrenamiento)
    # 1. Filtro pasa-bajas (eliminar ruido > 3.5 kHz)
    x_completo = filtrar_ruido_pasabajos(x_completo, FRECUENCIA_MUESTREO_OBJETIVO)
    
    # 2. Eliminar silencio (recortar inicio/fin)
    x_completo = eliminar_silencio_voz(x_completo, FRECUENCIA_MUESTREO_OBJETIVO)
    
    # Buscar la ventana de N muestras con MAYOR energía (donde está la voz)
    mejor_energia = -1
    mejor_inicio = 0
    
    # Deslizar ventana de N muestras
    for i in range(0, max(1, len(x_completo) - N_FFT), max(1, N_FFT // 4)):
        if i + N_FFT > len(x_completo):
            break
        ventana = x_completo[i:i + N_FFT]
        energia = np.sum(ventana ** 2)
        
        if energia > mejor_energia:
            mejor_energia = energia
            mejor_inicio = i
    
    # Extraer la ventana con mayor energía
    if mejor_inicio + N_FFT <= len(x_completo):
        x = x_completo[mejor_inicio:mejor_inicio + N_FFT]
    else:
        x = x_completo[:N_FFT] if len(x_completo) >= N_FFT else x_completo
    
def grabar_audio_microfono():
    """Graba audio desde el micrófono con PREPROCESADO COMPLETO (igual que entrenamiento).
    Graba 1 segundo, aplica filtros, y extrae ventana de N muestras con mayor energía."""
    
    # Grabar 1 segundo completo
    duracion_grabacion = 1.0  # segundos
    
    # Intentar obtener el dispositivo de entrada predeterminado
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
    
    # PREPROCESADO (igual que entrenamiento)
    # 1. Filtro pasa-bajas (eliminar ruido > 3.5 kHz)
    x_completo = filtrar_ruido_pasabajos(x_completo, FRECUENCIA_MUESTREO_OBJETIVO)
    
    # 2. Eliminar silencio (recortar inicio/fin)
    x_completo = eliminar_silencio_voz(x_completo, FRECUENCIA_MUESTREO_OBJETIVO)
    
    # Buscar la ventana de N muestras con MAYOR energía (donde está la voz)
    mejor_energia = -1
    mejor_inicio = 0
    
    # Deslizar ventana de N muestras
    for i in range(0, max(1, len(x_completo) - N_FFT), max(1, N_FFT // 4)):
        if i + N_FFT > len(x_completo):
            break
        ventana = x_completo[i:i + N_FFT]
        energia = np.sum(ventana ** 2)
        
        if energia > mejor_energia:
            mejor_energia = energia
            mejor_inicio = i
    
    # Extraer la ventana con mayor energía
    if mejor_inicio + N_FFT <= len(x_completo):
        x = x_completo[mejor_inicio:mejor_inicio + N_FFT]
    else:
        x = x_completo[:N_FFT] if len(x_completo) >= N_FFT else x_completo
    
    # Ajustar a N exactamente
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    
    # NORMALIZACIÓN DE AMPLITUD (crucial para compensar diferencias de volumen)
    # Normalizar a RMS = 0.1 (nivel estándar) - IGUAL que en procesamiento_audio.py
    rms = np.sqrt(np.mean(x ** 2))
    if rms > 1e-6:  # Evitar división por cero
        x = x * (0.1 / rms)
        print(f"[MIC] RMS normalizado: {rms:.6f} → 0.1")
    
    return x
