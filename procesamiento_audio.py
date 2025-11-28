import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample

from configuracion import FRECUENCIA_MUESTREO_OBJETIVO, FRECUENCIA_CORTE_PB, ORDEN_FILTRO, PREENFASIS_ALPHA, UMBRAL_ENERGIA_SILENCIO, MARGEN_SILENCIO_MS, N_FFT

def extraer_ventana_maxima_energia(senal, N):
    if len(senal) <= N:
        ventana = np.pad(senal, (0, N - len(senal)))
    else:
        mejor_energia = -1
        mejor_inicio = 0
        paso = N // 4
        
        for i in range(0, len(senal) - N, paso):
            ventana_temp = senal[i:i + N]
            energia = np.sum(ventana_temp ** 2)
            
            if energia > mejor_energia:
                mejor_energia = energia
                mejor_inicio = i
        
        ventana = senal[mejor_inicio:mejor_inicio + N]
    
    rms = np.sqrt(np.mean(ventana ** 2))
    if rms > 1e-6:
        ventana = ventana * (0.1 / rms)
    
    return ventana

def aplicar_preenfasis(senal, coef=PREENFASIS_ALPHA):
    return np.append(senal[0], senal[1:] - coef * senal[:-1])

def eliminar_silencio_voz(senal, fs, umbral_db=UMBRAL_ENERGIA_SILENCIO, margen_ms=MARGEN_SILENCIO_MS):
    energia = senal ** 2
    ventana_muestras = int(0.025 * fs)
    
    energia_ventana = np.convolve(energia, np.ones(ventana_muestras)/ventana_muestras, mode='same')
    energia_ventana = np.maximum(energia_ventana, 1e-10)
    energia_db = 10 * np.log10(energia_ventana)
    
    umbral = np.max(energia_db) + umbral_db
    
    mascara_voz = energia_db > umbral
    
    if not np.any(mascara_voz):
        return senal
    
    indices_voz = np.where(mascara_voz)[0]
    margen_muestras = int(margen_ms * fs / 1000)
    
    inicio = max(0, indices_voz[0] - margen_muestras)
    fin = min(len(senal), indices_voz[-1] + margen_muestras)
    
    return senal[inicio:fin]

def cargar_senal_desde_wav(ruta_archivo):
    fs, datos = wavfile.read(str(ruta_archivo))

    if datos.dtype == np.int16:
        datos = datos.astype(np.float32) / 32768.0
    elif datos.dtype == np.int32:
        datos = datos.astype(np.float32) / 2147483648.0
    else:
        datos = datos.astype(np.float32)

    if datos.ndim == 2:
        datos = np.mean(datos, axis=1)

    return fs, datos

def re_muestrear_senal(fs_original, senal):
    if fs_original == FRECUENCIA_MUESTREO_OBJETIVO:
        return senal

    duracion = len(senal) / fs_original
    nuevo_num_muestras = int(duracion * FRECUENCIA_MUESTREO_OBJETIVO)
    senal_remuestreada = resample(senal, nuevo_num_muestras)
    return senal_remuestreada

def filtrar_ruido_pasabajos(senal, fs, frecuencia_corte=FRECUENCIA_CORTE_PB, orden=ORDEN_FILTRO):
    b, a = butter(orden, frecuencia_corte / (fs / 2), btype='low')
    return filtfilt(b, a, senal)

def ajustar_longitud_potencia_de_dos(senal):
    longitud_actual = len(senal)
    nueva_longitud = 1 << (longitud_actual - 1).bit_length()
    if nueva_longitud == longitud_actual:
        return senal
    senal_ajustada = np.zeros(nueva_longitud, dtype=np.float32)
    senal_ajustada[:longitud_actual] = senal
    return senal_ajustada

def calcular_fft_magnitud(senal):
    return np.abs(np.fft.fft(senal))
