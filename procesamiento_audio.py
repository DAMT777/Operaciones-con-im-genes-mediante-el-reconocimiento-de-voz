import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample

from configuracion import FRECUENCIA_MUESTREO_OBJETIVO


def aplicar_preenfasis(senal, coef=0.97):
    """Aplica filtro de pre-énfasis para realzar frecuencias altas.
    Esto ayuda a balancear el espectro de frecuencias en señales de voz."""
    return np.append(senal[0], senal[1:] - coef * senal[:-1])


def eliminar_silencio_voz(senal, fs, umbral_db=-40, margen_ms=100):
    """Elimina silencios al inicio y final basándose en energía de la señal.
    Útil para normalizar grabaciones con diferentes cantidades de silencio."""
    # Calcular energía en dB
    energia = senal ** 2
    ventana_muestras = int(0.025 * fs)  # Ventana de 25ms
    
    # Energía por ventana
    energia_ventana = np.convolve(energia, np.ones(ventana_muestras)/ventana_muestras, mode='same')
    energia_ventana = np.maximum(energia_ventana, 1e-10)  # Evitar log(0)
    energia_db = 10 * np.log10(energia_ventana)
    
    # Umbral
    umbral = np.max(energia_db) + umbral_db
    
    # Encontrar regiones con voz
    mascara_voz = energia_db > umbral
    
    if not np.any(mascara_voz):
        return senal
    
    # Encontrar inicio y fin
    indices_voz = np.where(mascara_voz)[0]
    margen_muestras = int(margen_ms * fs / 1000)
    
    inicio = max(0, indices_voz[0] - margen_muestras)
    fin = min(len(senal), indices_voz[-1] + margen_muestras)
    
    return senal[inicio:fin]


def cargar_senal_desde_wav(ruta_archivo):
    """Carga una señal .wav y la devuelve en mono y como float normalizado."""
    fs, datos = wavfile.read(str(ruta_archivo))

    # Convertir a float en [-1, 1]
    if datos.dtype == np.int16:
        datos = datos.astype(np.float32) / 32768.0
    elif datos.dtype == np.int32:
        datos = datos.astype(np.float32) / 2147483648.0
    else:
        datos = datos.astype(np.float32)

    # Si es estéreo, pasarlo a mono
    if datos.ndim == 2:
        datos = np.mean(datos, axis=1)

    return fs, datos


def re_muestrear_senal(fs_original, senal):
    """Remuestrea la señal a FRECUENCIA_MUESTREO_OBJETIVO si es diferente."""
    if fs_original == FRECUENCIA_MUESTREO_OBJETIVO:
        return senal

    duracion = len(senal) / fs_original
    nuevo_num_muestras = int(duracion * FRECUENCIA_MUESTREO_OBJETIVO)
    senal_remuestreada = resample(senal, nuevo_num_muestras)
    return senal_remuestreada


def filtrar_ruido_pasabajos(senal, fs, frecuencia_corte=3500.0, orden=5):
    """Filtra la señal con un filtro pasa‑bajos Butterworth para reducir ruido.
    Según la teoría, se debe acondicionar la señal eliminando ruido antes del análisis."""
    nyquist = fs / 2.0
    wc = min(frecuencia_corte / nyquist, 0.95)  # Asegurar que no exceda Nyquist
    b, a = butter(orden, wc, btype="low")
    senal_filtrada = filtfilt(b, a, senal)
    return senal_filtrada


def ajustar_longitud_potencia_de_dos(senal):
    """Ajusta la longitud de la señal al siguiente número potencia de 2 (relleno con ceros)."""
    longitud_actual = len(senal)
    nueva_longitud = 1 << (longitud_actual - 1).bit_length()  # siguiente potencia de 2
    if nueva_longitud == longitud_actual:
        return senal
    senal_ajustada = np.zeros(nueva_longitud, dtype=np.float32)
    senal_ajustada[:longitud_actual] = senal
    return senal_ajustada


def calcular_fft_magnitud(senal):
    """Calcula la FFT y devuelve solo el espectro de magnitud de la parte positiva.
    Según la teoría, trabajamos con las frecuencias desde 0 hasta Nyquist (N/2)."""
    N = len(senal)
    espectro = np.fft.fft(senal)
    # Solo frecuencias positivas [0, N/2), excluyendo la frecuencia de Nyquist duplicada
    espectro_magnitud = np.abs(espectro[: N // 2])
    return espectro_magnitud
