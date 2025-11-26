import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample

from configuracion import FRECUENCIA_MUESTREO_OBJETIVO


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


def filtrar_ruido_pasabajos(senal, fs, frecuencia_corte=4000.0, orden=4):
    """Filtra la señal con un filtro pasa‑bajos Butterworth para reducir ruido."""
    nyquist = fs / 2.0
    wc = frecuencia_corte / nyquist
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
    """Calcula la FFT y devuelve solo el espectro de magnitud de la parte positiva."""
    N = len(senal)
    espectro = np.fft.fft(senal)
    espectro_magnitud = np.abs(espectro[: N // 2])  # solo frecuencias positivas
    return espectro_magnitud
