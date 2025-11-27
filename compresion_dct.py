"""
Módulo de compresión de imágenes usando DCT-2D manual.
Implementa las fórmulas de la Transformada Discreta del Coseno Bidimensional.
"""

import numpy as np
import cv2


def calcular_coeficientes_dct(N, M):
    """
    Calcula los coeficientes alfa y beta para DCT-2D según las fórmulas.
    
    Fórmulas:
    β_l = sqrt(1/M) si l=0, sino sqrt(2/M)
    α_k = sqrt(1/N) si k=0, sino sqrt(2/N)
    """
    # Coeficientes beta (columnas)
    beta = np.zeros(M)
    beta[0] = np.sqrt(1.0 / M)
    for l in range(1, M):
        beta[l] = np.sqrt(2.0 / M)
    
    # Coeficientes alfa (filas)
    alfa = np.zeros(N)
    alfa[0] = np.sqrt(1.0 / N)
    for k in range(1, N):
        alfa[k] = np.sqrt(2.0 / N)
    
    return alfa, beta


def dct_2d_manual(bloque):
    """
    Aplica la Transformada Discreta del Coseno 2D de forma manual.
    
    Fórmula DCT-2D:
    X(k,l) = Σ(n=0 to N-1) Σ(m=0 to M-1) α_k β_l x(n,m) cos((2m+1)πl/2M) cos((2n+1)πk/2N)
    
    Parámetros:
    -----------
    bloque : ndarray
        Bloque de imagen de tamaño NxM
        
    Retorna:
    --------
    ndarray : Coeficientes DCT del bloque
    """
    N, M = bloque.shape
    alfa, beta = calcular_coeficientes_dct(N, M)
    
    # Precalcular matrices de cosenos para optimizar
    cos_matrix_n = np.zeros((N, N))
    cos_matrix_m = np.zeros((M, M))
    
    for k in range(N):
        for n in range(N):
            cos_matrix_n[k, n] = np.cos((2 * n + 1) * np.pi * k / (2 * N))
    
    for l in range(M):
        for m in range(M):
            cos_matrix_m[l, m] = np.cos((2 * m + 1) * np.pi * l / (2 * M))
    
    # Matriz de salida
    X = np.zeros((N, M), dtype=np.float64)
    
    # Aplicar la fórmula DCT-2D usando matrices precalculadas
    for k in range(N):
        for l in range(M):
            suma = 0.0
            for n in range(N):
                for m in range(M):
                    suma += bloque[n, m] * cos_matrix_m[l, m] * cos_matrix_n[k, n]
            
            X[k, l] = alfa[k] * beta[l] * suma
    
    return X


def idct_2d_manual(coeficientes):
    """
    Aplica la Transformada Inversa Discreta del Coseno 2D de forma manual.
    
    Fórmula IDCT-2D:
    x(n,m) = Σ(k=0 to N-1) Σ(l=0 to M-1) α_k β_l X(k,l) cos((2m+1)πl/2M) cos((2n+1)πk/2N)
    
    Parámetros:
    -----------
    coeficientes : ndarray
        Coeficientes DCT del bloque
        
    Retorna:
    --------
    ndarray : Bloque reconstruido
    """
    N, M = coeficientes.shape
    alfa, beta = calcular_coeficientes_dct(N, M)
    
    # Precalcular matrices de cosenos para optimizar
    cos_matrix_n = np.zeros((N, N))
    cos_matrix_m = np.zeros((M, M))
    
    for k in range(N):
        for n in range(N):
            cos_matrix_n[k, n] = np.cos((2 * n + 1) * np.pi * k / (2 * N))
    
    for l in range(M):
        for m in range(M):
            cos_matrix_m[l, m] = np.cos((2 * m + 1) * np.pi * l / (2 * M))
    
    # Matriz de salida
    x = np.zeros((N, M), dtype=np.float64)
    
    # Aplicar la fórmula IDCT-2D usando matrices precalculadas
    for n in range(N):
        for m in range(M):
            suma = 0.0
            for k in range(N):
                for l in range(M):
                    suma += alfa[k] * beta[l] * coeficientes[k, l] * cos_matrix_m[l, m] * cos_matrix_n[k, n]
            
            x[n, m] = suma
    
    return x


def comprimir_imagen_dct(imagen, porcentaje_compresion, tamanio_bloque=8):
    """
    Comprime una imagen usando DCT-2D por bloques.
    
    Parámetros:
    -----------
    imagen : ndarray
        Imagen en escala de grises
    porcentaje_compresion : float
        Porcentaje de coeficientes a eliminar (0-100)
    tamanio_bloque : int
        Tamaño del bloque para DCT (8x8 por defecto)
        
    Retorna:
    --------
    tuple : (coeficientes_dct, forma_original, coeficientes_eliminados)
    """
    # Convertir a float si es necesario
    if imagen.dtype != np.float64:
        imagen = imagen.astype(np.float64)
    
    h, w = imagen.shape
    forma_original = imagen.shape
    
    # Padding para que sea múltiplo del tamaño de bloque
    pad_h = (tamanio_bloque - (h % tamanio_bloque)) % tamanio_bloque
    pad_w = (tamanio_bloque - (w % tamanio_bloque)) % tamanio_bloque
    
    if pad_h > 0 or pad_w > 0:
        imagen = np.pad(imagen, ((0, pad_h), (0, pad_w)), mode='edge')
    
    h_pad, w_pad = imagen.shape
    
    # Calcular total de bloques para mostrar progreso
    num_bloques_h = h_pad // tamanio_bloque
    num_bloques_w = w_pad // tamanio_bloque
    total_bloques = num_bloques_h * num_bloques_w
    print(f"  Procesando {total_bloques} bloques de {tamanio_bloque}x{tamanio_bloque}...")
    
    # Aplicar DCT por bloques
    dct_coefs = np.zeros_like(imagen, dtype=np.float64)
    bloque_actual = 0
    
    for i in range(0, h_pad, tamanio_bloque):
        for j in range(0, w_pad, tamanio_bloque):
            bloque = imagen[i:i+tamanio_bloque, j:j+tamanio_bloque]
            dct_bloque = dct_2d_manual(bloque)
            dct_coefs[i:i+tamanio_bloque, j:j+tamanio_bloque] = dct_bloque
            
            # Mostrar progreso cada 100 bloques
            bloque_actual += 1
            if bloque_actual % 100 == 0 or bloque_actual == total_bloques:
                porcentaje_progreso = (bloque_actual / total_bloques) * 100
                print(f"    Progreso: {bloque_actual}/{total_bloques} bloques ({porcentaje_progreso:.1f}%)")
    
    # Comprimir eliminando coeficientes pequeños
    coefs_filtrados, num_eliminados = eliminar_coeficientes_pequenos(
        dct_coefs, porcentaje_compresion
    )
    
    return coefs_filtrados, forma_original, num_eliminados


def aplicar_dct_bloques(imagen, tamanio_bloque=8):
    """
    Aplica DCT-2D por bloques a toda la imagen sin comprimir.
    Útil para visualizar el mapa completo de coeficientes DCT.
    
    Parámetros:
    -----------
    imagen : ndarray
        Imagen en escala de grises
    tamanio_bloque : int
        Tamaño del bloque para DCT (8x8 por defecto)
        
    Retorna:
    --------
    tuple : (dct_completa, forma_con_padding)
    """
    # Convertir a float si es necesario
    if imagen.dtype != np.float64:
        imagen = imagen.astype(np.float64)
    
    h, w = imagen.shape
    
    # Padding para que sea múltiplo del tamaño de bloque
    pad_h = (tamanio_bloque - (h % tamanio_bloque)) % tamanio_bloque
    pad_w = (tamanio_bloque - (w % tamanio_bloque)) % tamanio_bloque
    
    if pad_h > 0 or pad_w > 0:
        imagen = np.pad(imagen, ((0, pad_h), (0, pad_w)), mode='edge')
    
    h_pad, w_pad = imagen.shape
    
    # Calcular total de bloques para mostrar progreso
    num_bloques_h = h_pad // tamanio_bloque
    num_bloques_w = w_pad // tamanio_bloque
    total_bloques = num_bloques_h * num_bloques_w
    print(f"  Aplicando DCT a {total_bloques} bloques de {tamanio_bloque}x{tamanio_bloque}...")
    
    # Aplicar DCT por bloques
    dct_completa = np.zeros_like(imagen, dtype=np.float64)
    bloque_actual = 0
    
    for i in range(0, h_pad, tamanio_bloque):
        for j in range(0, w_pad, tamanio_bloque):
            bloque = imagen[i:i+tamanio_bloque, j:j+tamanio_bloque]
            dct_bloque = dct_2d_manual(bloque)
            dct_completa[i:i+tamanio_bloque, j:j+tamanio_bloque] = dct_bloque
            
            # Mostrar progreso cada 100 bloques
            bloque_actual += 1
            if bloque_actual % 100 == 0 or bloque_actual == total_bloques:
                porcentaje_progreso = (bloque_actual / total_bloques) * 100
                print(f"    Progreso: {bloque_actual}/{total_bloques} bloques ({porcentaje_progreso:.1f}%)")
    
    return dct_completa, (h_pad, w_pad)


def eliminar_coeficientes_pequenos(dct_coefs, porcentaje):
    """
    Elimina un porcentaje de los coeficientes DCT más pequeños.
    
    Parámetros:
    -----------
    dct_coefs : ndarray
        Coeficientes DCT
    porcentaje : float
        Porcentaje de coeficientes a eliminar (0-100)
        
    Retorna:
    --------
    tuple : (coeficientes_filtrados, número_eliminados)
    """
    # Aplanar los coeficientes
    coefs_planos = dct_coefs.flatten()
    total_coefs = len(coefs_planos)
    
    # Calcular cuántos eliminar
    num_eliminar = int((porcentaje / 100.0) * total_coefs)
    
    if num_eliminar < 1:
        return dct_coefs.copy(), 0
    
    # Encontrar índices de los coeficientes más pequeños
    indices_ordenados = np.argsort(np.abs(coefs_planos))
    
    # Crear copia y eliminar coeficientes pequeños
    coefs_filtrados = coefs_planos.copy()
    coefs_filtrados[indices_ordenados[:num_eliminar]] = 0
    
    # Reshape a forma original
    coefs_filtrados = coefs_filtrados.reshape(dct_coefs.shape)
    
    return coefs_filtrados, num_eliminar


def descomprimir_imagen_dct(coeficientes_dct, forma_original, tamanio_bloque=8):
    """
    Descomprime una imagen aplicando IDCT-2D.
    
    Parámetros:
    -----------
    coeficientes_dct : ndarray
        Coeficientes DCT comprimidos
    forma_original : tuple
        Forma original de la imagen (h, w)
    tamanio_bloque : int
        Tamaño del bloque usado en la compresión
        
    Retorna:
    --------
    ndarray : Imagen reconstruida
    """
    h_pad, w_pad = coeficientes_dct.shape
    
    # Calcular total de bloques
    num_bloques_h = h_pad // tamanio_bloque
    num_bloques_w = w_pad // tamanio_bloque
    total_bloques = num_bloques_h * num_bloques_w
    print(f"  Reconstruyendo imagen desde {total_bloques} bloques...")
    
    imagen_rec = np.zeros_like(coeficientes_dct, dtype=np.float64)
    bloque_actual = 0
    
    # Aplicar IDCT por bloques
    for i in range(0, h_pad, tamanio_bloque):
        for j in range(0, w_pad, tamanio_bloque):
            bloque_dct = coeficientes_dct[i:i+tamanio_bloque, j:j+tamanio_bloque]
            bloque_rec = idct_2d_manual(bloque_dct)
            imagen_rec[i:i+tamanio_bloque, j:j+tamanio_bloque] = bloque_rec
            
            # Mostrar progreso cada 100 bloques
            bloque_actual += 1
            if bloque_actual % 100 == 0 or bloque_actual == total_bloques:
                porcentaje_progreso = (bloque_actual / total_bloques) * 100
                print(f"    Progreso: {bloque_actual}/{total_bloques} bloques ({porcentaje_progreso:.1f}%)")
    
    # Recortar al tamaño original
    imagen_rec = imagen_rec[:forma_original[0], :forma_original[1]]
    
    # Clip a rango válido y convertir a uint8
    imagen_rec = np.clip(imagen_rec, 0, 255).astype(np.uint8)
    
    return imagen_rec


def calcular_metricas_compresion(imagen_original, imagen_comprimida, num_coefs_eliminados, total_coefs):
    """
    Calcula métricas de calidad de la compresión.
    
    Retorna:
    --------
    dict : Diccionario con métricas
    """
    # MSE (Error Cuadrático Medio)
    mse = np.mean((imagen_original.astype(float) - imagen_comprimida.astype(float)) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255**2 / mse)
    
    # Tasa de compresión
    tasa_compresion = (num_coefs_eliminados / total_coefs) * 100
    
    # Coeficientes mantenidos
    coefs_mantenidos = total_coefs - num_coefs_eliminados
    
    return {
        'mse': mse,
        'psnr': psnr,
        'tasa_compresion': tasa_compresion,
        'coefs_eliminados': num_coefs_eliminados,
        'coefs_mantenidos': coefs_mantenidos,
        'total_coefs': total_coefs
    }
