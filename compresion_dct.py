
import numpy as np
import cv2

def calcular_coeficientes_dct(N, M):
    beta = np.zeros(M)
    beta[0] = np.sqrt(1.0 / M)
    for l in range(1, M):
        beta[l] = np.sqrt(2.0 / M)
    
    alfa = np.zeros(N)
    alfa[0] = np.sqrt(1.0 / N)
    for k in range(1, N):
        alfa[k] = np.sqrt(2.0 / N)
    
    return alfa, beta

def dct_2d_manual(bloque):
    N, M = bloque.shape
    alfa, beta = calcular_coeficientes_dct(N, M)
    
    cos_matrix_n = np.zeros((N, N))
    cos_matrix_m = np.zeros((M, M))
    
    for k in range(N):
        for n in range(N):
            cos_matrix_n[k, n] = np.cos((2 * n + 1) * np.pi * k / (2 * N))
    
    for l in range(M):
        for m in range(M):
            cos_matrix_m[l, m] = np.cos((2 * m + 1) * np.pi * l / (2 * M))
    
    X = np.zeros((N, M), dtype=np.float64)
    
    for k in range(N):
        for l in range(M):
            suma = 0.0
            for n in range(N):
                for m in range(M):
                    suma += bloque[n, m] * cos_matrix_m[l, m] * cos_matrix_n[k, n]
            
            X[k, l] = alfa[k] * beta[l] * suma
    
    return X

def idct_2d_manual(coeficientes):
    N, M = coeficientes.shape
    alfa, beta = calcular_coeficientes_dct(N, M)
    
    cos_matrix_n = np.zeros((N, N))
    cos_matrix_m = np.zeros((M, M))
    
    for k in range(N):
        for n in range(N):
            cos_matrix_n[k, n] = np.cos((2 * n + 1) * np.pi * k / (2 * N))
    
    for l in range(M):
        for m in range(M):
            cos_matrix_m[l, m] = np.cos((2 * m + 1) * np.pi * l / (2 * M))
    
    x = np.zeros((N, M), dtype=np.float64)
    
    for n in range(N):
        for m in range(M):
            suma = 0.0
            for k in range(N):
                for l in range(M):
                    suma += alfa[k] * beta[l] * coeficientes[k, l] * cos_matrix_m[l, m] * cos_matrix_n[k, n]
            
            x[n, m] = suma
    
    return x

def comprimir_imagen_dct(imagen, porcentaje_compresion, tamanio_bloque=8):
    if imagen.dtype != np.float64:
        imagen = imagen.astype(np.float64)
    
    h, w = imagen.shape
    forma_original = imagen.shape
    
    pad_h = (tamanio_bloque - (h % tamanio_bloque)) % tamanio_bloque
    pad_w = (tamanio_bloque - (w % tamanio_bloque)) % tamanio_bloque
    
    if pad_h > 0 or pad_w > 0:
        imagen = np.pad(imagen, ((0, pad_h), (0, pad_w)), mode='edge')
    
    h_pad, w_pad = imagen.shape
    
    num_bloques_h = h_pad // tamanio_bloque
    num_bloques_w = w_pad // tamanio_bloque
    total_bloques = num_bloques_h * num_bloques_w
    print(f"  Procesando {total_bloques} bloques de {tamanio_bloque}x{tamanio_bloque}...")
    
    dct_coefs = np.zeros_like(imagen, dtype=np.float64)
    bloque_actual = 0
    
    for i in range(0, h_pad, tamanio_bloque):
        for j in range(0, w_pad, tamanio_bloque):
            bloque = imagen[i:i+tamanio_bloque, j:j+tamanio_bloque]
            dct_bloque = dct_2d_manual(bloque)
            dct_coefs[i:i+tamanio_bloque, j:j+tamanio_bloque] = dct_bloque
            
            bloque_actual += 1
            if bloque_actual % 100 == 0 or bloque_actual == total_bloques:
                porcentaje_progreso = (bloque_actual / total_bloques) * 100
                print(f"    Progreso: {bloque_actual}/{total_bloques} bloques ({porcentaje_progreso:.1f}%)")
    
    coefs_filtrados, num_eliminados = eliminar_coeficientes_pequenos(
        dct_coefs, porcentaje_compresion
    )
    
    return coefs_filtrados, forma_original, num_eliminados

def aplicar_dct_bloques(imagen, tamanio_bloque=8):
    if imagen.dtype != np.float64:
        imagen = imagen.astype(np.float64)
    
    h, w = imagen.shape
    
    pad_h = (tamanio_bloque - (h % tamanio_bloque)) % tamanio_bloque
    pad_w = (tamanio_bloque - (w % tamanio_bloque)) % tamanio_bloque
    
    if pad_h > 0 or pad_w > 0:
        imagen = np.pad(imagen, ((0, pad_h), (0, pad_w)), mode='edge')
    
    h_pad, w_pad = imagen.shape
    
    num_bloques_h = h_pad // tamanio_bloque
    num_bloques_w = w_pad // tamanio_bloque
    total_bloques = num_bloques_h * num_bloques_w
    print(f"  Aplicando DCT a {total_bloques} bloques de {tamanio_bloque}x{tamanio_bloque}...")
    
    dct_completa = np.zeros_like(imagen, dtype=np.float64)
    bloque_actual = 0
    
    for i in range(0, h_pad, tamanio_bloque):
        for j in range(0, w_pad, tamanio_bloque):
            bloque = imagen[i:i+tamanio_bloque, j:j+tamanio_bloque]
            dct_bloque = dct_2d_manual(bloque)
            dct_completa[i:i+tamanio_bloque, j:j+tamanio_bloque] = dct_bloque
            
            bloque_actual += 1
            if bloque_actual % 100 == 0 or bloque_actual == total_bloques:
                porcentaje_progreso = (bloque_actual / total_bloques) * 100
                print(f"    Progreso: {bloque_actual}/{total_bloques} bloques ({porcentaje_progreso:.1f}%)")
    
    return dct_completa, (h_pad, w_pad)

def eliminar_coeficientes_pequenos(dct_coefs, porcentaje):
    coefs_planos = dct_coefs.flatten()
    total_coefs = len(coefs_planos)
    
    num_eliminar = int((porcentaje / 100.0) * total_coefs)
    
    if num_eliminar < 1:
        return dct_coefs.copy(), 0
    
    indices_ordenados = np.argsort(np.abs(coefs_planos))
    
    coefs_filtrados = coefs_planos.copy()
    coefs_filtrados[indices_ordenados[:num_eliminar]] = 0
    
    coefs_filtrados = coefs_filtrados.reshape(dct_coefs.shape)
    
    return coefs_filtrados, num_eliminar

def descomprimir_imagen_dct(coeficientes_dct, forma_original, tamanio_bloque=8):
    h_pad, w_pad = coeficientes_dct.shape
    
    num_bloques_h = h_pad // tamanio_bloque
    num_bloques_w = w_pad // tamanio_bloque
    total_bloques = num_bloques_h * num_bloques_w
    print(f"  Reconstruyendo imagen desde {total_bloques} bloques...")
    
    imagen_rec = np.zeros_like(coeficientes_dct, dtype=np.float64)
    bloque_actual = 0
    
    for i in range(0, h_pad, tamanio_bloque):
        for j in range(0, w_pad, tamanio_bloque):
            bloque_dct = coeficientes_dct[i:i+tamanio_bloque, j:j+tamanio_bloque]
            bloque_rec = idct_2d_manual(bloque_dct)
            imagen_rec[i:i+tamanio_bloque, j:j+tamanio_bloque] = bloque_rec
            
            bloque_actual += 1
            if bloque_actual % 100 == 0 or bloque_actual == total_bloques:
                porcentaje_progreso = (bloque_actual / total_bloques) * 100
                print(f"    Progreso: {bloque_actual}/{total_bloques} bloques ({porcentaje_progreso:.1f}%)")
    
    imagen_rec = imagen_rec[:forma_original[0], :forma_original[1]]
    
    imagen_rec = np.clip(imagen_rec, 0, 255).astype(np.uint8)
    
    return imagen_rec

def calcular_metricas_compresion(imagen_original, imagen_comprimida, num_coefs_eliminados, total_coefs):
    mse = np.mean((imagen_original.astype(float) - imagen_comprimida.astype(float)) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255**2 / mse)
    
    tasa_compresion = (num_coefs_eliminados / total_coefs) * 100
    
    coefs_mantenidos = total_coefs - num_coefs_eliminados
    
    return {
        'mse': mse,
        'psnr': psnr,
        'tasa_compresion': tasa_compresion,
        'coefs_eliminados': num_coefs_eliminados,
        'coefs_mantenidos': coefs_mantenidos,
        'total_coefs': total_coefs
    }
