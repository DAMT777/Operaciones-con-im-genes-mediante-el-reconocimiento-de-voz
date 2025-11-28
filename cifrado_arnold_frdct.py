import numpy as np
import cv2
from scipy.fftpack import dct, idct

def transformacion_arnold(imagen, a, k, inversa=False):
    n, m = imagen.shape
    resultado = imagen.copy()
    
    es_cuadrada = (n == m)
    
    if inversa:
        for _ in range(k):
            temp = np.zeros_like(resultado)
            if es_cuadrada:
                for x in range(n):
                    for y in range(m):
                        x_new = ((a + 1) * x - y) % n
                        y_new = (-a * x + y) % m
                        temp[x_new, y_new] = resultado[x, y]
            else:
                for x in range(n):
                    for y in range(m):
                        x_new = (y - a * x) % n
                        y_new = x % m
                        temp[x_new, y_new] = resultado[x, y]
            resultado = temp
    else:
        for _ in range(k):
            temp = np.zeros_like(resultado)
            if es_cuadrada:
                for x in range(n):
                    for y in range(m):
                        x_new = (x + y) % n
                        y_new = (a * x + (a + 1) * y) % m
                        temp[x_new, y_new] = resultado[x, y]
            else:
                for x in range(n):
                    for y in range(m):
                        x_new = y % n
                        y_new = (x + a * y) % m
                        temp[x_new, y_new] = resultado[x, y]
            resultado = temp
    
    return resultado

def frdct_2d(imagen, alpha):
    dct_result = dct(dct(imagen.T, norm='ortho').T, norm='ortho')
    
    if abs(alpha) > 1e-6:
        N, M = imagen.shape
        u_vals = np.arange(N).reshape(-1, 1)
        v_vals = np.arange(M).reshape(1, -1)
        
        phase_u = alpha * u_vals / (2 * N)
        phase_v = alpha * v_vals / (2 * M)
        modulation = np.exp(-1j * np.pi * (phase_u + phase_v))
        
        dct_result = np.real(dct_result * modulation)
    
    return dct_result

def frdct_inversa_2d(matriz, alpha):
    matriz_proc = matriz.copy()
    
    if abs(alpha) > 1e-6:
        N, M = matriz.shape
        u_vals = np.arange(N).reshape(-1, 1)
        v_vals = np.arange(M).reshape(1, -1)
        
        phase_u = alpha * u_vals / (2 * N)
        phase_v = alpha * v_vals / (2 * M)
        modulation_inv = np.exp(1j * np.pi * (phase_u + phase_v))
        
        matriz_proc = np.real(matriz_proc * modulation_inv)
    
    resultado = idct(idct(matriz_proc.T, norm='ortho').T, norm='ortho')
    
    return resultado

def comprimir_dct(imagen, porcentaje_eliminacion):
    imagen_float = imagen.astype(np.float32)
    dct_coef = cv2.dct(imagen_float)
    
    coef_flat = dct_coef.flatten()
    umbral_comp = np.percentile(np.abs(coef_flat), porcentaje_eliminacion)
    dct_comprimida = dct_coef.copy()
    dct_comprimida[np.abs(dct_comprimida) < umbral_comp] = 0
    
    imagen_comprimida = cv2.idct(dct_comprimida)
    imagen_comprimida = np.clip(imagen_comprimida, 0, 255).astype(np.uint8)
    
    coef_eliminados = np.sum(dct_comprimida == 0) / dct_comprimida.size * 100
    
    return imagen_comprimida, dct_comprimida, coef_eliminados

def cifrar_imagen_completo(imagen_original, a, k, alpha, porcentaje_compresion=2.0):
    print(f"\n=== PROCESO DE CIFRADO ===")
    print(f"Parámetros: a={a}, k={k}, α={alpha}")
    print(f"Compresión: {porcentaje_compresion}% de coeficientes eliminados")
    
    print("\nPASO 1: Aplicando transformación de Arnold...")
    imagen_arnold = transformacion_arnold(imagen_original, a, k, inversa=False)
    print(f"✓ Arnold completado ({k} iteraciones)")
    
    print("\nPASO 2: Aplicando compresión DCT...")
    imagen_comprimida, matriz_dct_comprimida, coef_eliminados = comprimir_dct(
        imagen_arnold, porcentaje_compresion
    )
    print(f"✓ Compresión completada ({coef_eliminados:.2f}% coeficientes eliminados)")
    
    print("\nPASO 3: Aplicando FrDCT 2D...")
    imagen_norm = imagen_comprimida.astype(np.float64) / 255.0
    matriz_frdct = frdct_2d(imagen_norm, alpha)
    print(f"✓ FrDCT completado")
    
    imagen_cifrada = np.abs(matriz_frdct)
    imagen_cifrada = (imagen_cifrada - imagen_cifrada.min())
    imagen_cifrada = (imagen_cifrada / imagen_cifrada.max() * 255).astype(np.uint8)
    
    print("\n✓ CIFRADO COMPLETADO")
    
    return {
        'imagen_arnold': imagen_arnold,
        'imagen_comprimida': imagen_comprimida,
        'matriz_dct_comprimida': matriz_dct_comprimida,
        'matriz_frdct': matriz_frdct,
        'imagen_cifrada': imagen_cifrada,
        'coef_eliminados': coef_eliminados
    }

def descifrar_imagen_completo(matriz_frdct, a, k, alpha):
    print(f"\n=== PROCESO DE DESCIFRADO ===")
    print(f"Parámetros: a={a}, k={k}, α={alpha}")
    
    print("\nPASO 1: Aplicando FrDCT inversa...")
    imagen_desc_norm = frdct_inversa_2d(matriz_frdct, alpha)
    
    imagen_desc_arnold = np.abs(imagen_desc_norm)
    imagen_desc_arnold = (imagen_desc_arnold - imagen_desc_arnold.min())
    imagen_desc_arnold = (imagen_desc_arnold / imagen_desc_arnold.max() * 255).astype(np.uint8)
    print(f"✓ FrDCT inversa completado")
    
    print("\nPASO 2: Aplicando transformación de Arnold inversa...")
    imagen_descifrada = transformacion_arnold(imagen_desc_arnold, a, k, inversa=True)
    print(f"✓ Arnold inverso completado ({k} iteraciones)")
    
    print("\n✓ DESCIFRADO COMPLETADO")
    
    return imagen_descifrada
