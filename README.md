# Proyecto de Reconocimiento de Voz con Procesamiento de Imágenes

Sistema de reconocimiento de comandos de voz que ejecuta operaciones de procesamiento de imágenes (segmentación, compresión y cifrado).

## Estructura del Proyecto

### 🎯 Módulos Principales

#### **Interfaz y Control**
- `interfaz_principal.py` - Interfaz gráfica principal con reconocimiento de voz continuo
- `configuracion.py` - Configuración global del sistema (frecuencias, rutas, parámetros)

#### **Reconocimiento de Voz**
- `captura_microfono.py` - Captura y preprocesamiento de audio desde micrófono
- `procesamiento_audio.py` - Filtrado, pre-énfasis y extracción de características
- `banco_filtros.py` - Cálculo de energías espectrales por sub-bandas
- `reconocimiento_comandos.py` - Reconocimiento por distancia Euclidiana
- `entrenamiento_comandos.py` - Entrenamiento del modelo con audios de ejemplo
- `umbrales_comandos.json` - Modelo entrenado (vectores de energía promedio)

#### **Procesamiento de Imágenes - Lógica Matemática**
- `cifrado_arnold_frdct.py` - **Implementación del cifrado Arnold + FrDCT**
  - Transformación de Arnold (espacial)
  - FrDCT 2D (fraccional DCT)
  - FrDCT inversa
  - Compresión DCT previa al cifrado
  - Funciones completas de cifrado/descifrado

- `compresion_dct.py` - **Implementación de compresión DCT-2D**
  - DCT 2D manual (sin librerías)
  - IDCT 2D manual
  - Compresión por bloques
  - Eliminación de coeficientes
  - Métricas (MSE, PSNR, tasa de compresión)

#### **Procesamiento de Imágenes - Interfaces Gráficas**
- `ventana_cifrado.py` - Interfaz de cifrado (usa `cifrado_arnold_frdct.py`)
- `ventana_compresion.py` - Interfaz de compresión (usa `compresion_dct.py`)
- `ventana_segmentacion.py` - Interfaz de segmentación K-means

### 📊 Datos
- `datos_entrenamiento/` - Audios grabados para entrenamiento
  - `A/` - Comando "segmentar" (183 muestras)
  - `B/` - Comando "comprimir" (174 muestras)
  - `C/` - Comando "cifrar" (147 muestras)

## Flujo de Trabajo

### 1️⃣ Reconocimiento de Voz
```
Micrófono → Captura (1s) → Filtrado → Pre-énfasis → FFT → 
Banco de Filtros (16 sub-bandas) → Vector de Energías → 
Normalización → Distancia Euclidiana → Comando Reconocido
```

### 2️⃣ Operaciones de Imagen

#### **Segmentación (Comando A)**
- K-means clustering con 3-8 clusters
- Visualización de clusters y centroides

#### **Compresión (Comando B)**
- DCT 2D por bloques (8×8)
- Eliminación de coeficientes pequeños
- Múltiples porcentajes (0.5%, 1%, 1.5%, 2%)
- Métricas de calidad (MSE, PSNR)

#### **Cifrado (Comando C)**
Proceso de 3 pasos:
1. **Arnold Transform** - Scrambling espacial
2. **Compresión DCT** - Eliminación 2% coeficientes
3. **FrDCT** - DCT fraccional para cifrado

## Separación de Lógica

### ✅ Ventajas de la Arquitectura

**Módulos Matemáticos Puros:**
- `cifrado_arnold_frdct.py` - Solo algoritmos de cifrado
- `compresion_dct.py` - Solo algoritmos de compresión
- Sin dependencias de GUI (tkinter)
- Reutilizables en otros proyectos
- Fáciles de probar unitariamente

**Módulos de Interfaz:**
- `ventana_cifrado.py` - Solo GUI y eventos
- `ventana_compresion.py` - Solo GUI y eventos
- `ventana_segmentacion.py` - Solo GUI y eventos
- Importan funciones desde módulos matemáticos

### 🔧 Uso de los Módulos Matemáticos

```python
# Ejemplo: Usar cifrado sin GUI
from cifrado_arnold_frdct import cifrar_imagen_completo, descifrar_imagen_completo
import cv2

imagen = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)
resultado = cifrar_imagen_completo(imagen, a=2, k=5, alpha=0.5, porcentaje_compresion=2.0)
imagen_cifrada = resultado['imagen_cifrada']
```

```python
# Ejemplo: Usar compresión sin GUI
from compresion_dct import comprimir_imagen_dct, descomprimir_imagen_dct
import cv2

imagen = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
coefs, forma, n_elim = comprimir_imagen_dct(imagen, porcentaje_compresion=5.0)
imagen_rec = descomprimir_imagen_dct(coefs, forma)
```

## Requisitos

```
numpy
scipy
opencv-python
sounddevice
matplotlib
ttkbootstrap
scikit-learn
```

## Ejecución

```bash
python interfaz_principal.py
```

El sistema:
1. Carga automáticamente el modelo entrenado
2. Activa el micrófono continuamente
3. Escucha comandos: "segmentar", "comprimir", "cifrar"
4. Ejecuta la operación correspondiente sobre la imagen seleccionada

## Características

- ✅ Reconocimiento de voz en tiempo real
- ✅ Pausa automática del micrófono al abrir ventanas
- ✅ Confirmación de comandos detectados
- ✅ Visualizaciones interactivas con matplotlib
- ✅ Lógica matemática separada de la interfaz
- ✅ Código modular y reutilizable
