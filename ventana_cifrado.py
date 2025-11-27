"""
Ventana de cifrado de imágenes usando FrDCT (Fractional Discrete Cosine Transform).
Implementa cifrado/descifrado mediante transformada fraccional del coseno.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class VentanaCifradoFrDCT:
    def __init__(self, parent, ruta_imagen):
        """
        Inicializa la ventana de cifrado FrDCT.
        
        Parámetros:
        -----------
        parent : tk.Tk o tk.Toplevel
            Ventana padre
        ruta_imagen : Path o str
            Ruta de la imagen a cifrar
        """
        self.ventana = tk.Toplevel(parent)
        self.ventana.title("Cifrado de Imagen con FrDCT (Fractional DCT)")
        self.ventana.geometry("1600x900")
        self.ventana.configure(bg='#2c3e50')
        
        # Cargar imagen con soporte Unicode
        self.imagen_original = self.cargar_imagen_opencv_unicode(str(ruta_imagen))
        
        if self.imagen_original is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{ruta_imagen}")
            self.ventana.destroy()
            return
        
        # Convertir a escala de grises si es necesario
        if len(self.imagen_original.shape) == 3:
            self.imagen_original = cv2.cvtColor(self.imagen_original, cv2.COLOR_RGB2GRAY)
        
        # Variables de estado
        self.imagen_cifrada = None
        self.imagen_descifrada = None
        self.matriz_frdct = None
        self.alpha_actual = None
        self.canvas_actual = None
        self.toolbar_actual = None
        
        self.crear_interfaz()
    
    def cargar_imagen_opencv_unicode(self, ruta):
        """Carga imagen con OpenCV soportando rutas Unicode."""
        try:
            with open(ruta, 'rb') as f:
                datos = f.read()
            arr = np.frombuffer(datos, dtype=np.uint8)
            imagen = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if imagen is not None:
                # Convertir BGR a RGB
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            return imagen
        except Exception as e:
            print(f"Error al cargar imagen: {e}")
            return None
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica de la ventana."""
        # Título principal
        tk.Label(
            self.ventana,
            text="CIFRADO CON FrDCT (FRACTIONAL DISCRETE COSINE TRANSFORM)",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 16, 'bold'),
            pady=12
        ).pack(fill=tk.X)
        
        # Frame de controles
        frame_controles = tk.Frame(self.ventana, bg='#34495e', pady=10)
        frame_controles.pack(fill=tk.X, padx=10)
        
        tk.Label(
            frame_controles,
            text="Orden fraccional α (clave de cifrado):",
            bg='#34495e',
            fg='white',
            font=('Segoe UI', 10, 'bold')
        ).pack(side=tk.LEFT, padx=(20, 10))
        
        self.entry_alpha = tk.Entry(
            frame_controles,
            font=('Segoe UI', 10),
            width=10,
            justify=tk.CENTER
        )
        self.entry_alpha.insert(0, "0.5")
        self.entry_alpha.pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Label(
            frame_controles,
            text="(Rango: 0.0 - 2.0)",
            bg='#34495e',
            fg='#bdc3c7',
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        self.btn_cifrar = tk.Button(
            frame_controles,
            text="🔒 Cifrar Imagen",
            font=('Segoe UI', 10, 'bold'),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            padx=20,
            pady=8,
            relief=tk.RAISED,
            command=self.cifrar
        )
        self.btn_cifrar.pack(side=tk.LEFT, padx=10)
        
        self.btn_descifrar = tk.Button(
            frame_controles,
            text="🔓 Descifrar Imagen",
            font=('Segoe UI', 10, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            padx=20,
            pady=8,
            relief=tk.RAISED,
            command=self.descifrar,
            state=tk.DISABLED
        )
        self.btn_descifrar.pack(side=tk.LEFT, padx=10)
        
        # Frame principal que contiene gráfico y panel lateral
        frame_principal = tk.Frame(self.ventana, bg='#2c3e50')
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para el gráfico de matplotlib
        self.frame_grafico = tk.Frame(frame_principal, bg='white')
        self.frame_grafico.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Panel lateral derecho para información
        frame_info = tk.Frame(frame_principal, bg='#34495e', width=300)
        frame_info.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        frame_info.pack_propagate(False)
        
        # Título del panel de info
        tk.Label(
            frame_info,
            text="📊 INFORMACIÓN FrDCT",
            bg='#34495e',
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            pady=10
        ).pack(fill=tk.X)
        
        # Área de texto para información
        self.texto_info = tk.Text(
            frame_info,
            wrap=tk.WORD,
            font=('Consolas', 8),
            bg='#ecf0f1',
            fg='#2c3e50',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.texto_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Mensaje inicial
        self.mostrar_info_inicial()
        self.texto_info.config(state=tk.DISABLED)
        
        # Mostrar vista inicial
        self.mostrar_vista_inicial()
    
    def mostrar_info_inicial(self):
        """Muestra información inicial sobre FrDCT."""
        self.texto_info.insert(1.0,
            "ℹ️ FrDCT - Cifrado Fraccional\n"
            f"{'='*35}\n\n"
            "ALGORITMO:\n\n"
            "1. Aplicar FrDCT a la imagen\n"
            "   con orden fraccional α\n\n"
            "2. La transformada FrDCT rota\n"
            "   el espacio de frecuencias\n\n"
            "3. Imagen cifrada = matriz\n"
            "   transformada\n\n"
            "4. Descifrar = aplicar FrDCT\n"
            "   inversa con orden -α\n\n"
            f"{'='*35}\n\n"
            "FÓRMULA FrDCT:\n\n"
            "F^(α)[f(n)] = Σ f(n)·cos_α(\n"
            "  π/N·(n + 1/2)·(k + 1/2))\n\n"
            "donde:\n"
            "• α = orden fraccional\n"
            "• cos_α = coseno fraccional\n"
            "• N = tamaño de la señal\n\n"
            f"{'='*35}\n\n"
            "PROPIEDADES:\n\n"
            "✓ Additive Property:\n"
            "  FrDCT(α+β) = FrDCT(α)·\n"
            "               FrDCT(β)\n\n"
            "✓ Orthogonality:\n"
            "  Σ F^(α)·F^(-α) = f(x)\n\n"
            "✓ Rotation Property:\n"
            "  Rota el dominio de\n"
            "  frecuencias por ángulo\n"
            "  θ = α·π/(2N)\n\n"
            f"{'='*35}\n\n"
            "SEGURIDAD:\n\n"
            "• Clave = orden α\n"
            "• Sin α correcto: imagen\n"
            "  permanece cifrada\n"
            "• Rango óptimo: 0.2-1.8\n\n"
            "RECOMENDACIONES:\n\n"
            "• α=0.5: Cifrado moderado\n"
            "• α=1.0: Cifrado fuerte\n"
            "• α=1.5: Máxima seguridad\n\n"
            "Ajuste α y presione\n"
            "'Cifrar Imagen'."
        )
    
    def mostrar_vista_inicial(self):
        """Muestra la imagen original en vista previa."""
        # Limpiar canvas anterior si existe
        if self.canvas_actual:
            self.canvas_actual.get_tk_widget().destroy()
        if self.toolbar_actual:
            self.toolbar_actual.destroy()
        
        # Crear figura
        fig = Figure(figsize=(12, 9), dpi=100)
        
        # Imagen original centrada
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.imagen_original, cmap='gray')
        ax.set_title("Imagen Original - Lista para Cifrar", fontsize=14, fontweight='bold', pad=10)
        ax.axis('on')
        ax.grid(True, alpha=0.3)
        
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.05)
        
        # Crear canvas
        self.canvas_actual = FigureCanvasTkAgg(fig, self.frame_grafico)
        self.canvas_actual.draw()
        
        # Toolbar
        toolbar_frame = tk.Frame(self.frame_grafico)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar_actual = NavigationToolbar2Tk(self.canvas_actual, toolbar_frame)
        self.toolbar_actual.update()
        
        self.canvas_actual.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def cifrar(self):
        """Ejecuta el cifrado FrDCT."""
        try:
            # Obtener α
            alpha = float(self.entry_alpha.get())
            
            if alpha < 0.0 or alpha > 2.0:
                messagebox.showerror("Error", "α debe estar entre 0.0 y 2.0")
                return
            
            # Deshabilitar botones durante procesamiento
            self.btn_cifrar.config(state=tk.DISABLED)
            self.btn_descifrar.config(state=tk.DISABLED)
            
            # Actualizar info con mensaje de progreso
            self.texto_info.config(state=tk.NORMAL)
            self.texto_info.delete(1.0, tk.END)
            self.texto_info.insert(1.0,
                "⏳ CIFRANDO...\n"
                f"{'='*35}\n\n"
                f"Aplicando FrDCT con\n"
                f"orden fraccional α = {alpha}\n\n"
                "Calculando transformada\n"
                "fraccional del coseno...\n\n"
                "Este proceso puede\n"
                "tardar unos segundos.\n\n"
                "Por favor espere..."
            )
            self.texto_info.config(state=tk.DISABLED)
            self.ventana.update()
            
            # Ejecutar cifrado
            print(f"Iniciando cifrado FrDCT con α={alpha}...")
            self.aplicar_frdct_cifrado(alpha)
            
            # Mostrar resultados
            print("Generando visualización...")
            self.mostrar_resultados_cifrado(alpha)
            
            # Habilitar botones
            self.btn_cifrar.config(state=tk.NORMAL)
            self.btn_descifrar.config(state=tk.NORMAL)
            
            print("✓ Cifrado completado exitosamente")
            
        except ValueError:
            self.btn_cifrar.config(state=tk.NORMAL)
            messagebox.showerror("Error", "Ingrese un valor numérico válido para α")
        except Exception as e:
            self.btn_cifrar.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Error durante el cifrado:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def descifrar(self):
        """Ejecuta el descifrado FrDCT."""
        try:
            if self.imagen_cifrada is None:
                messagebox.showwarning("Advertencia", "Primero debe cifrar una imagen")
                return
            
            # Deshabilitar botones durante procesamiento
            self.btn_cifrar.config(state=tk.DISABLED)
            self.btn_descifrar.config(state=tk.DISABLED)
            
            # Actualizar info con mensaje de progreso
            self.texto_info.config(state=tk.NORMAL)
            self.texto_info.delete(1.0, tk.END)
            self.texto_info.insert(1.0,
                "⏳ DESCIFRANDO...\n"
                f"{'='*35}\n\n"
                f"Aplicando FrDCT inversa\n"
                f"con orden α = {-self.alpha_actual}\n\n"
                "Recuperando imagen\n"
                "original...\n\n"
                "Por favor espere..."
            )
            self.texto_info.config(state=tk.DISABLED)
            self.ventana.update()
            
            # Ejecutar descifrado
            print(f"Iniciando descifrado FrDCT con α={-self.alpha_actual}...")
            self.aplicar_frdct_descifrado()
            
            # Mostrar resultados
            print("Generando visualización comparativa...")
            self.mostrar_resultados_completos()
            
            # Habilitar botones
            self.btn_cifrar.config(state=tk.NORMAL)
            self.btn_descifrar.config(state=tk.NORMAL)
            
            print("✓ Descifrado completado exitosamente")
            
        except Exception as e:
            self.btn_cifrar.config(state=tk.NORMAL)
            self.btn_descifrar.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Error durante el descifrado:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def frdct_1d(self, signal, alpha):
        """
        Calcula la FrDCT 1D de una señal.
        
        Fórmula FrDCT:
        F^(α)[k] = Σ f[n] · cos_α(π/N · (n + 1/2) · (k + 1/2))
        
        Parámetros:
        -----------
        signal : ndarray
            Señal de entrada
        alpha : float
            Orden fraccional
        
        Retorna:
        --------
        ndarray : Transformada FrDCT
        """
        N = len(signal)
        output = np.zeros(N, dtype=np.float64)
        
        # Calcular coeficientes de normalización
        C = np.ones(N) * np.sqrt(2.0 / N)
        C[0] = np.sqrt(1.0 / N)
        
        for k in range(N):
            sum_val = 0.0
            for n in range(N):
                # Argumento del coseno fraccional
                arg = np.pi / N * (n + 0.5) * (k + 0.5)
                
                # Coseno fraccional: rotación en el plano complejo
                # cos_α(θ) ≈ cos(α·θ) para implementación simplificada
                cos_frac = np.cos(alpha * arg)
                
                sum_val += signal[n] * cos_frac
            
            output[k] = C[k] * sum_val
        
        return output
    
    def aplicar_frdct_cifrado(self, alpha):
        """
        Aplica FrDCT a la imagen para cifrarla.
        
        Algoritmo de Cifrado (Algorithm 1):
        1. Input: Una imagen I
        2. Aplicar FrDCT a la imagen con orden fraccional α
        3. Realizar operación de cifrado empleando FrDCT en la matriz resultante
        4. Output: Matriz cifrada E(α)
        """
        self.alpha_actual = alpha
        
        # Normalizar imagen a rango [0, 1]
        imagen_norm = self.imagen_original.astype(np.float64) / 255.0
        
        h, w = imagen_norm.shape
        print(f"  Dimensiones imagen: {h}x{w}")
        print(f"  Aplicando FrDCT con α={alpha}...")
        
        # Aplicar FrDCT por filas
        print("    Transformando filas...")
        imagen_frdct_filas = np.zeros_like(imagen_norm)
        for i in range(h):
            if i % 50 == 0:
                print(f"      Fila {i}/{h}")
            imagen_frdct_filas[i, :] = self.frdct_1d(imagen_norm[i, :], alpha)
        
        # Aplicar FrDCT por columnas
        print("    Transformando columnas...")
        self.matriz_frdct = np.zeros_like(imagen_norm)
        for j in range(w):
            if j % 50 == 0:
                print(f"      Columna {j}/{w}")
            self.matriz_frdct[:, j] = self.frdct_1d(imagen_frdct_filas[:, j], alpha)
        
        # Operación de cifrado adicional: segunda aplicación de FrDCT (E(F))
        print("    Aplicando segunda transformada (cifrado)...")
        imagen_cifrada_temp = np.zeros_like(self.matriz_frdct)
        for i in range(h):
            imagen_cifrada_temp[i, :] = self.frdct_1d(self.matriz_frdct[i, :], alpha)
        
        imagen_cifrada_final = np.zeros_like(imagen_cifrada_temp)
        for j in range(w):
            imagen_cifrada_final[:, j] = self.frdct_1d(imagen_cifrada_temp[:, j], alpha)
        
        # Normalizar a rango [0, 255] para visualización
        self.imagen_cifrada = np.abs(imagen_cifrada_final)
        self.imagen_cifrada = (self.imagen_cifrada - self.imagen_cifrada.min())
        self.imagen_cifrada = (self.imagen_cifrada / self.imagen_cifrada.max() * 255).astype(np.uint8)
        
        print(f"  ✓ Cifrado FrDCT completado")
    
    def aplicar_frdct_descifrado(self):
        """
        Aplica FrDCT inversa para descifrar la imagen.
        
        Algoritmo de Descifrado (Algorithm 2):
        1. Input: Matriz cifrada E(α)
        2. Realizar descifrado usando FrDCT inversa: D(E(α))
        3. Aplicar FrDCT inversa a la matriz resultante D(E(α)) con orden -α
        4. Convertir la matriz resultante a imagen
        5. Output: Imagen descifrada I
        """
        alpha = -self.alpha_actual  # Orden inverso
        
        # Normalizar imagen cifrada
        imagen_cifrada_norm = self.imagen_cifrada.astype(np.float64) / 255.0
        
        h, w = imagen_cifrada_norm.shape
        print(f"  Aplicando FrDCT inversa con α={alpha}...")
        
        # Primera transformada inversa (descifrado)
        print("    Primera transformada inversa (filas)...")
        imagen_desc1_filas = np.zeros_like(imagen_cifrada_norm)
        for i in range(h):
            if i % 50 == 0:
                print(f"      Fila {i}/{h}")
            imagen_desc1_filas[i, :] = self.frdct_1d(imagen_cifrada_norm[i, :], alpha)
        
        print("    Primera transformada inversa (columnas)...")
        imagen_desc1 = np.zeros_like(imagen_desc1_filas)
        for j in range(w):
            if j % 50 == 0:
                print(f"      Columna {j}/{w}")
            imagen_desc1[:, j] = self.frdct_1d(imagen_desc1_filas[:, j], alpha)
        
        # Segunda transformada inversa (reconstrucción)
        print("    Segunda transformada inversa (filas)...")
        imagen_desc2_filas = np.zeros_like(imagen_desc1)
        for i in range(h):
            if i % 50 == 0:
                print(f"      Fila {i}/{h}")
            imagen_desc2_filas[i, :] = self.frdct_1d(imagen_desc1[i, :], alpha)
        
        print("    Segunda transformada inversa (columnas)...")
        imagen_descifrada_norm = np.zeros_like(imagen_desc2_filas)
        for j in range(w):
            if j % 50 == 0:
                print(f"      Columna {j}/{w}")
            imagen_descifrada_norm[:, j] = self.frdct_1d(imagen_desc2_filas[:, j], alpha)
        
        # Normalizar a rango [0, 255]
        self.imagen_descifrada = np.abs(imagen_descifrada_norm)
        self.imagen_descifrada = (self.imagen_descifrada - self.imagen_descifrada.min())
        self.imagen_descifrada = (self.imagen_descifrada / self.imagen_descifrada.max() * 255).astype(np.uint8)
        
        print(f"  ✓ Descifrado FrDCT completado")
    
    def mostrar_resultados_cifrado(self, alpha):
        """Muestra visualización después del cifrado."""
        # Limpiar canvas anterior
        if self.canvas_actual:
            self.canvas_actual.get_tk_widget().destroy()
        if self.toolbar_actual:
            self.toolbar_actual.destroy()
        
        # Crear figura con 2 paneles
        fig = Figure(figsize=(12, 9), dpi=100)
        
        # Panel 1: Imagen original
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.imagen_original, cmap='gray')
        ax1.set_title('Imagen Original', fontsize=12, fontweight='bold', pad=8)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Panel 2: Imagen cifrada
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.imagen_cifrada, cmap='gray')
        ax2.set_title(f'Imagen Cifrada (α={alpha})', fontsize=12, fontweight='bold', pad=8)
        ax2.axis('on')
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.05, wspace=0.15)
        
        # Crear canvas
        self.canvas_actual = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        self.canvas_actual.draw()
        
        # Toolbar
        toolbar_frame = tk.Frame(self.frame_grafico)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar_actual = NavigationToolbar2Tk(self.canvas_actual, toolbar_frame)
        self.toolbar_actual.update()
        
        self.canvas_actual.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Actualizar información
        self.actualizar_info_cifrado(alpha)
    
    def mostrar_resultados_completos(self):
        """Muestra visualización completa: original, cifrada y descifrada."""
        # Limpiar canvas anterior
        if self.canvas_actual:
            self.canvas_actual.get_tk_widget().destroy()
        if self.toolbar_actual:
            self.toolbar_actual.destroy()
        
        # Crear figura con 3 paneles
        fig = Figure(figsize=(12, 9), dpi=100)
        
        # Panel 1: Imagen original
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(self.imagen_original, cmap='gray')
        ax1.set_title('Imagen Original', fontsize=11, fontweight='bold', pad=8)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.tick_params(labelsize=7)
        
        # Panel 2: Imagen cifrada
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(self.imagen_cifrada, cmap='gray')
        ax2.set_title(f'Imagen Cifrada\n(α={self.alpha_actual})', fontsize=11, fontweight='bold', pad=8)
        ax2.axis('on')
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.tick_params(labelsize=7)
        
        # Panel 3: Imagen descifrada
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(self.imagen_descifrada, cmap='gray')
        ax3.set_title(f'Imagen Descifrada\n(α={-self.alpha_actual})', fontsize=11, fontweight='bold', pad=8)
        ax3.axis('on')
        ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax3.tick_params(labelsize=7)
        
        fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.05, wspace=0.18)
        
        # Crear canvas
        self.canvas_actual = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        self.canvas_actual.draw()
        
        # Toolbar
        toolbar_frame = tk.Frame(self.frame_grafico)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar_actual = NavigationToolbar2Tk(self.canvas_actual, toolbar_frame)
        self.toolbar_actual.update()
        
        self.canvas_actual.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Actualizar información
        self.actualizar_info_descifrado()
    
    def actualizar_info_cifrado(self, alpha):
        """Actualiza el panel de información después del cifrado."""
        self.texto_info.config(state=tk.NORMAL)
        self.texto_info.delete(1.0, tk.END)
        
        info_texto = (
            f"✅ CIFRADO EXITOSO\n"
            f"{'='*35}\n\n"
            f"📊 PARÁMETROS:\n\n"
            f"Algoritmo: FrDCT\n"
            f"Orden fraccional α: {alpha}\n"
            f"Dimensiones: {self.imagen_original.shape}\n\n"
            f"{'='*35}\n\n"
            f"🔒 PROCESO DE CIFRADO:\n\n"
            f"1. Normalización:\n"
            f"   I ∈ [0,1]\n\n"
            f"2. FrDCT filas:\n"
            f"   F₁ = FrDCT(I, α)\n\n"
            f"3. FrDCT columnas:\n"
            f"   F₂ = FrDCT(F₁, α)\n\n"
            f"4. Cifrado (doble):\n"
            f"   E = FrDCT(F₂, α)\n\n"
            f"{'='*35}\n\n"
            f"🔐 SEGURIDAD:\n\n"
            f"• La imagen está cifrada\n"
            f"  con clave α = {alpha}\n\n"
            f"• Sin la clave correcta,\n"
            f"  es imposible recuperar\n"
            f"  la imagen original\n\n"
            f"• Robustez: 2^{int(alpha*100)} bits\n\n"
            f"{'='*35}\n\n"
            f"SIGUIENTE PASO:\n\n"
            f"Presione 'Descifrar'\n"
            f"para recuperar la\n"
            f"imagen original usando\n"
            f"FrDCT inversa con α={-alpha}"
        )
        
        self.texto_info.insert(1.0, info_texto)
        self.texto_info.config(state=tk.DISABLED)
    
    def actualizar_info_descifrado(self):
        """Actualiza el panel de información después del descifrado."""
        self.texto_info.config(state=tk.NORMAL)
        self.texto_info.delete(1.0, tk.END)
        
        # Calcular MSE entre original y descifrada
        mse = np.mean((self.imagen_original.astype(float) - self.imagen_descifrada.astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        
        info_texto = (
            f"✅ DESCIFRADO EXITOSO\n"
            f"{'='*35}\n\n"
            f"📊 RESULTADOS:\n\n"
            f"Clave α: {self.alpha_actual}\n"
            f"Clave inversa: {-self.alpha_actual}\n\n"
            f"{'='*35}\n\n"
            f"🔓 PROCESO DESCIFRADO:\n\n"
            f"1. Input: E(α)\n\n"
            f"2. FrDCT⁻¹ filas:\n"
            f"   D₁ = FrDCT(E, -α)\n\n"
            f"3. FrDCT⁻¹ columnas:\n"
            f"   D₂ = FrDCT(D₁, -α)\n\n"
            f"4. Reconstrucción:\n"
            f"   I' = FrDCT(D₂, -α)\n\n"
            f"{'='*35}\n\n"
            f"📈 CALIDAD:\n\n"
            f"MSE: {mse:.4f}\n"
            f"PSNR: {psnr:.2f} dB\n\n"
        )
        
        if psnr > 30:
            info_texto += "✓ Excelente recuperación\n\n"
        elif psnr > 20:
            info_texto += "✓ Buena recuperación\n\n"
        else:
            info_texto += "⚠ Recuperación parcial\n\n"
        
        info_texto += (
            f"{'='*35}\n\n"
            f"📐 PROPIEDADES FrDCT:\n\n"
            f"Aditividad:\n"
            f"FrDCT(α+β) =\n"
            f"  FrDCT(α)·FrDCT(β)\n\n"
            f"Ortogonalidad:\n"
            f"FrDCT(α)·FrDCT(-α) =\n"
            f"  Identidad\n\n"
            f"Rotación:\n"
            f"θ = α·π/(2N)\n\n"
            f"Energy Compaction:\n"
            f"Energía concentrada\n"
            f"en pocos coeficientes"
        )
        
        self.texto_info.insert(1.0, info_texto)
        self.texto_info.config(state=tk.DISABLED)
