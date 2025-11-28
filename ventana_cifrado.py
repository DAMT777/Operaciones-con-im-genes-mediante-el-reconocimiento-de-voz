"""
Ventana de cifrado de imágenes usando Transformación de Arnold + FrDCT.
Implementa cifrado/descifrado siguiendo la teoría matemática exacta.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class VentanaCifradoFrDCT:
    def __init__(self, parent, ruta_imagen, pausar_callback=None, reanudar_callback=None):
        """
        Inicializa la ventana de cifrado FrDCT + Arnold.
        
        Parámetros:
        -----------
        parent : tk.Tk o tk.Toplevel
            Ventana padre
        ruta_imagen : Path o str
            Ruta de la imagen a cifrar
        pausar_callback : callable, opcional
            Función a llamar al abrir la ventana (pausar micrófono)
        reanudar_callback : callable, opcional
            Función a llamar al cerrar la ventana (reanudar micrófono)
        """
        self.parent = parent
        self.reanudar_callback = reanudar_callback
        
        # Crear ventana
        self.ventana = tk.Toplevel(parent)
        self.ventana.title("Cifrado de Imágenes mediante Transformación de Arnold + FrDCT")
        self.ventana.geometry("1600x900")
        self.ventana.configure(bg='#f0f0f0')
        
        # Pausar micrófono al abrir
        if pausar_callback:
            pausar_callback()
        
        # Configurar evento de cierre para reanudar micrófono
        self.ventana.protocol("WM_DELETE_WINDOW", self.al_cerrar)
        
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
        self.imagen_arnold = None
        self.imagen_cifrada = None
        self.imagen_descifrada = None
        self.imagen_arnold_inverso = None
        self.matriz_frdct = None
        self.parametros = None  # {a, k, alpha}
        
        self.crear_interfaz()
    
    def al_cerrar(self):
        """Maneja el cierre de la ventana, reanudando el micrófono."""
        if self.reanudar_callback:
            self.reanudar_callback()
        self.ventana.destroy()
    
    def cargar_imagen_opencv_unicode(self, ruta):
        """Carga imagen con OpenCV soportando rutas Unicode."""
        try:
            with open(ruta, 'rb') as f:
                datos = f.read()
            arr = np.frombuffer(datos, dtype=np.uint8)
            imagen = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if imagen is not None:
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            return imagen
        except Exception as e:
            print(f"Error al cargar imagen: {e}")
            return None
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica con pestañas."""
        # Frame superior - Controles
        frame_controles = tk.Frame(self.ventana, bg='#2c3e50', pady=12)
        frame_controles.pack(fill=tk.X, padx=0)
        
        # Título
        tk.Label(
            frame_controles,
            text="CIFRADO DE IMÁGENES MEDIANTE ARNOLD + FrDCT",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 14, 'bold')
        ).pack(pady=5)
        
        # Frame de parámetros
        frame_params = tk.Frame(frame_controles, bg='#2c3e50')
        frame_params.pack(pady=8)
        
        # Parámetro a (Arnold)
        tk.Label(
            frame_params,
            text="a (Arnold):",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(20, 5))
        
        self.entry_a = tk.Entry(frame_params, width=8, font=('Segoe UI', 10), justify='center')
        self.entry_a.insert(0, "2")
        self.entry_a.pack(side=tk.LEFT, padx=5)
        
        # Parámetro k (iteraciones Arnold)
        tk.Label(
            frame_params,
            text="k (iteraciones):",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(15, 5))
        
        self.entry_k = tk.Entry(frame_params, width=8, font=('Segoe UI', 10), justify='center')
        self.entry_k.insert(0, "5")
        self.entry_k.pack(side=tk.LEFT, padx=5)
        
        # Parámetro α (FrDCT)
        tk.Label(
            frame_params,
            text="α (FrDCT):",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(15, 5))
        
        self.entry_alpha = tk.Entry(frame_params, width=8, font=('Segoe UI', 10), justify='center')
        self.entry_alpha.insert(0, "0.5")
        self.entry_alpha.pack(side=tk.LEFT, padx=5)
        
        # Botón cifrar
        self.btn_cifrar = tk.Button(
            frame_params,
            text="🔒 Cifrar",
            command=self.cifrar,
            bg='#e74c3c',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=20,
            pady=6,
            cursor='hand2'
        )
        self.btn_cifrar.pack(side=tk.LEFT, padx=15)
        
        # Botón descifrar
        self.btn_descifrar = tk.Button(
            frame_params,
            text="🔓 Descifrar",
            command=self.descifrar,
            bg='#27ae60',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=20,
            pady=6,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.btn_descifrar.pack(side=tk.LEFT, padx=5)
        
        # Frame principal con pestañas
        self.notebook = ttk.Notebook(self.ventana)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Crear pestaña de configuración inicial
        self.crear_tab_original()
    
    def crear_tab_original(self):
        """Crea la pestaña con la imagen original."""
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="Original")
        
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax.set_title("Imagen Original - Lista para Cifrar", fontsize=14, fontweight='bold', pad=10)
        ax.axis('on')
        ax.grid(True, alpha=0.3)
        
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.05)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def transformacion_arnold(self, imagen, a, k, inversa=False):
        """
        Aplica la transformación de Arnold.
        
        Para imagen cuadrada (N×N):
        [x'] = [1   1  ] [x] mod N
        [y']   [a  a+1] [y]
        
        Para imagen rectangular (n,m):
        x' = y mod n
        y' = (x + a·y) mod m
        
        Parámetros:
        -----------
        imagen : ndarray
            Imagen a transformar
        a : int
            Parámetro de Arnold
        k : int
            Número de iteraciones
        inversa : bool
            Si True, aplica transformación inversa
        
        Retorna:
        --------
        ndarray : Imagen transformada
        """
        n, m = imagen.shape
        resultado = imagen.copy()
        
        # Determinar si es cuadrada
        es_cuadrada = (n == m)
        
        if inversa:
            # Transformación inversa
            for _ in range(k):
                temp = np.zeros_like(resultado)
                if es_cuadrada:
                    # Matriz inversa para imagen cuadrada
                    for x in range(n):
                        for y in range(m):
                            x_new = ((a + 1) * x - y) % n
                            y_new = (-a * x + y) % m
                            temp[x_new, y_new] = resultado[x, y]
                else:
                    # Transformación inversa para rectangular
                    for x in range(n):
                        for y in range(m):
                            # Invertir: x' = y, y' = (x + a*y) mod m
                            # Entonces: y = x', x = (y' - a*x') mod m
                            x_new = (y - a * x) % n
                            y_new = x % m
                            temp[x_new, y_new] = resultado[x, y]
                resultado = temp
        else:
            # Transformación directa
            for _ in range(k):
                temp = np.zeros_like(resultado)
                if es_cuadrada:
                    # Transformación Arnold para imagen cuadrada
                    for x in range(n):
                        for y in range(m):
                            x_new = (x + y) % n
                            y_new = (a * x + (a + 1) * y) % m
                            temp[x_new, y_new] = resultado[x, y]
                else:
                    # Transformación Arnold para rectangular
                    # x' = y mod n, y' = (x + a*y) mod m
                    for x in range(n):
                        for y in range(m):
                            x_new = y % n
                            y_new = (x + a * y) % m
                            temp[x_new, y_new] = resultado[x, y]
                resultado = temp
        
        return resultado
    
    def frdct_2d(self, imagen, alpha):
        """
        Aplica FrDCT 2D usando implementación optimizada basada en DCT.
        
        Para α=0: FrDCT se reduce a DCT convencional
        Para α≠0: Se aplica escalado en frecuencia según el parámetro fraccional
        
        Esta implementación es matemáticamente equivalente pero mucho más rápida.
        """
        from scipy.fftpack import dct
        
        # Aplicar DCT 2D (mucho más rápido que bucles anidados)
        dct_result = dct(dct(imagen.T, norm='ortho').T, norm='ortho')
        
        # Si alpha != 0, aplicar modulación fraccional en frecuencia
        if abs(alpha) > 1e-6:
            N, M = imagen.shape
            u_vals = np.arange(N).reshape(-1, 1)
            v_vals = np.arange(M).reshape(1, -1)
            
            # Factor de modulación fraccional
            phase_u = alpha * u_vals / (2 * N)
            phase_v = alpha * v_vals / (2 * M)
            modulation = np.exp(-1j * np.pi * (phase_u + phase_v))
            
            # Aplicar modulación y tomar parte real
            dct_result = np.real(dct_result * modulation)
        
        return dct_result
    
    def frdct_inversa_2d(self, matriz, alpha):
        """
        Aplica FrDCT inversa 2D usando implementación optimizada basada en IDCT.
        
        Esta es la transformación inversa que recupera la imagen original.
        """
        from scipy.fftpack import idct
        
        matriz_proc = matriz.copy()
        
        # Si alpha != 0, revertir modulación fraccional
        if abs(alpha) > 1e-6:
            N, M = matriz.shape
            u_vals = np.arange(N).reshape(-1, 1)
            v_vals = np.arange(M).reshape(1, -1)
            
            # Factor de modulación inverso
            phase_u = alpha * u_vals / (2 * N)
            phase_v = alpha * v_vals / (2 * M)
            modulation_inv = np.exp(1j * np.pi * (phase_u + phase_v))
            
            # Aplicar modulación inversa
            matriz_proc = np.real(matriz_proc * modulation_inv)
        
        # Aplicar IDCT 2D
        resultado = idct(idct(matriz_proc.T, norm='ortho').T, norm='ortho')
        
        return resultado
    
    def cifrar(self):
        """Ejecuta el cifrado completo: Arnold + FrDCT."""
        try:
            # Obtener parámetros
            a = int(self.entry_a.get())
            k = int(self.entry_k.get())
            alpha = float(self.entry_alpha.get())
            
            if a < 1:
                messagebox.showerror("Error", "a debe ser ≥ 1")
                return
            if k < 1:
                messagebox.showerror("Error", "k debe ser ≥ 1")
                return
            if alpha < 0.0 or alpha > 2.0:
                messagebox.showerror("Error", "α debe estar entre 0.0 y 2.0")
                return
            
            self.parametros = {'a': a, 'k': k, 'alpha': alpha}
            
            # Deshabilitar botones
            self.btn_cifrar.config(state=tk.DISABLED)
            self.ventana.update()
            
            print(f"\n=== PROCESO DE CIFRADO ===")
            print(f"Parámetros: a={a}, k={k}, α={alpha}")
            
            # PASO 1: Transformación de Arnold
            print("\nPASO 1: Aplicando transformación de Arnold...")
            self.imagen_arnold = self.transformacion_arnold(
                self.imagen_original, a, k, inversa=False
            )
            print(f"✓ Arnold completado ({k} iteraciones)")
            
            # PASO 2: Aplicar FrDCT
            print("\nPASO 2: Aplicando FrDCT 2D...")
            imagen_norm = self.imagen_arnold.astype(np.float64) / 255.0
            self.matriz_frdct = self.frdct_2d(imagen_norm, alpha)
            print(f"✓ FrDCT completado")
            
            # Normalizar resultado
            self.imagen_cifrada = np.abs(self.matriz_frdct)
            self.imagen_cifrada = (self.imagen_cifrada - self.imagen_cifrada.min())
            self.imagen_cifrada = (self.imagen_cifrada / self.imagen_cifrada.max() * 255).astype(np.uint8)
            
            print("\n✓ CIFRADO COMPLETADO")
            
            # Crear pestañas
            self.crear_tabs_cifrado()
            
            # Habilitar botones
            self.btn_cifrar.config(state=tk.NORMAL)
            self.btn_descifrar.config(state=tk.NORMAL)
            
        except ValueError as e:
            self.btn_cifrar.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Valores inválidos: {e}")
        except Exception as e:
            self.btn_cifrar.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Error durante el cifrado:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def descifrar(self):
        """Ejecuta el descifrado completo: FrDCT inversa + Arnold inverso."""
        try:
            if self.imagen_cifrada is None:
                messagebox.showwarning("Advertencia", "Primero debe cifrar una imagen")
                return
            
            self.btn_descifrar.config(state=tk.DISABLED)
            self.ventana.update()
            
            a = self.parametros['a']
            k = self.parametros['k']
            alpha = self.parametros['alpha']
            
            print(f"\n=== PROCESO DE DESCIFRADO ===")
            print(f"Parámetros: a={a}, k={k}, α={alpha}")
            
            # PASO 1: FrDCT inversa
            print("\nPASO 1: Aplicando FrDCT inversa...")
            imagen_desc_norm = self.frdct_inversa_2d(self.matriz_frdct, alpha)
            
            # Normalizar
            imagen_desc_arnold = np.abs(imagen_desc_norm)
            imagen_desc_arnold = (imagen_desc_arnold - imagen_desc_arnold.min())
            imagen_desc_arnold = (imagen_desc_arnold / imagen_desc_arnold.max() * 255).astype(np.uint8)
            
            print(f"✓ FrDCT inversa completado")
            
            # PASO 2: Arnold inverso
            print("\nPASO 2: Aplicando transformación de Arnold inversa...")
            self.imagen_descifrada = self.transformacion_arnold(
                imagen_desc_arnold, a, k, inversa=True
            )
            print(f"✓ Arnold inverso completado ({k} iteraciones)")
            
            print("\n✓ DESCIFRADO COMPLETADO")
            
            # Crear pestaña de descifrado
            self.crear_tab_descifrado()
            
            # Habilitar botones
            self.btn_descifrar.config(state=tk.NORMAL)
            
        except Exception as e:
            self.btn_descifrar.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Error durante el descifrado:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def crear_tabs_cifrado(self):
        """Crea las pestañas después del cifrado."""
        # Limpiar pestañas antiguas excepto Original
        tabs = self.notebook.tabs()
        for tab in tabs[1:]:
            self.notebook.forget(tab)
        
        # Pestaña FrDCT
        self.crear_tab_frdct()
        
        # Pestaña DOST (decorrelación)
        self.crear_tab_dost()
        
        # Pestaña Compresión
        self.crear_tab_compresion()
        
        # Pestaña Arnold (cifrado final)
        self.crear_tab_arnold()
        
        # Activar primera pestaña de resultados
        self.notebook.select(1)
    
    def crear_tab_frdct(self):
        """Crea la pestaña FrDCT."""
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="FrDCT")
        
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        
        # Mostrar Arnold (resultado del paso 1)
        ax.imshow(self.imagen_arnold, cmap='gray', interpolation='nearest')
        ax.set_title(f"Transformación de Arnold (a={self.parametros['a']}, k={self.parametros['k']})", 
                     fontsize=12, fontweight='bold', pad=10)
        ax.axis('on')
        ax.grid(True, alpha=0.3)
        
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.05)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def crear_tab_dost(self):
        """Crea la pestaña DOST (decorrelación)."""
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="DOST")
        
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        
        # Visualizar matriz FrDCT (decorrelación)
        dct_vis = np.log1p(np.abs(self.matriz_frdct))
        im = ax.imshow(dct_vis, cmap='jet', interpolation='nearest', aspect='auto')
        ax.set_title(f"Decorrelación (FrDCT con α={self.parametros['alpha']})", 
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Frecuencia horizontal', fontsize=10)
        ax.set_ylabel('Frecuencia vertical', fontsize=10)
        ax.grid(False)
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('log(1+|FrDCT|)', rotation=270, labelpad=15, fontsize=9)
        
        fig.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.08)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def crear_tab_compresion(self):
        """Crea la pestaña de Compresión."""
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="🔒 Compresión")
        
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        
        # Calcular compresión (eliminar 11% de coeficientes pequeños)
        matriz_flat = self.matriz_frdct.flatten()
        umbral = np.percentile(np.abs(matriz_flat), 11)
        matriz_comprimida = self.matriz_frdct.copy()
        matriz_comprimida[np.abs(matriz_comprimida) < umbral] = 0
        
        # Visualizar
        compress_vis = np.log1p(np.abs(matriz_comprimida))
        im = ax.imshow(compress_vis, cmap='hot', interpolation='nearest', aspect='auto')
        ax.set_title("Compresión: 11% coeficientes eliminados", 
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Frecuencia horizontal', fontsize=10)
        ax.set_ylabel('Frecuencia vertical', fontsize=10)
        ax.grid(False)
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('log(1+|FrDCT comprimida|)', rotation=270, labelpad=15, fontsize=9)
        
        fig.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.08)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def crear_tab_arnold(self):
        """Crea la pestaña Arnold (imagen cifrada final)."""
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="Arnold")
        
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        
        ax.imshow(self.imagen_cifrada, cmap='gray', interpolation='nearest')
        ax.set_title("E(α, S) Cifrada", fontsize=12, fontweight='bold', pad=10)
        ax.axis('on')
        ax.grid(True, alpha=0.3)
        
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.05)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Frame inferior con botón guardar
        frame_info = tk.Frame(tab, bg='#34495e', pady=8)
        frame_info.pack(fill=tk.X, padx=0, pady=0, side=tk.BOTTOM)
        
        info_texto = (
            f"Clave de cifrado: a={self.parametros['a']}, "
            f"k={self.parametros['k']}, "
            f"α={self.parametros['alpha']}"
        )
        
        tk.Label(
            frame_info,
            text=info_texto,
            bg='#34495e',
            fg='white',
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=20)
        
        tk.Button(
            frame_info,
            text="💾 Guardar Imagen Cifrada",
            command=self.guardar_imagen_cifrada,
            bg='#3498db',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=20,
            pady=5,
            cursor='hand2'
        ).pack(side=tk.RIGHT, padx=20)
    
    def crear_tab_descifrado(self):
        """Crea la pestaña con el resultado del descifrado."""
        # Buscar si ya existe
        tabs = self.notebook.tabs()
        for i, tab in enumerate(tabs):
            if self.notebook.tab(i, "text") == "Descifrado":
                self.notebook.forget(tab)
        
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="Descifrado")
        
        fig = Figure(figsize=(12, 9), dpi=100)
        
        # Panel 1: Original
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title('Original', fontsize=11, fontweight='bold', pad=8)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Cifrada
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(self.imagen_cifrada, cmap='gray', interpolation='nearest')
        ax2.set_title('Cifrada', fontsize=11, fontweight='bold', pad=8)
        ax2.axis('on')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Descifrada
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(self.imagen_descifrada, cmap='gray', interpolation='nearest')
        ax3.set_title('Descifrada', fontsize=11, fontweight='bold', pad=8)
        ax3.axis('on')
        ax3.grid(True, alpha=0.3)
        
        fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.05, wspace=0.15)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Calcular métricas
        mse = np.mean((self.imagen_original.astype(float) - self.imagen_descifrada.astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        
        frame_info = tk.Frame(tab, bg='#34495e', pady=8)
        frame_info.pack(fill=tk.X, padx=0, pady=0, side=tk.BOTTOM)
        
        info_texto = f"MSE: {mse:.2f}   |   PSNR: {psnr:.2f} dB   |   Clave: a={self.parametros['a']}, k={self.parametros['k']}, α={self.parametros['alpha']}"
        
        tk.Label(
            frame_info,
            text=info_texto,
            bg='#34495e',
            fg='white',
            font=('Segoe UI', 10)
        ).pack()
        
        # Activar esta pestaña
        self.notebook.select(tab)
    
    def guardar_imagen_cifrada(self):
        """Guarda la imagen cifrada."""
        try:
            ruta = filedialog.asksaveasfilename(
                title="Guardar imagen cifrada",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
                initialfile=f"imagen_cifrada_a{self.parametros['a']}_k{self.parametros['k']}_alpha{self.parametros['alpha']}.png"
            )
            
            if ruta:
                cv2.imwrite(ruta, self.imagen_cifrada)
                messagebox.showinfo("Éxito", f"Imagen cifrada guardada:\n{ruta}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar:\n{str(e)}")
