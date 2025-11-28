
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from cifrado_arnold_frdct import (
    transformacion_arnold,
    frdct_2d,
    frdct_inversa_2d,
    comprimir_dct,
    cifrar_imagen_completo,
    descifrar_imagen_completo
)

class VentanaCifradoFrDCT:
    def __init__(self, parent, ruta_imagen, pausar_callback=None, reanudar_callback=None):
        self.parent = parent
        self.reanudar_callback = reanudar_callback
        
        self.ventana = tk.Toplevel(parent)
        self.ventana.title("Cifrado de Imágenes mediante Transformación de Arnold + FrDCT")
        self.ventana.geometry("1600x900")
        self.ventana.configure(bg='#f0f0f0')
        
        if pausar_callback:
            pausar_callback()
        
        self.ventana.protocol("WM_DELETE_WINDOW", self.al_cerrar)
        
        self.imagen_original = self.cargar_imagen_opencv_unicode(str(ruta_imagen))
        
        if self.imagen_original is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{ruta_imagen}")
            self.ventana.destroy()
            return
        
        if len(self.imagen_original.shape) == 3:
            self.imagen_original = cv2.cvtColor(self.imagen_original, cv2.COLOR_RGB2GRAY)
        
        self.imagen_arnold = None
        self.imagen_comprimida = None
        self.matriz_dct_comprimida = None
        self.porcentaje_compresion = 2.0
        self.imagen_cifrada = None
        self.imagen_descifrada = None
        self.imagen_arnold_inverso = None
        self.matriz_frdct = None
        self.parametros = None
        
        self.crear_interfaz()
    
    def al_cerrar(self):
        if self.reanudar_callback:
            self.reanudar_callback()
        self.ventana.destroy()
    
    def cargar_imagen_opencv_unicode(self, ruta):
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
        frame_controles = tk.Frame(self.ventana, bg='#2c3e50', pady=12)
        frame_controles.pack(fill=tk.X, padx=0)
        
        tk.Label(
            frame_controles,
            text="CIFRADO DE IMÁGENES MEDIANTE ARNOLD + FrDCT",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 14, 'bold')
        ).pack(pady=5)
        
        frame_params = tk.Frame(frame_controles, bg='#2c3e50')
        frame_params.pack(pady=8)
        
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
        
        self.notebook = ttk.Notebook(self.ventana)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.crear_tab_original()
    
    def crear_tab_original(self):
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
    
    def cifrar(self):
        try:
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
            
            self.btn_cifrar.config(state=tk.DISABLED)
            self.ventana.update()
            
            resultado = cifrar_imagen_completo(
                self.imagen_original, a, k, alpha, self.porcentaje_compresion
            )
            
            self.imagen_arnold = resultado['imagen_arnold']
            self.imagen_comprimida = resultado['imagen_comprimida']
            self.matriz_dct_comprimida = resultado['matriz_dct_comprimida']
            self.matriz_frdct = resultado['matriz_frdct']
            self.imagen_cifrada = resultado['imagen_cifrada']
            
            self.crear_tabs_cifrado()
            
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
        try:
            if self.imagen_cifrada is None:
                messagebox.showwarning("Advertencia", "Primero debe cifrar una imagen")
                return
            
            self.btn_descifrar.config(state=tk.DISABLED)
            self.ventana.update()
            
            a = self.parametros['a']
            k = self.parametros['k']
            alpha = self.parametros['alpha']
            
            self.imagen_descifrada = descifrar_imagen_completo(
                self.matriz_frdct, a, k, alpha
            )
            
            self.crear_tab_descifrado()
            
            self.btn_descifrar.config(state=tk.NORMAL)
            
        except Exception as e:
            self.btn_descifrar.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Error durante el descifrado:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def crear_tabs_cifrado(self):
        tabs = self.notebook.tabs()
        for tab in tabs[1:]:
            self.notebook.forget(tab)
        
        self.crear_tab_frdct()
        
        self.crear_tab_dost()
        
        self.crear_tab_compresion()
        
        self.crear_tab_arnold()
        
        self.notebook.select(1)
    
    def crear_tab_frdct(self):
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="FrDCT")
        
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        
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
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="DOST")
        
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        
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
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="🗃 Compresión")
        
        fig = Figure(figsize=(12, 7), dpi=100)
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.imagen_arnold, cmap='gray', interpolation='nearest')
        ax1.set_title("Imagen tras Arnold (antes compresión)", fontsize=11, fontweight='bold', pad=10)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(1, 2, 2)
        dct_vis = np.log1p(np.abs(self.matriz_dct_comprimida))
        im = ax2.imshow(dct_vis, cmap='hot', interpolation='nearest', aspect='auto')
        coef_eliminados = np.sum(self.matriz_dct_comprimida == 0) / self.matriz_dct_comprimida.size * 100
        ax2.set_title(f"DCT Comprimida ({coef_eliminados:.1f}% coef. eliminados)", 
                     fontsize=11, fontweight='bold', pad=10)
        ax2.set_xlabel('Frecuencia horizontal', fontsize=9)
        ax2.set_ylabel('Frecuencia vertical', fontsize=9)
        ax2.grid(False)
        
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('log(1+|DCT|)', rotation=270, labelpad=15, fontsize=8)
        
        fig.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.08, wspace=0.15)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def crear_tab_arnold(self):
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="Arnold")
        
        fig = Figure(figsize=(12, 7), dpi=100)
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title("Imagen Original", fontsize=11, fontweight='bold', pad=10)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.imagen_cifrada, cmap='gray', interpolation='nearest')
        ax2.set_title("E(α, S) Cifrada", fontsize=11, fontweight='bold', pad=10)
        ax2.axis('on')
        ax2.grid(True, alpha=0.3)
        
        fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.05, wspace=0.12)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
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
        tabs = self.notebook.tabs()
        for i, tab in enumerate(tabs):
            if self.notebook.tab(i, "text") == "Descifrado":
                self.notebook.forget(tab)
        
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="Descifrado")
        
        fig = Figure(figsize=(12, 9), dpi=100)
        
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title('Original', fontsize=11, fontweight='bold', pad=8)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(self.imagen_cifrada, cmap='gray', interpolation='nearest')
        ax2.set_title('Cifrada', fontsize=11, fontweight='bold', pad=8)
        ax2.axis('on')
        ax2.grid(True, alpha=0.3)
        
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
        ).pack(side=tk.LEFT, padx=20)
        
        tk.Button(
            frame_info,
            text="📊 Ver Descompresión",
            command=self.mostrar_descompresion_descifrado,
            bg='#27ae60',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=20,
            pady=5,
            cursor='hand2'
        ).pack(side=tk.RIGHT, padx=20)
        
        self.notebook.select(tab)
    
    def guardar_imagen_cifrada(self):
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
    
    def mostrar_descompresion_descifrado(self):
        if self.imagen_descifrada is None:
            messagebox.showwarning("Advertencia", "Primero debe descifrar la imagen")
            return
        
        ventana_descomp = tk.Toplevel(self.ventana)
        ventana_descomp.title("Descompresión de Imagen Descifrada")
        ventana_descomp.geometry("1400x800")
        
        fig = Figure(figsize=(14, 8), dpi=100)
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title('Original', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(self.imagen_cifrada, cmap='gray', interpolation='nearest')
        ax2.set_title('Cifrada', fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(self.imagen_descifrada, cmap='gray', interpolation='nearest')
        ax3.set_title('Descifrada', fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        dct_coef = cv2.dct(self.imagen_descifrada.astype(np.float32))
        ax4 = fig.add_subplot(2, 2, 4)
        dct_vis = np.log(np.abs(dct_coef) + 1)
        im = ax4.imshow(dct_vis, cmap='hot', interpolation='nearest')
        ax4.set_title('DCT de Imagen Descifrada', fontsize=11, fontweight='bold')
        ax4.axis('off')
        fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        fig.suptitle('Análisis de Descompresión Post-Descifrado', fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, ventana_descomp)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
