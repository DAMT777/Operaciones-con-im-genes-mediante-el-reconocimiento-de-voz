
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from compresion_dct import (
    comprimir_imagen_dct,
    descomprimir_imagen_dct,
    calcular_metricas_compresion
)

def cargar_imagen_unicode(ruta):
    try:
        with open(ruta, 'rb') as f:
            datos = f.read()
        
        arr = np.frombuffer(datos, dtype=np.uint8)
        
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return None

class VentanaCompresionDCT:
    def __init__(self, parent, ruta_imagen, pausar_callback=None, reanudar_callback=None):
        self.parent = parent
        self.ruta_imagen = ruta_imagen
        self.reanudar_callback = reanudar_callback
        
        self.imagen_original = cargar_imagen_unicode(str(ruta_imagen))
        
        if self.imagen_original is None:
            messagebox.showerror(
                "Error", 
                f"No se pudo cargar la imagen.\n\n"
                f"Ruta: {ruta_imagen}\n\n"
                f"Sugerencia: Evite rutas con tildes, ñ u otros caracteres especiales."
            )
            return
        
        self.coeficientes_dct = None
        self.dct_completa = None
        self.forma_original = None
        self.imagen_comprimida = None
        self.num_coefs_eliminados = 0
        self.metricas = None
        self.canvas_actual = None
        self.toolbar_actual = None
        
        self.resultados_porcentajes = {}
        self.porcentajes_procesados = []
        
        self.ventana = tk.Toplevel(parent)
        self.ventana.title("Compresión de Imágenes mediante DCT-2D")
        self.ventana.geometry("1600x900")
        self.ventana.configure(bg='#f0f0f0')
        
        if pausar_callback:
            pausar_callback()
        
        self.ventana.protocol("WM_DELETE_WINDOW", self.al_cerrar)
        
        self.crear_interfaz()
    
    def al_cerrar(self):
        if self.reanudar_callback:
            self.reanudar_callback()
        self.ventana.destroy()
        
    def crear_interfaz(self):
        frame_controles = tk.Frame(self.ventana, bg='#2c3e50', pady=12)
        frame_controles.pack(fill=tk.X, padx=0)
        
        tk.Label(
            frame_controles,
            text="COMPRESIÓN DE IMÁGENES MEDIANTE DCT-2D",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 14, 'bold')
        ).pack(pady=5)
        
        frame_botones2 = tk.Frame(frame_controles, bg='#2c3e50')
        frame_botones2.pack(pady=5)
        
        tk.Label(
            frame_botones2,
            text="Porcentajes (% a eliminar):",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT, padx=8)
        
        self.entry_porcentajes = tk.Entry(
            frame_botones2,
            width=25,
            font=('Segoe UI', 11),
            justify='center'
        )
        self.entry_porcentajes.insert(0, "0.5,1,1.5,2")
        self.entry_porcentajes.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            frame_botones2,
            text="Ej: 0.5,1,2,3 (mínimo 3 valores)",
            bg='#2c3e50',
            fg='#bdc3c7',
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=8)
        
        self.btn_comprimir = tk.Button(
            frame_botones2,
            text="Procesar",
            command=self.comprimir,
            bg='#27ae60',
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            padx=25,
            pady=6,
            cursor='hand2',
            relief=tk.RAISED,
            bd=2
        )
        self.btn_comprimir.pack(side=tk.LEFT, padx=10)
        
        self.notebook = ttk.Notebook(self.ventana)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.crear_tab_configuracion()
    
    def crear_tab_configuracion(self):
        tab = tk.Frame(self.notebook, bg='#f0f0f0')
        self.notebook.add(tab, text="Configuración")
        
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax.set_title("Imagen Original - Lista para Comprimir", fontsize=14, fontweight='bold', pad=10)
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
        
    def comprimir(self):
        try:
            porcentajes_texto = self.entry_porcentajes.get()
            porcentajes = [float(p.strip()) for p in porcentajes_texto.replace(';', ',').split(',') if p.strip()]
            
            if len(porcentajes) < 3:
                messagebox.showerror("Error", "Debe ingresar al menos 3 porcentajes")
                return
            
            for p in porcentajes:
                if p < 0 or p > 100:
                    messagebox.showerror("Error", "Todos los porcentajes deben estar entre 0 y 100")
                    return
            
            self.btn_comprimir.config(state=tk.DISABLED)
            self.ventana.update()
            
            from compresion_dct import aplicar_dct_bloques
            print("Aplicando DCT-2D completa...")
            self.dct_completa, _ = aplicar_dct_bloques(self.imagen_original)
            
            tabs = self.notebook.tabs()
            for tab in tabs[1:]:
                self.notebook.forget(tab)
            
            self.resultados_porcentajes = {}
            self.porcentajes_procesados = sorted(porcentajes)
            
            for i, porcentaje in enumerate(self.porcentajes_procesados, 1):
                print(f"\n[{i}/{len(self.porcentajes_procesados)}] Procesando {porcentaje}%...")
                self.ventana.title(f"Compresión DCT - Procesando {i}/{len(self.porcentajes_procesados)} ({porcentaje}%)")
                
                coefs_dct, forma, num_eliminados = comprimir_imagen_dct(
                    self.imagen_original, porcentaje, tamanio_bloque=8
                )
                
                img_reconstruida = descomprimir_imagen_dct(
                    coefs_dct, forma, tamanio_bloque=8
                )
                
                total_coefs = coefs_dct.size
                metricas = calcular_metricas_compresion(
                    self.imagen_original,
                    img_reconstruida,
                    num_eliminados,
                    total_coefs
                )
                
                self.resultados_porcentajes[porcentaje] = {
                    'coeficientes': coefs_dct,
                    'imagen_reconstruida': img_reconstruida,
                    'num_eliminados': num_eliminados,
                    'metricas': metricas
                }
            
            print("\nCreando pestañas...")
            self.crear_tab_resumen_general()
            for porcentaje in self.porcentajes_procesados:
                self.crear_tab_porcentaje(porcentaje)
            
            self.notebook.select(1)
            
            self.btn_comprimir.config(state=tk.NORMAL)
            self.ventana.title("Compresión de Imágenes mediante DCT-2D")
            
            print("✓ Compresión completada exitosamente")
            
        except ValueError:
            self.btn_comprimir.config(state=tk.NORMAL)
            messagebox.showerror("Error", "Ingrese un valor numérico válido para el porcentaje")
        except Exception as e:
            self.btn_comprimir.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Error durante la compresión:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def crear_tab_resumen_general(self):
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text="Resumen general")
        
        fig = Figure(figsize=(14, 7), dpi=100)
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title('Imagen original', fontsize=12, fontweight='bold', pad=10)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        ax2 = fig.add_subplot(1, 2, 2)
        dct_log_completo = np.log1p(np.abs(self.dct_completa))
        im2 = ax2.imshow(dct_log_completo, cmap='hot', interpolation='nearest', aspect='auto')
        ax2.set_title('Mapa DCT completa (log)', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel('Frecuencia horizontal', fontsize=10)
        ax2.set_ylabel('Frecuencia vertical', fontsize=10)
        ax2.grid(False)
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('log(1+|DCT|)', rotation=270, labelpad=15, fontsize=9)
        
        fig.tight_layout(pad=2.0)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def crear_tab_porcentaje(self, porcentaje):
        tab = tk.Frame(self.notebook, bg='white')
        self.notebook.add(tab, text=f"Imagen {porcentaje:.1f}%")
        
        datos = self.resultados_porcentajes[porcentaje]
        
        fig = Figure(figsize=(12, 9), dpi=100)
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title('Imagen original', fontsize=11, fontweight='bold', pad=8)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(datos['imagen_reconstruida'], cmap='gray', interpolation='nearest')
        ax2.set_title(f'Reconstruida ({porcentaje:.1f}% coef. eliminados)', fontsize=11, fontweight='bold', pad=8)
        ax2.axis('on')
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        ax3 = fig.add_subplot(2, 2, 3)
        dct_log = np.log1p(np.abs(datos['coeficientes']))
        im3 = ax3.imshow(dct_log, cmap='hot', interpolation='nearest', aspect='auto')
        ax3.set_title('Mapa DCT filtrada (log)', fontsize=11, fontweight='bold', pad=8)
        ax3.set_xlabel('Frecuencia horizontal', fontsize=9)
        ax3.set_ylabel('Frecuencia vertical', fontsize=9)
        ax3.grid(False)
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('log(1+|DCT|)', rotation=270, labelpad=15, fontsize=8)
        
        ax4 = fig.add_subplot(2, 2, 4)
        diferencia = np.abs(self.imagen_original.astype(float) - datos['imagen_reconstruida'].astype(float))
        im4 = ax4.imshow(diferencia, cmap='hot', interpolation='nearest')
        ax4.set_title('Diferencia absoluta |Original - Reconstruida|', fontsize=11, fontweight='bold', pad=8)
        ax4.set_xlabel('x', fontsize=9)
        ax4.set_ylabel('y', fontsize=9)
        ax4.grid(False)
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('Error', rotation=270, labelpad=15, fontsize=8)
        
        fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.06, hspace=0.32, wspace=0.25)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        
        toolbar_frame = tk.Frame(tab)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        frame_info = tk.Frame(tab, bg='#34495e', pady=8)
        frame_info.pack(fill=tk.X, padx=0, pady=0, side=tk.BOTTOM)
        
        metricas = datos['metricas']
        
        frame_metricas = tk.Frame(frame_info, bg='#34495e')
        frame_metricas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        info_texto = (
            f"(x, y) = (coordenada del mouse)   |   "
            f"PSNR: {metricas['psnr']:.2f} dB   |   "
            f"MSE: {metricas['mse']:.2f}   |   "
            f"Compresión: {metricas['tasa_compresion']:.2f}%"
        )
        
        tk.Label(
            frame_metricas,
            text=info_texto,
            bg='#34495e',
            fg='white',
            font=('Segoe UI', 10)
        ).pack()
        
        frame_boton = tk.Frame(frame_info, bg='#34495e')
        frame_boton.pack(side=tk.RIGHT, padx=10)
        
        tk.Button(
            frame_boton,
            text="🔍 Ver Descompresión",
            command=lambda p=porcentaje: self.mostrar_descompresion(p),
            bg='#9b59b6',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=15,
            pady=5,
            cursor='hand2',
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            frame_boton,
            text="💾 Guardar Imagen",
            command=lambda p=porcentaje: self.guardar_imagen_descomprimida(p),
            bg='#3498db',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=15,
            pady=5,
            cursor='hand2',
            relief=tk.RAISED,
            bd=2
        ).pack(side=tk.LEFT, padx=5)
    
    def guardar_imagen_descomprimida(self, porcentaje):
        try:
            datos = self.resultados_porcentajes.get(porcentaje)
            if datos is None:
                messagebox.showwarning("Advertencia", f"No hay datos para el porcentaje {porcentaje}%")
                return
            
            from tkinter import filedialog
            ruta_guardar = filedialog.asksaveasfilename(
                title=f"Guardar imagen descomprimida ({porcentaje}%)",
                defaultextension=".png",
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg"),
                    ("BMP", "*.bmp"),
                    ("Todos los archivos", "*.*")
                ],
                initialfile=f"imagen_descomprimida_{porcentaje}pct.png"
            )
            
            if ruta_guardar:
                imagen_guardar = datos['imagen_reconstruida'].astype(np.uint8)
                cv2.imwrite(ruta_guardar, imagen_guardar)
                
                messagebox.showinfo(
                    "Éxito",
                    f"Imagen descomprimida guardada exitosamente:\n\n"
                    f"{ruta_guardar}\n\n"
                    f"Porcentaje: {porcentaje}%\n"
                    f"PSNR: {datos['metricas']['psnr']:.2f} dB\n"
                    f"Compresión: {datos['metricas']['tasa_compresion']:.2f}%"
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar imagen:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def mostrar_descompresion(self, porcentaje):
        try:
            datos = self.resultados_porcentajes.get(porcentaje)
            if datos is None:
                messagebox.showwarning("Advertencia", f"No hay datos para el porcentaje {porcentaje}%")
                return
            
            ventana_decomp = tk.Toplevel(self.ventana)
            ventana_decomp.title(f"Descompresión {porcentaje}% - Visualización Detallada")
            ventana_decomp.geometry("1600x900")
            ventana_decomp.configure(bg='#2c3e50')
            
            tk.Label(
                ventana_decomp,
                text=f"PROCESO DE DESCOMPRESIÓN ({porcentaje}% COEFICIENTES ELIMINADOS)",
                bg='#2c3e50',
                fg='white',
                font=('Segoe UI', 14, 'bold'),
                pady=12
            ).pack(fill=tk.X)
            
            frame_grafico = tk.Frame(ventana_decomp, bg='white')
            frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            from compresion_dct import aplicar_dct_bloques
            dct_reconstruida, _ = aplicar_dct_bloques(datos['imagen_reconstruida'])
            
            fig = Figure(figsize=(14, 9), dpi=100)
            
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
            ax1.set_title('Imagen Original', fontsize=11, fontweight='bold', pad=8)
            ax1.axis('on')
            ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(datos['imagen_reconstruida'], cmap='gray', interpolation='nearest')
            ax2.set_title(f'Imagen Descomprimida (Reconstruida)\n({porcentaje:.1f}% coef. eliminados)', 
                         fontsize=11, fontweight='bold', pad=8)
            ax2.axis('on')
            ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            ax3 = fig.add_subplot(2, 2, 3)
            dct_log_filtrado = np.log1p(np.abs(datos['coeficientes']))
            im3 = ax3.imshow(dct_log_filtrado, cmap='hot', interpolation='nearest', aspect='auto')
            ax3.set_title(f'Mapa DCT Filtrada (log)\n({porcentaje:.1f}% eliminados)', 
                         fontsize=10, fontweight='bold', pad=8)
            ax3.set_xlabel('Frecuencia horizontal', fontsize=8)
            ax3.set_ylabel('Frecuencia vertical', fontsize=8)
            ax3.grid(False)
            cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('log(1+|DCT|)', rotation=270, labelpad=10, fontsize=7)
            cbar3.ax.tick_params(labelsize=7)
            
            ax4 = fig.add_subplot(2, 2, 4)
            dct_log_reconstruida = np.log1p(np.abs(dct_reconstruida))
            im4 = ax4.imshow(dct_log_reconstruida, cmap='hot', interpolation='nearest', aspect='auto')
            ax4.set_title('Mapa DCT Reconstruida (log)\n(Después de aplicar IDCT)', 
                         fontsize=10, fontweight='bold', pad=8)
            ax4.set_xlabel('Frecuencia horizontal', fontsize=8)
            ax4.set_ylabel('Frecuencia vertical', fontsize=8)
            ax4.grid(False)
            cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.set_label('log(1+|DCT|)', rotation=270, labelpad=10, fontsize=7)
            cbar4.ax.tick_params(labelsize=7)
            
            fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.06, hspace=0.32, wspace=0.22)
            
            canvas = FigureCanvasTkAgg(fig, frame_grafico)
            canvas.draw()
            
            toolbar_frame = tk.Frame(frame_grafico)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            frame_info = tk.Frame(ventana_decomp, bg='#34495e', pady=10)
            frame_info.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            metricas = datos['metricas']
            info_texto = (
                f"📊 MÉTRICAS DE DESCOMPRESIÓN:   "
                f"PSNR: {metricas['psnr']:.2f} dB   |   "
                f"MSE: {metricas['mse']:.2f}   |   "
                f"Compresión: {metricas['tasa_compresion']:.2f}%   |   "
                f"Coeficientes eliminados: {datos['num_eliminados']:,}"
            )
            
            tk.Label(
                frame_info,
                text=info_texto,
                bg='#34495e',
                fg='white',
                font=('Consolas', 10)
            ).pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar descompresión:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def descomprimir(self):
        try:
            if self.imagen_comprimida is None:
                messagebox.showwarning("Advertencia", "Primero debe comprimir una imagen")
                return
            
            ventana_decomp = tk.Toplevel(self.ventana)
            ventana_decomp.title("Descompresión DCT - Comparación Detallada")
            ventana_decomp.geometry("1600x900")
            ventana_decomp.configure(bg='#2c3e50')
            
            tk.Label(
                ventana_decomp,
                text="DESCOMPRESIÓN MEDIANTE IDCT-2D",
                bg='#2c3e50',
                fg='white',
                font=('Segoe UI', 16, 'bold'),
                pady=12
            ).pack(fill=tk.X)
            
            frame_grafico = tk.Frame(ventana_decomp, bg='white')
            frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            from compresion_dct import aplicar_dct_bloques
            dct_comprimida, _ = aplicar_dct_bloques(self.imagen_comprimida)
            
            fig = Figure(figsize=(14, 9), dpi=100)
            
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(self.imagen_comprimida, cmap='gray', interpolation='nearest')
            porcentaje = (self.num_coefs_eliminados / self.coeficientes_dct.size) * 100
            ax1.set_title(f'Imagen Comprimida\n({porcentaje:.1f}% coef. eliminados)', 
                         fontsize=11, fontweight='bold', pad=8)
            ax1.axis('on')
            ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(self.imagen_comprimida, cmap='gray', interpolation='nearest')
            ax2.set_title('Imagen Descomprimida\n(Aplicando IDCT-2D)', 
                         fontsize=11, fontweight='bold', pad=8)
            ax2.axis('on')
            ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            ax3 = fig.add_subplot(2, 2, 3)
            dct_log_filtrado = np.log1p(np.abs(self.coeficientes_dct))
            im3 = ax3.imshow(dct_log_filtrado, cmap='inferno', interpolation='nearest', aspect='auto')
            ax3.set_title(f'Mapa DCT Comprimida (log)\n({porcentaje:.1f}% eliminados)', 
                         fontsize=10, fontweight='bold', pad=8)
            ax3.set_xlabel('Frecuencia horizontal', fontsize=8)
            ax3.set_ylabel('Frecuencia vertical', fontsize=8)
            ax3.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
            ax3.tick_params(labelsize=7)
            cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('log(1+|DCT|)', rotation=270, labelpad=10, fontsize=7)
            cbar3.ax.tick_params(labelsize=7)
            
            ax4 = fig.add_subplot(2, 2, 4)
            dct_log_descomprimida = np.log1p(np.abs(dct_comprimida))
            im4 = ax4.imshow(dct_log_descomprimida, cmap='inferno', interpolation='nearest', aspect='auto')
            ax4.set_title('Mapa DCT Descomprimida (log)\n(Después de IDCT-2D)', 
                         fontsize=10, fontweight='bold', pad=8)
            ax4.set_xlabel('Frecuencia horizontal', fontsize=8)
            ax4.set_ylabel('Frecuencia vertical', fontsize=8)
            ax4.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
            ax4.tick_params(labelsize=7)
            cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.set_label('log(1+|DCT|)', rotation=270, labelpad=10, fontsize=7)
            cbar4.ax.tick_params(labelsize=7)
            
            fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.06, hspace=0.32, wspace=0.22)
            
            canvas = FigureCanvasTkAgg(fig, frame_grafico)
            canvas.draw()
            
            toolbar_frame = tk.Frame(frame_grafico)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            frame_info = tk.Frame(ventana_decomp, bg='#34495e', pady=10)
            frame_info.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            info_texto = (
                f"📊 MÉTRICAS DE DESCOMPRESIÓN:   "
                f"PSNR: {self.metricas['psnr']:.2f} dB   |   "
                f"MSE: {self.metricas['mse']:.2f}   |   "
                f"Compresión: {self.metricas['tasa_compresion']:.2f}%   |   "
                f"Coeficientes retenidos: {self.coeficientes_dct.size - self.num_coefs_eliminados:,} / {self.coeficientes_dct.size:,}"
            )
            
            tk.Label(
                frame_info,
                text=info_texto,
                bg='#34495e',
                fg='white',
                font=('Consolas', 10)
            ).pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al descomprimir:\n{str(e)}")
            import traceback
            traceback.print_exc()
