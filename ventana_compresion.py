"""
Ventana de compresión de imágenes usando DCT-2D manual.
"""

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
    """Carga una imagen desde una ruta con caracteres Unicode (tildes, ñ, etc.)."""
    try:
        # Leer archivo como bytes
        with open(ruta, 'rb') as f:
            datos = f.read()
        
        # Convertir a numpy array
        arr = np.frombuffer(datos, dtype=np.uint8)
        
        # Decodificar con OpenCV
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return None


class VentanaCompresionDCT:
    def __init__(self, parent, ruta_imagen):
        self.parent = parent
        self.ruta_imagen = ruta_imagen
        
        # Cargar imagen original usando función que soporta Unicode
        self.imagen_original = cargar_imagen_unicode(str(ruta_imagen))
        
        if self.imagen_original is None:
            messagebox.showerror(
                "Error", 
                f"No se pudo cargar la imagen.\n\n"
                f"Ruta: {ruta_imagen}\n\n"
                f"Sugerencia: Evite rutas con tildes, ñ u otros caracteres especiales."
            )
            return
        
        # Variables de estado
        self.coeficientes_dct = None
        self.dct_completa = None
        self.forma_original = None
        self.imagen_comprimida = None
        self.num_coefs_eliminados = 0
        self.metricas = None
        self.canvas_actual = None
        self.toolbar_actual = None
        
        # Crear ventana
        self.ventana = tk.Toplevel(parent)
        self.ventana.title("Compresión de Imagen usando DCT-2D Manual")
        self.ventana.geometry("1600x900")
        self.ventana.configure(bg='#f0f0f0')
        
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Frame superior - Controles
        frame_controles = tk.Frame(self.ventana, bg='#2c3e50', pady=12)
        frame_controles.pack(fill=tk.X, padx=0)
        
        # Título
        tk.Label(
            frame_controles,
            text="COMPRESIÓN CON DCT-2D MANUAL",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 16, 'bold')
        ).pack(pady=5)
        
        # Frame de controles
        frame_botones = tk.Frame(frame_controles, bg='#2c3e50')
        frame_botones.pack(pady=8)
        
        # Label de porcentaje
        tk.Label(
            frame_botones,
            text="Porcentaje de compresión:",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT, padx=8)
        
        # Entry para porcentaje
        self.entry_porcentaje = tk.Entry(
            frame_botones,
            width=10,
            font=('Segoe UI', 11),
            justify='center'
        )
        self.entry_porcentaje.insert(0, "60")
        self.entry_porcentaje.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            frame_botones,
            text="%",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT, padx=2)
        
        # Botón comprimir
        self.btn_comprimir = tk.Button(
            frame_botones,
            text="🗜️ Comprimir",
            command=self.comprimir,
            bg='#27ae60',
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            padx=25,
            pady=8,
            cursor='hand2',
            relief=tk.RAISED,
            bd=2
        )
        self.btn_comprimir.pack(side=tk.LEFT, padx=10)
        
        # Botón descomprimir
        self.btn_descomprimir = tk.Button(
            frame_botones,
            text="📂 Descomprimir",
            command=self.descomprimir,
            bg='#3498db',
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            padx=25,
            pady=8,
            cursor='hand2',
            relief=tk.RAISED,
            bd=2,
            state=tk.DISABLED
        )
        self.btn_descomprimir.pack(side=tk.LEFT, padx=10)
        
        # Frame contenedor principal con panel lateral para métricas
        frame_principal = tk.Frame(self.ventana, bg='#f0f0f0')
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para la figura de matplotlib (izquierda)
        self.frame_grafico = tk.Frame(frame_principal, bg='white')
        self.frame_grafico.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Frame para métricas (derecha)
        frame_metricas = tk.Frame(frame_principal, bg='#ecf0f1', width=280, relief=tk.RIDGE, bd=2)
        frame_metricas.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        frame_metricas.pack_propagate(False)
        
        # Título de métricas
        tk.Label(
            frame_metricas,
            text="📊 MÉTRICAS",
            bg='#34495e',
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            pady=10
        ).pack(fill=tk.X)
        
        # Área de texto para métricas
        self.texto_metricas = tk.Text(
            frame_metricas,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#ecf0f1',
            fg='#2c3e50',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.texto_metricas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Mensaje inicial
        self.texto_metricas.insert(1.0, 
            "ℹ️ Información:\n\n"
            "Ajuste el porcentaje de\n"
            "compresión (0-100%) y\n"
            "presione 'Comprimir'.\n\n"
            "Un porcentaje mayor\n"
            "elimina más coeficientes\n"
            "DCT, reduciendo el tamaño\n"
            "pero afectando la calidad.\n\n"
            "Valores recomendados:\n"
            "  • 40-60%: Alta calidad\n"
            "  • 60-80%: Calidad media\n"
            "  • 80-95%: Baja calidad"
        )
        self.texto_metricas.config(state=tk.DISABLED)
        
        # Mostrar vista inicial
        self.mostrar_vista_inicial()
        
    def mostrar_vista_inicial(self):
        """Muestra la imagen original en un panel de matplotlib."""
        # Limpiar canvas anterior si existe
        if self.canvas_actual:
            self.canvas_actual.get_tk_widget().destroy()
        if self.toolbar_actual:
            self.toolbar_actual.destroy()
        
        # Crear figura con tamaño adecuado
        fig = Figure(figsize=(14, 10), dpi=100)
        
        # Imagen original centrada
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title("Imagen Original - Lista para Comprimir", fontsize=14, fontweight='bold')
        ax1.axis('on')
        ax1.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Crear canvas
        self.canvas_actual = FigureCanvasTkAgg(fig, self.frame_grafico)
        self.canvas_actual.draw()
        
        # Agregar toolbar PRIMERO (en la parte superior)
        toolbar_frame = tk.Frame(self.frame_grafico)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar_actual = NavigationToolbar2Tk(self.canvas_actual, toolbar_frame)
        self.toolbar_actual.update()
        
        # Luego el canvas (debajo del toolbar)
        self.canvas_actual.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def comprimir(self):
        """Ejecuta la compresión usando DCT-2D manual y muestra resultados."""
        try:
            # Obtener porcentaje
            porcentaje = float(self.entry_porcentaje.get())
            
            if porcentaje < 0 or porcentaje > 100:
                messagebox.showerror("Error", "El porcentaje debe estar entre 0 y 100")
                return
            
            # Mostrar mensaje de progreso
            messagebox.showinfo(
                "Procesando",
                "Comprimiendo imagen usando DCT-2D manual...\n"
                "Este proceso puede tardar varios segundos.\n\n"
                "Por favor espere..."
            )
            self.ventana.update()
            
            # Comprimir (también guarda DCT completa)
            from compresion_dct import aplicar_dct_bloques
            self.dct_completa, _ = aplicar_dct_bloques(self.imagen_original)
            
            self.coeficientes_dct, self.forma_original, self.num_coefs_eliminados = \
                comprimir_imagen_dct(self.imagen_original, porcentaje, tamanio_bloque=8)
            
            # Descomprimir para visualizar
            self.imagen_comprimida = descomprimir_imagen_dct(
                self.coeficientes_dct,
                self.forma_original,
                tamanio_bloque=8
            )
            
            # Calcular métricas
            total_coefs = self.coeficientes_dct.size
            self.metricas = calcular_metricas_compresion(
                self.imagen_original,
                self.imagen_comprimida,
                self.num_coefs_eliminados,
                total_coefs
            )
            
            # Mostrar visualización
            self.mostrar_resultados_compresion(porcentaje)
            
            # Habilitar botón descomprimir
            self.btn_descomprimir.config(state=tk.NORMAL)
            
            messagebox.showinfo(
                "Éxito",
                f"¡Imagen comprimida exitosamente!\n\n"
                f"Compresión: {self.metricas['tasa_compresion']:.2f}%\n"
                f"PSNR: {self.metricas['psnr']:.2f} dB\n"
                f"MSE: {self.metricas['mse']:.2f}"
            )
            
        except ValueError:
            messagebox.showerror("Error", "Ingrese un valor numérico válido para el porcentaje")
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la compresión:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def descomprimir(self):
        """Descomprime la imagen y la guarda."""
        try:
            if self.imagen_comprimida is None:
                messagebox.showwarning("Advertencia", "Primero debe comprimir una imagen")
                return
            
            # Pedir nombre de archivo para guardar
            archivo_salida = filedialog.asksaveasfilename(
                title="Guardar imagen descomprimida",
                defaultextension=".png",
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg"),
                    ("Todos los archivos", "*.*")
                ]
            )
            
            if archivo_salida:
                # Guardar imagen descomprimida
                cv2.imwrite(archivo_salida, self.imagen_comprimida)
                
                messagebox.showinfo(
                    "Éxito",
                    f"Imagen descomprimida guardada en:\n{archivo_salida}\n\n"
                    f"Calidad PSNR: {self.metricas['psnr']:.2f} dB"
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar imagen:\n{str(e)}")
    
    def mostrar_resultados_compresion(self, porcentaje):
        """Muestra visualización de 4 paneles con resultados de compresión."""
        # Limpiar canvas anterior si existe
        if self.canvas_actual:
            self.canvas_actual.get_tk_widget().destroy()
        if self.toolbar_actual:
            self.toolbar_actual.destroy()
        
        # Crear nueva figura con 4 subplots - tamaño igual a lab6
        fig = Figure(figsize=(14, 10), dpi=100)
        
        # Panel 1: Imagen original
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title('Imagen original', fontsize=12, fontweight='bold')
        ax1.axis('on')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Imagen reconstruida
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(self.imagen_comprimida, cmap='gray', interpolation='nearest')
        ax2.set_title(f'Reconstruida ({porcentaje:.1f}% coef. eliminados)', 
                     fontsize=12, fontweight='bold')
        ax2.axis('on')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Mapa DCT (escala logarítmica)
        ax3 = fig.add_subplot(2, 2, 3)
        dct_log = np.log1p(np.abs(self.dct_completa))
        im3 = ax3.imshow(dct_log, cmap='inferno', interpolation='nearest')
        ax3.set_title('Mapa DCT filtrada (log)', fontsize=12, fontweight='bold')
        ax3.axis('on')
        ax3.grid(True, alpha=0.3)
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Panel 4: Mapa de diferencia
        ax4 = fig.add_subplot(2, 2, 4)
        diferencia = np.abs(self.imagen_original.astype(float) - self.imagen_comprimida.astype(float))
        im4 = ax4.imshow(diferencia, cmap='hot', interpolation='nearest')
        ax4.set_title('Diferencia absoluta |Original - Reconstruida|', fontsize=12, fontweight='bold')
        ax4.axis('on')
        ax4.grid(True, alpha=0.3)
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        fig.tight_layout()
        
        # Crear canvas de matplotlib
        self.canvas_actual = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        self.canvas_actual.draw()
        
        # Agregar toolbar de navegación PRIMERO (en la parte superior)
        toolbar_frame = tk.Frame(self.frame_grafico)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar_actual = NavigationToolbar2Tk(self.canvas_actual, toolbar_frame)
        self.toolbar_actual.update()
        
        # Luego el canvas (debajo del toolbar)
        self.canvas_actual.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Actualizar texto de métricas
        self.texto_metricas.config(state=tk.NORMAL)
        self.texto_metricas.delete(1.0, tk.END)
        metricas_texto = (
            f"✅ COMPRESIÓN EXITOSA\n"
            f"{'='*30}\n\n"
            f"📊 MÉTRICAS DE CALIDAD:\n\n"
            f"🗜️ Compresión:\n"
            f"   {self.metricas['tasa_compresion']:.2f}%\n\n"
            f"📉 PSNR (calidad):\n"
            f"   {self.metricas['psnr']:.2f} dB\n\n"
            f"📈 MSE (error):\n"
            f"   {self.metricas['mse']:.2f}\n\n"
            f"🔢 Coeficientes:\n"
            f"   Eliminados:\n"
            f"   {self.num_coefs_eliminados:,}\n\n"
            f"{'='*30}\n\n"
            f"🔍 HERRAMIENTAS:\n\n"
            f"Use la barra de\n"
            f"herramientas para:\n\n"
            f"  • Hacer zoom\n"
            f"  • Desplazarse\n"
            f"  • Guardar figura\n\n"
            f"Presione 'Descomprimir'\n"
            f"para guardar el\n"
            f"resultado."
        )
        self.texto_metricas.insert(1.0, metricas_texto)
        self.texto_metricas.config(state=tk.DISABLED)
