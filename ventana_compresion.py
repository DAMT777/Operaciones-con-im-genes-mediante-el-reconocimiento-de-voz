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
            text="Porcentajes (mínimo 3):",
            bg='#2c3e50',
            fg='white',
            font=('Segoe UI', 11)
        ).pack(side=tk.LEFT, padx=8)
        
        # Entry para porcentajes
        self.entry_porcentajes = tk.Entry(
            frame_botones,
            width=20,
            font=('Segoe UI', 11),
            justify='center'
        )
        self.entry_porcentajes.insert(0, "50, 70, 90")
        self.entry_porcentajes.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            frame_botones,
            text="(Ej: 50, 70, 90)",
            bg='#2c3e50',
            fg='#bdc3c7',
            font=('Segoe UI', 9)
        ).pack(side=tk.LEFT, padx=8)
        
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
            text="📂 Descomprimir y Guardar",
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
        fig = Figure(figsize=(12, 9), dpi=100)
        
        # Imagen original centrada
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title("Imagen Original - Lista para Comprimir", fontsize=14, fontweight='bold', pad=10)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3)
        
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.05)
        
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
            # Obtener y parsear porcentajes
            porcentajes_texto = self.entry_porcentajes.get()
            porcentajes = [float(p.strip()) for p in porcentajes_texto.replace(';', ',').split(',') if p.strip()]
            
            if len(porcentajes) < 3:
                messagebox.showerror("Error", "Debe ingresar al menos 3 porcentajes")
                return
            
            for p in porcentajes:
                if p < 0 or p > 100:
                    messagebox.showerror("Error", "Todos los porcentajes deben estar entre 0 y 100")
                    return
            
            # Deshabilitar botones durante procesamiento
            self.btn_comprimir.config(state=tk.DISABLED)
            self.btn_descomprimir.config(state=tk.DISABLED)
            
            # Actualizar métricas con mensaje de progreso
            self.texto_metricas.config(state=tk.NORMAL)
            self.texto_metricas.delete(1.0, tk.END)
            self.texto_metricas.insert(1.0, 
                "⏳ PROCESANDO...\n"
                f"{'='*30}\n\n"
                "Aplicando DCT-2D manual\n"
                "por bloques de 8x8...\n\n"
                "Este proceso puede\n"
                "tardar entre 10-60\n"
                "segundos dependiendo\n"
                "del tamaño de la imagen.\n\n"
                "Por favor espere..."
            )
            self.texto_metricas.config(state=tk.DISABLED)
            self.ventana.update()
            
            # Comprimir (también guarda DCT completa)
            from compresion_dct import aplicar_dct_bloques
            print("Aplicando DCT-2D completa...")
            self.dct_completa, _ = aplicar_dct_bloques(self.imagen_original)
            
            # Usar el primer porcentaje por ahora (TODO: implementar múltiples)
            porcentaje = porcentajes[0]
            
            print(f"Comprimiendo al {porcentaje}%...")
            self.coeficientes_dct, self.forma_original, self.num_coefs_eliminados = \
                comprimir_imagen_dct(self.imagen_original, porcentaje, tamanio_bloque=8)
            
            # Descomprimir para visualizar
            print("Reconstruyendo imagen...")
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
            
            print("Generando visualización...")
            # Mostrar visualización
            self.mostrar_resultados_compresion(porcentaje)
            
            # Habilitar botón descomprimir
            self.btn_comprimir.config(state=tk.NORMAL)
            self.btn_descomprimir.config(state=tk.NORMAL)
            
            print("✓ Compresión completada exitosamente")
            
        except ValueError:
            self.btn_comprimir.config(state=tk.NORMAL)
            messagebox.showerror("Error", "Ingrese un valor numérico válido para el porcentaje")
        except Exception as e:
            self.btn_comprimir.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Error durante la compresión:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def descomprimir(self):
        """Muestra ventana comparando imagen comprimida vs descomprimida con sus mapas DCT."""
        try:
            if self.imagen_comprimida is None:
                messagebox.showwarning("Advertencia", "Primero debe comprimir una imagen")
                return
            
            # Crear ventana de comparación
            ventana_decomp = tk.Toplevel(self.ventana)
            ventana_decomp.title("Descompresión DCT - Comparación Detallada")
            ventana_decomp.geometry("1600x900")
            ventana_decomp.configure(bg='#2c3e50')
            
            # Título
            tk.Label(
                ventana_decomp,
                text="DESCOMPRESIÓN MEDIANTE IDCT-2D",
                bg='#2c3e50',
                fg='white',
                font=('Segoe UI', 16, 'bold'),
                pady=12
            ).pack(fill=tk.X)
            
            # Frame para el gráfico
            frame_grafico = tk.Frame(ventana_decomp, bg='white')
            frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Calcular DCT de la imagen comprimida para visualización
            from compresion_dct import aplicar_dct_bloques
            dct_comprimida, _ = aplicar_dct_bloques(self.imagen_comprimida)
            
            # Crear figura con 4 paneles (2x2)
            fig = Figure(figsize=(14, 9), dpi=100)
            
            # Panel 1: Imagen comprimida
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(self.imagen_comprimida, cmap='gray', interpolation='nearest')
            porcentaje = (self.num_coefs_eliminados / self.coeficientes_dct.size) * 100
            ax1.set_title(f'Imagen Comprimida\n({porcentaje:.1f}% coef. eliminados)', 
                         fontsize=11, fontweight='bold', pad=8)
            ax1.axis('on')
            ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Panel 2: Imagen descomprimida (misma, ya que IDCT se aplicó)
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(self.imagen_comprimida, cmap='gray', interpolation='nearest')
            ax2.set_title('Imagen Descomprimida\n(Aplicando IDCT-2D)', 
                         fontsize=11, fontweight='bold', pad=8)
            ax2.axis('on')
            ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Panel 3: Mapa DCT de imagen comprimida (filtrada)
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
            
            # Panel 4: Mapa DCT de imagen descomprimida
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
            
            # Ajustar espaciado
            fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.06, hspace=0.32, wspace=0.22)
            
            # Agregar canvas
            canvas = FigureCanvasTkAgg(fig, frame_grafico)
            canvas.draw()
            
            # Toolbar
            toolbar_frame = tk.Frame(frame_grafico)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Frame inferior con métricas
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
    
    def mostrar_resultados_compresion(self, porcentaje):
        """Muestra visualización de 4 paneles con resultados de compresión."""
        # Limpiar canvas anterior si existe
        if self.canvas_actual:
            self.canvas_actual.get_tk_widget().destroy()
        if self.toolbar_actual:
            self.toolbar_actual.destroy()
        
        # Crear figura con 4 paneles 2x2 - optimizada para ocupar espacio
        fig = Figure(figsize=(12, 9), dpi=100)
        
        # FILA SUPERIOR: Imágenes
        # Panel 1: Imagen original
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(self.imagen_original, cmap='gray', interpolation='nearest')
        ax1.set_title('Imagen Original', fontsize=10, fontweight='bold', pad=3)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.tick_params(labelsize=7)
        
        # Panel 2: Imagen reconstruida/comprimida
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(self.imagen_comprimida, cmap='gray', interpolation='nearest')
        ax2.set_title(f'Comprimida ({porcentaje:.1f}% coef. eliminados)', 
                     fontsize=10, fontweight='bold', pad=3)
        ax2.axis('on')
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.tick_params(labelsize=7)
        
        # FILA INFERIOR: Mapas DCT
        # Panel 3: Mapa DCT completo (sin filtrar)
        ax3 = fig.add_subplot(2, 2, 3)
        dct_log_completo = np.log1p(np.abs(self.dct_completa))
        im3 = ax3.imshow(dct_log_completo, cmap='inferno', interpolation='nearest', aspect='auto')
        ax3.set_title('Mapa DCT Completo', 
                     fontsize=9, fontweight='bold', pad=3)
        ax3.set_xlabel('Frecuencia horizontal', fontsize=7)
        ax3.set_ylabel('Frecuencia vertical', fontsize=7)
        ax3.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        ax3.tick_params(labelsize=6)
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('log(1+|DCT|)', rotation=270, labelpad=8, fontsize=6)
        cbar3.ax.tick_params(labelsize=6)
        
        # Panel 4: Mapa DCT filtrado
        ax4 = fig.add_subplot(2, 2, 4)
        dct_log_filtrado = np.log1p(np.abs(self.coeficientes_dct))
        im4 = ax4.imshow(dct_log_filtrado, cmap='inferno', interpolation='nearest', aspect='auto')
        ax4.set_title(f'Mapa DCT Filtrado ({porcentaje:.1f}%)', 
                     fontsize=9, fontweight='bold', pad=3)
        ax4.set_xlabel('Frecuencia horizontal', fontsize=7)
        ax4.set_ylabel('Frecuencia vertical', fontsize=7)
        ax4.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        ax4.tick_params(labelsize=6)
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('log(1+|DCT|)', rotation=270, labelpad=8, fontsize=6)
        cbar4.ax.tick_params(labelsize=6)
        
        # Ajustar con subplots_adjust para control preciso - márgenes mínimos
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.04, hspace=0.25, wspace=0.18)
        
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
