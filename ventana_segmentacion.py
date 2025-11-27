"""
Ventana de segmentación de imágenes usando K-means clustering.
Implementa segmentación por color utilizando machine learning.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class VentanaSegmentacionKMeans:
    def __init__(self, parent, ruta_imagen):
        """
        Inicializa la ventana de segmentación K-means.
        
        Parámetros:
        -----------
        parent : tk.Tk o tk.Toplevel
            Ventana padre
        ruta_imagen : Path o str
            Ruta de la imagen a segmentar
        """
        self.ventana = tk.Toplevel(parent)
        self.ventana.title("Segmentación de Imagen con K-means Clustering")
        self.ventana.geometry("1600x900")
        self.ventana.configure(bg='#2c3e50')
        
        # Cargar imagen con soporte Unicode
        self.imagen_original = self.cargar_imagen_opencv_unicode(str(ruta_imagen))
        
        if self.imagen_original is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{ruta_imagen}")
            self.ventana.destroy()
            return
        
        # Variables de estado
        self.imagen_segmentada = None
        self.etiquetas_kmeans = None
        self.centros_kmeans = None
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
            text="SEGMENTACIÓN CON K-MEANS CLUSTERING",
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
            text="Número de clusters (K):",
            bg='#34495e',
            fg='white',
            font=('Segoe UI', 10, 'bold')
        ).pack(side=tk.LEFT, padx=(20, 10))
        
        self.entry_k = tk.Entry(
            frame_controles,
            font=('Segoe UI', 10),
            width=8,
            justify=tk.CENTER
        )
        self.entry_k.insert(0, "4")
        self.entry_k.pack(side=tk.LEFT, padx=(0, 15))
        
        self.btn_segmentar = tk.Button(
            frame_controles,
            text="🎯 Segmentar Imagen",
            font=('Segoe UI', 10, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            padx=20,
            pady=8,
            relief=tk.RAISED,
            command=self.segmentar
        )
        self.btn_segmentar.pack(side=tk.LEFT, padx=10)
        
        # Frame principal que contiene gráfico y panel lateral
        frame_principal = tk.Frame(self.ventana, bg='#2c3e50')
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para el gráfico de matplotlib
        self.frame_grafico = tk.Frame(frame_principal, bg='white')
        self.frame_grafico.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Panel lateral derecho para información
        frame_info = tk.Frame(frame_principal, bg='#34495e', width=280)
        frame_info.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        frame_info.pack_propagate(False)
        
        # Título del panel de info
        tk.Label(
            frame_info,
            text="📊 INFORMACIÓN",
            bg='#34495e',
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            pady=10
        ).pack(fill=tk.X)
        
        # Área de texto para información
        self.texto_info = tk.Text(
            frame_info,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#ecf0f1',
            fg='#2c3e50',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.texto_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Mensaje inicial
        self.texto_info.insert(1.0,
            "ℹ️ K-means Clustering\n"
            f"{'='*30}\n\n"
            "Algoritmo de Machine Learning\n"
            "para segmentación de imágenes\n"
            "por similaridad de color.\n\n"
            "PASOS:\n\n"
            "1. Cada píxel se representa\n"
            "   como un vector [R, G, B]\n\n"
            "2. K-means agrupa píxeles\n"
            "   similares en K clusters\n\n"
            "3. Cada cluster representa\n"
            "   una región de la imagen\n\n"
            "RECOMENDACIONES:\n\n"
            "• K=2-3: Segmentación simple\n"
            "• K=4-6: Segmentación media\n"
            "• K=7-10: Segmentación fina\n\n"
            "Valores muy altos de K\n"
            "pueden sobre-segmentar\n"
            "la imagen.\n\n"
            "Ajuste K y presione\n"
            "'Segmentar Imagen'."
        )
        self.texto_info.config(state=tk.DISABLED)
        
        # Mostrar vista inicial
        self.mostrar_vista_inicial()
    
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
        ax.imshow(self.imagen_original)
        ax.set_title("Imagen Original - Lista para Segmentar", fontsize=14, fontweight='bold', pad=10)
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
    
    def segmentar(self):
        """Ejecuta la segmentación K-means."""
        try:
            # Obtener K
            k = int(self.entry_k.get())
            
            if k < 2 or k > 20:
                messagebox.showerror("Error", "K debe estar entre 2 y 20")
                return
            
            # Deshabilitar botón durante procesamiento
            self.btn_segmentar.config(state=tk.DISABLED)
            
            # Actualizar info con mensaje de progreso
            self.texto_info.config(state=tk.NORMAL)
            self.texto_info.delete(1.0, tk.END)
            self.texto_info.insert(1.0,
                "⏳ PROCESANDO...\n"
                f"{'='*30}\n\n"
                f"Aplicando K-means con\n"
                f"K = {k} clusters...\n\n"
                "Este proceso puede\n"
                "tardar unos segundos.\n\n"
                "Por favor espere..."
            )
            self.texto_info.config(state=tk.DISABLED)
            self.ventana.update()
            
            # Ejecutar segmentación
            print(f"Iniciando segmentación K-means con K={k}...")
            self.aplicar_kmeans(k)
            
            # Mostrar resultados
            print("Generando visualización...")
            self.mostrar_resultados_segmentacion(k)
            
            # Habilitar botón
            self.btn_segmentar.config(state=tk.NORMAL)
            
            print("✓ Segmentación completada exitosamente")
            
        except ValueError:
            self.btn_segmentar.config(state=tk.NORMAL)
            messagebox.showerror("Error", "Ingrese un valor numérico válido para K")
        except Exception as e:
            self.btn_segmentar.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Error durante la segmentación:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def aplicar_kmeans(self, k):
        """
        Aplica el algoritmo K-means clustering a la imagen.
        
        Algoritmo K-means:
        1. Inicializar K centroides aleatoriamente
        2. Asignar cada píxel al centroide más cercano
        3. Recalcular centroides como el promedio de píxeles asignados
        4. Repetir pasos 2-3 hasta convergencia
        
        Parámetros:
        -----------
        k : int
            Número de clusters
        """
        # Reshape imagen a array 2D: cada fila es un píxel [R, G, B]
        h, w, c = self.imagen_original.shape
        pixels = self.imagen_original.reshape((-1, 3))
        
        # Convertir a float32 para K-means
        pixels = np.float32(pixels)
        
        print(f"  Dimensiones imagen: {h}x{w} = {h*w} píxeles")
        print(f"  Aplicando K-means con K={k}...")
        
        # Criterios de parada: 100 iteraciones O epsilon=1.0
        criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        
        # Aplicar K-means
        # compactness: suma de distancias cuadradas de cada punto a su centro
        # labels: etiqueta de cluster para cada píxel (0 a K-1)
        # centers: coordenadas RGB de los K centroides
        compactness, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criterios,
            attempts=10,
            flags=cv2.KMEANS_PP_CENTERS  # Inicialización inteligente K-means++
        )
        
        # Convertir centros a uint8
        centers = np.uint8(centers)
        
        # Asignar a cada píxel el color de su centroide
        imagen_segmentada = centers[labels.flatten()]
        
        # Reshape de vuelta a imagen
        self.imagen_segmentada = imagen_segmentada.reshape((h, w, 3))
        self.etiquetas_kmeans = labels.reshape((h, w))
        self.centros_kmeans = centers
        
        print(f"  ✓ K-means completado")
        print(f"  Compacidad: {compactness:.2f}")
        print(f"  Centroides (colores RGB):")
        for i, centro in enumerate(centers):
            print(f"    Cluster {i}: RGB{tuple(centro)}")
    
    def mostrar_resultados_segmentacion(self, k):
        """Muestra visualización de resultados de segmentación."""
        # Limpiar canvas anterior
        if self.canvas_actual:
            self.canvas_actual.get_tk_widget().destroy()
        if self.toolbar_actual:
            self.toolbar_actual.destroy()
        
        # Crear figura con 4 paneles
        fig = Figure(figsize=(12, 9), dpi=100)
        
        # Panel 1: Imagen original
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(self.imagen_original)
        ax1.set_title('Imagen Original', fontsize=11, fontweight='bold', pad=8)
        ax1.axis('on')
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.tick_params(labelsize=7)
        
        # Panel 2: Imagen segmentada
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(self.imagen_segmentada)
        ax2.set_title(f'Imagen Segmentada (K={k} clusters)', fontsize=11, fontweight='bold', pad=8)
        ax2.axis('on')
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.tick_params(labelsize=7)
        
        # Panel 3: Mapa de etiquetas (clusters)
        ax3 = fig.add_subplot(2, 2, 3)
        im3 = ax3.imshow(self.etiquetas_kmeans, cmap='tab10', interpolation='nearest')
        ax3.set_title(f'Mapa de Clusters (0 a {k-1})', fontsize=10, fontweight='bold', pad=8)
        ax3.set_xlabel('X (píxeles)', fontsize=8)
        ax3.set_ylabel('Y (píxeles)', fontsize=8)
        ax3.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        ax3.tick_params(labelsize=7)
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('ID Cluster', rotation=270, labelpad=10, fontsize=7)
        cbar3.ax.tick_params(labelsize=7)
        
        # Panel 4: Paleta de colores (centroides)
        ax4 = fig.add_subplot(2, 2, 4)
        paleta = np.zeros((100, k * 50, 3), dtype=np.uint8)
        for i in range(k):
            paleta[:, i*50:(i+1)*50] = self.centros_kmeans[i]
        ax4.imshow(paleta)
        ax4.set_title('Paleta de Colores (Centroides K-means)', fontsize=10, fontweight='bold', pad=8)
        ax4.set_xlabel('Clusters', fontsize=8)
        ax4.set_yticks([])
        ax4.set_xticks([i*50 + 25 for i in range(k)])
        ax4.set_xticklabels([f'C{i}' for i in range(k)], fontsize=7)
        ax4.grid(False)
        
        # Ajustar espaciado
        fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06, hspace=0.30, wspace=0.18)
        
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
        self.texto_info.config(state=tk.NORMAL)
        self.texto_info.delete(1.0, tk.END)
        
        # Calcular estadísticas de clusters
        h, w = self.etiquetas_kmeans.shape
        total_pixeles = h * w
        
        info_texto = (
            f"✅ SEGMENTACIÓN EXITOSA\n"
            f"{'='*30}\n\n"
            f"📊 PARÁMETROS:\n\n"
            f"Algoritmo: K-means\n"
            f"Clusters (K): {k}\n"
            f"Imagen: {w}x{h} px\n"
            f"Total píxeles: {total_pixeles:,}\n\n"
            f"{'='*30}\n\n"
            f"🎨 DISTRIBUCIÓN:\n\n"
        )
        
        for i in range(k):
            count = np.sum(self.etiquetas_kmeans == i)
            porcentaje = (count / total_pixeles) * 100
            rgb = self.centros_kmeans[i]
            info_texto += (
                f"Cluster {i}:\n"
                f"  {count:,} px ({porcentaje:.1f}%)\n"
                f"  RGB{tuple(rgb)}\n\n"
            )
        
        info_texto += (
            f"{'='*30}\n\n"
            f"📐 ALGORITMO K-MEANS:\n\n"
            f"1. Inicialización: K\n"
            f"   centroides aleatorios\n\n"
            f"2. Asignación: Cada píxel\n"
            f"   al centroide + cercano\n\n"
            f"3. Actualización: Centros\n"
            f"   = promedio de píxeles\n\n"
            f"4. Convergencia: Repetir\n"
            f"   hasta estabilidad\n\n"
            f"Distancia: Euclidiana\n"
            f"en espacio RGB\n\n"
            f"d = √[(R₁-R₂)² +\n"
            f"     (G₁-G₂)² +\n"
            f"     (B₁-B₂)²]"
        )
        
        self.texto_info.insert(1.0, info_texto)
        self.texto_info.config(state=tk.DISABLED)
