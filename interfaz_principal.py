import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import queue
import numpy as np

import ttkbootstrap as tb
from ttkbootstrap.constants import *

from configuracion import ARCHIVO_UMBRALES, ETIQUETAS_COMANDOS, N_FFT, FRECUENCIA_MUESTREO_OBJETIVO
from entrenamiento_comandos import entrenar_modelo_comandos
from captura_microfono import grabar_audio_microfono
from reconocimiento_comandos import (
    cargar_umbrales_desde_archivo,
    procesar_senal_para_reconocimiento,
    reconocer_comando_por_energia,
    ejecutar_operacion_imagen,
)


class AplicacionReconocimiento(tb.Window):
    def __init__(self):
        super().__init__(themename="flatly")
        self.title("Reconocimiento de Comandos por Bancos de Filtros")
        self.geometry("720x400")

        self.ruta_imagen = None
        self.umbrales = None
        self.microfono_activo = False
        self.hilo_microfono = None

        self.crear_componentes_interfaz()
        
        # Auto-cargar entrenamiento al iniciar
        self.after(500, self.auto_cargar_entrenamiento)
        
        # Auto-iniciar micrófono
        self.after(1000, self.activar_microfono_continuo)

    # ------------------------------------------------------------------
    def crear_componentes_interfaz(self):
        marco_principal = tb.Frame(self, padding=20)
        marco_principal.pack(fill=BOTH, expand=YES)

        titulo = tb.Label(
            marco_principal,
            text="Reconocimiento de voz con bancos de filtros (3 comandos)",
            bootstyle="inverse-primary",
            font=("Segoe UI", 14, "bold"),
        )
        titulo.pack(fill=X, pady=(0, 15))

        # Estado del micrófono
        self.label_microfono = tb.Label(
            marco_principal,
            text="🎤 Micrófono: Inicializando...",
            bootstyle="warning",
            font=("Segoe UI", 12, "bold"),
        )
        self.label_microfono.pack(fill=X, pady=5)

        # Botones principales
        marco_botones = tb.Frame(marco_principal)
        marco_botones.pack(fill=X, pady=10)

        btn_seleccionar_imagen = tb.Button(
            marco_botones,
            text="📁 Seleccionar imagen para operaciones",
            bootstyle="info",
            command=self.seleccionar_imagen,
        )
        btn_seleccionar_imagen.pack(fill=X, pady=5)
        
        self.btn_toggle_mic = tb.Button(
            marco_botones,
            text="⏸️ Pausar Micrófono",
            bootstyle="danger",
            command=self.toggle_microfono,
        )
        self.btn_toggle_mic.pack(fill=X, pady=5)

        # Area de estado
        self.texto_estado = tb.Text(
            marco_principal,
            height=8,
            wrap="word",
        )
        self.texto_estado.pack(fill=BOTH, expand=YES, pady=(15, 0))

        # Estilo oscuro manual
        self.texto_estado.configure(
            background="#1e1e1e",
            foreground="white",
            insertbackground="white",
        )

        self.agregar_linea_estado(
            "Bienvenido. Configure la base de datos de audio y siga los pasos 1 - 4."
        )

    # ------------------------------------------------------------------
    def agregar_linea_estado(self, mensaje):
        # Siempre actualiza el Text en el hilo principal
        self.after(
            0,
            lambda: (
                self.texto_estado.insert("end", mensaje + "\n"),
                self.texto_estado.see("end"),
            ),
        )

    def _mostrar_info(self, titulo, mensaje):
        self.after(0, lambda: messagebox.showinfo(titulo, mensaje))

    def _mostrar_advertencia(self, titulo, mensaje):
        self.after(0, lambda: messagebox.showwarning(titulo, mensaje))

    def _mostrar_error(self, titulo, mensaje):
        self.after(0, lambda: messagebox.showerror(titulo, mensaje))

    def _mostrar_confirmacion(self, titulo, mensaje):
        """Muestra un diálogo de confirmación y devuelve True si el usuario acepta.
        Debe ser llamado desde un hilo secundario."""
        resultado_queue = queue.Queue()
        
        def mostrar():
            respuesta = messagebox.askyesno(titulo, mensaje)
            resultado_queue.put(respuesta)
        
        self.after(0, mostrar)
        # Esperar la respuesta del usuario
        return resultado_queue.get()

    # ------------------------------------------------------------------
    def ejecutar_entrenamiento_en_hilo(self):
        hilo = threading.Thread(target=self._tarea_entrenamiento, daemon=True)
        hilo.start()

    def _tarea_entrenamiento(self):
        try:
            self.agregar_linea_estado("Iniciando entrenamiento de comandos...")
            entrenar_modelo_comandos()
            self.agregar_linea_estado(
                f"Entrenamiento finalizado. Archivo de umbrales guardado en: {ARCHIVO_UMBRALES}"
            )
            self._mostrar_info(
                "Entrenamiento finalizado",
                "El modelo ha sido entrenado correctamente.",
            )
        except Exception as e:
            self.agregar_linea_estado(f"[ERROR] Durante el entrenamiento: {e}")
            self._mostrar_error("Error en entrenamiento", str(e))

    # ------------------------------------------------------------------
    def cargar_umbrales_interfaz(self):
        try:
            self.umbrales = cargar_umbrales_desde_archivo()
            self.agregar_linea_estado("Umbrales cargados correctamente.")
            self._mostrar_info("Umbrales cargados", "Listo para reconocer comandos.")
        except Exception as e:
            self.agregar_linea_estado(f"[ERROR] Al cargar umbrales: {e}")
            self._mostrar_error("Error al cargar umbrales", str(e))

    # ------------------------------------------------------------------
    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccione la imagen base",
            filetypes=[
                ("Imagenes", "*.png;*.jpg;*.jpeg;*.bmp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if ruta:
            self.ruta_imagen = Path(ruta)
            self.agregar_linea_estado(f"Imagen seleccionada: {self.ruta_imagen}")
    
    def _seleccionar_y_aplicar_comando(self, comando, etiqueta):
        """Abre diálogo para seleccionar imagen y aplica el comando."""
        ruta = filedialog.askopenfilename(
            title=f"Seleccione imagen para aplicar '{etiqueta}'",
            filetypes=[
                ("Imagenes", "*.png;*.jpg;*.jpeg;*.bmp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        
        if ruta:
            self.ruta_imagen = Path(ruta)
            self.agregar_linea_estado(f"✓ Imagen seleccionada: {self.ruta_imagen.name}")
            
            # Aplicar el comando directamente
            self.agregar_linea_estado(f"Ejecutando: {etiqueta}...")
            ejecutar_operacion_imagen(comando, self.ruta_imagen)
            self.agregar_linea_estado(f"✓ {etiqueta} completado")
        else:
            self.agregar_linea_estado(f"✗ No se seleccionó imagen. Operación '{etiqueta}' cancelada")

    # ------------------------------------------------------------------
    def grabar_y_reconocer_en_hilo(self):
        hilo = threading.Thread(target=self._tarea_grabar_y_reconocer, daemon=True)
        hilo.start()

    def _tarea_grabar_y_reconocer(self):
        if self.umbrales is None:
            self.agregar_linea_estado(
                "Primero debe cargar los umbrales (paso 2) antes de reconocer."
            )
            self._mostrar_advertencia(
                "Umbrales no cargados",
                "Por favor cargue los umbrales entrenados antes de reconocer.",
            )
            return
        
        # Validar que se haya seleccionado una imagen
        if self.ruta_imagen is None:
            self.agregar_linea_estado(
                "Debe seleccionar una imagen (paso 3) antes de grabar el comando."
            )
            self._mostrar_advertencia(
                "Imagen no seleccionada",
                "Por favor seleccione una imagen antes de grabar el comando de voz.",
            )
            return

        self.agregar_linea_estado("Grabando audio desde el microfono...")
        senal = grabar_audio_microfono()

        self.agregar_linea_estado("Procesando senal y calculando energias...")
        vector_energias = procesar_senal_para_reconocimiento(senal)

        comando, puntaje = reconocer_comando_por_energia(
            vector_energias, self.umbrales
        )

        if comando is None:
            mensaje = "No se reconocio ningun comando dentro de los umbrales entrenados."
            self.agregar_linea_estado(mensaje)
            self._mostrar_info("Resultado", mensaje)
        else:
            etiqueta = ETIQUETAS_COMANDOS.get(comando, comando)
            mensaje = f"Comando reconocido: {etiqueta} (puntaje = {puntaje:.5e})"
            self.agregar_linea_estado(mensaje)
            
            # Mostrar diálogo de confirmación antes de ejecutar la operación
            confirmacion = self._mostrar_confirmacion(
                "Confirmar operacion",
                f"¿Desea aplicar la operacion '{etiqueta}' a la imagen seleccionada?\n\n"
                f"Imagen: {self.ruta_imagen.name}\n"
                f"Operacion: {etiqueta}\n"
                f"Confianza: {puntaje:.2f}"
            )
            
            if confirmacion:
                self.agregar_linea_estado(
                    f"Aplicando operacion de imagen asociada a {etiqueta}..."
                )
                ejecutar_operacion_imagen(comando, self.ruta_imagen)
                self.agregar_linea_estado(f"Operacion '{etiqueta}' completada exitosamente.")
            else:
                self.agregar_linea_estado(
                    f"Operación '{etiqueta}' cancelada por el usuario."
                )
    
    # ------------------------------------------------------------------
    # Métodos para micrófono continuo
    # ------------------------------------------------------------------
    
    def auto_cargar_entrenamiento(self):
        """Carga automáticamente los umbrales al iniciar si existen."""
        try:
            from pathlib import Path
            if Path(ARCHIVO_UMBRALES).exists():
                self.agregar_linea_estado("Cargando umbrales entrenados automáticamente...")
                self.umbrales = cargar_umbrales_desde_archivo()
                self.agregar_linea_estado("✓ Umbrales cargados. Sistema listo.")
            else:
                self.agregar_linea_estado("⚠ No se encontraron umbrales. Entrenando modelo...")
                self.ejecutar_entrenamiento_en_hilo()
        except Exception as e:
            self.agregar_linea_estado(f"⚠ Error al cargar umbrales: {e}")
    
    def activar_microfono_continuo(self):
        """Activa el micrófono en modo continuo para escuchar comandos."""
        if self.umbrales is None:
            self.agregar_linea_estado("⏳ Esperando carga de umbrales...")
            self.after(2000, self.activar_microfono_continuo)
            return
        
        self.microfono_activo = True
        self.label_microfono.config(text="🎤 Micrófono: ACTIVO (escuchando...)", bootstyle="success")
        self.btn_toggle_mic.config(text="⏸️ Pausar Micrófono")
        self.agregar_linea_estado("🎤 Micrófono activado. Diga un comando...")
        
        # Iniciar hilo de escucha
        self.hilo_microfono = threading.Thread(target=self._bucle_escucha_microfono, daemon=True)
        self.hilo_microfono.start()
    
    def toggle_microfono(self):
        """Pausa o reanuda el micrófono."""
        if self.microfono_activo:
            self.microfono_activo = False
            self.label_microfono.config(text="🎤 Micrófono: PAUSADO", bootstyle="warning")
            self.btn_toggle_mic.config(text="▶️ Reanudar Micrófono")
            self.agregar_linea_estado("🎤 Micrófono pausado")
        else:
            self.activar_microfono_continuo()
    
    def _bucle_escucha_microfono(self):
        """Bucle que escucha continuamente el micrófono (método SIMPLIFICADO)."""
        import time
        ultimo_reconocimiento = 0
        TIEMPO_ESPERA = 2.0  # Segundos entre grabaciones
        
        print("[MICRÓFONO] ✅ Listo. Escuchando...")
        print("[CONSEJO] Habla cuando veas 'Grabando...' en consola\n")
        
        while True:
            if not self.microfono_activo:
                break
            
            try:
                # Esperar tiempo mínimo entre reconocimientos
                tiempo_actual = time.time()
                if tiempo_actual - ultimo_reconocimiento < TIEMPO_ESPERA:
                    time.sleep(0.1)
                    continue
                
                print(f"\n[GRABANDO...] 1.0s (buscando voz...)")
                
                # Grabar audio (ahora graba 1s y busca ventana con más energía)
                senal = grabar_audio_microfono()
                
                if not self.microfono_activo:
                    break
                
                # Calcular RMS para logging
                rms_val = np.sqrt(np.mean(senal ** 2))
                db = 20.0 * np.log10(max(1e-12, rms_val))
                
                print(f"[CAPTURA] RMS={rms_val:.6f}, dB={db:.1f}")
                
                # Verificar que haya señal de audio (no solo silencio)
                if rms_val < 0.001:  # Umbral mínimo de señal
                    print(f"[DESCARTADO] Señal muy débil (silencio)\n")
                    continue
                
                # Procesar señal (usa método lab5)
                vector_energias = procesar_senal_para_reconocimiento(senal)
                
                # Reconocer comando (usa distancia euclidiana)
                comando, distancia = reconocer_comando_por_energia(vector_energias, self.umbrales)
                
                # UMBRAL DE DISTANCIA (más estricto para evitar falsos positivos)
                DISTANCIA_MAXIMA_ACEPTABLE = 0.05  # Muy estricto: solo acepta muy similares
                
                etiqueta = ETIQUETAS_COMANDOS.get(comando, comando)
                print(f"[RECONOCIMIENTO] {etiqueta}: distancia={distancia:.4f}, umbral={DISTANCIA_MAXIMA_ACEPTABLE}")
                
                if distancia < DISTANCIA_MAXIMA_ACEPTABLE:
                        ultimo_reconocimiento = tiempo_actual
                        self.agregar_linea_estado(f"✓ Comando detectado: {etiqueta} (dist: {distancia:.3f})")
                        
                        # Validar que hay imagen seleccionada
                        if self.ruta_imagen is None:
                            self.agregar_linea_estado(f"⚠ No hay imagen. Solicitando selección...")
                            
                            # Preguntar si quiere seleccionar una imagen ahora
                            quiere_seleccionar = self._mostrar_confirmacion(
                                "Imagen no seleccionada",
                                f"Comando '{etiqueta}' detectado.\n\n"
                                f"No hay imagen seleccionada.\n"
                                f"¿Desea seleccionar una imagen ahora?"
                            )
                            
                            if quiere_seleccionar:
                                # Abrir diálogo de selección en el hilo principal
                                self.after(0, lambda: self._seleccionar_y_aplicar_comando(comando, etiqueta))
                            else:
                                self.agregar_linea_estado(f"✗ Operación '{etiqueta}' cancelada (sin imagen)")
                            continue
                        
                        # Mostrar confirmación
                        confirmacion = self._mostrar_confirmacion(
                            "Confirmar operación",
                            f"¿Aplicar '{etiqueta}' a la imagen?\n\n"
                            f"Imagen: {self.ruta_imagen.name}\n"
                            f"Distancia: {distancia:.3f}"
                        )
                        
                        if confirmacion:
                            self.agregar_linea_estado(f"Ejecutando: {etiqueta}...")
                            ejecutar_operacion_imagen(comando, self.ruta_imagen)
                            self.agregar_linea_estado(f"✓ {etiqueta} completado")
                        else:
                            self.agregar_linea_estado(f"✗ {etiqueta} cancelado")
                else:
                    print(f"[RECHAZADO] {etiqueta}: distancia {distancia:.3f} > umbral {DISTANCIA_MAXIMA_ACEPTABLE}")
                    self.agregar_linea_estado(f"⚠ '{etiqueta}' detectado pero distancia alta ({distancia:.3f})")
                
            except Exception as e:
                self.agregar_linea_estado(f"Error en reconocimiento: {e}")
                time.sleep(0.5)


def main():
    app = AplicacionReconocimiento()
    app.mainloop()


if __name__ == "__main__":
    main()
