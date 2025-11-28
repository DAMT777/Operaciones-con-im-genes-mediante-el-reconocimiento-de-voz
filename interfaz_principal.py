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
        
        self.after(500, self.auto_cargar_entrenamiento)
        
        self.after(1000, self.activar_microfono_continuo)

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

        self.label_microfono = tb.Label(
            marco_principal,
            text="🎤 Micrófono: Inicializando...",
            bootstyle="warning",
            font=("Segoe UI", 12, "bold"),
        )
        self.label_microfono.pack(fill=X, pady=5)

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

        self.texto_estado = tb.Text(
            marco_principal,
            height=8,
            wrap="word",
        )
        self.texto_estado.pack(fill=BOTH, expand=YES, pady=(15, 0))

        self.texto_estado.configure(
            background="#1e1e1e",
            foreground="white",
            insertbackground="white",
        )

        self.agregar_linea_estado(
            "Bienvenido. Configure la base de datos de audio y siga los pasos 1 - 4."
        )

    def agregar_linea_estado(self, mensaje):
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
        result = messagebox.askyesno(titulo, mensaje)
        return result
    
    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccione imagen para operaciones",
            filetypes=[
                ("Imagenes", "*.png;*.jpg;*.jpeg;*.bmp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        
        if ruta:
            self.ruta_imagen = Path(ruta)
            self.agregar_linea_estado(f"✓ Imagen seleccionada: {self.ruta_imagen.name}")
        else:
            self.agregar_linea_estado(f"✗ No se seleccionó imagen")
    
    def _seleccionar_y_aplicar_comando(self, comando, etiqueta):
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
            
            self.agregar_linea_estado(f"Ejecutando: {etiqueta}...")
            ejecutar_operacion_imagen(comando, self.ruta_imagen, self.pausar_microfono, self.reanudar_microfono)
            self.agregar_linea_estado(f"✓ {etiqueta} completado")
        else:
            self.agregar_linea_estado(f"✗ No se seleccionó imagen. Operación '{etiqueta}' cancelada")

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
                ejecutar_operacion_imagen(comando, self.ruta_imagen, self.pausar_microfono, self.reanudar_microfono)
                self.agregar_linea_estado(f"Operacion '{etiqueta}' completada exitosamente.")
            else:
                self.agregar_linea_estado(
                    f"Operación '{etiqueta}' cancelada por el usuario."
                )
    
    def auto_cargar_entrenamiento(self):
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
        if self.umbrales is None:
            self.agregar_linea_estado("⏳ Esperando carga de umbrales...")
            self.after(2000, self.activar_microfono_continuo)
            return
        
        self.microfono_activo = True
        self.label_microfono.config(text="🎤 Micrófono: ACTIVO (escuchando...)", bootstyle="success")
        self.btn_toggle_mic.config(text="⏸️ Pausar Micrófono")
        self.agregar_linea_estado("🎤 Micrófono activado. Diga un comando...")
        
        self.hilo_microfono = threading.Thread(target=self._bucle_escucha_microfono, daemon=True)
        self.hilo_microfono.start()
    
    def toggle_microfono(self):
        if self.microfono_activo:
            self.microfono_activo = False
            self.label_microfono.config(text="🎤 Micrófono: PAUSADO", bootstyle="warning")
            self.btn_toggle_mic.config(text="▶️ Reanudar Micrófono")
            self.agregar_linea_estado("🎤 Micrófono pausado")
        else:
            self.activar_microfono_continuo()
    
    def pausar_microfono(self):
        if self.microfono_activo:
            self.microfono_activo = False
            self.label_microfono.config(text="🎤 Micrófono: PAUSADO (procesando imagen)", bootstyle="info")
            self.btn_toggle_mic.config(text="▶️ Reanudar Micrófono")
            self.agregar_linea_estado("🎤 Micrófono pausado (ventana de procesamiento abierta)")
    
    def reanudar_microfono(self):
        if not self.microfono_activo and self.umbrales is not None:
            self.activar_microfono_continuo()
            self.agregar_linea_estado("🎤 Micrófono reanudado (ventana de procesamiento cerrada)")
    
    def _bucle_escucha_microfono(self):
        import time
        ultimo_reconocimiento = 0
        TIEMPO_ESPERA = 1.5
        
        print("[MICRÓFONO] ✅ Listo. Escuchando...")
        print("[CONSEJO] Habla CLARO y FUERTE cuando veas 'Grabando...'\n")
        
        ultimo_comando = None
        contador_mismo_comando = 0
        CONFIRMACIONES_NECESARIAS = 1
        
        while True:
            if not self.microfono_activo:
                break
            
            try:
                tiempo_actual = time.time()
                if tiempo_actual - ultimo_reconocimiento < TIEMPO_ESPERA:
                    time.sleep(0.1)
                    continue
                
                print(f"\n[GRABANDO...] 1.0s (buscando voz...)")
                
                senal = grabar_audio_microfono()
                
                if not self.microfono_activo:
                    break
                
                rms_val = np.sqrt(np.mean(senal ** 2))
                db = 20.0 * np.log10(max(1e-12, rms_val))
                
                print(f"[CAPTURA] RMS={rms_val:.6f}, dB={db:.1f}")
                
                if rms_val < 0.0001:
                    print(f"[DESCARTADO] Señal muy débil (silencio)\n")
                    continue
                
                print(f"[OK] Señal detectada (RMS={rms_val:.6f}), procesando...")
                
                vector_energias = procesar_senal_para_reconocimiento(senal)
                
                comando, distancia = reconocer_comando_por_energia(vector_energias, self.umbrales)
                
                if comando is None:
                    print(f"[RECHAZADO] Ningún comando cumple umbral (mejor dist={distancia:.4f})")
                    continue
                
                etiqueta = ETIQUETAS_COMANDOS.get(comando, comando)
                print(f"[RECONOCIMIENTO] {etiqueta}: distancia={distancia:.4f}")
                
                if comando == ultimo_comando:
                    contador_mismo_comando += 1
                    print(f"[CONFIRMACIÓN] {etiqueta} ({contador_mismo_comando}/{CONFIRMACIONES_NECESARIAS})")
                else:
                    ultimo_comando = comando
                    contador_mismo_comando = 1
                    print(f"[NUEVO] {etiqueta} detectado (1/{CONFIRMACIONES_NECESARIAS})")
                    continue
                
                if contador_mismo_comando >= CONFIRMACIONES_NECESARIAS:
                    ultimo_reconocimiento = tiempo_actual
                    self.agregar_linea_estado(f"✓ Comando detectado: {etiqueta} (dist: {distancia:.3f})")
                    
                    if self.ruta_imagen is None:
                        self.agregar_linea_estado(f"⚠ No hay imagen. Solicitando selección...")
                        
                        quiere_seleccionar = self._mostrar_confirmacion(
                            "Imagen no seleccionada",
                            f"Comando '{etiqueta}' detectado.\n\n"
                            f"No hay imagen seleccionada.\n"
                            f"¿Desea seleccionar una imagen ahora?"
                        )
                        
                        if quiere_seleccionar:
                            self.after(0, lambda: self._seleccionar_y_aplicar_comando(comando, etiqueta))
                        else:
                            self.agregar_linea_estado(f"✗ Operación '{etiqueta}' cancelada (sin imagen)")
                        continue
                    
                    confirmacion = self._mostrar_confirmacion(
                        "Confirmar operación",
                        f"¿Aplicar '{etiqueta}' a la imagen?\n\n"
                        f"Imagen: {self.ruta_imagen.name}\n"
                        f"Distancia: {distancia:.3f}"
                    )
                    
                    if confirmacion:
                        self.agregar_linea_estado(f"Ejecutando: {etiqueta}...")
                        ejecutar_operacion_imagen(comando, self.ruta_imagen, self.pausar_microfono, self.reanudar_microfono)
                        self.agregar_linea_estado(f"✓ {etiqueta} completado")
                    else:
                        self.agregar_linea_estado(f"✗ {etiqueta} cancelado")
                    
                    ultimo_comando = None
                    contador_mismo_comando = 0
                
            except Exception as e:
                self.agregar_linea_estado(f"Error en reconocimiento: {e}")
                ultimo_comando = None
                contador_mismo_comando = 0
                time.sleep(0.5)

def main():
    app = AplicacionReconocimiento()
    app.mainloop()

if __name__ == "__main__":
    main()
