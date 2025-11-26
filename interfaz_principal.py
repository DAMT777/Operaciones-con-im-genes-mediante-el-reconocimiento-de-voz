import threading
from pathlib import Path

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox

from configuracion import ARCHIVO_UMBRALES
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

        self.crear_componentes_interfaz()

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

        # Botones principales
        marco_botones = tb.Frame(marco_principal)
        marco_botones.pack(fill=X, pady=10)

        btn_entrenar = tb.Button(
            marco_botones,
            text="1. Entrenar modelo con base de datos",
            bootstyle="primary",
            command=self.ejecutar_entrenamiento_en_hilo,
        )
        btn_entrenar.pack(fill=X, pady=5)

        btn_cargar_umbrales = tb.Button(
            marco_botones,
            text="2. Cargar umbrales entrenados",
            bootstyle="secondary",
            command=self.cargar_umbrales_interfaz,
        )
        btn_cargar_umbrales.pack(fill=X, pady=5)

        btn_seleccionar_imagen = tb.Button(
            marco_botones,
            text="3. Seleccionar imagen para operaciones",
            bootstyle="info",
            command=self.seleccionar_imagen,
        )
        btn_seleccionar_imagen.pack(fill=X, pady=5)

        btn_grabar_reconocer = tb.Button(
            marco_botones,
            text="4. Grabar desde micrófono y reconocer comando",
            bootstyle="success",
            command=self.grabar_y_reconocer_en_hilo,
        )
        btn_grabar_reconocer.pack(fill=X, pady=5)

        # Área de estado
        self.texto_estado = tb.Text(
            marco_principal,
            height=8,
            bootstyle="dark",
            wrap="word",
        )
        self.texto_estado.pack(fill=BOTH, expand=YES, pady=(15, 0))

        self.agregar_linea_estado(
            "Bienvenido. Configure la base de datos de audio y siga los pasos 1 → 4."
        )

    # ------------------------------------------------------------------
    def agregar_linea_estado(self, mensaje):
        self.texto_estado.insert("end", mensaje + "\n")
        self.texto_estado.see("end")

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
            messagebox.showinfo(
                "Entrenamiento finalizado",
                "El modelo ha sido entrenado correctamente.",
            )
        except Exception as e:
            self.agregar_linea_estado(f"[ERROR] Durante el entrenamiento: {e}")
            messagebox.showerror("Error en entrenamiento", str(e))

    # ------------------------------------------------------------------
    def cargar_umbrales_interfaz(self):
        try:
            self.umbrales = cargar_umbrales_desde_archivo()
            self.agregar_linea_estado("Umbrales cargados correctamente.")
            messagebox.showinfo("Umbrales cargados", "Listo para reconocer comandos.")
        except Exception as e:
            self.agregar_linea_estado(f"[ERROR] Al cargar umbrales: {e}")
            messagebox.showerror("Error al cargar umbrales", str(e))

    # ------------------------------------------------------------------
    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccione la imagen base",
            filetypes=[
                ("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if ruta:
            self.ruta_imagen = Path(ruta)
            self.agregar_linea_estado(f"Imagen seleccionada: {self.ruta_imagen}")

    # ------------------------------------------------------------------
    def grabar_y_reconocer_en_hilo(self):
        hilo = threading.Thread(target=self._tarea_grabar_y_reconocer, daemon=True)
        hilo.start()

    def _tarea_grabar_y_reconocer(self):
        if self.umbrales is None:
            self.agregar_linea_estado(
                "Primero debe cargar los umbrales (paso 2) antes de reconocer."
            )
            messagebox.showwarning(
                "Umbrales no cargados",
                "Por favor cargue los umbrales entrenados antes de reconocer.",
            )
            return

        self.agregar_linea_estado("Grabando audio desde el micrófono...")
        senal = grabar_audio_microfono()

        self.agregar_linea_estado("Procesando señal y calculando energías...")
        vector_energias = procesar_senal_para_reconocimiento(senal)

        comando, distancia = reconocer_comando_por_energia(
            vector_energias, self.umbrales
        )

        if comando is None:
            mensaje = (
                "No se reconoció ningún comando dentro del margen de error máximo del 5 %."  # noqa: E501
            )
            self.agregar_linea_estado(mensaje)
            messagebox.showinfo("Resultado", mensaje)
        else:
            mensaje = f"Comando reconocido: {comando} (distancia = {distancia:.5e})"
            self.agregar_linea_estado(mensaje)
            messagebox.showinfo("Resultado", mensaje)

            if self.ruta_imagen is not None:
                self.agregar_linea_estado(
                    f"Aplicando operación de imagen asociada a {comando}..."
                )
                ejecutar_operacion_imagen(comando, self.ruta_imagen)
            else:
                self.agregar_linea_estado(
                    "No se ha seleccionado imagen. Solo se muestra el comando reconocido."  # noqa: E501
                )


def main():
    app = AplicacionReconocimiento()
    app.mainloop()


if __name__ == "__main__":
    main()
