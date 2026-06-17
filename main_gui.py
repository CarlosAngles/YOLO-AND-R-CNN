import sys
import cv2
import time
import os
import json
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QSplitter, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont
from detection_engine import HiloCamara, TrabajadorInferencia

class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laboratorio de Visión Artificial - Comparativa de Arquitecturas")
        self.setMinimumSize(1300, 850)
        
        # --- ESTILO PROFESIONAL ---
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; }
            QLabel { color: #e2e8f0; font-family: 'Segoe UI', sans-serif; }
            QFrame#tarjeta_yolo {
                background-color: #1e293b; border-radius: 8px; border: 1px solid #334155;
                border-top: 4px solid #10b981;
            }
            QFrame#tarjeta_rcnn {
                background-color: #1e293b; border-radius: 8px; border: 1px solid #334155;
                border-top: 4px solid #3b82f6;
            }
            QLabel#titulo { font-size: 28px; font-weight: 800; color: #f8fafc; }
            QLabel#subtitulo { font-size: 15px; color: #94a3b8; }
            QLabel#titulo_modelo { font-size: 18px; font-weight: 700; color: #f1f5f9; padding: 15px; }
            QLabel#metricas {
                font-size: 14px; font-weight: 600; color: #94a3b8; padding: 15px;
                background-color: #0f172a; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px;
            }
            QPushButton {
                background-color: #1e293b; color: #f1f5f9; border: 1px solid #475569;
                border-radius: 6px; padding: 10px 20px; font-weight: 600;
            }
            QPushButton:hover { background-color: #334155; }
            QPushButton#btn_iniciar { background-color: #059669; color: white; border: none; }
            QPushButton#btn_iniciar:hover { background-color: #10b981; }
            QPushButton#btn_detener { color: #ef4444; border: 1px solid #ef4444; }
            QLabel#area_visualizacion { background-color: #000000; border-radius: 4px; margin: 15px; }
        """)

        self.inicializar_ui()
        self.hilo_cam = None
        self.trabajador_yolo = None
        self.trabajador_rcnn = None
        self.ruta_fuente = ""
        self.historial_metricas = {'YOLO': [], 'RCNN': []}

    def inicializar_ui(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        diseno_principal = QVBoxLayout(widget_central)
        diseno_principal.setContentsMargins(30, 30, 30, 30)
        diseno_principal.setSpacing(25)
        
        # ENCABEZADO
        diseno_encabezado = QHBoxLayout()
        caja_titulo = QVBoxLayout()
        lbl_titulo = QLabel("Visión Artificial")
        lbl_titulo.setObjectName("titulo")
        lbl_subtitulo = QLabel("Análisis comparativo de latencia y precisión: YOLOv8 vs Faster R-CNN")
        lbl_subtitulo.setObjectName("subtitulo")
        caja_titulo.addWidget(lbl_titulo); caja_titulo.addWidget(lbl_subtitulo)
        diseno_encabezado.addLayout(caja_titulo)
        diseno_encabezado.addStretch()
        
        # BOTONES
        self.btn_cargar = QPushButton("Cargar Archivo"); self.btn_cargar.clicked.connect(self.cargar_fuente)
        self.btn_camara = QPushButton("Usar Cámara"); self.btn_camara.clicked.connect(self.usar_camara)
        self.btn_iniciar = QPushButton("Iniciar Análisis"); self.btn_iniciar.setObjectName("btn_iniciar"); self.btn_iniciar.setEnabled(False); self.btn_iniciar.clicked.connect(self.iniciar)
        self.btn_detener = QPushButton("Detener"); self.btn_detener.setObjectName("btn_detener"); self.btn_detener.setEnabled(False); self.btn_detener.clicked.connect(self.detener)
        
        for btn in [self.btn_cargar, self.btn_camara, self.btn_iniciar, self.btn_detener]:
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            diseno_encabezado.addWidget(btn)
        diseno_principal.addLayout(diseno_encabezado)
        
        # ÁREA DE VISUALIZACIÓN
        divisor = QSplitter(Qt.Orientation.Horizontal)
        
        # TARJETA YOLO
        tarjeta_yolo = QFrame(); tarjeta_yolo.setObjectName("tarjeta_yolo")
        diseno_y = QVBoxLayout(tarjeta_yolo)
        tit_y = QLabel("YOLOv8 Nano"); tit_y.setObjectName("titulo_modelo"); diseno_y.addWidget(tit_y)
        self.lbl_yolo = QLabel("Esperando..."); self.lbl_yolo.setObjectName("area_visualizacion"); self.lbl_yolo.setAlignment(Qt.AlignmentFlag.AlignCenter); diseno_y.addWidget(self.lbl_yolo, 1)
        self.lbl_yolo_m = QLabel("Latencia: -- ms\nObjetos: --\nConfianza: --%"); self.lbl_yolo_m.setObjectName("metricas"); diseno_y.addWidget(self.lbl_yolo_m)
        
        # TARJETA RCNN
        tarjeta_rcnn = QFrame(); tarjeta_rcnn.setObjectName("tarjeta_rcnn")
        diseno_r = QVBoxLayout(tarjeta_rcnn)
        tit_r = QLabel("Faster R-CNN V2"); tit_r.setObjectName("titulo_modelo"); diseno_r.addWidget(tit_r)
        self.lbl_rcnn = QLabel("Esperando..."); self.lbl_rcnn.setObjectName("area_visualizacion"); self.lbl_rcnn.setAlignment(Qt.AlignmentFlag.AlignCenter); diseno_r.addWidget(self.lbl_rcnn, 1)
        self.lbl_rcnn_m = QLabel("Latencia: -- ms\nObjetos: --\nConfianza: --%"); self.lbl_rcnn_m.setObjectName("metricas"); diseno_r.addWidget(self.lbl_rcnn_m)
        
        divisor.addWidget(tarjeta_yolo); divisor.addWidget(tarjeta_rcnn)
        diseno_principal.addWidget(divisor, 1)

    def usar_camara(self):
        self.ruta_fuente = 0
        self.btn_iniciar.setEnabled(True)
        self.lbl_yolo.setText("Cámara detectada."); self.lbl_rcnn.setText("Cámara detectada.")

    def cargar_fuente(self):
        f, _ = QFileDialog.getOpenFileName(self, "Cargar Archivo", "input", "Multimedia (*.jpg *.png *.mp4 *.avi)")
        if f:
            self.ruta_fuente = f; self.btn_iniciar.setEnabled(True)
            self.lbl_yolo.setText(os.path.basename(f)); self.lbl_rcnn.setText(os.path.basename(f))

    def iniciar(self):
        self.detener()
        self.btn_iniciar.setEnabled(False); self.btn_detener.setEnabled(True)
        self.hilo_cam = HiloCamara(self.ruta_fuente)
        self.trabajador_yolo = TrabajadorInferencia('YOLO', self.hilo_cam)
        self.trabajador_yolo.resultado_listo.connect(lambda f, s, c, conf: self.actualizar_vista(f, s, c, conf, 'YOLO'))
        self.trabajador_rcnn = TrabajadorInferencia('RCNN', self.hilo_cam)
        self.trabajador_rcnn.resultado_listo.connect(lambda f, s, c, conf: self.actualizar_vista(f, s, c, conf, 'RCNN'))
        self.hilo_cam.start(); self.trabajador_yolo.start(); self.trabajador_rcnn.start()

    def detener(self):
        if len(self.historial_metricas['YOLO']) > 0:
            metricas = {m: {'latency_ms': round(sum(d)/len(d), 2), 'fps': round(1000/(sum(d)/len(d)), 2)} 
                       for m, d in self.historial_metricas.items() if d}
            os.makedirs('output', exist_ok=True)
            with open('output/real_time_metrics.json', 'w') as f: json.dump(metricas, f, indent=4)
        
        if self.trabajador_yolo: self.trabajador_yolo.detener(); self.trabajador_yolo = None
        if self.trabajador_rcnn: self.trabajador_rcnn.detener(); self.trabajador_rcnn = None
        if self.hilo_cam: self.hilo_cam.detener(); self.hilo_cam = None
        self.btn_iniciar.setEnabled(True); self.btn_detener.setEnabled(False)
        self.historial_metricas = {'YOLO': [], 'RCNN': []}

    def actualizar_vista(self, cuadro, velocidad, conteo, confianza, tipo_m):
        self.historial_metricas[tipo_m].append(velocidad)
        rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img_qt = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img_qt).scaled(580, 440, Qt.AspectRatioMode.KeepAspectRatio)
        
        metricas_texto = f"Latencia: {velocidad:.1f} ms | FPS: {1000/velocidad:.1f}\nObjetos: {conteo}\nConfianza: {confianza*100:.1f}%"
        
        if tipo_m == 'YOLO':
            self.lbl_yolo.setPixmap(pix)
            self.lbl_yolo_m.setText(metricas_texto)
        else:
            self.lbl_rcnn.setPixmap(pix)
            self.lbl_rcnn_m.setText(metricas_texto)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal(); ventana.show()
    sys.exit(app.exec())
