import cv2
import torch
import torchvision
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as T
from PIL import Image

# Lista oficial COCO 91 clases
COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class HiloCamara(QThread):
    nuevo_cuadro = pyqtSignal(np.ndarray)

    def __init__(self, fuente):
        super().__init__()
        self.fuente = fuente
        self.ejecutando = True
        self.ultimo_cuadro = None
        self.id_cuadro = 0
        self.mutex = QMutex()

    def run(self):
        es_imagen = False
        if isinstance(self.fuente, str) and self.fuente.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif')):
            es_imagen = True

        if es_imagen:
            cuadro = cv2.imread(self.fuente)
            if cuadro is not None:
                self.mutex.lock()
                self.ultimo_cuadro = cuadro.copy()
                self.id_cuadro += 1
                self.mutex.unlock()
                self.nuevo_cuadro.emit(self.ultimo_cuadro)
                while self.ejecutando: time.sleep(0.5)
            return

        cap = cv2.VideoCapture(self.fuente)
        if not cap.isOpened(): return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        tiempo_espera = 1.0 / fps if fps > 0 else 0.033

        while self.ejecutando:
            inicio_bucle = time.time()
            ret, cuadro = cap.read()
            if not ret:
                if isinstance(self.fuente, int): continue
                break
            
            # Aplicar efecto espejo si es una cámara en vivo
            if isinstance(self.fuente, int):
                cuadro = cv2.flip(cuadro, 1)
            
            self.mutex.lock()
            self.ultimo_cuadro = cuadro.copy()
            self.id_cuadro += 1
            self.mutex.unlock()
            self.nuevo_cuadro.emit(cuadro)
            
            transcurrido = time.time() - inicio_bucle
            tiempo_dormir = tiempo_espera - transcurrido
            if tiempo_dormir > 0:
                time.sleep(tiempo_dormir)
            
        cap.release()

    def obtener_ultimo_cuadro(self):
        self.mutex.lock()
        cuadro = (self.ultimo_cuadro.copy(), self.id_cuadro) if self.ultimo_cuadro is not None else (None, 0)
        self.mutex.unlock()
        return cuadro

    def detener(self):
        self.ejecutando = False
        self.wait()

class TrabajadorInferencia(QThread):
    resultado_listo = pyqtSignal(np.ndarray, float, int, float) # cuadro, tiempo, conteo, confianza

    def __init__(self, tipo_modelo, hilo_camara):
        super().__init__()
        self.tipo_modelo = tipo_modelo
        self.hilo_camara = hilo_camara
        self.ejecutando = True
        self.ultimo_id_procesado = -1
        self.dispositivo = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        if tipo_modelo == 'YOLO':
            self.modelo = YOLO('yolov8n.pt')
            self.modelo.to(self.dispositivo)
        else:
            np.random.seed(42)
            self.colores = np.random.randint(0, 255, size=(len(COCO_NAMES), 3), dtype=np.uint8)
            pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.modelo = fasterrcnn_resnet50_fpn_v2(weights=pesos)
            self.modelo.eval().to(self.dispositivo)
            self.transformacion = T.Compose([T.ToTensor()])

    def run(self):
        while self.ejecutando:
            cuadro, id_cuadro = self.hilo_camara.obtener_ultimo_cuadro()
            
            if cuadro is None or id_cuadro == self.ultimo_id_procesado:
                time.sleep(0.01)
                continue
            
            self.ultimo_id_procesado = id_cuadro
            inicio_tiempo = time.time()
            conteo = 0
            confianza_total = 0.0
            
            if self.tipo_modelo == 'YOLO':
                resultados = self.modelo(cuadro, verbose=False)
                cuadro_anotado = resultados[0].plot()
                
                conteo = len(resultados[0].boxes)
                if conteo > 0:
                    confianza_total = float(torch.mean(resultados[0].boxes.conf))
            else:
                cuadro_rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
                
                h, w = cuadro_rgb.shape[:2]
                tamano_max = 800
                escala = 1.0
                if max(h, w) > tamano_max:
                    escala = tamano_max / max(h, w)
                    nuevo_w, nuevo_h = int(w * escala), int(h * escala)
                    cuadro_rgb = cv2.resize(cuadro_rgb, (nuevo_w, nuevo_h))

                tensor_img = self.transformacion(Image.fromarray(cuadro_rgb)).to(self.dispositivo)
                with torch.no_grad():
                    prediccion = self.modelo([tensor_img])[0]
                
                cuadro_anotado = cuadro.copy()
                
                cajas = prediccion['boxes'].cpu().numpy()
                etiquetas = prediccion['labels'].cpu().numpy()
                puntajes = prediccion['scores'].cpu().numpy()
                
                if escala != 1.0:
                    cajas = cajas / escala
                
                cajas_finales = []
                etiquetas_finales = []
                puntajes_finales = []
                
                for j in range(len(cajas)):
                    if etiquetas[j] == 0: continue
                    es_persona = (etiquetas[j] == 1)
                    umbral = 0.65 if es_persona else 0.75
                    
                    if puntajes[j] >= umbral:
                        cajas_finales.append(cajas[j])
                        etiquetas_finales.append(etiquetas[j])
                        puntajes_finales.append(puntajes[j])
                        
                if len(cajas_finales) > 0:
                    t_cajas = torch.tensor(np.array(cajas_finales))
                    t_puntajes = torch.tensor(np.array(puntajes_finales))
                    t_etiquetas = torch.tensor(np.array(etiquetas_finales))
                    
                    mantener = torchvision.ops.nms(t_cajas, t_puntajes, iou_threshold=0.45)
                    
                    cajas = t_cajas[mantener].numpy()
                    etiquetas = t_etiquetas[mantener].numpy()
                    puntajes = t_puntajes[mantener].numpy()
                    
                    conteo = len(cajas)
                    confianza_total = float(np.mean(puntajes))
                    
                    grosor = max(1, int(w / 400))
                    escala_fuente = w / 1000
                    
                    for i in range(len(cajas)):
                        id_clase = etiquetas[i]
                        if COCO_NAMES[id_clase] == 'N/A': continue
                        caja = cajas[i].astype(int)
                        color = tuple(int(c) for c in self.colores[id_clase])
                        
                        cv2.rectangle(cuadro_anotado, (caja[0], caja[1]), (caja[2], caja[3]), color, grosor)
                        texto_etiqueta = f"{COCO_NAMES[id_clase]} {puntajes[i]:.2f}"
                        (tw, th), _ = cv2.getTextSize(texto_etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, grosor)
                        ty = max(caja[1], th + 10)
                        cv2.rectangle(cuadro_anotado, (caja[0], ty - th - 10), (caja[0] + tw, ty), color, -1)
                        cv2.putText(cuadro_anotado, texto_etiqueta, (caja[0], ty - 5), cv2.FONT_HERSHEY_SIMPLEX, escala_fuente, (255, 255, 255), grosor)
            
            tiempo_inferencia = (time.time() - inicio_tiempo) * 1000
            self.resultado_listo.emit(cuadro_anotado, tiempo_inferencia, conteo, confianza_total)

    def detener(self):
        self.ejecutando = False
        self.wait()
