# Detección de Objetos: Faster R-CNN vs YOLOv8

Comparación experimental entre dos enfoques de detección de objetos en visión artificial: **Faster R-CNN ResNet-50-FPN V2** (modelo de dos etapas) y **YOLOv8n** (modelo de una sola etapa), evaluados sobre imágenes estáticas y video en tiempo real. El dispositivo se selecciona automáticamente (GPU si está disponible, CPU en caso contrario).

Proyecto desarrollado como Actividad 06 del curso de Visión Artificial — Escuela Profesional de Ingeniería de Sistemas, Universidad Nacional del Altiplano (Puno, Perú).

## Características

- Detección con YOLOv8n (Ultralytics) y Faster R-CNN ResNet-50-FPN V2 (TorchVision), sobre imágenes, video o cámara web
- Interfaz gráfica unificada (PyQt6) que ejecuta ambos modelos en paralelo (hilos independientes) y muestra resultados lado a lado
- Registro automático de métricas (latencia y FPS) en `output/real_time_metrics.json` al detener el análisis
- Generación de gráfico comparativo y reporte en Markdown a partir de esas métricas

## Tecnologías

Python · PyTorch · TorchVision · Ultralytics YOLOv8 · OpenCV · PyQt6

## Estructura del proyecto

| Archivo / carpeta | Descripción |
| --- | --- |
| `main_gui.py` | Interfaz gráfica (PyQt6): carga de imagen/video/cámara, ejecuta YOLO y Faster R-CNN en paralelo, muestra resultados lado a lado y guarda métricas al detener |
| `detection_engine.py` | Lógica de captura (`HiloCamara`) e inferencia (`TrabajadorInferencia`) para YOLOv8n y Faster R-CNN ResNet-50-FPN V2 |
| `compare_results.py` | Genera la gráfica comparativa (`output/comparison_metrics.png`) y el reporte en Markdown (`output/REPORTE_COMPARATIVO.md`), usando las métricas reales si existen o valores por defecto |
| `input/` | Imágenes o videos de prueba |
| `output/` | Métricas y reportes generados al ejecutar la GUI y `compare_results.py` |
| `assets/` | Imágenes usadas en este README |
| `yolov8n.pt` | Pesos preentrenados de YOLOv8 Nano |

## Instalación

```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
pip install torch torchvision ultralytics opencv-python PyQt6 matplotlib pillow numpy
```

## Uso

1. Ejecuta la interfaz gráfica:

   ```bash
   python main_gui.py
   ```

2. Haz clic en **Cargar Archivo** (imagen o video) o **Usar Cámara**.
3. Haz clic en **Iniciar Análisis** para correr YOLOv8n y Faster R-CNN en paralelo.
4. Haz clic en **Detener** para finalizar; las métricas se guardan en `output/real_time_metrics.json`.
5. (Opcional) Genera el gráfico y el reporte comparativo a partir de las métricas capturadas:

   ```bash
   python compare_results.py
   ```

## Resultados

Valores de referencia (son los valores por defecto en `compare_results.py`; se reemplazan automáticamente si existe `output/real_time_metrics.json` con métricas reales). El dispositivo se selecciona automáticamente (estas pruebas se ejecutaron sobre CPU). El umbral de confianza no es fijo en 0.5 para ambos modelos: YOLOv8n usa el umbral por defecto de Ultralytics, mientras que Faster R-CNN aplica 0.65 para la clase "persona" y 0.75 para el resto, con un filtrado adicional por NMS (IoU 0.45).

| Métrica | YOLOv8n | Faster R-CNN | Diferencia |
| --- | --- | --- | --- |
| Tiempo de inferencia (imagen) | 192.00 ms | 1,951.79 ms | R-CNN ~10.2× más lento |
| Velocidad de video (FPS) | 10.37 | 0.52 | YOLO ~19.9× más rápido |
| Categorías COCO soportadas | 80 | 91 | R-CNN +11 categorías |

**Conclusión principal:** YOLOv8n es significativamente más adecuado para aplicaciones en tiempo real, mientras que Faster R-CNN ofrece mayor precisión de localización cuando la latencia no es crítica, especialmente con hardware GPU.

### Detección sobre imagen estática

![Comparación YOLOv8 vs Faster R-CNN sobre imagen estática](assets/figura1_deteccion_imagen.jpg)

*Faster R-CNN asignó mayor confianza a los objetos (manzana: 1.00, plátano: 0.99) frente a YOLOv8 (0.94 y 0.88), mostrando mayor precisión de detección a costa de más tiempo de cómputo.*

### Detección sobre video en tiempo real

![Comparación YOLOv8 vs Faster R-CNN sobre video en tiempo real](assets/figura2_deteccion_video.jpg)

*YOLOv8 detectó a la persona en 104 ms (9.6 FPS); Faster R-CNN, con 296 ms (3.4 FPS), identificó además un objeto adicional ("dining table"), a costa de mayor latencia.*

## Dataset

Ambos modelos se evaluaron sobre **COCO** (Common Objects in Context).

## Trabajo futuro

- Evaluar variantes más grandes de YOLOv8 (m, l, x) y Mask R-CNN sobre GPU
- Incorporar métricas de precisión media (mAP@0.5 y mAP@0.5:0.95)

## Autores

- Quispe Ticona, Angel Pedro
- Angles Quispe, Carlos Mauricio
- Mamani Turpo, Elfer
- Huanca Chambi, Cristian Brayan

**Curso:** Visión Artificial — **Docente:** Fernandez Chambi, Mayenka — Semestre X, Grupo B

## Referencias

- Girshick et al. (2014). *Rich feature hierarchies for accurate object detection and semantic segmentation.* CVPR.
- He et al. (2016). *Deep residual learning for image recognition.* CVPR.
- Jocher, Chaurasia & Qiu (2023). *Ultralytics YOLO (v8.0.0)* [Software]. https://github.com/ultralytics/ultralytics
- Lin et al. (2014). *Microsoft COCO: Common objects in context.* ECCV.
- Lin et al. (2017). *Feature pyramid networks for object detection.* CVPR.
- Redmon et al. (2016). *You only look once: Unified, real-time object detection.* CVPR.
- Ren et al. (2015). *Faster R-CNN: Towards real-time object detection with region proposal networks.* NeurIPS.
