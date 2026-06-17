# Reporte de Rendimiento Real: YOLO vs R-CNN

## 1. Introducción
Este reporte presenta una comparativa técnica basada en el rendimiento **real** medido en este hardware. Se comparan dos arquitecturas:
- **YOLOv8 (One-Stage)**: Optimizada para velocidad y tiempo real.
- **Faster R-CNN (Two-Stage)**: Orientada a la precisión estructural.

## 2. Metodología y Hardware
- **Contexto**: Hardware Real detectado durante la ejecución en la GUI.
- **Origen de datos**: Captura automática durante la ejecución de la interfaz gráfica.
- **Métrica**: Promedio ponderado de latencia por frame.

## 3. Resultados Cuantitativos (Mediciones Reales)

| Modelo | Latencia Promedio (ms) | FPS Promedio | Eficiencia Relativa |
|--------|-----------------------|--------------|---------------------|
| **YOLOv8n** | 148.26 ms | 6.75 | 100% (Referencia) |
| **Faster R-CNN** | 294.79 ms | 3.39 | 50.22% |

## 4. Análisis de Resultados
- **Latencia**: YOLOv8 es aproximadamente **2.0 veces** más rápido que Faster R-CNN en este sistema.
- **Fluidez**: Con **6.75 FPS**, YOLOv8 se comporta de manera aceptable para video en vivo. 
- **Cuello de Botella**: Faster R-CNN muestra una latencia de **294.79 ms**, lo que sugiere que el hardware está usando CUDA.

## 5. Conclusión
Basado en las métricas obtenidas, para este entorno específico, **YOLOv8** es la única arquitectura capaz de mantener un flujo de trabajo de visión artificial en tiempo real. Faster R-CNN queda reservado para tareas de análisis de imágenes estáticas de alta precisión.
