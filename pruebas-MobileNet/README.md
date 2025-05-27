# Pruebas Modelo MobileNet SSD

Está carpeta contiene las prubas desarrolladas con el modelo MobileNet

---

## 📁 Está es la estructura de está carpeta

```bash
.
├── backend/          
├── frontend/       
├── scripts/          
└── README.md       

detect_test.py - Es la primera prueba que se realizo del model mobilnet para saber si funciona google coral
detect_rtsp.py - Abre la camara y detecta objetos, no tiene etiqueta de vehiculos, los FPS son muy altos, muy lenta la captura de video
pruebas_rtsp.py - Este archivo es mas fuida la obtención de fps, detecta objetos aún solo vehiculos no.
SSD_MobileNet.py - Este archivo es la versión mas estable y mas completa ya que detecta vehiculas, no tiene boundery box, pero sí en cuanto detecta un vehiculo recorta la posible zona en la qu eeste el vehiculo.

recortes - Está carpeta guarda los recortes de los posibles zonas en donde se encuentre una placa.

ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite - Modelo descargado de la liga oficial de MobileNet.
coco_labels.txt - Son las etoquetas del modelo, también descargadas de la pagina oficial.