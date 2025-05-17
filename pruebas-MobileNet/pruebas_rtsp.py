import cv2
import numpy as np
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
import os

# Opciones para forzar RTSP por TCP y buffer más amplio
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000"

# URL del substream RTSP optimizado
RTSP_URL = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'

# Modelo Coral optimizado + etiquetas
MODEL = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
LABELS_FILE = 'coco_labels.txt'

# Cargar etiquetas
def read_labels(filename):
    with open(filename, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = read_labels(LABELS_FILE)

# Cargar modelo en Coral
interpreter = make_interpreter(MODEL)
interpreter.allocate_tensors()
input_size = common.input_size(interpreter)

# Captura de video
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print(" No se pudo abrir el stream RTSP.")
    exit()

cv2.namedWindow('Coral - Detección', cv2.WINDOW_NORMAL)

# Medición de FPS
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Error al leer frame. Reintentando...")
        time.sleep(1)
        continue

    frame_count += 1

    # Procesar 1 de cada 3 frames para aliviar carga
    if frame_count % 3 != 0:
        continue

    # Preprocesamiento para Coral
    resized = cv2.resize(frame, input_size)
    common.set_input(interpreter, resized)
    interpreter.invoke()
    objs = detect.get_objects(interpreter, score_threshold=0.5)

    # Dibujar detecciones sobre el frame original
    scale_x = frame.shape[1] / input_size[0]
    scale_y = frame.shape[0] / input_size[1]

    for obj in objs:
        bbox = obj.bbox
        x0, y0 = int(bbox.xmin * scale_x), int(bbox.ymin * scale_y)
        x1, y1 = int(bbox.xmax * scale_x), int(bbox.ymax * scale_y)
        label = labels.get(obj.id, obj.id)
        score = obj.score

        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(frame, f'{label} ({score:.2f})', (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # FPS
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar frame
    cv2.imshow('Coral - Detección', frame)

    # Salida
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
