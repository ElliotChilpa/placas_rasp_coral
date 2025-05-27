import cv2
import numpy as np
import time
import os
import uuid
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

# Forzar RTSP por TCP y ampliar buffer
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000"

# RTSP del substream
RTSP_URL = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'

# Modelo y etiquetas
MODEL = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
LABELS_FILE = 'coco_labels.txt'

# Leer etiquetas
def read_labels(filename):
    with open(filename, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = read_labels(LABELS_FILE)

# Crear carpeta para placas recortadas
os.makedirs("recortes", exist_ok=True)

# Cargar modelo Coral
interpreter = make_interpreter(MODEL)
interpreter.allocate_tensors()
input_size = common.input_size(interpreter)

# Captura de video
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("No se pudo abrir el stream RTSP.")
    exit()

cv2.namedWindow('Coral - Vehículos', cv2.WINDOW_NORMAL)

# Medición de FPS
frame_count = 0
start_time = time.time()

# Filtrar solo vehículos
vehiculos_validos = ['car', 'truck', 'bus']

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error al leer frame. Reintentando...")
        time.sleep(1)
        continue

    frame_count += 1

    # Procesar 1 de cada 3 frames
    if frame_count % 3 != 0:
        continue

    # Preprocesamiento Coral
    resized = cv2.resize(frame, input_size)
    common.set_input(interpreter, resized)
    interpreter.invoke()
    objs = detect.get_objects(interpreter, score_threshold=0.5)

    # Escalado a resolución original
    scale_x = frame.shape[1] / input_size[0]
    scale_y = frame.shape[0] / input_size[1]

    for obj in objs:
        label = labels.get(obj.id, obj.id)
        if label not in vehiculos_validos:
            continue  # ignora todo lo que no sea vehículo

        bbox = obj.bbox
        x0, y0 = int(bbox.xmin * scale_x), int(bbox.ymin * scale_y)
        x1, y1 = int(bbox.xmax * scale_x), int(bbox.ymax * scale_y)
        score = obj.score

        # Dibujar solo vehículos
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(frame, f'{label} ({score:.2f})', (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Recorte de la región inferior del vehículo (posible placa)
        plate_y_start = int(y1 - (y1 - y0) * 0.3)
        plate_crop = frame[plate_y_start:y1, x0:x1]

        if plate_crop.size > 0:
            cv2.imshow("Placa recortada", plate_crop)
            filename = f"placa_{uuid.uuid4().hex[:8]}.jpg"
            path = os.path.join("recortes", filename)
            cv2.imwrite(path, plate_crop)
            print(f"Placa recortada guardada: {path}")

    # Mostrar frame principal con FPS
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Coral - Vehículos', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Finalizando...")
        break

cap.release()
cv2.destroyAllWindows()
