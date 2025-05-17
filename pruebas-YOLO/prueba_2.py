import cv2
import numpy as np
import time
import os
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from PIL import Image

# Configuración RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000"
RTSP_URL = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'

# Cargar modelo Coral
MODEL = 'best_clean_edgetpu.tflite'
interpreter = make_interpreter(MODEL)
interpreter.allocate_tensors()
input_size = common.input_size(interpreter)  # Esperado: (320, 320)

# Captura de video
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("No se pudo abrir el stream RTSP.")
    exit()

cv2.namedWindow('YOLOv5 Coral', cv2.WINDOW_NORMAL)

# Control de FPS
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer frame. Reintentando...")
        time.sleep(1)
        continue

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    # Redimensionar y preparar entrada
    resized = cv2.resize(frame, input_size)
    image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    common.set_input(interpreter, image)
    interpreter.invoke()

    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]  # (6300, 6)
    height, width, _ = frame.shape

    for det in output:
        cls_id, score, cx, cy, w, h = det

        # Filtro de score bajo (int8)
        if score < 32:
            continue

        # Conversión cx, cy, w, h → x0, y0, x1, y1 y escalado a tamaño real
        x0 = int(((cx - w / 2) / 255) * width)
        y0 = int(((cy - h / 2) / 255) * height)
        x1 = int(((cx + w / 2) / 255) * width)
        y1 = int(((cy + h / 2) / 255) * height)

        # Validar límites
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue

        label = f'Placa ({score / 255:.2f})'
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, label, (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar FPS
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('YOLOv5 Coral', frame)

    # Salida
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Finalizando.")
        break

cap.release()
cv2.destroyAllWindows()
