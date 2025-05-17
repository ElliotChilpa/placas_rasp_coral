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

# Cargar modelo YOLOv5s EdgeTPU
MODEL = 'best_clean_edgetpu.tflite'
interpreter = make_interpreter(MODEL)
interpreter.allocate_tensors()
input_size = common.input_size(interpreter)  # (320, 320)

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

    # Redimensionar para la red
    resized = cv2.resize(frame, input_size)
    image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    common.set_input(interpreter, image)
    interpreter.invoke()

    # Obtener y procesar salida del modelo
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # (6300, 6)
    print("Ejemplo de detección:", output[0])

    height, width, _ = frame.shape

    for det in output:
        cls_id, score, x1, y1, x2, y2 = det

        # Filtro por score (cuantizado a 0-255, usamos >128 como ~0.5)
        if score < 64:
            continue

        # Normalizar coordenadas (dividir entre input size)
        x1 /= 320
        y1 /= 320
        x2 /= 320
        y2 /= 320

        # Escalar a tamaño del frame original
        x0 = int(x1 * width)
        y0 = int(y1 * height)
        x1 = int(x2 * width)
        y1 = int(y2 * height)

        # Etiqueta genérica (puedes usar un diccionario si luego quieres etiquetas reales)
        label = f'Vehículo ({score / 255:.2f})'
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, label, (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar FPS
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('YOLOv5 Coral', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Finalizando.")
        break

cap.release()
cv2.destroyAllWindows()
