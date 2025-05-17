import cv2
import numpy as np
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
from PIL import Image, ImageDraw, ImageFont

# Configura tu RTSP aquí
RTSP_URL = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_main'

# Modelo y etiquetas
MODEL = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
LABELS = 'coco_labels.txt'

# Leer etiquetas
def read_label_file(file_path):
    with open(file_path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = read_label_file(LABELS)
interpreter = make_interpreter(MODEL)
interpreter.allocate_tensors()

# Abrir el stream RTSP
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: No se pudo abrir el stream RTSP")
    exit()

cv2.namedWindow('Detección Coral TPU - RTSP', cv2.WINDOW_NORMAL)

# Para calcular FPS
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error leyendo el frame del stream. Reintentando...")
        time.sleep(1)
        continue

    # Prepara la imagen para el TPU
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_image = image.resize(common.input_size(interpreter), Image.Resampling.LANCZOS)
    common.set_input(interpreter, resized_image)
    interpreter.invoke()

    objs = detect.get_objects(interpreter, score_threshold=0.5)

    # Dibuja las detecciones
    draw = ImageDraw.Draw(image)
    for obj in objs:
        bbox = obj.bbox
        label = labels.get(obj.id, obj.id)
        score = obj.score
        draw.rectangle([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], outline='red', width=2)
        draw.text((bbox.xmin, bbox.ymin), f'{label}: {score:.2f}', fill='red')

    # Convierte a formato OpenCV y muestra
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame_count += 1

    # Calcula FPS cada 30 frames
    #if frame_count >= 30:
    if frame_count % 3 != 0:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0
        print(f"FPS: {fps:.2f}")

    cv2.imshow('Detección Coral TPU - RTSP', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Finalizando detección.")
        break

cap.release()
cv2.destroyAllWindows()
