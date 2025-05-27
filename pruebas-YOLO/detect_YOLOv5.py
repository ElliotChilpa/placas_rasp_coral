#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Detección de placas con modelo YOLOv5-EdgeTPU (.tflite) en tiempo real
sobre un stream RTSP, dibujando la caja en la imagen.
Compatible con Raspberry Pi + Coral USB / PCIe.
"""
import os
import time
import cv2
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common


# ───────── Configuración ───────── #
RTSP_URL = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'
# MODEL_PATH = 'best_clean_edgetpu.tflite'
MODEL_PATH = 'Prueba-int8_edgetpu.tflite'
SCORE_THRESHOLD = 0.25          # confianza mínima
SKIP_EVERY_N_FRAMES = 3         # procesa 1 de cada 3
WINDOW_NAME = 'YOLOv5 Coral'

# Ajustes de FFmpeg (RTSP por TCP + búfer grande)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000"

# ───────── Cargar modelo ───────── #
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_w, input_h = common.input_size(interpreter)  # (320, 320)

# Metadatos de cuantización para des-escalar salidas INT8
out_info = interpreter.get_output_details()[0]
out_scale, out_zp = out_info['quantization']

# ───────── Abrir stream ───────── #
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit('No se pudo abrir el stream RTSP.')

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# ───────── Bucle principal ───────── #
frame_count, start_time = 0, time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Error al leer frame. Reintentando…')
            time.sleep(1)
            continue

        frame_count += 1
        if frame_count % SKIP_EVERY_N_FRAMES:
            # saltar este frame para aligerar carga
            continue

        # --- Preparar entrada modelo ---
        frame_resized = cv2.resize(frame, (input_w, input_h))
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        common.set_input(interpreter, Image.fromarray(rgb))

        interpreter.invoke()

        # --- Leer detecciones ---
        det = interpreter.get_tensor(out_info['index'])[0].astype(np.int32)
        det = (det - out_zp) * out_scale        # float32 entre 0-1

        h_orig, w_orig = frame.shape[:2]

        for cx, cy, bw, bh, conf, cls_id in det:
            if conf < SCORE_THRESHOLD:
                continue

            # Convertir centro-ancho-alto a esquinas
            x0 = int((cx - bw / 2) * w_orig)
            y0 = int((cy - bh / 2) * h_orig)
            x1 = int((cx + bw / 2) * w_orig)
            y1 = int((cy + bh / 2) * h_orig)

            # Validar límites
            if x0 < 0 or y0 < 0 or x1 > w_orig or y1 > h_orig:
                continue

            # Dibujar
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f'Placa {conf:.2f}', (x0, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Mostrar FPS ---
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        # Salir con “q”
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
