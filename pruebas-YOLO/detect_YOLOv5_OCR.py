#!/usr/bin/env python3
# -- coding: utf-8 --
"""
YOLOv5-EdgeTPU + OCR de placas con corrección de perspectiva y
umbralización adaptativa. Escribe un JSON con hora y caracteres.

Requisitos extra:
    pip install easyocr imutils
"""
import os
import cv2
import json
import time
import numpy as np
from datetime import datetime
from PIL import Image
from collections import deque
from imutils.perspective import four_point_transform
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import easyocr

# ───────── Configuración ───────── #
RTSP_URL = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'
MODEL_PATH = 'best_clean_edgetpu.tflite'
JSON_LOG = 'placas_detectadas.json'
SCORE_TH = 0.25
SKIP_EVERY_N_FRAMES = 3
ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
DUPLICATE_SECONDS = 5
MIN_W, MIN_H = 10, 10          # área mínima del ROI para OCR

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;2048000"
)

# ───────── Cargar modelo Edge-TPU ───────── #
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
in_w, in_h = common.input_size(interpreter)
out_info = interpreter.get_output_details()[0]
out_scale, out_zp = out_info['quantization']

# ───────── OCR ───────── #
reader = easyocr.Reader(['en'], gpu=False)

# ───────── Utilidades ───────── #
def rectify_and_threshold(crop):
    """Corrige perspectiva (si hay contorno rectangular) y umbraliza."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            crop = four_point_transform(crop, approx.reshape(-1, 2))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 21, 15)
    return thr

def ocr_plate(img):
    """Ejecuta EasyOCR y devuelve el texto con mayor confianza."""
    result = reader.readtext(img, detail=1,
                             allowlist=ALLOWLIST, paragraph=False)
    if not result:
        return None
    best = max(result, key=lambda r: r[2])
    return best[1].replace(' ', '').upper() or None

def save_json(record):
    """Agrega un registro (dict) a JSON persistente."""
    data = []
    if os.path.isfile(JSON_LOG):
        try:
            with open(JSON_LOG) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    data.append(record)
    with open(JSON_LOG, 'w') as f:
        json.dump(data, f, indent=2)

# ───────── Abrir stream ───────── #
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit('❌  No se pudo abrir el stream RTSP.')

WIN = 'YOLOv5 Coral'
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

frame_count, start_time = 0, time.time()
last_seen = deque(maxlen=20)   # [(texto, timestamp)]

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print('⚠️  Error al leer frame; reintentando…')
            time.sleep(1)
            continue

        frame_count += 1
        if frame_count % SKIP_EVERY_N_FRAMES:
            continue

        # ---- Inferencia Edge-TPU ----
        inp = cv2.resize(frame, (in_w, in_h))
        common.set_input(interpreter,
                         Image.fromarray(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)))
        interpreter.invoke()

        det = interpreter.get_tensor(out_info['index'])[0].astype(np.int32)
        det = (det - out_zp) * out_scale  # float32 0-1

        h0, w0 = frame.shape[:2]

        for cx, cy, bw, bh, conf, cls_id in det:
            if conf < SCORE_TH:
                continue

            # Convertir centro → esquinas y clamp
            x0 = int((cx - bw/2) * w0)
            y0 = int((cy - bh/2) * h0)
            x1 = int((cx + bw/2) * w0)
            y1 = int((cy + bh/2) * h0)
            x0 = max(0, min(x0, w0-1))
            y0 = max(0, min(y0, h0-1))
            x1 = max(0, min(x1, w0-1))
            y1 = max(0, min(y1, h0-1))

            # ---- NUEVO: descartar cajas pequeñas o sin área ----
            if x1 - x0 < MIN_W or y1 - y0 < MIN_H:
                continue

            crop = frame[y0:y1, x0:x1].copy()
            if crop.size == 0:
                continue

            proc = rectify_and_threshold(crop)
            text = ocr_plate(proc)

            now_iso = datetime.now().isoformat(timespec='seconds')
            label = f'{text or "??"} {conf:.2f}'

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), 2)
            cv2.putText(frame, label, (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # ---- Guardar JSON evitando duplicados ----
            if text:
                recent = [t for t, ts in last_seen
                          if t == text and time.time() - ts < DUPLICATE_SECONDS]
                if not recent:
                    record = {'time': now_iso, 'plate': text}
                    save_json(record)
                    last_seen.append((text, time.time()))
                    print('🔸', record)

        # FPS
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow(WIN, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
