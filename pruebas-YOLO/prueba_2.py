#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Edge-TPU + EasyOCR   — logea la placa *solo cuando la ha visto ≥ 3 veces seguidas*
* Analiza 1 de cada 4 cuadros (más denso que antes)
* No guarda imágenes ni CSV; solo actualiza JSON
* Solo caracteres A-Z y 0-9 (sin guiones/espacios)
"""

import os, cv2, json, time, sys, numpy as np
from datetime import datetime
from collections import deque, defaultdict
from PIL import Image
from imutils.perspective import four_point_transform
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import easyocr

# ───────── Configuración básica ───────── #
RTSP_URL    = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'
MODEL_PATH  = 'best_clean_edgetpu.tflite'
SKIP_FRAMES = 4                 # procesa 25 % de los cuadros
HEADLESS    = False
JSON_LOG    = 'placas_detectadas.json'

# ───────── Umbrales / tamaños ───────── #
SCORE_TH          = 0.35
ALLOWLIST         = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
MIN_W, MIN_H      = 8, 8
INP_SIZE          = 320          # redimensionado para Edge-TPU
REPEAT_N          = 3            # ← mínimo de repeticiones requeridas
REPEAT_WINDOW_S   = 10           # las 3 lecturas deben ocurrir dentro de 10 s

# ───────── Inicialización ───────── #
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;2048000"
)

interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
in_w = in_h = INP_SIZE
out_det = interpreter.get_output_details()[0]
scale, zp = out_det['quantization']

reader = easyocr.Reader(['es'], gpu=False)

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit('❌ No se pudo abrir el stream RTSP')

show = not HEADLESS
if show:
    cv2.namedWindow('Placas', cv2.WINDOW_NORMAL)

frame_cnt, t0 = 0, time.time()
plate_hits = defaultdict(deque)      # texto → deque([timestamp,…])

# ───────── Ayudantes ───────── #
def rectify(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx) == 4:
            bgr = four_point_transform(bgr, approx.reshape(-1, 2))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,21,15)

def ocr(roi):
    txts = reader.readtext(roi, allowlist=ALLOWLIST, detail=1, paragraph=False)
    if not txts:
        return None
    _, raw, _ = max(txts, key=lambda r: r[2])
    clean = ''.join(ch for ch in raw if ch.isalnum()).upper()
    return clean or None

def save_json(record):
    try:
        data = json.load(open(JSON_LOG))
    except Exception:
        data = []
    data.append(record)
    json.dump(data, open(JSON_LOG,'w'), indent=2)

# ───────── Bucle principal ───────── #
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.3); continue

        frame_cnt += 1
        if frame_cnt % SKIP_FRAMES and show is False:
            continue          # en headless salta el dibujado

        # Edge-TPU
        inp = cv2.resize(frame, (in_w, in_h))
        common.set_input(interpreter,
                         Image.fromarray(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)))
        interpreter.invoke()
        det = interpreter.get_tensor(out_det['index'])[0].astype(np.int32)
        det = (det - zp) * scale

        h, w = frame.shape[:2]
        now = time.time()

        for cx, cy, bw, bh, conf, _ in det:
            if conf < SCORE_TH:  continue
            x0 = int((cx - bw/2)*w); y0 = int((cy - bh/2)*h)
            x1 = int((cx + bw/2)*w); y1 = int((cy + bh/2)*h)
            x0,y0 = max(0,x0), max(0,y0)
            x1,y1 = min(w-1,x1), min(h-1,y1)
            if x1-x0 < MIN_W or y1-y0 < MIN_H:  continue

            roi = frame[y0:y1, x0:x1]
            plate = ocr(rectify(roi))
            if not plate:      continue

            hits = plate_hits[plate]
            hits.append(now)
            # descarta lecturas fuera de ventana
            while hits and now - hits[0] > REPEAT_WINDOW_S:
                hits.popleft()

            # ¿se alcanzó el umbral de repeticiones?
            if len(hits) == REPEAT_N:
                record = {'time': datetime.now().isoformat(timespec='seconds'),
                          'plate': plate}
                save_json(record)
                print('✅ placas confirmada:', record)
                hits.clear()   # evita múltiples registros seguidos

            if show:
                lbl = f'{plate} {conf:.2f}'
                cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),2)
                cv2.putText(frame,lbl,(x0,y0-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        if show:
            fps = frame_cnt/(time.time()-t0)
            cv2.putText(frame,f'FPS:{fps:.1f}',(10,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
            cv2.imshow('Placas', frame)
            k = cv2.waitKey(1) & 0xFF
            if k==ord('q'): break
            elif k==ord('v'):
                show=False
                cv2.destroyWindow('Placas')

finally:
    cap.release()
    if show: cv2.destroyAllWindows()