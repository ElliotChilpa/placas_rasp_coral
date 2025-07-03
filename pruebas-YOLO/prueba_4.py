#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge-TPU + EasyOCR  —  registra la placa sólo cuando se lee ≥ 3 veces
• Entrada 320×320 — se descarta guiones; sólo A-Z, 0-9
• Procesa 1 de cada SKIP_FRAMES cuadros (fluido)
• No guarda imágenes ni CSV; sólo escribe en placas_detectadas.json
• Ventana opcional (HEADLESS = True la desactiva); tecla v oculta/muestra
"""

import os, cv2, json, time, sys, threading, queue, numpy as np
from datetime import datetime
from collections import deque, defaultdict
from PIL import Image
from imutils.perspective import four_point_transform
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import easyocr

# ───────── Parámetros editables ───────── #
RTSP_URL     = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'
MODEL_PATH   = 'best_clean_edgetpu.tflite'
INP_SIZE     = 320          # redimensiona 320×320
SKIP_FRAMES  = 4            # procesa 25 % de cuadros
HEADLESS     = False

# ───────── Umbrales / nombres coherentes ───────── #
SCORE_TH          = 0.35
ALLOWLIST         = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
MIN_W, MIN_H      = 8, 8
REPEAT_N          = 5
REPEAT_WINDOW_S   = 8
JSON_LOG          = 'placas_detectadas.json'
WIN_NAME          = 'Placas Coral'

# ───────── Entorno OpenCV RTSP ───────── #
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;2048000"
)

# ───────── Modelo Edge-TPU ───────── #
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
in_w = in_h = INP_SIZE
out_det = interpreter.get_output_details()[0]
scale, zp = out_det['quantization']

# ───────── OCR hilo separado ───────── #
reader = easyocr.Reader(['es'], gpu=False)
ocr_queue = queue.Queue(maxsize=30)
plate_hits = defaultdict(deque)   # texto → deque([timestamps])

def rectify(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx) == 4:
            bgr = four_point_transform(bgr, approx.reshape(-1, 2))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(g, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 15)

# ───────── MODIFICACIÓN CLAVE ───────── #
def save_json(rec):
    """
    Sobrescribe placas_detectadas.json con un solo registro (el más reciente).
    """
    with open(JSON_LOG, 'w') as f:
        json.dump([rec], f, indent=2)

def ocr_worker():
    while True:
        roi = ocr_queue.get()
        if roi is None:
            break
        proc = rectify(roi)
        res = reader.readtext(proc, allowlist=ALLOWLIST,
                              detail=1, paragraph=False)
        if res:
            _, raw, _ = max(res, key=lambda r: r[2])
            plate = ''.join(ch for ch in raw if ch.isalnum()).upper()
            if plate:
                now = time.time()
                dq = plate_hits[plate]
                dq.append(now)
                while dq and now - dq[0] > REPEAT_WINDOW_S:
                    dq.popleft()
                if len(dq) == REPEAT_N:         # se repitió 3 veces
                    rec = {'time': datetime.now().isoformat(timespec='seconds'),
                           'plate': plate}
                    save_json(rec)              # ← siempre deja un solo registro
                    print('✅ confirmada:', rec)
                    dq.clear()
        ocr_queue.task_done()

threading.Thread(target=ocr_worker, daemon=True).start()

# ───────── Captura RTSP ───────── #
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit('❌ No se pudo abrir el stream RTSP')

show = not HEADLESS
if show:
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

frame_cnt, t0 = 0, time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.3)
            continue

        frame_cnt += 1
        if frame_cnt % SKIP_FRAMES and not show:
            continue

        # --- Edge-TPU ---
        inp = cv2.resize(frame, (in_w, in_h))
        common.set_input(interpreter,
                         Image.fromarray(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)))
        interpreter.invoke()
        det = interpreter.get_tensor(out_det['index'])[0].astype(np.int32)
        det = (det - zp) * scale

        h, w = frame.shape[:2]
        for cx, cy, bw, bh, conf, _ in det:
            if conf < SCORE_TH:
                continue
            x0 = int((cx - bw/2) * w); y0 = int((cy - bh/2) * h)
            x1 = int((cx + bw/2) * w); y1 = int((cy + bh/2) * h)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            if x1 - x0 < MIN_W or y1 - y0 < MIN_H:
                continue

            # envía recorte al hilo OCR sin bloquear
            try:
                ocr_queue.put_nowait(frame[y0:y1, x0:x1].copy())
            except queue.Full:
                pass

            if show:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # -------- GUI --------
        if show:
            fps = frame_cnt / (time.time() - t0)
            cv2.putText(frame, f'FPS:{fps:.1f}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow(WIN_NAME, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('v'):
                show = False
                cv2.destroyWindow(WIN_NAME)

finally:
    cap.release()
    if show:
        cv2.destroyAllWindows()
    ocr_queue.put(None)           # detiene hilo OCR
    ocr_queue.join()
