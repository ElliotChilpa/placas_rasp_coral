#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Detección de placas con YOLOv5-EdgeTPU y OCR (EasyOCR).
Guarda un registro JSON-Lines con hora ISO-8601 y texto reconocido.
Compatible con Python 3.9.18.
"""

import os
import cv2
import json
import time
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from imutils.perspective import four_point_transform
from PIL import Image
import easyocr
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# ───────── Parámetros generales ───────── #
RTSP_URL        = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'
MODEL_PATH      = 'best_clean_edgetpu.tflite'
JSON_PATH       = Path('placas_detectadas.jsonl')        # formato JSON-Lines
ALLOWLIST       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
SKIP_EVERY_N    = 3         # procesa 1 de cada N frames
DUPLICATE_TOL_S = 5         # ventana para evitar duplicados
MARGIN_RATIO    = 0.10      # margen extra alrededor de la bbox
INIT_SCORE_TH   = 0.25
MIN_ROI_RATIO   = 0.015     # 1.5 % del ancho/alto de frame

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
    "rtsp_transport;tcp|buffer_size;2048000"

# ───────── Modelo Edge-TPU ───────── #
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
IN_W, IN_H = common.input_size(interpreter)
out0 = interpreter.get_output_details()[0]
SCALE, ZP = out0['quantization']

# ───────── OCR ───────── #
reader = easyocr.Reader(['en'], gpu=False)

# ───────── Funciones auxiliares ───────── #
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))

def enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    """CLAHE + bilateral + adaptive threshold + cierre morfológico."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blur = cv2.bilateralFilter(clahe, d=7, sigmaColor=75, sigmaSpace=75)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

def rectify_plate(crop: np.ndarray) -> np.ndarray:
    """Corrige perspectiva si encuentra un contorno cuadrilátero."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return four_point_transform(crop, approx.reshape(-1, 2))
    return crop

def easyocr_text(img_bw: np.ndarray) -> Optional[str]:
    """Ejecuta EasyOCR (imagen escalada ×2) y devuelve el mejor texto."""
    up = cv2.resize(img_bw, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    res = reader.readtext(up, detail=1, allowlist=ALLOWLIST, paragraph=False)
    if not res:
        return None
    best = max(res, key=lambda r: r[2])
    return best[1].replace(' ', '').upper() or None

def log_plate(fh, plate: str) -> None:
    entry = {'time': datetime.now().isoformat(timespec='seconds'),
             'plate': plate}
    fh.write(json.dumps(entry) + '\n')

# ───────── Inicialización ───────── #
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit('❌  No se pudo abrir el stream RTSP.')

cv2.namedWindow('Placas', cv2.WINDOW_NORMAL)

frame_i, t0 = 0, time.time()
seen: deque[Tuple[str, float]] = deque(maxlen=50)
score_th = INIT_SCORE_TH

with JSON_PATH.open('a', buffering=1) as json_file:  # auto-flush
    # ───────── Bucle principal ───────── #
    while True:
        ok, frm = cap.read()
        if not ok:
            print('⚠️  Frame drop'); time.sleep(1); continue

        frame_i += 1
        if frame_i % SKIP_EVERY_N:
            continue

        # ---- Inferencia Edge-TPU ----
        inp = cv2.resize(frm, (IN_W, IN_H))
        common.set_input(
            interpreter,
            Image.fromarray(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
        )
        interpreter.invoke()
        det = (interpreter.get_tensor(out0['index'])[0].astype(np.int16) - ZP) * SCALE
        H, W = frm.shape[:2]

        plates_found = 0
        for cx, cy, bw, bh, conf, _ in det:
            if conf < score_th:
                continue

            # BBox a píxeles + margen
            w, h = bw * W, bh * H
            x, y = cx * W, cy * H
            x0 = clamp(int(x - w / 2 - MARGIN_RATIO * w), 0, W - 1)
            y0 = clamp(int(y - h / 2 - MARGIN_RATIO * h), 0, H - 1)
            x1 = clamp(int(x + w / 2 + MARGIN_RATIO * w), 0, W - 1)
            y1 = clamp(int(y + h / 2 + MARGIN_RATIO * h), 0, H - 1)
            if x1 - x0 < MIN_ROI_RATIO * W or y1 - y0 < MIN_ROI_RATIO * H:
                continue

            roi = frm[y0:y1, x0:x1].copy()
            roi = rectify_plate(roi)
            bw_img = enhance_for_ocr(roi)
            txt = easyocr_text(bw_img)

            color = (0, 255, 0) if txt else (0, 0, 255)
            cv2.rectangle(frm, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frm, txt or '??', (x0, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if txt:
                recent = [p for p, t in seen
                          if p == txt and time.time() - t < DUPLICATE_TOL_S]
                if not recent:
                    log_plate(json_file, txt)
                    seen.append((txt, time.time()))
            plates_found += 1

        # Ajuste dinámico del umbral
        score_th = np.clip(
            score_th + (0.02 if plates_found > 3 else -0.01),
            0.15, 0.40
        )

        # FPS
        fps = frame_i / (time.time() - t0)
        cv2.putText(frm, f'{fps:.1f} FPS', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Placas', frm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ───────── Limpieza ───────── #
cap.release()
cv2.destroyAllWindows()
