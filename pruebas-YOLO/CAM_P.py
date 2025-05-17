#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, cv2, json, platform, numpy as np
from collections import deque
from datetime import datetime
import easyocr


TFLITE_MODEL = "best_clean_edgetpu.tflite"   # modelo entrrenado 
RTSP_URL     = "rtsp://admin:Chaparrito10@192.168.0.4:554/h264Preview_01_main"
OUTPUT_DIR   = "Output"
JSONL_PATH   = os.path.join(OUTPUT_DIR, "placas_detectadas.jsonl")
SCORE_TH     = 0.25                                     # umbral confianza

# ───────────────────────── Carga modelo Edge TPU ───────────────────── #
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:                                     # fallback si falta wheel
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate

delegates = []
if platform.system() != "Windows":                      # TPU no soportado en Win
    try:
        delegates.append(load_delegate("libedgetpu.so.1"))
    except ValueError:
        print("Coral no encontrado → usando CPU")

interpreter = Interpreter(TFLITE_MODEL,
                          experimental_delegates=delegates)
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
IMG_SZ  = in_det['shape'][2]

# ───────────────────────── Funciones auxiliares ────────────────────── #
def preproc_bgr(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SZ, IMG_SZ))
    return np.expand_dims(resized.astype(np.uint8), 0)

def infer_yolo(img):
    h, w = img.shape[:2]
    interpreter.set_tensor(in_det['index'], preproc_bgr(img))
    interpreter.invoke()
    preds = interpreter.get_tensor(out_det['index'])[0]     # (N, 6)
    boxes = []
    for cx, cy, bw, bh, obj, cls in preds:
        conf = obj * cls
        if conf < SCORE_TH: continue
        x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
        x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
        boxes.append((x1, y1, x2, y2))
    return boxes

def prep_gray(g):
    den  = cv2.bilateralFilter(g, 11, 17, 17)
    res  = cv2.resize(den, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.equalizeHist(res)

reader  = easyocr.Reader(['en'], gpu=False)
deque_pl = deque(maxlen=5)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────────────────────────── Loop principal ──────────────────────── #
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara RTSP")

def log_json(hora, placa):
    registro = {"hora": hora, "placa": placa}
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    print(json.dumps(registro, ensure_ascii=False))

paused = False
while True:
    if not paused:
        ok, frame = cap.read()
        if not ok: break
        for (x1, y1, x2, y2) in infer_yolo(frame):
            roi = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            if roi.size == 0: continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            proc = prep_gray(gray)
            res  = reader.readtext(proc)
            if res:
                placa = "".join(filter(str.isalnum, max(res, key=lambda r:r[2])[1])).strip()
                if placa and placa not in deque_pl:
                    deque_pl.appendleft(placa)
                    hora = datetime.now().isoformat(timespec="seconds")
                    log_json(hora, placa)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imshow("LPR EdgeTPU", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('p'): paused = not paused
    if k == ord('q'): break

cap.release(); cv2.destroyAllWindows()
