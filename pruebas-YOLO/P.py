#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, json, time, threading, queue, numpy as np
from datetime import datetime
from collections import defaultdict, deque
from PIL import Image
from imutils.perspective import four_point_transform
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
import easyocr

# ───────── Cámaras ───────── #
CAMERAS = [
    {"name": "entrada",
     "url":  "rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub"},
    {"name": "salida",
     "url":  "rtsp://admin:Chaparrito10@192.168.0.4:554/h264Preview_01_sub"},
]

# ───────── Parámetros ───────── #
MODEL_PATH, INP_SIZE = 'best_clean_edgetpu.tflite', 320
SCORE_TH, SKIP_INFER = 0.30, 4
ALLOWLIST, MIN_W, MIN_H = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 8, 8
REPEAT_N, REPEAT_WINDOW_S = 3, 10
COOLDOWN_S             = 5
MAX_ROI_QUEUE          = 20
DISCARD_AFTER_S        = 1.0

# ───────── Low-latency RTSP ───────── #
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer"

RUNNING = threading.Event(); RUNNING.set()
PREVIEW = threading.Event(); PREVIEW.set()
frames_lock, last_frames = threading.Lock(), {}

infer_queue = queue.Queue(maxsize=60)
ocr_queue   = queue.Queue(maxsize=MAX_ROI_QUEUE)

plate_hits  = defaultdict(deque)
last_saved  = defaultdict(dict)          # gate → {plate: last_time}

# ═════════ UTILIDADES ═════════ #
def rectify(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cnts,_ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx) == 4:
            bgr = four_point_transform(bgr, approx.reshape(-1, 2))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(g, 255,
             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 15)

def save_json(rec):
    """
    Guarda **solo el último registro** en entrada.json o salida.json.
    """
    fname = 'entrada.json' if rec['gate'] == 'entrada' else 'salida.json'
    with open(fname, 'w', encoding='utf8') as f:
        json.dump([rec], f, indent=2, ensure_ascii=False)   # ← sobrescribe

# ═════════ OCR WORKERS ═════════ #
def ocr_worker():
    reader = easyocr.Reader(['es'], gpu=False)
    while RUNNING.is_set() or not ocr_queue.empty():
        try:
            roi, gate, ts = ocr_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if roi is None:
            break
        if time.time() - ts > DISCARD_AFTER_S:
            ocr_queue.task_done(); continue

        txt = ''
        try:
            proc = rectify(roi)
            r = reader.readtext(proc, allowlist=ALLOWLIST,
                                 detail=1, paragraph=False)
            if r:
                txt = ''.join(ch for ch in max(r, key=lambda x: x[2])[1]
                              if ch.isalnum())
        except Exception:
            pass

        if txt:
            txt = txt.upper()
            if time.time() - last_saved[gate].get(txt, 0) < COOLDOWN_S:
                ocr_queue.task_done(); continue

            now = time.time(); key = (gate, txt); dq = plate_hits[key]
            dq.append(now)
            while dq and now - dq[0] > REPEAT_WINDOW_S:
                dq.popleft()

            if len(dq) == REPEAT_N:
                rec = {"time": datetime.now().isoformat(timespec='seconds'),
                       "plate": txt,
                       "gate": gate}
                save_json(rec)          # ← escribe el archivo correcto
                print('✅', rec)
                last_saved[gate][txt] = now
                dq.clear()
        ocr_queue.task_done()

for _ in range(3):
    threading.Thread(target=ocr_worker, daemon=True).start()

# ═════════ TPU WORKER ═════════ #
def tpu_worker():
    inter = make_interpreter(MODEL_PATH); inter.allocate_tensors()
    outs  = inter.get_output_details()
    multi = len(outs)>=4
    if not multi: scale,zp = outs[0]['quantization']

    while RUNNING.is_set() or not infer_queue.empty():
        try: frame, cam = infer_queue.get(timeout=0.2)
        except queue.Empty: continue
        if frame is None: break

        inp=cv2.resize(frame,(INP_SIZE,INP_SIZE))
        common.set_input(inter,Image.fromarray(
                         cv2.cvtColor(inp,cv2.COLOR_BGR2RGB)))
        inter.invoke()

        h,w = frame.shape[:2]; rois=[]
        if multi:
            sx,sy=w/INP_SIZE,h/INP_SIZE
            for o in detect.get_objects(inter,SCORE_TH):
                x0=int(o.bbox.xmin*sx); y0=int(o.bbox.ymin*sy)
                x1=int(o.bbox.xmax*sx); y1=int(o.bbox.ymax*sy)
                if (x1-x0)>=MIN_W and (y1-y0)>=MIN_H: rois.append((x0,y0,x1,y1))
        else:
            det=inter.get_tensor(outs[0]['index'])[0].astype(np.int32)
            det=(det-zp)*scale
            for cx,cy,bw,bh,conf,_ in det:
                if conf<SCORE_TH: continue
                x0=int((cx-bw/2)*w); y0=int((cy-bh/2)*h)
                x1=int((cx+bw/2)*w); y1=int((cy+bh/2)*h)
                if (x1-x0)>=MIN_W and (y1-y0)>=MIN_H: rois.append((x0,y0,x1,y1))

        ts=time.time()                                  
        for x0,y0,x1,y1 in rois:
            if ocr_queue.full(): break
            ocr_queue.put((frame[y0:y1,x0:x1].copy(), cam, ts))  
            cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),2)

        with frames_lock: last_frames[cam]=frame
        infer_queue.task_done()

threading.Thread(target=tpu_worker,daemon=True).start()

# ═════════ CAPTURE THREADS ═════════ #
def open_cap(url):
    cap=cv2.VideoCapture(url,cv2.CAP_FFMPEG)
    if cap.isOpened():
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        except cv2.error: pass
    return cap

def capture_loop(name,url):
    cap=None; frame_cnt=0; last_ok=0
    while RUNNING.is_set():
        if cap is None or not cap.isOpened():
            if cap: cap.release()
            cap=open_cap(url)
            if not cap.isOpened():
                print(f'[{name}] sin stream…'); time.sleep(2); continue
            last_ok=time.time()

        ok,frame=cap.read()
        if not ok:
            if time.time()-last_ok>2: cap.release(); cap=None
            continue
        last_ok=time.time(); frame_cnt+=1

        if (frame_cnt%SKIP_INFER)==0 and not infer_queue.full():
            infer_queue.put((frame.copy(),name))

        with frames_lock:
            if name not in last_frames: last_frames[name]=frame.copy()
    if cap: cap.release()

for cam in CAMERAS:
    threading.Thread(target=capture_loop,
                     args=(cam['name'],cam['url']),daemon=True).start()

# ═════════ UI PRINCIPAL ═════════ #
open_w=set()
cv2.namedWindow('control'); cv2.imshow('control',np.zeros((30,300),np.uint8))

try:
    while RUNNING.is_set():
        k=cv2.waitKey(25)&0xFF
        if k==ord('v'): PREVIEW.clear() if PREVIEW.is_set() else PREVIEW.set()
        elif k==ord('q'): RUNNING.clear()

        if not PREVIEW.is_set():
            for w in list(open_w):
                try: cv2.destroyWindow(w)
                except cv2.error: pass
                open_w.discard(w)
            continue

        with frames_lock:
            for cam in CAMERAS:
                n=cam['name']
                if n in last_frames:
                    cv2.imshow(n,last_frames[n]); open_w.add(n)

finally:
    RUNNING.clear()
    for q in (infer_queue,ocr_queue): q.put(None)
    cv2.destroyAllWindows()