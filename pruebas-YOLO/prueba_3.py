import os, cv2, time, numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from PIL import Image

# ── Configuración ──
RTSP_URL = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'
MODEL    = 'best_clean_edgetpu.tflite'
CONF_TH, NMS_TH = 0.25, 0.45
PROCESS_N = 3                  # procesa 1 de cada N frames

# ── Cargar modelo ──
inter = make_interpreter(MODEL); inter.allocate_tensors()
IMG_SZ = common.input_size(inter)          # (320, 320)
o_det  = inter.get_output_details()[0]
scale, zp = o_det['quantization']


cv2.namedWindow('YOLOv5 Coral', cv2.WINDOW_NORMAL)
f_cnt, t0 = 0, time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print('Frame vacío…'); time.sleep(1); continue
    f_cnt += 1

    if f_cnt % PROCESS_N:                 # saltar inferencia pero mostrar
        cv2.imshow('YOLOv5 Coral', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # ── Inferencia ──
    rgb = cv2.cvtColor(cv2.resize(frame, IMG_SZ), cv2.COLOR_BGR2RGB)
    common.set_input(inter, Image.fromarray(rgb)); inter.invoke()
    preds = inter.get_tensor(o_det['index'])[0]
    if o_det['dtype'] == np.int8:
        preds = (preds.astype(np.float32) - zp) * scale

    H, W = frame.shape[:2]
    boxes, scores = [], []

    # cx, cy, w, h, obj, cls_conf
    for cx, cy, w, h, obj, cls in preds:
        conf = obj * cls
        if conf < CONF_TH: continue
        boxes.append([cx - w/2, cy - h/2, w, h])   # x, y, w, h  (0-1)
        scores.append(float(conf))

    # ── NMS y dibujado ──
    if boxes:
        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_TH, NMS_TH)
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            x0, y0 = int(x * W), int(y * H)
            x1, y1 = int((x + w) * W), int((y + h) * H)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), 2)
            cv2.putText(frame, f'{scores[i]:.2f}', (x0, y0-6),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)

    fps = f_cnt / (time.time() - t0)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,0), 2)
    cv2.imshow('YOLOv5 Coral', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()
