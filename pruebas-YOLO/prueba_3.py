import os, cv2, time, numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from PIL import Image

# ───────── Configuración básica ───────── #
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;2048000"
RTSP_URL  = 'rtsp://admin:Chaparrito10@192.168.0.3:554/h264Preview_01_sub'
MODEL     = 'best_clean_edgetpu.tflite'             # nombre exacto
CONF_TH   = 0.25                                         # confianza mínima
NMS_TH    = 0.45                                         # IoU para NMS

# ───────── Cargar modelo ───────── #
interpreter = make_interpreter(MODEL)
interpreter.allocate_tensors()
input_size  = common.input_size(interpreter)             # (320,320)
out_det     = interpreter.get_output_details()[0]
out_scale, out_zp = out_det['quantization']              # p.ej (0.027f, 0)

# ───────── Stream RTSP ───────── #
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el stream RTSP")

cv2.namedWindow('YOLOv5 Coral', cv2.WINDOW_NORMAL)
frame_cnt, t0 = 0, time.time()

# ───────── Bucle principal ───────── #
while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame vacío, reconectando…"); time.sleep(1); continue
    frame_cnt += 1
    if frame_cnt % 3:                      # procesar 1 de cada 3 frames
        cv2.imshow('YOLOv5 Coral', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # ---------- Inferencia ----------
    rgb   = cv2.cvtColor(cv2.resize(frame, input_size), cv2.COLOR_BGR2RGB)
    common.set_input(interpreter, Image.fromarray(rgb))
    interpreter.invoke()
    preds = interpreter.get_tensor(out_det['index'])[0]

    # Des-cuantizar INT8 → float32
    if out_det['dtype'] == np.int8:
        preds = (preds.astype(np.float32) - out_zp) * out_scale

    # ---------- Filtrar + NMS ----------
    # cx,cy,w,h,obj,cls_conf
    bboxes, scores = [], []
    for cx, cy, w, h, obj, cls in preds:
        conf = obj * cls
        if conf < CONF_TH: continue
        bboxes.append([(cx - w/2), (cy - h/2), (cx + w/2), (cy + h/2)])
        scores.append(float(conf))

    if bboxes:
        idxs = cv2.dnn.NMSBoxes(bboxes, scores, CONF_TH, NMS_TH, top_k=1000)
        h_i, w_i = frame.shape[:2]
        for i in idxs.flatten():
            x0, y0, x1, y1 = bboxes[i]
            x0, y0 = int(x0 * w_i), int(y0 * h_i)
            x1, y1 = int(x1 * w_i), int(y1 * h_i)
            label = f'Placa {scores[i]:.2f}'
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), 2)
            cv2.putText(frame, label, (x0, y0-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ---------- FPS ----------
    fps = frame_cnt / (time.time() - t0)
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.imshow('YOLOv5 Coral', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
