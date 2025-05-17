from tflite_runtime.interpreter import Interpreter, load_delegate
import numpy as np
import time
import os

MODEL_PATH = "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"

def check_coral_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modelo no encontrado: {MODEL_PATH}")
        return

    try:
        interpreter = Interpreter(
            model_path=MODEL_PATH,
            experimental_delegates=[load_delegate("libedgetpu.so.1")]
        )
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()

        dummy_input = np.ones(input_details[0]['shape'], dtype=input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], dummy_input)

        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()

        print("✅ Coral Accelerator detectado y modelo ejecutado.")
        print(f"🕒 Tiempo de inferencia: {t1 - t0:.4f} segundos")
    except Exception as e:
        print("❌ Error al usar Coral.")
        print("Detalles:", e)

if __name__ == "__main__":
    check_coral_model()
