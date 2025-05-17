from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
from PIL import Image, ImageDraw

MODEL = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
LABELS = 'coco_labels.txt'
IMAGE = 'grace_hopper.bmp'

def read_label_file(file_path):
    with open(file_path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = read_label_file(LABELS)

# Configurar el intérprete
interpreter = make_interpreter(MODEL)
interpreter.allocate_tensors()

# Cargar y preparar la imagen
image = Image.open(IMAGE)
image_resized = image.resize(common.input_size(interpreter), Image.Resampling.LANCZOS)
common.set_input(interpreter, image_resized)

# Ejecutar inferencia
interpreter.invoke()
objs = detect.get_objects(interpreter, score_threshold=0.5)

# Dibujar los resultados
draw = ImageDraw.Draw(image)
for obj in objs:
    bbox = obj.bbox
    label = labels.get(obj.id, obj.id)
    score = obj.score
    print(f'Detectado: {label} con confianza {score:.2f}')
    draw.rectangle([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], outline='red')
    draw.text((bbox.xmin, bbox.ymin), f'{label}: {score:.2f}', fill='red')

# Guardar la imagen con las detecciones
image.save('resultado_deteccion.jpg')
print('Resultado guardado como resultado_deteccion.jpg')
