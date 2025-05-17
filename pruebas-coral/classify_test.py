from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import classify
from pycoral.adapters import common
from PIL import Image

# Rutas del modelo, etiquetas y la imagen
MODEL = 'mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite'
LABELS = 'inat_bird_labels.txt'
IMAGE = 'parrot.jpg'

# Leer las etiquetas
def read_label_file(file_path):
    with open(file_path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = read_label_file(LABELS)

# Configurar el intérprete con el modelo
interpreter = make_interpreter(MODEL)
interpreter.allocate_tensors()

# Preparar la imagen
from PIL import Image, ImageOps

image = Image.open(IMAGE).convert('RGB')
image = ImageOps.fit(image, common.input_size(interpreter), Image.Resampling.LANCZOS)

common.set_input(interpreter, image)

# Ejecutar la inferencia
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Mostrar los resultados
for c in classes:
    print(f"Resultado: {labels.get(c.id, c.id)} - Score: {c.score:.5f}")
