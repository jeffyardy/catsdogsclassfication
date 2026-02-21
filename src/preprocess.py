
import numpy as np
from PIL import Image

def load_and_resize(image_bytes, size=(224,224)):
    img = Image.open(image_bytes).convert("RGB")
    img = img.resize(size)
    arr = np.array(img)/255.0
    return arr.astype("float32")

def batch_stack(images):
    return np.stack(images, axis=0)
