from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import cv2

model = load_model("./Model/luppa.h5")
labels = ["Soil", "Tree"]
img_name = None

def decode_image_from_bytes(data)->str:
    """
    Executes string decode to image binary format.
    
    Parameters
    ----------
    data: str
        Image formatted in string
    """
    image = np.fromstring(data, np.uint8)
    image_cv = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image_cv

def get_image_from_base64(base64_data: str):
    """Convert base64 img to uint8.

    Parameters
    ----------
    base64_data: str
        Image formatted in base64.
    """
    nparr = base64.b64decode(base64_data)
    return decode_image_from_bytes(nparr)

def inference(data: str) -> str:
    """Run inference on image and return predictions.

    Parameters
    ----------
    data: str
        Image formatted in string
    """
    img = get_image_from_base64(data)
    global labels
    img = cv2.imread(img)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    result = np.argmax(result)
    return "Imagem: %s" %labels[result]