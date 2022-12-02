import cv2
import numpy as np

from .detect_chars import detect_text, get_coordinates_cols
from .img_resizer import image_resize

# Load the pretrained model
from tensorflow.keras.models import load_model
model = load_model('final_codebase/models/model_cnn4.h5')

class_names = ['add', 'divide', 'eight', 'five', 'four', 'multiply', 'nine', 
               'one', 'seven', 'six', 'subtract', 'three', 'two', 'zero']

def __detect_chars(image):
    "Use a pretrained model to classify the image"
    img = image.copy()
    
    # If the image is not in grayscale, convert it to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img.shape[0] > img.shape[1]:
        diff = img.shape[0] - img.shape[1]
        left = diff // 2
        right = diff - left

        img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=255)

    elif img.shape[1] > img.shape[0]:
        diff = img.shape[1] - img.shape[0]
        top = diff // 2
        bottom = diff - top

        img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=255)

    _, img = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)
    
    img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, 150, 150, 1)
    img = img / 255

    classes = model.predict(img, verbose=0)

    class_index = np.argmax(classes)
    
    probability = classes[0][class_index]

    return (class_names[class_index], probability)


map_dict = {'one':'1', 'two':'2', 'three':'3', 'four':'4', 'five':'5', 
            'six':'6', 'seven':'7', 'eight':'8', 'nine':'9', 'zero':'0', 
            'multiply':'*', 'divide':'/', 'add':'+', 'subtract':'-'}

def detect_cnn(img_to_work):
    eq_img = img_to_work.copy()
    eq_img = image_resize(eq_img, width=1000)

    out_box, out_text = detect_text(eq_img)

    out_cords = get_coordinates_cols(out_text)

    final = []
    for i in out_cords:
        final.append(__detect_chars(eq_img[i[2]:i[3], i[0]:i[1]]))
    
    output = ''
    for val, prob in final:
        output = output+map_dict[val]
        
    return out_cords, output