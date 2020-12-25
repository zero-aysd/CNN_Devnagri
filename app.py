from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import base64
import io
from tensorflow import keras
import pandas as pd

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_image', methods=['GET', 'POST'])
def get_image():
    print('Incoming...')
    image_string = request.get_json()
    print('image recieved')
    print('.............................................................................')
    im = Image.open(io.BytesIO(base64.b64decode(image_string['jsontext'].split(',')[1])))
    im.save('test_image.png', 'PNG')
    digit, label = predict('./test_image.png')
    label2 = devanagri_label(label)
    return label2, 200


keras.backend.clear_session()
model = keras.models.load_model('devanagari_minst.h5')


def predict(image_path):
    image_tensor = keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    input_arr = np.array([keras.preprocessing.image.img_to_array(image_tensor)])
    input_arr = input_arr / 255
    res2 = model.predict(input_arr)
    return res2, np.argmax(res2)


with open('class_labels.json', 'r') as f:
    import json
    class_labels = json.load(f)

# inverting class labels dict

inv_class_labels = {v: k for k, v in class_labels.items()}
labels = pd.read_csv('./archive/labels.csv')


def devanagri_label(label):
    devanagri = labels.at[int(inv_class_labels.get(label)), "Devanagari label"]
    return devanagri





if __name__ == '__main__':
    app.run(debug=True)
