import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
model = load_model(r"tumor.h5", compile=False)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['images']

        # Use BytesIO to handle the image in memory
        img = Image.open(io.BytesIO(f.read()))
        img = img.resize((224, 224))  # Resize the image to match the model's expected input size

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x), axis=1)
        index = ['No', 'Yes']

        text = "Having Brain Tumor: " + str(index[pred[0]])
        return text

if __name__ == '__main__':
    app.run(debug=True)
