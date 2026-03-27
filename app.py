import os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Load model TFLite
MODEL_PATH = "fashion_model.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if file:
                img = Image.open(file).convert('L').resize((28, 28))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]['index'])

                prediction = classes[np.argmax(preds)]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)
