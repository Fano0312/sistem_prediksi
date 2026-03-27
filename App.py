import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# 1. Konfigurasi Path Model
# Jika di laptop, pastikan file .h5 ada di folder yang sama dengan app.py
MODEL_PATH = 'fashion_model.h5'

# Memuat model AI
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat!")
else:
    print("❌ Model tidak ditemukan! Pastikan file fashion_model.h5 ada di folder yang sama.")

# Definisi Label Fashion MNIST sesuai urutan kelas
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if file:
                # Preprocessing Gambar: Grayscale, resize 28x28, dan normalisasi
                img = Image.open(file).convert('L').resize((28, 28))
                img_array = np.array(img) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                # Melakukan Prediksi menggunakan model
                preds = model.predict(img_array)
                prediction = classes[np.argmax(preds)]
        except Exception as e:
            prediction = f"Error Prediksi: {e}"
            
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    # Menggunakan port dari environment (penting untuk hosting seperti Vercel/Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)