from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your model
MODEL_PATH = 'MCAR.keras'  
model = load_model(MODEL_PATH)

# Read class names from text file
with open('classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    if file:
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Preprocess the image (adjust target_size to your model's input shape)
        img = image.load_img(filepath, target_size=(224, 224))  # adjust as needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize if needed

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('index.html', prediction=predicted_class, image_url=filepath)

if __name__ == '__main__':
    app.run(debug=True)
