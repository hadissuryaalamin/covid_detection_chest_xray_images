from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('covid_detection_model.keras')

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Class labels
class_labels = ['Covid', 'Normal', 'Pneumonia']

def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array = img_array / 255.0  # Normalize the image

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    return class_labels[predicted_class_index]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Predict the image class
            prediction = predict_image(file_path)

            return render_template('index.html', prediction=prediction, img_path=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
