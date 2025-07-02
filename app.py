import os
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('model/deepfake_quick_model.h5')


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction) * 100
    label = "Fake" if prediction < 0.5 else "Real"
    adjusted_confidence = confidence if label == "Real" else 100 - confidence
    return label, round(adjusted_confidence, 2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        prediction, confidence = predict_image(file_path)

        return render_template('result.html',
                               image_path=filename,
                               prediction=prediction,
                               confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)
