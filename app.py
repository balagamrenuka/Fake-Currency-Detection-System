from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
MODEL_PATH = 'decision_tree_60_40_model.pkl'
clf = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mean_bgr = img.mean(axis=(0, 1))
    std_bgr  = img.std(axis=(0, 1))
    mean_hsv = hsv.mean(axis=(0, 1))
    std_hsv  = hsv.std(axis=(0, 1))
    lap_var  = cv2.Laplacian(gray, cv2.CV_64F).var()

    features = np.hstack([mean_bgr, std_bgr, mean_hsv, std_hsv, lap_var])
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction=None)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict
    features = extract_features(filepath)
    pred = clf.predict([features])[0]

    # Return **only** the simple label string:
    if pred == 0:
        prediction = "Real Currency"
    else:
        prediction = "Fake Currency"

    return render_template('index.html', prediction=prediction, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
