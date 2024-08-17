from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load the pre-trained smile detector model
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')


@app.route('/detect_smile', methods=['POST'])
def detect_smile():
    # Receive the image file from the request
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect smiles
    smiles = smile_cascade.detectMultiScale(
        gray, scaleFactor=1.8, minNeighbors=20)

    # Check if any smiles were detected
    if len(smiles) > 0:
        return jsonify({"smile_detected": True})
    else:
        return jsonify({"smile_detected": False})


if __name__ == '__main__':
    app.run(debug=True)
