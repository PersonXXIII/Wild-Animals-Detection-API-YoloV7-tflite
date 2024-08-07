from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load TFLite model
model_path = "models/best.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define labels
labels = [
    "Bear", "Boar", "Coyote", "Deer", "Fox", "Goat", "Hyena", "Leopard",
    "Lion", "Monkey", "Rabbit", "Squirrel", "Wolves"
]

def preprocess_image(image, height, width):
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_output(output, threshold=0.4):
    detections = output[0]
    detected_classes = set()
    for detection in detections:
        for det in detection:
            score = det[4]
            class_probs = det[5:]
            if score >= threshold:
                class_id = np.argmax(class_probs)
                detected_classes.add(class_id)
    return detected_classes

@app.route('/')
def index():
    return "Hello,\nWelcome to the WA API."

@app.route('/image', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    height, width = 640, 640
    preprocessed_image = preprocess_image(image, height, width)

    expected_shape = input_details[0]['shape']
    if preprocessed_image.shape != tuple(expected_shape):
        return jsonify({'error': f'Input shape mismatch: expected {expected_shape}, got {preprocessed_image.shape}'}), 400

    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    output = [interpreter.get_tensor(detail['index']) for detail in output_details]

    detected_classes = postprocess_output(output)
    detected_labels = [labels[cls] for cls in detected_classes]

    return jsonify({'labels': detected_labels})

@app.route('/realtime', methods=['POST'])
def detect_objects_realtime():
    data = request.form['image']
    image_data = base64.b64decode(data.split(',')[1])
    image = Image.open(BytesIO(image_data))
    image = np.array(image)
    
    height, width = 640, 640
    preprocessed_image = preprocess_image(image, height, width)

    expected_shape = input_details[0]['shape']
    if preprocessed_image.shape != tuple(expected_shape):
        return jsonify({'error': f'Input shape mismatch: expected {expected_shape}, got {preprocessed_image.shape}'}), 400

    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    output = [interpreter.get_tensor(detail['index']) for detail in output_details]

    detected_classes = postprocess_output(output)
    detected_labels = [labels[cls] for cls in detected_classes]

    return jsonify({'labels': detected_labels})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
