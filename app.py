from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)   # enable CORS so dashboard can access API

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tomato_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Exact class names from training
classes = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

@app.route("/")
def home():
    return {"status": "Tomato Disease API is online"}, 200


@app.route("/predict", methods=["POST"])
def predict():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img_bytes = np.frombuffer(file.read(), np.uint8)

        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (128, 128))
        img = img / 255.0

        img = np.expand_dims(img, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "disease": classes[class_index],
            "confidence": round(confidence, 4),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)