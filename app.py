from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tomato_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

# ---------------------------------------
# Check if image contains plant leaf
# ---------------------------------------

def is_leaf_image(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.sum(mask > 0) / mask.size

    if green_ratio < 0.10:
        return False

    return True


# ---------------------------------------
# Image enhancement
# ---------------------------------------

def enhance_image(img):

    img = cv2.GaussianBlur(img, (3,3), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img


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

        # ---------------------------------------
        # Leaf validation
        # ---------------------------------------

        if not is_leaf_image(img):
            return jsonify({
                "error": "No plant leaf detected. Please capture tomato leaf."
            }), 400


        # ---------------------------------------
        # Image enhancement
        # ---------------------------------------

        img = enhance_image(img)

        # ---------------------------------------
        # Resize and normalize
        # ---------------------------------------

        img = cv2.resize(img, (128,128))
        img = img / 255.0

        img = np.expand_dims(img, axis=0).astype(np.float32)

        # ---------------------------------------
        # Run prediction
        # ---------------------------------------

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        disease_name = classes[class_index]

        return jsonify({

            "disease": disease_name,
            "confidence": round(confidence,4),
            "status": "success"

        })


    except Exception as e:

        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)