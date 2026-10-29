from flask import Flask
from flask import request
from flask import jsonify

import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model(
    "mnist_model.h5"
)

@app.route("/")
def home():

    return "MNIST Digit Classifier API"

@app.route("/predict", methods=["POST"])

def predict():

    data = request.get_json()

    image = np.array(
        data["image"]
    )

    image = image.reshape(
        1,28,28
    )

    prediction = model.predict(
        image
    )

    digit = int(
        np.argmax(prediction)
    )

    return jsonify(
        {"predicted_digit": digit}
    )

if __name__ == "__main__":

    app.run(debug=True)