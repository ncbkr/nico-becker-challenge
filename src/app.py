"""This script runs a flask app for Building Footprint inference

This inference application loads a Tensorflow model to make predictions
on images received via the '/inference' endpoint.

Before running this app, you might want to train a model first. You can
do this with the train.py script. After you successfully trained a
model, please set the path in the call to 'load_model' to point to your
new model. If you used any custom metric during training, remember to
provide those functions via the 'custom_objects'.
"""

import tensorflow as tf
from flask import Flask, request
import numpy as np

from unet.metrics import MeanIoU_Greater
from unet.data import prepare_image


app = Flask(__name__)

model = tf.keras.models.load_model(
    # TODO: To use another model, please change this path
    "data/models/20210419_073252/model",
    custom_objects={"MeanIoU_Greater": MeanIoU_Greater},
)


@app.route("/inference", methods=["POST"])
def inference():
    data = request.json
    img_arr = np.array(data["image"], dtype=np.uint8)

    # preprocess image
    img_tensor = tf.convert_to_tensor(img_arr)
    img_tensor = prepare_image(img_tensor, 128, 128)

    # make prediction
    prediction = model(img_tensor[tf.newaxis, ...])
    prediction = prediction[0]

    # create binary mask
    prediction = tf.greater(prediction, 0.5)
    prediction = tf.cast(prediction, tf.int8)

    # resize prediction to the original shape
    prediction = tf.image.resize(prediction, img_arr.shape[:2], method="nearest")
    prediction = tf.squeeze(prediction)

    return {"prediction": prediction.numpy().tolist()}


if __name__ == "__main__":
    app.run(debug=True)
