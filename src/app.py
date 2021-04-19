import tensorflow as tf
from flask import Flask, request
import numpy as np
from unet.metrics import MeanIoU_Greater

app = Flask(__name__)

model = tf.keras.models.load_model(
    "data/models/20210419_073252/model",
    custom_objects={"MeanIoU_Greater": MeanIoU_Greater},
)


@app.route("/inference", methods=["POST"])
def inference():
    data = request.json
    img_arr = np.array(data["image"], dtype=np.uint8)

    img_tensor = tf.convert_to_tensor(img_arr)
    img_tensor = tf.image.resize(img_tensor, (128, 128))
    img_tensor = tf.cast(img_tensor, tf.float32) / 255.0

    prediction = model(img_tensor[tf.newaxis, ...])
    prediction = prediction[0]
    prediction = tf.greater(prediction, 0.5)
    prediction = tf.cast(prediction, tf.int8)

    prediction = tf.image.resize(prediction, img_arr.shape[:2], method="nearest")
    prediction = tf.squeeze(prediction)

    return {"prediction": prediction.numpy().tolist()}


if __name__ == "__main__":
    app.run(debug=True)
