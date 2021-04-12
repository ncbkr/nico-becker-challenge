
import tensorflow as tf
from flask import Flask, request
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('data/models/20210411_190339/model')

@app.route('/inference', methods=["POST"])
def inference():
    data = request.json
    img_arr = np.array(data["image"], dtype=np.uint8)
    # TODO: Replace with a call to your model
    print(img_arr.shape)
    img_tensor = tf.convert_to_tensor(img_arr)
    img_tensor = tf.image.resize(img_tensor, (128, 128))
    img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
    #random_mask = (np.random.uniform(size=(img_arr.shape[:2])) > 0.5).astype(np.uint8)

    prediction = model(img_tensor[tf.newaxis, ...])
    prediction = prediction[0]

    prediction = tf.image.resize(prediction, img_arr.shape[:2])
    prediction = tf.squeeze(prediction)

    return {"prediction": prediction.numpy().tolist()}

if __name__ == '__main__':
    app.run(debug=True)

