from unet import get_unet
from data import inputs_and_targets
from visualize import export_prediction

import tensorflow as tf

import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = Path("data/models") / training_id
model_dir.mkdir()

sample_image = tf.io.read_file("data/sample_image.jpg")
sample_image = tf.image.decode_jpeg(sample_image)
sample_image = tf.image.resize(sample_image, (128, 128))

sample_mask = tf.io.read_file("data/sample_mask.jpg")
sample_mask = tf.image.decode_jpeg(sample_mask)
sample_mask = tf.image.resize(sample_mask, (128, 128), method="nearest")
sample_mask = tf.image.rgb_to_grayscale(sample_mask)


class ExportCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        filename = str(model_dir / f"epoch_{epoch}.jpg")
        export_prediction(self.model,  sample_image[tf.newaxis, ...], sample_mask[tf.newaxis, ...], filename)

model = get_unet()


# todo add metrics: accuracy, precision, recall, mean IoU.
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.MeanIoU(num_classes=2)]) 

TEST_LENGTH = 24
TRAIN_LENGTH = 240 - TEST_LENGTH
BATCH_SIZE = 1
BUFFER_SIZE = 4
EPOCHS = 1
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VAL_SUBSPLITS = 2
VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE // VAL_SUBSPLITS

dataset = inputs_and_targets("data/data.csv", 128, 128)
dataset = tf.data.Dataset.shuffle(dataset, seed=23, buffer_size=BUFFER_SIZE)

test = tf.data.Dataset.take(dataset, 24)
train = tf.data.Dataset.skip(dataset, 24)

test = test.batch(BATCH_SIZE)
train = train.cache().batch(BATCH_SIZE)
train = train.prefetch(buffer_size=tf.data.AUTOTUNE)

model_history = model.fit(train, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test,
                          callbacks=[ExportCallback()])

tf.saved_model.save(model, str(model_dir / "model"))