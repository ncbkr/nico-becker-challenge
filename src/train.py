"""Train UNet for Building Footprints

This script can be used for training a UNet model for creating building
footprints from aerial imagery. The UNet is implemented in Tensorflow. 
The dataset loading and preprocessing steps are implemented using the 
tf.data.Dataset API. The model architecture and the functions used for
preprocessing, the custom metrics and functions for visualization are
implemented in the unet module.

It is expected that input data comes as a csv with these columns:

- mask_url
- image_url

In this script you can manipulate how the data is preprocessed before
training and you can change the training hyper parameters.

On the end of each epoch, a sample prediction is made and a comparison
of the input image, the original mask and the prediction is stored to
a local directory. In addition, the tensorboard logs are also stored.
Each training run writes into a separate directory within the data
directory.

Notice: For calculating the MeanIoU during training, a custom metric is
used. If you want to load a trained model in another script, make sure
to also import the same MeanIoU function that was used during training.
"""

from datetime import datetime
from pathlib import Path

import tensorflow as tf

from unet.unet import get_unet
from unet.data import inputs_and_targets
from unet.data import get_csv_dataset
from unet.data import augment
from unet.data import prepare_image
from unet.data import prepare_mask
from unet.visualize import export_prediction
from unet.metrics import MeanIoU_Greater


class ExportCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        filename = str(model_dir / f"epoch_{epoch}.jpg")
        export_prediction(self.model,  sample_image[tf.newaxis, ...], sample_mask[tf.newaxis, ...], filename)


training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = Path("data/models") / training_id
model_dir.mkdir()

sample_image = tf.io.read_file("data/provided_image.jpg")
sample_image = tf.image.decode_jpeg(sample_image)
sample_image = prepare_image(sample_image, 128, 128)

sample_mask = tf.io.read_file("data/provided_mask.jpg")
sample_mask = tf.image.decode_jpeg(sample_mask)
sample_mask = prepare_mask(sample_mask, 128, 128)

model = get_unet()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), MeanIoU_Greater(num_classes=2)]) 

TEST_LENGTH = 24
TRAIN_LENGTH = 240 - TEST_LENGTH
BATCH_SIZE = 1
SHUFFLE_BUFFER = 250
EPOCHS = 7
TRAIN_REPEAT = 3

# first load csv dataset
dataset = inputs_and_targets("data/data.csv", 128, 128)
dataset = dataset.cache("data/cache/cached")

# then split into train and test
dataset = dataset.shuffle(SHUFFLE_BUFFER)
test = tf.data.Dataset.take(dataset, 24)
train = tf.data.Dataset.skip(dataset, 24)

# repeat train
train = train.repeat(3)

# shuffle train again
train = train.shuffle(SHUFFLE_BUFFER * TRAIN_REPEAT)

# batch data
test = test.batch(BATCH_SIZE)
train = train.batch(BATCH_SIZE)

# apply augmentations to train
train = train.map(lambda image, mask: augment(image, mask))

# prefetch training data
train = train.prefetch(buffer_size=tf.data.AUTOTUNE)

model_history = model.fit(train, epochs=EPOCHS,
                          validation_data=test,
                          callbacks=[
                              ExportCallback(),
                            tf.keras.callbacks.TensorBoard(log_dir=str(model_dir / "logs"), histogram_freq=1)])

tf.saved_model.save(model, str(model_dir / "model"))