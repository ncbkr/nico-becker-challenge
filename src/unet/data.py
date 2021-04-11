import tensorflow as tf
import pandas as pd

from PIL import Image
import requests
from io import BytesIO


def get_csv_dataset(data_filepath):
    df = pd.read_csv(data_filepath)
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))

    return dataset


def tf_load_image(tf_image_url, image_height, image_width):

    def load_image(image_url, image_height, image_widthe):
        img = tf.image.decode_jpeg(requests.get(image_url.numpy()).content)
        img = tf.image.resize(img, (image_height, image_width))
        return img

    img = tf.py_function(load_image, [tf_image_url, image_height, image_width], Tout=tf.float32)
    img.set_shape((image_height, image_width, 3))
    return img


def tf_load_mask(tf_image_url, image_height, image_width):

    def load_mask(image_url, image_height, image_width):
        mask = tf.image.decode_jpeg(requests.get(image_url.numpy()).content)
        mask = tf.where(mask == 255, 1, mask)
        mask = tf.image.resize(mask, (image_height, image_width), method="nearest")
        mask = tf.cast(mask, tf.float32) 
        return mask

    mask = tf.py_function(load_mask, [tf_image_url, image_height, image_width], Tout=tf.float32)
    mask.set_shape((image_height, image_width, 1))
    return mask

# TODO: check if this would work with categorical cross entropy
def tf_load_mask_with_2_channels(tf_image_url, image_height, image_width):

    def load_mask(image_url, image_height, image_width):
        img = tf.image.decode_jpeg(requests.get(image_url.numpy()).content)
        img = tf.image.resize(img, (image_height, image_width), method="nearest")
        img = tf.where(img == 255, 1, img)
        img = tf.keras.utils.to_categorical(img, num_classes=2, dtype='float32')
        return img

    img = tf.py_function(load_mask, [tf_image_url, image_height, image_width], Tout=tf.float32)
    img.set_shape((image_height, image_width, 2))
    return img

def input_and_target_urls_from_csv(data_filepath):
    csv_dataset = get_csv_dataset(data_filepath)

    mask_url_dataset = csv_dataset.map(lambda x: x["mask_url"])
    image_url_dataset = csv_dataset.map(lambda x: x["image_url"])

    return mask_url_dataset, image_url_dataset


def inputs_and_targets(data_filepath, image_height, image_width):
    mask_url_dataset, image_url_dataset = input_and_target_urls_from_csv(data_filepath)

    # TODO: Mask could also be two channels? tf.to_categorical
    mask_dataset = mask_url_dataset.map(lambda x: tf_load_mask_with_2_channels(x, image_height, image_width))
    image_dataset = image_url_dataset.map(lambda x: tf_load_image(x, image_height, image_width))

    dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))

    return dataset
