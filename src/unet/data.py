import tensorflow as tf
import pandas as pd

from PIL import Image
import requests
from io import BytesIO


def get_csv_dataset(data_filepath):
    # data_file = tf.keras.utils.get_file("data.csv", data_filepath)
    df = pd.read_csv(data_filepath)
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))

    return dataset


def tf_load_image(tf_image_url):

    def load_image(image_url):
        return tf.image.decode_jpeg(requests.get(image_url.numpy()).content)

    imgs = tf.py_function(load_image, [tf_image_url], Tout=tf.uint8)
    return imgs


def input_and_target_urls_from_csv(data_filepath):
    csv_dataset = get_csv_dataset(data_filepath)

    mask_url_dataset = csv_dataset.map(lambda x: x["mask_url"])
    image_url_dataset = csv_dataset.map(lambda x: x["image_url"])

    return mask_url_dataset, image_url_dataset


def inputs_and_targets(data_filepath):
    mask_url_dataset, image_url_dataset = input_and_target_urls_from_csv(data_filepath)

    mask_dataset = mask_url_dataset.map(lambda x: tf_load_image(x))
    image_dataset = image_url_dataset.map(lambda x: tf_load_image(x))

    return mask_dataset, image_dataset
