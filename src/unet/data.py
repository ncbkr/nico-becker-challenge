import requests
from io import BytesIO

import tensorflow as tf
import pandas as pd


def prepare_image(image, image_height, image_width):
    image = tf.image.resize(image, (image_height, image_width))
    image = tf.cast(image, tf.float32) / 255.0
    return image


def prepare_mask(mask, mask_height, mask_width):
    mask = tf.where(mask == 255, 1, mask)
    mask = tf.image.resize(mask, (mask_height, mask_width), method="nearest")
    mask = tf.cast(mask, tf.float32)

    if mask.shape[-1] > 1:
        mask = tf.image.rgb_to_grayscale(mask)

    return mask


def get_csv_dataset(data_filepath):
    df = pd.read_csv(data_filepath)
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))

    return dataset


def tf_load_image(tf_image_url, image_height, image_width):
    def load_image(image_url, image_height, image_width):
        img = tf.image.decode_jpeg(requests.get(image_url.numpy()).content)
        img = prepare_image(img, image_height, image_width)
        return img

    img = tf.py_function(
        load_image, [tf_image_url, image_height, image_width], Tout=tf.float32
    )
    img.set_shape((image_height, image_width, 3))
    return img


def tf_load_mask(tf_image_url, image_height, image_width):
    def load_mask(image_url, image_height, image_width):
        mask = tf.image.decode_jpeg(requests.get(image_url.numpy()).content)
        mask = prepare_mask(mask, image_height, image_width)
        return mask

    mask = tf.py_function(
        load_mask, [tf_image_url, image_height, image_width], Tout=tf.float32
    )
    mask.set_shape((image_height, image_width, 1))
    return mask


def input_and_target_urls_from_csv(data_filepath):
    csv_dataset = get_csv_dataset(data_filepath)

    mask_url_dataset = csv_dataset.map(lambda x: x["mask_url"])
    image_url_dataset = csv_dataset.map(lambda x: x["image_url"])

    return mask_url_dataset, image_url_dataset


def inputs_and_targets(data_filepath, image_height, image_width):
    mask_url_dataset, image_url_dataset = input_and_target_urls_from_csv(data_filepath)

    mask_dataset = mask_url_dataset.map(
        lambda x: tf_load_mask(x, image_height, image_width)
    )
    image_dataset = image_url_dataset.map(
        lambda x: tf_load_image(x, image_height, image_width)
    )

    dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))

    return dataset


@tf.function
def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_brightness(input_image, max_delta=0.5)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_saturation(input_image, 0.1, 0.8)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_contrast(input_image, 0.1, 0.8)

    input_image = tf.clip_by_value(input_image, 0, 1)
    return input_image, input_mask
