import tensorflow as tf
import matplotlib.pyplot as plt


def create_masks(pred_masks):
    pred_masks = tf.argmax(pred_masks, axis=-1)
    pred_masks = pred_masks[..., tf.newaxis]
    return pred_masks


def export(export_list, export_file):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(export_list)):
        plt.subplot(1, len(export_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(export_list[i]))
        plt.axis('off')

    plt.savefig(export_file)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_prediction(model, images, targets):
    """assumes input batch"""
    pred_mask = model.predict(images)
    display([images[0], targets[0], create_masks(pred_mask)[0]])


def export_prediction(model, images, targets, filename):
    """assumes input batch"""
    pred_mask = model.predict(images)
    export([images[0], targets[0], create_masks(pred_mask)[0]], filename)