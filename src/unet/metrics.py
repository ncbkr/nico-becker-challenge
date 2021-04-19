import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU


class MeanIoU_Greater(MeanIoU):
    """
    Class to make tf.keras.metrics.MeanIoU compliant with the
    predictions of the unet. The UNet predicts probabilities, floats
    between 0 and 1. In order to correctly calculate the MeanIoU, the
    probabilities need to be transformed into a binary mask.
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.greater(y_pred, 0.5)
        return super().update_state(y_true, y_pred, sample_weight)
