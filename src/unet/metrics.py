import tensorflow as tf

from tensorflow.keras.metrics import MeanIoU


class MeanIoU_Greater(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.greater(y_pred, 0.5)
        return super().update_state(y_true, y_pred, sample_weight)


class MeanIoU_Argmax(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = y_pred[..., tf.newaxis]

        return super().update_state(y_true, y_pred, sample_weight)
