from unet.data import get_csv_dataset
from unet.data import input_and_target_urls_from_csv
from unet.data import inputs_and_targets
from functools import reduce

from pathlib import Path
import tensorflow as tf
import requests

def test_dataset_can_be_loaded():
    file_path = Path("data/data.csv")

    dataset = get_csv_dataset(file_path)
    assert dataset

    for feature_batch in dataset.take(1):
        for key, value in feature_batch.items():
            assert key in ["image_url", "mask_url"]

def test_inputs_and_targets_are_splitted():
    file_path = Path("data/data.csv")

    mask_url_dataset, image_url_dataset = input_and_target_urls_from_csv(file_path)

    mask_url = "https://d1h90vpqo1860x.cloudfront.net/ab53069c-5a8a-4376-ae33-d4aae82c906f?Expires=1620835688&Signature=BLGppTV6oAfydoCOFk2KazWYPPzSgIZy-k4FU9JIqMvDCKmU-vQMX-sVyqLA1uC-vjESCX0PV0FberLZBI3k~FdkMCWmD7s9Tf3Cozus9-rGOI0tTryMhIoEkFJ-AsE2GwMqOVfA7~aoSnN~dz2V5RAH9BoCVNVAqd5y7UxytH4spiGuqAzvRBWXOI2a0v845jL16I4B47vQUO9w1XidE3Xrj1oStvZ0w0YTbQqiyq8R-5Mz05Q7WxKlIwqIGiVCyPRx2MIrNEz9zvHBzWhGP~f9kWGtEogxajZksKUuHAcWUvAaag0ITiTgExyPvnXcgHGhuAGWX3L6bBJnGG1Qrw__&Key-Pair-Id=APKAJI6YUNLVJA3JZROQ"
    image_url = "https://d1h90vpqo1860x.cloudfront.net/3e18e6da-eb40-4dd8-a720-b9343f474339?Expires=1620835689&Signature=nc6bNdvCqjZSLJnZAxb~vl4knSWiUFk1BxtrLzn6HXa5vcu0IHYWDNJvqj9ASw7JgyMPuDybT~2WZ53oH7lgPqnSWRXfQWELV9Lcuu8vP0OUCkCbDrz-CklayKTWegn7Qc2Ococwz4GBPy-hK6B9ijGGEwctUiI6XZMSjzCIEgF3-HC0x2FDHzOU7eP7qhQMIq1~~Ca3GxaFdBmI0k0TSdyAvnIuHULDb4w-gaxFT0wnrTZ9AY6SD6wwO0~vPTEi5~k~DNKkb4DeGwFMcV~lWj2kmXe~zDApE2q3tm7x-OiYoJnXG1G5urOCdgSGXIqlKuWMKHaDEndpAa2e6R8MUQ__&Key-Pair-Id=APKAJI6YUNLVJA3JZROQ"

    for x in mask_url_dataset.take(1).as_numpy_iterator():
        assert x.decode("utf-8") == mask_url
    
    for x in image_url_dataset.take(1).as_numpy_iterator():
        assert x.decode("utf-8") == image_url                                                                                                                                                                                                                                                                                                                                              


def test_input_and_target_images_are_loaded():
    file_path = Path("data/data.csv")

    image_height = 128
    image_width = 128

    mask_url = "https://d1h90vpqo1860x.cloudfront.net/ab53069c-5a8a-4376-ae33-d4aae82c906f?Expires=1620835688&Signature=BLGppTV6oAfydoCOFk2KazWYPPzSgIZy-k4FU9JIqMvDCKmU-vQMX-sVyqLA1uC-vjESCX0PV0FberLZBI3k~FdkMCWmD7s9Tf3Cozus9-rGOI0tTryMhIoEkFJ-AsE2GwMqOVfA7~aoSnN~dz2V5RAH9BoCVNVAqd5y7UxytH4spiGuqAzvRBWXOI2a0v845jL16I4B47vQUO9w1XidE3Xrj1oStvZ0w0YTbQqiyq8R-5Mz05Q7WxKlIwqIGiVCyPRx2MIrNEz9zvHBzWhGP~f9kWGtEogxajZksKUuHAcWUvAaag0ITiTgExyPvnXcgHGhuAGWX3L6bBJnGG1Qrw__&Key-Pair-Id=APKAJI6YUNLVJA3JZROQ"
    image_url = "https://d1h90vpqo1860x.cloudfront.net/3e18e6da-eb40-4dd8-a720-b9343f474339?Expires=1620835689&Signature=nc6bNdvCqjZSLJnZAxb~vl4knSWiUFk1BxtrLzn6HXa5vcu0IHYWDNJvqj9ASw7JgyMPuDybT~2WZ53oH7lgPqnSWRXfQWELV9Lcuu8vP0OUCkCbDrz-CklayKTWegn7Qc2Ococwz4GBPy-hK6B9ijGGEwctUiI6XZMSjzCIEgF3-HC0x2FDHzOU7eP7qhQMIq1~~Ca3GxaFdBmI0k0TSdyAvnIuHULDb4w-gaxFT0wnrTZ9AY6SD6wwO0~vPTEi5~k~DNKkb4DeGwFMcV~lWj2kmXe~zDApE2q3tm7x-OiYoJnXG1G5urOCdgSGXIqlKuWMKHaDEndpAa2e6R8MUQ__&Key-Pair-Id=APKAJI6YUNLVJA3JZROQ"

    mask = tf.image.decode_jpeg(requests.get(mask_url).content)
    mask = tf.where(mask == 255, 1, mask)
    mask = tf.image.resize(mask, (image_height, image_width), method="nearest")
    mask = tf.cast(mask, tf.float32)
    # mask = tf.keras.utils.to_categorical(mask, num_classes=2, dtype='float32')

    image = tf.image.decode_jpeg(requests.get(image_url).content)
    image = tf.image.resize(image, (image_height, image_width))
    image = tf.cast(image, tf.float32) / 255.0

    n_values = reduce(lambda a, b: a*b, mask.shape)

    image_dataset = inputs_and_targets(file_path, image_height, image_width)
    
    for x, y in image_dataset.take(1):
        assert tf.is_tensor(y)
        assert y.shape.as_list() == [128, 128, 1]
        tf.debugging.assert_equal(y, mask)

        assert tf.is_tensor(x)
        assert x.shape.as_list() == [128, 128, 3]
        tf.debugging.assert_equal(x, image)
