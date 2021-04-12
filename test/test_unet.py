import tensorflow as tf

from PIL import Image
import requests
from io import BytesIO

from unet.unet import get_unet

import pytest

@pytest.fixture(scope="module")
def model():
    model = get_unet()
    return model

def test_unet_can_be_created(model):
    assert model

def test_unet_returns_result_in_input_shape(model):
    test_url = "https://d1h90vpqo1860x.cloudfront.net/3e18e6da-eb40-4dd8-a720-b9343f474339?Expires=1620835689&Signature=nc6bNdvCqjZSLJnZAxb~vl4knSWiUFk1BxtrLzn6HXa5vcu0IHYWDNJvqj9ASw7JgyMPuDybT~2WZ53oH7lgPqnSWRXfQWELV9Lcuu8vP0OUCkCbDrz-CklayKTWegn7Qc2Ococwz4GBPy-hK6B9ijGGEwctUiI6XZMSjzCIEgF3-HC0x2FDHzOU7eP7qhQMIq1~~Ca3GxaFdBmI0k0TSdyAvnIuHULDb4w-gaxFT0wnrTZ9AY6SD6wwO0~vPTEi5~k~DNKkb4DeGwFMcV~lWj2kmXe~zDApE2q3tm7x-OiYoJnXG1G5urOCdgSGXIqlKuWMKHaDEndpAa2e6R8MUQ__&Key-Pair-Id=APKAJI6YUNLVJA3JZROQ"
    response = requests.get(test_url)
    img = Image.open(BytesIO(response.content))
    img_244 = img.resize((128, 128))
    input_tensor = tf.keras.preprocessing.image.img_to_array(img_244)
    input_batch = tf.expand_dims(input_tensor, 0)
    y = model(input_batch)

    assert tf.is_tensor(y)
    assert y.shape.as_list() == [1, 128, 128, 1]