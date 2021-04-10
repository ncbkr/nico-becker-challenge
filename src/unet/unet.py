import tensorflow

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import UpSampling2D

#https://www.tensorflow.org/guide/keras/functional


def get_encoder():
    base = MobileNetV2(input_shape=[128, 128, 3], weights="imagenet", include_top=False, alpha=0.35)
    base.trainable = False

    extraction_layers = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    extracted_outputs = [base.get_layer(name).output for name in extraction_layers]

    # Create the feature extraction model
    return tensorflow.keras.Model(inputs=base.input, outputs=extracted_outputs)
    

def upsampling_block(filters, size):
    initializer = tensorflow.random_normal_initializer(0., 0.02)

    decoder_block = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False
        ),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.ReLU()
    ])

    return decoder_block 


def get_unet():    
    inputs = Input(shape=(128, 128, 3), name="input_image")

    encoder = get_encoder()

    # TODO put this into separate model with multiple outputs?
    # Or also a model that takes the skip connections as input?
    # TODO find out how upsampling should look like when using 244 x 244 images
    upsampling_blocks = [
        upsampling_block(512, 3),
        upsampling_block(256, 3),
        upsampling_block(128, 3),
        upsampling_block(64, 3),
    ]

    # implement pass through

    # downsampling input
    encoder_outputs = encoder(inputs)

    # get 'bottom' features
    x = encoder_outputs[-1]

    # reverse ordering of skip connections for expanding path
    skip_connections = reversed(encoder_outputs[:-1])

    # upsampling and skip connections
    for upsampling, skip in zip(upsampling_blocks, skip_connections):
        x = upsampling(x)
        print("x: ", x.shape)
        print("skip: ", skip.shape)
        tensorflow.keras.layers.concatenate([skip, x])

    x = tensorflow.keras.layers.Conv2DTranspose(2, 3, strides=2, padding='same')(x)

    return tensorflow.keras.Model(inputs=inputs, outputs=x)