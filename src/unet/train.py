from unet import get_unet
from data import inputs_and_targets

import tensorflow as tf

model = get_unet()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


TEST_LENGTH = 24
TRAIN_LENGTH = 240 - TEST_LENGTH
BATCH_SIZE = 1
BUFFER_SIZE = 4
EPOCHS = 1
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VAL_SUBSPLITS = 2
VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE // VAL_SUBSPLITS

dataset = inputs_and_targets("data/data.csv", 128, 128)
dataset = tf.data.Dataset.shuffle(dataset, seed=23, buffer_size=BUFFER_SIZE)

test = tf.data.Dataset.take(dataset, 24)
train = tf.data.Dataset.skip(dataset, 24)

test = test.batch(BATCH_SIZE)
train = train.cache().batch(BATCH_SIZE)
train = train.prefetch(buffer_size=tf.data.AUTOTUNE)

print(train.element_spec)

model_history = model.fit(train, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test)#,
                          #callbacks=[DisplayCallback()])

tf.saved_model.save(model, "data/models/model")