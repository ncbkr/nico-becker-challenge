# 1. Sketch the network & get it to train

Starting point: https://www.tensorflow.org/tutorials/images/segmentation

- They also use a pretrained MobileNetV2
- I could take the encoder part from their example

The the upsampling step of the decoder is adapted from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

Final layer: Conv2DTranspose, 2 Channels, One per class, labels are one hot encoded, loss is calculated on logits

training: - 1 epoch, batch size 1, no preprocessing but resizing, no augmentation, metric: accuracy

see for example: https://github.com/ncbkr/nico-becker-challenge/commit/5027cfbb1862ef7b808408d994608c5d0b381d5a


# 2. Get valid metrics
In order to get the metrics working, I iterated quite a bit:

I added the other desired metrics
- Precision
- Recall
- MeanIoU

observed behavior:
- precision and recall where the same as accuracy (https://stackoverflow.com/questions/61835742/same-value-for-keras-2-3-0-metrics-accuracy-precision-and-recall)
- MeanIoU was constant

---

I then changed the final layer to Conv2DTranspose with 1 Channel. As losses, I tried SparseCategoricalCrossEntropy and BinaryCrossEntropy. Loss was calculated on logits.

observed behavior:
- precision and recall where showing different from accuracy and from each other
- MeanIoU was still messed up

---

Next, I added an Activation layer after the final Conv2DTranspose. I used a Sigmoid activation because I wanted the values to be close to 0 and 1, respectively.

observed behavior:
- accuracy, precision and recall were still looking good
- MeanIoU was still messed up
- visualizations of the predictions were looking not footprint-like at all
- the visuals explained the bad MeanIoU but were contradictory to good accuracy, precision and recall

---

I came to the conclusion that MeanIoU and the plots were looking weird because my approach was lacking some post processing of the plain model probabilities. So I used a threshold to convert the predictions into a binary mask with values either 0 or 1.

observed behavior:
- Visualizations looked a lot more like footprints

---

Having solved the issue with the visualization, I decided implement a custom metric for the MeanIoU. I subclassed the original tf.keras.metrics MeanIoU and applied the threshold before passing the predictions to the actual calculation of the metric.

observed behavior:
- also MeanIoU was finally making sense


# 3. Improve results

After having set up the metrics, I tried various approaches for improving the model

1. normalization of the image (values between 0 and 1)
2. increase number of training examples with dataset.repeat()
3. adding random augmentation to the training set to cope for repetiton
4. play with hyperparameters like n_epochs and batch_size

observed behavior:
1. did improve results drastically on train and validation set
2. did improve results drastically on train and validation set
3. performance decreased and did never reach results I got without augmentation
4. had no real effect. Any number of epochs > 5 did not lead to improved results. batch sizes had no noticeable effect

conclusion:
- observed behaviors for 1 and 2 were not surprising but very welcome
- observed behavior for 3 was both, expected and surprising. I expected the training metrics to be worse because with ranom augmentation I wanted to prevent overfitting. However, it was suprising that the results were a lot worse and the metrics also plateaued after a few epochs. The reason for this could also be a bug in my augmentation function
- observed behavior for 4 was not surprising. I used a pretrained model, so I expected the results to be okayish from the start. The dataset is still small, so I expected the model to know it by heart very quickly.

# 4. Biggest Learnings

- Tensorflow Functional API is very nice for feature extraction
- Tensorflow Dataset API is not always fun to debug, but handy in general. Especially caching datasets comes very handy compared to downloading images for every training.
- Every Final Layer / Activation has its purpose and metrics do not necessarily work out of the box

# runs

**No normalization:** https://tensorboard.dev/experiment/twRhXkI5R8iljU2RQT1XUA/

**augmentation, short training:** https://tensorboard.dev/experiment/NQxuYw5MSOaIXz0ZFbByKg/#scalars

**augmentation, long training:** https://tensorboard.dev/experiment/bkzl2lGeS3OwVDAjCwAgoA/

**No augmentation, repeat 3:** https://tensorboard.dev/experiment/vhzDeQQjTZCjgFZFv90MBQ/#scalars

**augmentation, repeat 10:** https://tensorboard.dev/experiment/AmtvatlpQYaJ6VoG6IPhnw/

**Best run, no augmentation, repeat 10:** https://tensorboard.dev/experiment/MEx7TNQoQ3OLSEJgjwPPrg/
