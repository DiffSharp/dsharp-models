# ResNet-50 with CIFAR-10

This example demonstrates how to train the [ResNet-50 network]( https://arxiv.org/abs/1512.03385) against the [CIFAR-10 image classification dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

A modified ResNet-50 network is instantiated from the ImageClassificationModels library of standard models, and applied to an instance of the CIFAR-10 dataset. A custom training loop is defined, and the training and test losses and accuracies for each epoch are shown during training.

## Setup

