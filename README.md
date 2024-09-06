# MNIST Digit Classification with TensorFlow and Keras

This project demonstrates several approaches to building and training neural networks using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. It includes simple neural networks (ANNs) as well as Convolutional Neural Networks (CNNs) to illustrate the progression from basic to more advanced models for image classification tasks.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
  - [Simple Neural Network](#simple-neural-network)
  - [Deep Neural Network](#deep-neural-network)
  - [Neural Network with Flatten Layer](#neural-network-with-flatten-layer)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
- [Confusion Matrix](#confusion-matrix)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to classify images of handwritten digits (0-9) using various neural network architectures. This project explores the use of different model architectures, activation functions, and techniques to achieve accurate classification of digits from the MNIST dataset.

## Dataset

The dataset used in this project is the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which consists of 60,000 training images and 10,000 test images. Each image is a 28x28 pixel grayscale image of a single handwritten digit.

## Models

### Simple Neural Network

A basic neural network with a single dense layer containing 10 neurons (one for each digit) and a sigmoid activation function.

```python
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])


### Deep Neural Network
A deeper neural network with an additional hidden layer of 100 neurons with ReLU activation and an output layer with 10 neurons using sigmoid activation.

model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

