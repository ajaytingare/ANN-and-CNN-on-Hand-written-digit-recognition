# ANN-and-CNN-on-Hand-written-digit-recognition
This repository, titled "ANN and CNN on Hand-written Digit Recognition," showcases a comparative analysis of Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) on the MNIST dataset. It explores how each model performs in recognizing hand-written digits, with a focus on accuracy, model architecture, and processing efficiency.

  Project Structure:

ANN-and-CNN-on-Handwritten-Digit-Recognition.ipynb: The main Jupyter Notebook that contains the implementation and comparison of the ANN and CNN models on the MNIST dataset.
README.md: This file provides an overview of the project, instructions for usage, and details about the models and dataset.

  Dataset:

The MNIST dataset is a widely used dataset in the field of machine learning and computer vision. It contains 60,000 training images and 10,000 test images of handwritten digits. Each image is grayscale and has a resolution of 28x28 pixels.

e.g. image of digit 5 (28x28 pixels with grayscale)
 
0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 3___ 18__ 18__ 18__ 126_ 136_ 175_ 26__ 166_ 255_ 247_ 127_ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 30__ 36__ 94__ 154_ 170_ 253_ 253_ 253_ 253_ 253_ 225_ 172_ 253_ 242_ 195_ 64__ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 49__ 238_ 253_ 253_ 253_ 253_ 253_ 253_ 253_ 253_ 251_ 93__ 82__ 82__ 56__ 39__ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 18__ 219_ 253_ 253_ 253_ 253_ 253_ 198_ 182_ 247_ 241_ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 80__ 156_ 107_ 253_ 253_ 205_ 11__ 0___ 43__ 154_ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 14__ 1___ 154_ 253_ 90__ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 139_ 253_ 190_ 2___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 11__ 190_ 253_ 70__ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 35__ 241_ 225_ 160_ 108_ 1___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 81__ 240_ 253_ 253_ 119_ 25__ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 45__ 186_ 253_ 253_ 150_ 27__ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 16__ 93__ 252_ 253_ 187_ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 249_ 253_ 249_ 64__ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 46__ 130_ 183_ 253_ 253_ 207_ 2___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 39__ 148_ 229_ 253_ 253_ 253_ 250_ 182_ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 24__ 114_ 221_ 253_ 253_ 253_ 253_ 201_ 78__ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 23__ 66__ 213_ 253_ 253_ 253_ 253_ 198_ 81__ 2___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 18__ 171_ 219_ 253_ 253_ 253_ 253_ 195_ 80__ 9___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 55__ 172_ 226_ 253_ 253_ 253_ 253_ 244_ 133_ 11__ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 136_ 253_ 253_ 253_ 212_ 135_ 132_ 16__ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___ 0___
 
  Objective:

This is dummy dataset as all knows, but main objective of this work is to predict the own hand-written digit by those both model.
And check the performance of both ANN and CNN models.


  Models Implemented:
  
  1. Artificial Neural Network (ANN):

A basic neural network with fully connected layers.
Input layer: 784 neurons (28x28 pixels).
Hidden layers: Multiple layers with varying numbers of neurons.
Output layer: 10 neurons, one for each digit (0-9).
Activation functions: ReLU for hidden layers and softmax for the output layer.

  2. Convolutional Neural Network (CNN):

A deep learning model specifically designed for image data.
Convolutional layers: Extract features from the input images using filters.
Pooling layers: Reduce the dimensionality of feature maps.
Fully connected layers: Make predictions based on extracted features.
Output layer: Similar to ANN, with 10 neurons for digit classification.

  Performance Comparison:
  
The notebook compares the performance of ANN and CNN models in terms of:

Accuracy: How well the model predicts the correct digit.
Training time: The time taken to train each model.
Model complexity: The number of parameters and layers in each model.

  Key Findings:
  
CNN typically outperforms ANN in terms of accuracy due to its ability to capture spatial hierarchies in image data.
ANN is faster to train but may not achieve the same level of accuracy as CNN, especially for image recognition tasks.

Requirements To run the notebook, you need the following dependencies:

Python 3.x

Jupyter Notebook

TensorFlow or PyTorch (depending on the implementation)

NumPy

Matplotlib

If you want to impliment your own thoughts on these models, there is colab file:
https://colab.research.google.com/drive/10Z88jH6Cjcm15PbOM0s4dM0YcsN4bUIG?usp=sharing

