# ANN-and-CNN-on-Hand-written-digit-recognition
This repository, titled "ANN and CNN on Hand-written Digit Recognition," showcases a comparative analysis of Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) on the MNIST dataset. It explores how each model performs in recognizing hand-written digits, with a focus on accuracy, model architecture, and processing efficiency.

  Project Structure:

ANN-and-CNN-on-Handwritten-Digit-Recognition.ipynb: The main Jupyter Notebook that contains the implementation and comparison of the ANN and CNN models on the MNIST dataset.
README.md: This file provides an overview of the project, instructions for usage, and details about the models and dataset.

  Dataset:

The MNIST dataset is a widely used dataset in the field of machine learning and computer vision. It contains 60,000 training images and 10,000 test images of handwritten digits. Each image is grayscale and has a resolution of 28x28 pixels.

e.g. image of digit 5 (28x28 pixels with grayscale)
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    3    18   18   18   126  136  175  26   166  255  247  127  0    0    0    0   
 0    0    0    0    0    0    0    0    30   36   94   154  170  253  253  253  253  253  225  172  253  242  195  64   0    0    0    0   
 0    0    0    0    0    0    0    49   238  253  253  253  253  253  253  253  253  251  93   82   82   56   39   0    0    0    0    0   
 0    0    0    0    0    0    0    18   219  253  253  253  253  253  198  182  247  241  0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    80   156  107  253  253  205  11   0    43   154  0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    14   1    154  253  90   0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    139  253  190  2    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    11   190  253  70   0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    35   241  225  160  108  1    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    81   240  253  253  119  25   0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    45   186  253  253  150  27   0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    16   93   252  253  187  0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    249  253  249  64   0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    46   130  183  253  253  207  2    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    39   148  229  253  253  253  250  182  0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    24   114  221  253  253  253  253  201  78   0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    23   66   213  253  253  253  253  198  81   2    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    18   171  219  253  253  253  253  195  80   9    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    55   172  226  253  253  253  253  244  133  11   0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    136  253  253  253  212  135  132  16   0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   
 0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0  

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

