# TF Multi-Task Learning Project

## About  the Project
Multi-task classification project for Computer Vision (CV) tasks. It is 
built on top of TensorFlow (TF) 2.2+ framework, with the usage of 
"tf.data.Dataset" object for handling training, validation and test data.

The model is built from the template project (CNN-template), that tries
to follow OOP paradigm and implement modular architectural structures 
which can be easily reused in different CV tasks.

The model is trained to solve the following two tasks:
1. Multi-class classification
   * "Classic" MNIST task: label each image with one of 10 non-overlapping
   classes.
2. Binary image classification
   * Classify each image to one of 2 non-overlapping classes: even or odd.

## Supported features
1. Setting up the "tf.data.Dataset" object for Multi-task setting.
2. Classic training approach: "compile" & "fit" methods.
3. Custom training approach: train loop implementation from scratch.
4. Training loop optimization: "tf.function".
5. Evaluation on the test dataset from scratch.
6. Model tracking via MLflow: track training experiments and compare 
performance.

Features supported from the CNN-template project:
1. OOP best practices for training pipeline.
2. Visualisation of the raw train dataset.
3. Visualisation of the augmented images from the train dataset.

### "tf.data.Dataset" for Multi-Task Learning
In order to perform a Multi-task training, it is necessary to have 
target labels for each task, per training / validation / test 
example from the dataset.

"tf.data.Dataset" class provide a good abstraction when dealing with
datasets of different sizes. Besides that, it encapsulates the batching,
shuffling and prefetching of the data, so the end user is not required
to implement these mechanisms themselves.

This is the reason to utilize "tf.data.Dataset" in this project and
adapt classical data preparation pipelines to use multiple labels - 
one per task (e.g. ground truth label for the first task, ground truth label for
the second task, etc.).

### Training Pipelines
1. Classic approach: "compile" & "fit"
   * The training pipeline is built by using TF methods "compile" and 
   "fit".
2. Custom approach: train loop implementation from scratch.
   * The main features of "compile" and "fit" method are implemented
   from scratch, by using the "GradientTape" object and "tf.function" 
   annotation.

The "compile" method is used to initialize optimizer, loss function(s), 
metrics, etc. If we create these objects manually and update them
with each training step/epoch, there is no need for compiling the model.

The "fit" method encapsulates everything that happens to the model with
each batch of data. In order to implement our custom train loop, we 
would use "GradientTape", which keeps track of the trainable variables.

### Optimization: "tf.function"

Not only the "tf.function" annotation decreases the training time, but 
it also seems that it is responsible for stabilized training. 
Surprisingly, without the proposed annotation, the training procedure is
more prone to **divergence**.

## Pipeline Details
The project is separated into the modules which are combined to form
the following pipeline:
1. Dataset loading, batching and prefetching using 'tf.data' Dataset
2. Dataset visualisation: inspection of both original samples from the
    dataset, and the images after the augmentation layer is applied.
3. Model build-up: create the custom architecture specified in the
    configuration file. All parameters (number of ConvLayers,
    existence of Batch Normalization and Pooling layers, etc.) could
    be specified via configuration file.
4. Model training: the whole process is supported by logging tool -
    MLflow, so we are able to track performance across individual
    experiments (where each experiment is denoted with one set of
    the hyper-parameters).
5. Model evaluation on the test dataset: simple accuracy metric.


## Set Up the Project

### Install Necessary Requirements

    make install

### Run Pipeline

    make run

### MLflow Support
In order to track training procedure, the MLflow tracks logs of
training parameters, which could be seen on MLflow standalone server.

Run the MLflow UI from the local terminal:

    mlflow ui

## References
* https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e: 
great introductory material for TF Sub-classing API and custom training
loop.
* https://towardsdatascience.com/tensorflow-2-2-and-a-custom-training-logic-16fa72934ac3: 
custom train loop in TF 2.2.
* https://towardsdatascience.com/multi-task-learning-for-computer-vision-classification-with-keras-36c52e6243d2:
very neat introduction for the Multi-task learning in Keras (TF). 
* https://github.com/The-AI-Summer/Deep-Learning-In-Production: very 
good materials on Deep Learning in general, including the best 
practices on writing Deep Learning code.