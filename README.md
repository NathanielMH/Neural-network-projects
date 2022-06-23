# Neural-network-projects

These are the notes and code I gathered from a series of guided project from Coursera. 
It taught the basic tools, libraries and concepts needed to solve real life problems with Neural Networks using the Keras framework, having Tensorflow as its backend.

## House price prediction
We used regression to solve for the price of a house in terms of parameters such as distance to city, year, etc.

### Architecture 
We used a 5 layer dense network with layer sizes 6, 10, 20, 5 and 1. 

### Parameters
The optimization algorithm is a version of stochastic gradient descent commonly known as Adam.
The loss function that was used for error is MSE (Mean squared error).

### Training
We used 100 epochs to train the model, coupled with an EarlyStopping callback from Keras to stop training the model is the validation loss stops descreasing after a few epochs. That way we can opt for a higher number of epochs for the training, resulting in higher accuracy in the predictions (but not so big as to cause overfitting).

## Image classification (Digits from 0 to 9)
We used a dataset from MNIST to train a classification model.

### Architecture 
We used a 4 layer dense network with layer sizes 784, 128, 128 and 10.

### Parameters
The optimization algorithm is stochastic gradient descent.

### Training
We used 3 epochs to train the model.

## Bank Loan Approvals

### Architecture
We used a 12 layer network with 2 dense layers with 250 neurons, 3 dense layers with 500 neurons, the input and output layers and 5 dropout layers to avoid overfitting.

## Parameters
The optimization algorithm is a version of stochastic gradient descent commonly knwon as Adam.
The loss function used for error is MSE.

## Training
We used 20 epochs to train the model with 0.2 validation loss.
