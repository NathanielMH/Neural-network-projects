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
We used data from UniversalBank to train a classification model to predict creditworthiness.

### Architecture
We used a 12 layer network with 2 dense layers with 250 neurons, 3 dense layers with 500 neurons, the input and output layers and 5 dropout layers to avoid overfitting.

### Parameters
The optimization algorithm is a version of stochastic gradient descent commonly knwon as Adam.
The loss function used for error is MSE.

### Training
We used 20 epochs to train the model with 0.2 validation loss.

## Facial Expression recognition
The data set used to train the network is very large, hence it not being included in this repository. The model takes hours to train if we want decent performance (50 epochs at least), we have therefore saved a trained version in a .json that is included in the repository.

### Architecture
We used a CNN that includes an input layer, zero padding, Conv2d, BatchNorm and ReLU, MaxPool2D, two RES blocks, AveragePooling2D layer, Flatten layer, 2 Dense layers with ReLU and a Dropout layer and finally a last dense layer with ReLU and the output layer.

### Parameters
The optimization algorithm is Adam, a version of stochastic gradient descent.
The loss function is MSE (Mean squared error).

### Training 
We used 50 epochs to train the model that is included in the .json file.

## Happy sad classifier
Used data downloaded from zip to train the model, that receives images of emoji like happy or sad faces and classifies them.

### Architecture
3 Conv2D and MaxPooling2D layers followed by a Dense 512 layer with reLU activation, plus the output layer with sigmoid activation.

### Parameters
The optimization algorithm is RMSProp, the loss function is binary crossentropy.

### Training
We trained the model for a maximum of 20 epochs, however with the custom callback that we created the model stops training after achieving 0.999 accuracy on training data to avoid overfitting.

## Sign language detector
The model receives black and white images of signs corresponding to characters in the sign language and classifies them according to the character they represent. 

### Architecture
2 Conv2D and MaxPooling2D layers, a flatten layer, a Dense layer with 1024 neurons and reLU activation, a Dense 512 layer with reLU activation and an output layer with 26 neurons corresponding to the 26 characters in sign language, with softmax activation.

### Parameters
The optimization algorithm is RMSProp and the loss function is sparse categorical cross entropy.

### Training
We trained the model for 15 epochs, reaching 99% training accuracy and 95% validation accuracy.

## Sentiment text classifier
The model receives text (tweets in particular) and predicts whether they are positive or negative in sentiment.

### Architecture
We apply transfer learning and load embedding layers from a pretrained dataset. After, we add a dropout layer to avoid overfitting and add adaptability to the new dataset. We then have a Conv1D layer with relu activation, a bidirectional LSTM layer with 16 dimensions, and two dense layers with activation functions relu and sigmoid respectively, with 512 neurons and 1 neuron for the output.

### Parameters
The optimization algorithm is rmsprop and the loss function is binary crossentropy as they have proved to perform well on binary classification problems.

### Training
The model was trained for 20 epochs.

## Poetry generator
Given a corpus of words, the model generates a continuation based on previous sentences.

### Architecture
Embedding layer with embedding dimension/ output dimension of 100, Bidirectional LSTM layer with 150 dimensions and an output layer of shape the amount of words present in the corpus, with softmax activation.

### Parameters
The optimization algorithm is adam, a variant of stochastic gradient descent and the loss function is categorical cross_entropy.

### Training
The model was trained for 50 epochs.
