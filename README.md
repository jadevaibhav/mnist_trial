# mnist_trial
Fun with mnist dataset using cnn in both keras and tensorflow
## Getting Started
Before starting this tutorial, I hope you already have tensorflow and Keras installed on your machine.I have Python 3 on my machine so be sure to use Python 3 and install tensorflow and Keras acoordingly. In This repository contains 2 files mnist_tensorflow.py and mnist_cnn.py. Latter contains the code for classifier in Keras. In order to code to work, you need to download the mnist dataset from Kaggle-https://www.kaggle.com/scolianni/mnistasjpg , and unzip in same folder as the code. Check if the unzipped folder has name 'trainingSample'. If not, you either change it to 'trainingSample' or change in the code
```
path= 'trainingSample'
```
to the folder name.

## Dataset specification
Mnist(training) contains total 600 samples, from which I have used 550 for training and 50 for validation.  

## mnist_tensorflow.py
File contains data-preprocessing, model defination and training. For training I have used stochastic gradient descent instead of usual batch gradient descent. For batch gradient descent, you can follow the tutorial provided by tensorflow in https://www.tensorflow.org/versions/r1.2/get_started/mnist/pros .

## mnist_cnn.py
I have used Keras for this code. Details about model architecture would be shown when you run the code. I have run the model for 50 epochs with batch size of 16 samples/batch. After running the code, you would have a 'mnist_50.h5' file which has saved model architecture as well as trained model. I you want to load the model for further use, just add
```
model = keras.models.load_model('mnist_50.h5')
```
in your code. If you don't want to save the trained model, remove the above mentioned line.
