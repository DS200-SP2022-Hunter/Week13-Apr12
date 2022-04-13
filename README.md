# Lab Assignment 12, Due on [Canvas](https://psu.instructure.com/courses/2174978/modules/items/35287099), Apr. 20 at 11:59pm
## Train and Test a Neural Network on Recognizing Handwritten Digits

The main objective of today's lab is to train a neural network classifier using a famous dataset (MNIST) involving images of handwritten digits.

**Objective**:  Use the [`MNIST` dataset](http://yann.lecun.com/exdb/mnist/) (see also Deng, 2012, _IEEE Signal Processing Magazine_ 29(6)) to train a neural network
to recognize handwritten digits via the `keras` and `tensorflow` libraries.  

The code in this lab is adapted from a Jupyter notebook created by [Xavier Snelgrove](https://wxs.ca/) that is [available on GitHub](https://github.com/wxs/keras-mnist-tutorial).  There are _many_ online tutorials that deal with the MNIST data.  All python in this lab will be widely recognized by data scientists who use these tools professionally.

**Your assignment** is as follows:

1. Open a colab window on your browser.  As usual, we'll first load the tools we need (which do not include the `datascience` library this week):
```
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
path_data = 'http://personal.psu.edu/drh20/200DS/assets/data/'
from urllib.request import urlopen
```

2. The data consist of 70,000 total images, each one labeled as a digit 0 through 9.  The images are divided into a training set of 60,000 and a test set of 10,000.  Each image is a 28x28 array of integers ranging from 0 to 255, where 0 is black, 255 is white, and values in between are shades of gray. The data are already part of `keras.datasets` so we can just load them:
```
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

3.  The code above created some new `numpy` array objects.  We can use [`numpy.shape`](https://numpy.org/doc/stable/reference/generated/numpy.shape.html) to find the dimensions of these arrays:
```
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)
```
Notice that the `X` objects are 3-dimensional arrays.  For instance, the `X` training data consist of 60K images, each one 28x28.

4. This code can be used to create a 3x3 array of 9 training set images selected at random. Make sure you understand how it works.
```
plt.rcParams['figure.figsize'] = (10,10) # You can experiment with changing these size settings

for i in range(9):
    plt.subplot(3, 3, i+1) # Produce a 3x3 array of plots, one at a time
    j = np.random.choice(y_train.size) # Choose a training image at random
    plt.imshow(X_train[j], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[j]))
```
5. We're going to use the training data to train a neural network, which is basically a very complicated regression model. Our neural network is going to take a single vector for each training example, so we need to reshape the input so that each 28x28 image becomes a single 784-dimensional vector. We'll also scale the inputs to be real numbers in the range [0, 1] rather than integers in the range [0, 255]:
```
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train / 255.0
X_test = X_test / 255.0
print("Training array shape", X_train.shape)
print("Testing array shape", X_test.shape)
```

6. The labels `y` for both the training and test sets are numbers from 0 to 9.  These will need to be converted into the format needed by the neural net software, namely, the so-called "one-hot" or "dummy variable" form.  Study how the code below converts the values in `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` to this format:
```
my_range = np.arange(10)
num_categories = 10

np_utils.to_categorical(my_range, num_categories)
```
Based on what you observe, describe how the one-hot format works.

7. Now convert the `y` labels to one-hot format:
```
Y_train = np_utils.to_categorical(y_train, num_categories)
Y_test = np_utils.to_categorical(y_test, num_categories)
```
8. Now it's time to build the neural network.  Our network will have three layers, though discussion of the details is way beyond the scope of this lab. As we said above, we're basically setting up a very complicated regression model.  The code below is copied directly from [Xavier Snelgrove's tutorial](https://github.com/wxs/keras-mnist-tutorial):
```
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
                           
model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probaility distribution, that is
                                 # that its values are all non-negative and sum to 1.
```
9. The so-called compilation stage involves determining a loss function and an optimization routine.  The loss function is analagous to the MSE, or root mean squared error, that we saw in the case of linear regression.  It is the function that is to be minimized in finding the best values for our regression model.  In addition to the loss function and optimizer, we also add the `accuracy` metric, which tells what fraction of the images are correctly classified:
```
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
10. Now that the model is built and compiled, it is time to do the heavy-duty computing involved in training it.  That is, we now must find the minimizer of our loss function.  This is the stage when we use the training dataset:
```
model.fit(X_train, Y_train,
          batch_size=128, epochs=4,
          verbose=1,
          validation_data=(X_test, Y_test))
```
Notice the `accuracy` metric as reported during the training process.  Keep in mind that the reported accuracy is misleading, since we expect to achieve better accuracy on the training data than on the test data.

11. We can now use the model to predict categories for the test data.  Predictions using the `softmax` activation are in the form of arrays of 10 probabilities, so we will use the `numpy` function called `argmax` to find the index (0, through 9) of the maximum value of each 10-item array. 
```
predicted_digits = np.argmax(model.predict(X_test), axis=1)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_digits == y_test)[0]
incorrect_indices = np.nonzero(predicted_digits != y_test)[0]
```
12. In `numpy`, you can find the length of array `x` by typing `x.size`.  Use this `size` method to determine the lengths of the `correct_indices` and `incorrect_indices` arrays.  Based on these results, what percent of the test data images were classified correctly?

13. We can adapt the code from Step 4 above to look at the first 9 correctly classified images and the first 9 incorrectly classified images:
```
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_digits[correct], y_test[correct]))
    
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_digits[incorrect], y_test[incorrect]))
```
14. In the summer of 2019, an undergraduate named Giovanni Smith spent eight weeks at Penn State as a research assistant working on the digit recognition problem.  During this time, he created two new datasets:  One in which he wrote digits himself, and one in which his friend Darius wrote digits.  Each of them produced a total of 240 digits, 24 of each digit from 0 to 9.  Let's read in the datasets and get them in the correct format:
```
X_Giovanni = np.loadtxt(urlopen(path_data + "X_Giovanni.csv"), dtype=np.uint8, skiprows=1, delimiter=',')
y_Giovanni = np.loadtxt(urlopen(path_data + "y_Giovanni.csv"), dtype=np.uint8, skiprows=1, delimiter=',')
X_Darius = np.loadtxt(urlopen(path_data + "X_Darius.csv"), dtype=np.uint8, skiprows=1, delimiter=',')
y_Darius = np.loadtxt(urlopen(path_data + "y_Darius.csv"), dtype=np.uint8, skiprows=1, delimiter=',')

# Scale the X images to be in the [0, 1] real number range instead of integers in the [0, 255] range
X_Giovanni = X_Giovanni / 255.0
X_Darius = X_Darius / 255.0

# Convert the y labels to one-hot format
Y_Giovanni = np_utils.to_categorical(y_Giovanni, num_categories)
Y_Darius = np_utils.to_categorical(y_Darius, num_categories)
```
15. By adapting code from Step 4, plot some of the digits that Giovanni and Darius produced. Do you notice anything different about them, relative to the MNIST digits?

16. Find the accuracy of the MNIST-trained neural net on both the Giovanni and the Darius datasets by adapting the code from Steps 11 and 12.  Do these two test sets perform as well as the MNIST test set?  Plot the first nine correctly classified and incorrectly classified images, for either the Giovanni or the Darius dataset, as in Step 13.

17. Train a new network on the Giovanni dataset and test it on the Darius dataset.  How does this network perform in terms of prediction accuracy?  Do you have any impressions about how well these neural networks perform at recognizing digits based on what you saw in Steps 15 through 17?

18.  Finally, make sure that your Jupyter notebook only includes code and text that is relevant to this assignment.  For instance, if you have been completing this assignment by editing the original code from Section 13.2, make sure to delete the material that isn't relevant before turning in your work.

When you've completed this, you should select "Print" from the File menu, then save to pdf using this option.  The pdf file that you create in this way is the file that you should upload to Canvas for grading.  If you have trouble with this step, try selecting the "A3" paper size from the advanced options and making sure that your colab is zoomed out all the way (using ctrl-minus or command-minus).  As an alternative, you can create the pdf within your google drive space and then download it from there.  Here's a [Jupyter noteboook](https://github.com/DS200-SP2022-Hunter/Week11-Mar29/blob/main/convert_pdf.ipynb) shared by Xinyu Dou that creates the pdf within the google drive space (you may need to modify it depending on your directory names and the name of your lab file).

