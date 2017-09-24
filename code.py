from mnist import MNIST
import numpy as np
import random

mndata = MNIST('/home/ahmed/MyWorkspace/Hand Writing Digits Recognizer/samples')

images_train_orig, labels_train_orig = mndata.load_training()
images_test, labels_test = mndata.load_testing()

X_train = np.asarray(images_train_orig)
X_test = np.asarray(images_test)

m_train = X_train.shape[0]

m_test = X_test.shape[0]

image_size = X_train.shape[1]

index = random.randrange(0, len(images))  # choose an index ;-)

print(mndata.display(images[index]))

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """ 
    s = 1/(1+np.exp(-z))
    
    return s
