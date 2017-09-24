from mnist import MNIST
import numpy as np
import random

NUM_DIGItS

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
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim, NUM_DIGItS))
    b = np.zeros((NUM_DIGItS,1))
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
