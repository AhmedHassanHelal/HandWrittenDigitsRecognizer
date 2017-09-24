from mnist import MNIST
import numpy as np
import random

NUM_DIGItS=10

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
    This function creates a vector of zeros of shape (dim, 10) for w and initializes b to 0.
    
    Argument:
    dim --  number of parameters in this case
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    
    w = np.zeros((dim, NUM_DIGItS))
    b = np.zeros((NUM_DIGItS,1))
    
    return w, b



def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    Z = np.dot(w.T, X)+b
    A = sigmoid(Z)                                    # compute activation
    cost = None                                 # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = None
    db = None
    ### END CODE HERE ###
    
    cost = np.squeeze(cost
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
