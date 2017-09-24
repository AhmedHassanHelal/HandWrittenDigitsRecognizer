from mnist import MNIST
import numpy as np
import random

mndata = MNIST('/home/ahmed/MyWorkspace/Hand Writing Digits Recognizer/samples')

images, labels = mndata.load_training()

X_train = np.asarray(images)

m_train = X_train.shape[0]

image_size = X_train.shape[1]

index = random.randrange(0, len(images))  # choose an index ;-)

print(mndata.display(images[index]))
