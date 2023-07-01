from tensorflow.keras.layers import Dense,Input,Conv2D,\
Flatten,Conv2DTranspose,Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
(x_train,_),(x_test,_)=cifar10.load_data()
def image_display(x_train:np.array,x_test:np.array):
    """cifar10 display image

    Args:
        x_train (np.array): (n,32,32,3) uint8
        x_test (np.array): (n,32,32,3) uint8
    """
    img_rows=x_train.shape[1]
    img_cols=x_train.shape[2]
    channels=x_train.shape[3]
    imgs=x_test[:100]
    imgs=imgs.reshape((10,10,img_rows,img_cols,channels))
    imgs=np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis("off")
    plt.imshow(imgs,interpolation="none")
    plt.show()
if __name__=="__main__":
    image_display(x_train,x_test)