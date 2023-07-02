from tensorflow.keras.layers import Dense,Input,Conv2D,\
Flatten,Conv2DTranspose,Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from typing import *

def image_display(x_train:np.ndarray,x_test:np.ndarray):
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
def data_preprocess(x_train:np.ndarray,x_test:np.ndarray):
    """prepare color and grey color

    Args:
        x_train (np.array): _description_
        x_test (np.array): _description_

    Returns:
        x_train: (n,32,32,3) float32
        x_train_grey: (n,32,32,1) float32
        x_test: (n,32,32,3) float32
        x_test_grey: (n,32,32,1) float32
    """
    x_train=x_train.astype(np.float32)/255.0
    print(x_train.shape)
    x_test=x_test.astype(np.float32)/255.0
    x_train_grey=tf.image.rgb_to_grayscale(x_train)
    x_test_grey=tf.image.rgb_to_grayscale(x_test)
    #reshape image to row*col*channel
    return x_train,x_train_grey.numpy(),x_test,x_test_grey.numpy() # type: ignore
def autoencoder(input_shape:Tuple[int,int,int],
              kernel_size:int=3,
              latent_dim:int=256,
              layer_filters:List[int]=[64,128,256]):
    inputs=Input(shape=input_shape,name="encoder_input")
    x=inputs
    #stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
    for filters in layer_filters:
        x=Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 strides=2,
                 activation=tf.nn.relu,
                 padding="same")(x)
    #shape info to build decoder model
    shape=K.int_shape(x)
    x=Flatten()(x)
    x=Dense(latent_dim,name="latent_vector")(x)
    x=Dense(shape[1]*shape[2]*shape[3])(x)
    #create decoder
    x=Reshape((shape[1],shape[2],shape[3]))(x)
    for filters in layer_filters[::-1]:
        x=Conv2DTranspose(filters=filters,
                          kernel_size=kernel_size,
                          strides=2,
                          activation=tf.nn.relu,
                          padding="same")(x)
    outputs=Conv2DTranspose(filters=3,
                            kernel_size=kernel_size,
                            activation="sigmoid",
                            padding="same",
                            name="decoder_output")(x)
    coder=Model(inputs,outputs,name="autoencoder")
    coder.summary()
    plot_model(coder,to_file="color_autoencoder.png",show_shapes=True)
    return coder
    
if __name__=="__main__":
    (x_train,_),(x_test,_)=cifar10.load_data()
    x_train,x_train_grey,x_test,x_test_grey=data_preprocess(x_train,x_test)
    print(type(x_test_grey))
    autoencoder(input_shape=x_train_grey.shape[1:])
    