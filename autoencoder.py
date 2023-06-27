from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Conv2D,Flatten
from tensorflow.keras.layers import Reshape,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import *
#global variables
shape=0
image_size=28
def data_load():
    (x_train,_),(x_test,_)=mnist.load_data()
    #resize and normalize
    image_size=x_train.shape[1]
    x_train=np.reshape(x_train,[-1,image_size,image_size])
    x_test=np.reshape(x_test,[-1,image_size,image_size])
    x_train=x_train.astype(np.float32)/255
    x_test=x_test.astype(np.float32)/255
    return x_train,x_test
#network parameters
def encoder(inputs,
            layer_filters:List[int]=[32,64],
                    kernel_size:int=3,
                    latent_dim:int=6):
    """encoder to pick down

    Args:
        input_shape Tuple[int,int,int]: input shape. Defaults to input_shape.
        layer_filters List[int]: filters. Defaults to [32,64].
        kernel_size (int, optional):kernel size. Defaults to 3.
    Returns:
        encoder(Model): tensor model
    """
    x=inputs
    for filters in layer_filters:
        x=Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation=tf.keras.activations.relu,
                 strides=2,
                 padding="same")(x)
    global shape
    shape=K.int_shape(x)
    x=Flatten()(x)
    latent=Dense(latent_dim,name="latent_vector")(x)
    encoder=Model(inputs,
                  latent,
                  name="encoder")
    encoder.summary()
    plot_model(encoder,
               to_file="encoder.png",
               show_shapes=True)
    return encoder
def decoder(kernel_size:int=3,
            layer_filters:List[int]=[32,64],
            latent_dim:int=6):
    latent_inputs=Input(shape=(latent_dim,),
                        name="decoder_input")
    x=Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x=Reshape((shape[1],shape[2],shape[3]))(x)
    for filters in layer_filters[::-1]:
        x=Conv2DTranspose(filters=filters,
                          kernel_size=(kernel_size,kernel_size),
                          activation=tf.keras.activations.relu,
                          strides=2,
                          padding="same")(x)
    outputs=Conv2DTranspose(filters=1,
                            kernel_size=(kernel_size,kernel_size),
                            activation=tf.keras.activations.sigmoid,
                            padding="same",
                            name="decoder_output")(x)
    decoder=Model(latent_inputs,outputs,name="decoder")
    decoder.summary()
    plot_model(decoder,to_file="decoder.png",show_shapes=True)
    return decoder
def autoencoder(input_shape=(image_size,image_size,1)):
    inputs=Input(shape=input_shape,name="encoder_input")
    a=encoder(inputs)
    coder=Model(inputs,decoder(a),name="autoencoder")
    plot_model(coder,to_file="autoencoder.png",show_shapes=True)
    return coder
if __name__ == "__main__":
    autoencoder()
    
