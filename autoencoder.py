from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Conv2D,Flatten
from tensorflow.keras.layers import Reshape,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import *
import os
#global variables
shape=0
image_size=28
BASE=os.path.join(os.path.dirname(__file__))
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
def encoder(input_shape,
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
    inputs=Input(shape=input_shape,name="encoder_input")
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
                          kernel_size=kernel_size,
                          activation=tf.keras.activations.relu,
                          strides=2,
                          padding="same")(x)
    outputs=Conv2DTranspose(filters=1,
                            kernel_size=kernel_size,
                            activation=tf.keras.activations.sigmoid,
                            padding="same",
                            name="decoder_output")(x)
    decoder=Model(latent_inputs,outputs,name="decoder")
    decoder.summary()
    plot_model(decoder,to_file="decoder.png",show_shapes=True)
    return decoder
def autoencoder(input_shape:Tuple[int,int,int]=(image_size,image_size,1)):
    """combine with encoder and decoder

    Args:
        input_shape (Tuple[int,int,int], optional): input shape. Defaults to (image_size,image_size,1).

    Returns:
        coder(model):autoencoder 
    """
    model1=encoder(input_shape)
    model2=decoder()
    outputs=model2(model1.output)
    coder=Model(inputs=model1.input,
                outputs=outputs)
    plot_model(coder,to_file="coder.png",show_shapes=True)
    return coder
def train(x_train,x_test):
    model=autoencoder()
    #save model
    save_dir=os.path.join(BASE,"model_saves")
    model_name=f"autocoder.h5"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath=os.path.join(save_dir,model_name)
    #use this to save the best model
    checkpoint=ModelCheckpoint(filepath=filepath,
                               monitor="loss",
                               verbose=1,
                               save_best_only=True
                               ) 
    #training
    model.compile(loss=tf.keras.losses.mse,
                  optimizer=tf.keras.optimizers.Adam())
    model.fit(x_train,x_train,validation_data=(x_test,x_test),epochs=1,
              batch_size=32,callbacks=[checkpoint,])
def evalutate(x_gold:np.array,
              model_path:str=r"model_saves\autocoder.h5"):
    x_gold=np.expand_dims(x_gold,axis=-1)
    model=tf.keras.models.load_model(model_path)
    x_decode=model.predict(x_gold)
    imgs=np.concatenate([x_gold[:8],x_decode[:8]])
    imgs=imgs.reshape((4,4,image_size,image_size))
    imgs=np.vstack([np.hstack(i) for i in imgs])
    
    plt.imshow(imgs)
    plt.show()
    return x_decode
if __name__ == "__main__":
    x_train,x_test=data_load()
    a=evalutate(x_test)
    
