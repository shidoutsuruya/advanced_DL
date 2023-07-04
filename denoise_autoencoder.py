from tensorflow.keras.layers import Input, Dense,Conv2D,Flatten,Reshape,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import *
from PIL import Image
def data_load():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    #resize and normalize
    image_size=x_train.shape[1]
    x_train=np.reshape(x_train,[-1,image_size,image_size])
    x_test=np.reshape(x_test,[-1,image_size,image_size])
    x_train=x_train.astype(np.float32)/255
    x_test=x_test.astype(np.float32)/255
    #add noise
    noise=np.random.normal(loc=0.5,scale=0.5,size=x_train.shape)
    x_train_noisy=x_train+noise
    noise=np.random.normal(loc=0.5,scale=0.5,size=x_test.shape)
    x_test_noisy=x_test+noise
    #clip pixel to range(0,1)
    x_train_noisy=np.clip(x_train_noisy,0.,1.)
    x_test_noisy=np.clip(x_test_noisy,0.,1.)
    return (x_train,x_train_noisy),(x_test,x_test_noisy),image_size
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
    #plot_model(encoder,
               #to_file="encoder.png",
               #show_shapes=True)
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
    #plot_model(decoder,to_file="decoder.png",show_shapes=True)
    return decoder
def autoencoder(input_shape:Tuple[int,int,int]):
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
    #plot_model(coder,to_file="coder.png",show_shapes=True)
    return coder
def train(model,
          x_train,
          x_train_noisy,
          x_test,
          x_test_noisy,
          batch_size:int=128,
          epochs:int=10):
    checkpoint=tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join("model_saves","noisy_autoencoder.h5"),
    monitor='val_loss',
    save_best_only=True)
    model.compile(optimizer="adam",
                  loss="mse")
    model.fit(x_train_noisy,
              x_train,
              validation_data=(x_test_noisy,x_test),
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[checkpoint])
    print("model train successfully")
    return model
def evaluate(model_path:str,
             x_test:np.ndarray,
             x_test_noisy:np.ndarray,image_size:int=28):
    model=tf.keras.models.load_model(model_path)
    x_decoder=model.predict(x_test_noisy)
    fig_draw(np.expand_dims(x_test,-1),np.expand_dims(x_test_noisy,-1),x_decoder,image_size)
def fig_draw(x_test,x_test_noisy,x_decoder,image_size):
    rows,cols=3,9
    num=rows*cols
    imgs=np.concatenate([x_test[:num],x_test_noisy[:num],x_decoder[:num]])
    imgs=imgs.reshape((rows*3,cols,image_size,image_size))
    imgs=np.vstack(np.split(imgs,rows,axis=1))
    imgs=imgs.reshape((rows*3,-1,image_size,image_size))
    imgs=np.vstack([np.hstack(i) for i in imgs])
    imgs=(imgs*255).astype(np.uint8)
    plt.figure()
    plt.axis("off")
    plt.imshow(imgs,interpolation="none",cmap="gray")
    plt.show()
if __name__=="__main__":
    (x_train,x_train_noisy),(x_test,x_test_noisy),image_size=data_load()
    input_shape=(image_size,image_size,1)
    #train model
    #model=autoencoder(input_shape)
    #train(model,x_train,x_train_noisy,x_test,x_test_noisy)
    evaluate(r"model_saves/noisy_autoencoder.h5",
             x_test,x_test_noisy,image_size)
    
    