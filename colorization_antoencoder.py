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
def rgb2gray(rgb:np.ndarray):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])
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
    x_test=x_test.astype(np.float32)/255.0
    #for cpu
    #x_train_grey=tf.image.rgb_to_grayscale(x_train).numpy()
    #x_test_grey=tf.image.rgb_to_grayscale(x_test).numpy()
    #for gpu
    x_train_grey=np.expand_dims(rgb2gray(x_train),-1)
    x_test_grey=np.expand_dims(rgb2gray(x_test),-1)
    return x_train,x_train_grey,x_test,x_test_grey
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
def train(model,file_name:str,
          x_train:np.ndarray,
          x_train_grey:np.ndarray,
          x_test:np.ndarray,
          x_test_grey:np.ndarray):
    """traininig

    Args:
        model (tf.model): model
        file_name (str): same model file name
        x_train (np.ndarray): np.ndarray (n,32,32,3)
        x_train_grey (np.ndarray): np.ndarray (n,32,32,1)
        x_test (np.ndarray): np.ndarray (n,32,32,3)
        x_test_grey (np.ndarray): np.ndarray (n,32,32,1)
    """
    file_path=os.path.join("model_saves",file_name)
    lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),
                                 cooldown=0,
                                 patience=5,
                                 verbose=1,
                                 min_lr=0.5e-6)   
    checkpoint=ModelCheckpoint(filepath=file_path,
                               monitor="val_loss",
                               verbose=1,
                               save_best_only=True)  
    model.compile(loss=tf.keras.losses.mse,
                        optimizer=tf.keras.optimizers.Adam())
    callbacks=[lr_reducer,checkpoint]
    model.fit(x_train_grey,x_train,
                    validation_data=(x_test_grey,x_test),
                    epochs=30,
                    batch_size=128,
                    callbacks=callbacks,
                    verbose=1)
    print("finish training")
def evaluate(model_path:str,
             x_test_gray:np.ndarray,
             x_test:np.ndarray):
    model=tf.keras.models.load_model(model_path)
    x_decoded=model.predict(x_test_gray)
    #predict
    predict=x_decoded[:100]
    predict=predict.reshape((10,10,32,32,3))
    predict=np.vstack([np.hstack(i) for i in predict])
    #gold
    gold=x_test[:100]
    gold=gold.reshape((10,10,32,32,3))
    gold=np.vstack([np.hstack(i) for i in gold])
    fig,(ax1,ax2)=plt.subplots(1,2)
    ax1.set_title("predict")
    ax1.imshow(predict,interpolation="none")
    ax2.set_title("gold")
    ax2.imshow(gold,interpolation="none")
    ax1.axis("off")
    ax2.axis("off")
    plt.show()
if __name__=="__main__":
    (x_train,_),(x_test,_)=cifar10.load_data()
    x_train,x_train_grey,x_test,x_test_grey=data_preprocess(x_train,x_test)
    #train model
    #model=autoencoder(input_shape=x_train_grey.shape[1:])
    #train(model,"color_autoencoder.h5",x_train,x_train_grey,x_test,x_test_grey)
    evaluate("model_saves/color_autoencoder.h5",x_test_grey,x_test)
    