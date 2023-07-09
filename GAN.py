from tensorflow.keras.layers import Input,concatenate,Dense,Reshape,LeakyReLU,Flatten
from tensorflow.keras.layers import BatchNormalization,Activation,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from typing import *
def generator(inputs:Layer,
              image_size:int,
              activation:str="sigmoid",
              labels: Optional[np.ndarray]=None,
              code:Optional[list]=None): 
    """build a generator model

    Args:
        inputs (Layer): Input layer
        image_size (int): image size for mnist
        activation (str, optional): activation function. Defaults to "sigmoid".
        labels (Optional[np.ndarray], optional): whether load label. Defaults to None.
        code (Optional[list], optional): for infogan. Defaults to None.
    Returns:
        Model: generator model
    """
    image_resize=image_size//4
    layer_filters=[128,64,32,1]
    if labels is not None:
        if code is None:
            inputs=[inputs,labels] #type:ignore
        else:
            inputs=[inputs,labels]+code #type:ignore
        x=concatenate(inputs,axis=1)
    elif code is not None:
        inputs=[inputs,codes]#type:ignore
        x=concatenate(inputs,axis=1)
    else:
        #default input is 100-dim noise
        x=inputs
    x=Dense(image_resize*image_resize*layer_filters[0])(x)
    x=Reshape((image_resize,image_resize,layer_filters[0]))(x)
    #filter 
    for filters in layer_filters:
        if filters>layer_filters[-2]:
            strides=2
        else:
            strides=1
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=Conv2DTranspose(filters=filters,
                          kernel_size=5,
                          strides=strides,
                          padding="same")(x)
    if activation is not None:
        x=Activation(activation)(x)
    return Model(inputs,x,name="generator")
def discriminator(inputs: Layer,
                  activation:str="sigmoid",
                  num_labels:Optional[int]=None,
                  num_codes:Optional[list]=None):
    """build discriminator model

    Args:
        inputs (Layer): input tensorflow layer
        activation (str, optional): activation function. Defaults to "sigmoid".
        num_labels (Optional[int], optional): tag number. Defaults to None.
        num_codes (Optional[list], optional): infogan. Defaults to None.
    """
    x=inputs
    layer_filters=[8,32,64,128]
    for filters in layer_filters:
        if filters==layer_filters[-1]:
            strides=1
        else:
            strides=2
        x=LeakyReLU(alpha=0.2)(x)
        x=Conv2DTranspose(filters=filters,
                          kernel_size=5,
                          strides=strides,
                          padding="same")(x)
    x=Flatten()(x)
    outputs=Dense(1)(x)
    if activation is not None:
        outputs=Activation(activation)(outputs)
    if num_labels:
        layer=Dense(layer_filters[-2])(x)
        labels=Dense(num_labels)(layer)
        labels=Activation(tf.nn.softmax,name="label")(labels)
        if num_codes is None:
            outputs=[outputs,labels]
        else:
            code1=Dense(1)(layer)
            code1=Activation("sigmoid",name="code1")(code1)
            code2=Dense(1)(layer)
            code2=Activation("sigmoid",name="code2")(code2)
            outputs=[outputs,labels,code1,code2]
    elif num_codes is not None:
        z0_recon=Dense(num_codes[0])(x)
        z0_recon=Activation("tanh",name="z0")(z0_recon)
        outputs=[outputs,z0_recon]
    return Model(inputs,outputs,name="discriminator")
def plot_images(model:Model,
                img_dir:str,
                step:int,
                num_labels:Optional[str]=None,
                latent_size:int=100,
                img_num:int=16,
                show_shape:tuple=(4,4),
                show=False,
                ):
    """plot the images
    Args:
        model (Model): load generator model
        latent_size (int): latent size
        img_num (int): show image num 16
        show_shape (tuple): to shape 16->(4,4) or (2,8)
        img_dir (str): save image dir
        show (bool, optional): whether plt.show. Defaults to False.
        step (int, optional): training save step. Defaults to 0.
    """    
    os.makedirs(img_dir,exist_ok=True)
    file_name=os.path.join(img_dir,f"mnist_{step}.png")
    noise_input=np.random.uniform(-1.0,1.0,size=[img_num,latent_size])
    predict_images=model.predict(noise_input)
    predict_images=predict_images.reshape(*show_shape,28,28,1)
    imgs=np.vstack([np.hstack(i) for i in predict_images])
    plt.figure()
    plt.imshow(imgs,interpolation='none',cmap='gray')
    plt.axis('off')
    if num_labels:
        plt.title(num_labels)
    plt.savefig(file_name)
    if show:
        plt.show()
    else:
        plt.close("all")