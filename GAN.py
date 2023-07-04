from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical,plot_model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os 
from typing import *
def build_generator(input_shape:Tuple[int,...],image_size:int):
    """generator is the model to generate fake images
        
    Args:
        input_shape (Tuple[int,...]): one dimension random vector
        image_size (int): generate image size
    Returns:
        Model: generator model
    """
    #initial parameters
    image_resize=image_size//4
    layer_filters=[128,64,32,1]
    #model create
    inputs=Input(shape=input_shape,name='generator_input')
    x=Dense(image_resize*image_resize*layer_filters[0])(inputs)
    x=Reshape((image_resize,image_resize,layer_filters[0]))(x)
    for filters in layer_filters:
        if filters>layer_filters[-2]:
            strides=2
        else:
            strides=1
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2DTranspose(filters=filters,
                          kernel_size=5,
                          strides=strides,
                          padding='same')(x)
    x=Activation(tf.nn.sigmoid)(x)
    generator=Model(inputs,x,name='generator')
    return generator
def build_discriminator(input_shape:Tuple[int,...]):
    """discriminator

    Args:
        input_shape (Tuple[int,...]): input layer of the discriminator

    Returns:
        Model: discriminator model
    """
    layer_filters=[32,64,128,256]
    inputs=Input(shape=input_shape,name='discriminator_input')
    x=inputs
    for filters in layer_filters:
        if filters==layer_filters[-1]:
            strides=1
        else:
            strides=2
        x=LeakyReLU(alpha=0.2)(x)
        x=Conv2D(filters=filters,
                 kernel_size=5,
                 strides=strides,
                 padding='same')(x)
    x=Flatten()(x)
    x=Dense(units=1)(x)
    x=Activation(tf.nn.sigmoid)(x)
    discriminator=Model(inputs,x,name='discriminator')
    return discriminator

def build_and_train_models(image_size:int,latent_size:int=100):
    """ build adversarial model from the generator and distriminator
    """
    lr=2e-4
    decay=6e-8
    input_shape=(image_size,image_size,1)
    #discrominator model
    discriminator=build_discriminator(input_shape)
    discriminator.compile(loss=tf.keras.losses.binary_crossentropy,
                          optimizer=tf.keras.optimizers.Adam(lr=lr,
                                                             decay=decay),
                          metrics=[tf.keras.metrics.Accuracy()])
    discriminator.summary()
    plot_model(discriminator,to_file='discriminator.png',show_shapes=True)
    #generator model
    input_shape_g=(latent_size,)
    generator=build_generator(input_shape_g,image_size)
    plot_model(generator,to_file='generator.png',show_shapes=True)
    generator.summary()
    #build adversarial model=generator+discriminator
    discriminator.trainable=False
    adversarial=Model(inputs=generator.input,
                      outputs=discriminator(generator.output),
                      name="adversarial_model")
    adversarial.compile(loss=tf.keras.losses.binary_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(lr=lr*0.5,decay=decay*0.5),
                        metrics=[tf.keras.metrics.Accuracy()])
    adversarial.summary()
    plot_model(adversarial,to_file='adversarial.png',show_shapes=True)
    models=(generator,discriminator,adversarial)
    return models
def train(models,
          save_interval=100,
          latent_size:int=100,
          img_num:int=16):
    """_train the Discriminator and Adversarial Networks
    """
    #data preparation
    (x_train,_),(_,_) = mnist.load_data()
    image_size=x_train.shape[1]
    x_train=np.reshape(x_train,[-1,image_size,image_size,1])
    x_train=x_train.astype(np.float32)/255
    #network parameters
    train_steps=10000
    img_dir="gan_images"
    #load model
    generator,discriminator,adversarial=models
    #noise vector
    noise_input=np.random.uniform(-1.0,1.0,size=[img_num,latent_size])
    #number of elements in train dataset
    train_size=x_train.shape[0]
    batch_size=64
    for i in range(train_steps):
        #random sampling
        rand_indexes=np.random.randint(0,train_size,size=batch_size)
        real_images=x_train[rand_indexes]
        #generate fake images
        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        fake_images=generator.predict(noise)
        #train the discriminator network
        x=np.concatenate((real_images,fake_images))
        y=np.ones([2*batch_size,1])
        y[batch_size:,:]=0.0
        loss,acc=discriminator.train_on_batch(x,y)
        log=f"{i:d}:[discriminator loss:{loss:.3f},acc:{acc:.3f}]"
        print(log)
        #train the adversarial network
        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        y=np.ones([batch_size,1])
        #noise the real
        loss,acc=adversarial.train_on_batch(noise,y)
        log=f"{log}[adversarial loss:{loss:.3f},acc:{acc:.3f}]"
        print(log)
        if (i+1)%save_interval==0:
            plot_images(generator,
                        noise_input=noise_input,
                        img_dir=img_dir,
                        show=False,
                        step=(i+1))
            generator.save(os.path.join("model_saves","GAN_generator.h5"))
def plot_images(generator,
                noise_input,
                img_dir,
                show=False,
                step=0,
                ):
    """generate fake images

    Args:
        generator (_type_): _description_
        noise_input (_type_): _description_
        show (bool, optional): _description_. Defaults to False.
        step (int, optional): _description_. Defaults to 0.
        model_name (str, optional): _description_. Defaults to "gan".
    """
    os.makedirs(img_dir,exist_ok=True)
    file_name=os.path.join(img_dir,f"mnist_{step}.png")
    predict_images=generator.predict(noise_input)
    predict_images=predict_images.reshape(4,4,28,28,1)
    imgs=np.vstack([np.hstack(i) for i in predict_images])
    plt.figure()
    plt.imshow(imgs,interpolation='none',cmap='gray')
    plt.savefig(file_name)
    if show:
        plt.show()
    else:
        plt.close("all")
    
if __name__ == "__main__":
    latent_size=100
    image_size=28
    models=build_and_train_models(image_size,latent_size)
    train(models,latent_size)