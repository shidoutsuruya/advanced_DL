from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import *
import gan
import os
def data_preprocess():
    """load minst data and preprocess it

    Returns:
        (np.ndarray,np.ndarray: x and y
    """
    (x_train,y_train),(_,_)=mnist.load_data()
    x_train=x_train.reshape(-1,28,28,1).astype(np.float32)/255.0
    y_train=to_categorical(y_train)
    return x_train,y_train
def build_and_train_models(latent_size:int=100):
        #network parameters
        lr=2e-4
        #data processing and initial parameters
        x_train,y_train=data_preprocess()
        image_size=x_train.shape[1]
        num_labels=y_train.shape[1]
        #discriminator
        inputs=Input(shape=(image_size,image_size,1),name="discriminator_input")
        discriminator=gan.discriminator(inputs=inputs,num_labels=num_labels)
        discriminator.compile(loss=[tf.keras.losses.binary_crossentropy,
                                    tf.keras.losses.categorical_crossentropy],
                              optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=lr),
                              metrics=tf.keras.metrics.Accuracy())
        #generator model
        inputs=Input(shape=(latent_size,),name="z_input")
        labels=Input(shape=(num_labels,),name="labels")
        generator=gan.generator(inputs=inputs,image_size=image_size,labels=labels)
        #create adversarial model
        discriminator.trainable=False
        adversial=Model([inputs,labels],
                        discriminator(generator([inputs,labels])),
                        name="adversarial_model")
        models=(generator,discriminator,adversial)
        return models
def train(model_name:str,
          img_size:int=16,
          train_steps:int=40000,
        batch_size:int=32,
        latent_size:int=100,
        save_interval=100):
    #load models
    generator,discriminator,adversarial=build_and_train_models()
    x_train,y_train=data_preprocess()
    train_size=x_train.shape[0]
    num_labels=y_train.shape[1]
    for i in range(train_steps):
        rand_indexes=np.random.randint(0,
                                       train_size,
                                       size=batch_size)
        real_images=x_train[rand_indexes]
        real_labels=y_train[rand_indexes]
        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        fake_labels=np.eye(num_labels)[np.random.choice(num_labels,batch_size)]
        fake_images=generator.predict([noise,fake_labels])
        x=np.concatenate((real_images,fake_images))
        labels=np.concatenate((real_labels,fake_labels))
        #label real and fake images
        y=np.ones([2*batch_size,1])
        y[batch_size:,:]=0.0
        #train discriminator
        metrics=discriminator.train_on_batch(x,[y,labels])
        print(metrics)
        fmt=f"{i}: [discriminator loss:{metrics[0]},\
        src loss:{metrics[1]},label loss:{metrics[2]},\
        src acc:{metrics[3]},label acc:{metrics[4]}]"
        print(fmt)
        #train the adversarial network
        if (i+1)%save_interval==0:
            gan.plot_images(generator,
                        img_dir="ACGAN_images",
                        num_labels=num_labels,
                        step=(i+1),
                        )
        generator.save(os.path.join("model_saves",model_name))
if __name__=="__main__":
    train(model_name="acgan_mnist.h5")