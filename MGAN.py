from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import tensorflow as tf
from GAN import build_generator,build_discriminator
import numpy as np
import sys

def build_and_train_models():
    #load training data
    (x_train,_),(_,_)=mnist.load_data()
    image_size=x_train.shape[1]
    x_train=np.reshape(x_train,[-1,image_size,image_size,1])
    x_train=x_train.astype(np.float32)/255
    #initial parameters
    latent_size=100
    #discriminator model
    discriminator=build_discriminator(input_shape=(image_size,image_size,1))
    discriminator.compile(loss=wasserstein_loss,
                          optimizer=tf.keras.optimizers.RMSprop(lr=0.00005),
                          metrics=['accuracy'])
    discriminator.summary()
    #generator model
    generator=build_generator((latent_size,),image_size)
    #adversarial model=generator+discriminator
    discriminator.trainable=False
    adversarial=Model(inputs=generator.input,
                      outputs=discriminator(generator.output)
                      ,name='adversarial_model')
    adversarial.compile(loss=wasserstein_loss,
                        optimizer=tf.keras.optimizers.RMSprop(lr=0.00005),
                        metrics=['accuracy'])
    models=(generator,discriminator,adversarial)
    return models