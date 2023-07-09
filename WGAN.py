from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import tensorflow as tf
from classical_GAN import build_generator,build_discriminator,plot_images
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
    return models,x_train
def wasserstein_loss(y_true,y_pred):
    return -K.mean(y_true*y_pred)
def train(models,
          x_train:np.ndarray,
          save_interval:int=100,
          train_steps:int=2000,
          img_num:int=16,
          batch_size:int=64,
          latent_size:int=100,
          n_critic:int=5,
          clip_value:float=0.01,
          ):
    generator,discriminator,adversarial=models
    noise_input=np.random.uniform(-1.0,1.0,size=[img_num,latent_size])
    train_size=x_train.shape[0]
    real_labels=np.ones([batch_size,1])
    for i in range(train_steps):
        #train discriminator n_critic times
        loss=0
        acc=0
        for _ in range(n_critic):
            rand_indexes=np.random.randint(0,
                                           train_size,
                                           size=batch_size)
            real_images=x_train[rand_indexes]
            noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
            fake_images=generator.predict(noise)
            #compute real and fake loss and acc
            real_loss,real_acc=discriminator.train_on_batch(real_images,
                                                            real_labels)
            fake_loss,fake_acc=discriminator.train_on_batch(fake_images,
                                                            -real_labels)
            #accumulate average loss and acc
            loss+=0.5*(real_loss+fake_loss)
            acc+=0.5*(real_acc+fake_acc)
            #clip discriminator weights to satisfy Lipschitz condition
            for layers in discriminator.layers:
                weights=layers.get_weights()
                weights=[np.clip(weight,
                                 -clip_value,
                                 clip_value) for weight in weights]
                layers.set_weights(weights)
        #average loss and accuracy per n_critic training iteration
        loss/=n_critic
        acc/=n_critic
        log1=f"{i}:[discriminator loss:{loss:.3f},acc:{acc:.3f}]"
        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        #adversial network training
        loss,acc=adversarial.train_on_batch(noise,real_labels)
        log2=f"{i}:[adversarial loss:{loss:.3f},acc:{acc:.3f}]"
        print(log1+log2)
        if (i+1)%save_interval==0:
            plot_images(generator,
                        latent_size=100,
                        img_num=16,
                        show_shape=(4,4),
                        img_dir='wgan_image',
                        step=(i+1)
            )
        generator.save("wgan.h5")
if __name__=='__main__':
    train(*build_and_train_models())
    
    
    
