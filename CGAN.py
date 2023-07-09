from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model,to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os

def build_generator(inputs,
                    labels,
                    image_size:int):
    """generator model

    Args:
        inputs (Layer):Input(shape,)
        labels (Layer):Input(shape,)
        image_size (int): image size

    Returns:
        Model: tensorflow model
    """
    image_resize=image_size//4
    layer_filters=[128,64,32,1]
    x=concatenate([inputs,labels],axis=1)
    x=Dense(image_resize*image_resize*layer_filters[0])(x)
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
    x=Activation('sigmoid')(x)
    generator=Model([inputs,labels],x,name="generator")
    return generator

def build_discriminator(inputs,
                        labels,
                        image_size:int):
    """discriminator model

    Args:
        inputs (Layer): Input(shape,)
        labels (Layer): Input(shape,)
        image_size (int): image size

    Returns:
        Model: tensorflow model
    """
    layer_filters=[32,64,128,256]
    #input
    x=inputs
    y=Dense(image_size*image_size)(labels)
    y=Reshape((image_size,image_size,1))(y)
    x=concatenate([x,y])
    for filter in layer_filters:
        if filter==layer_filters[-1]:
            strides=1
        else:
            strides=2
        x=LeakyReLU(alpha=0.2)(x)
        x=Conv2D(filter,
                 kernel_size=5,
                 strides=strides,
                 padding='same')(x)
    x=Flatten()(x)
    x=Dense(1)(x)
    x=Activation('sigmoid')(x)
    discriminator=Model([inputs,labels],x,name="discriminator")
    return discriminator

def build_models(image_size:int,latent_size:int=100,num_labels=10):
    """create adversarial model
    
    Args:
        image_size (int): image size
        latent_size (int, optional): latent size. Defaults to 100.
        num_labels (int, optional): categories number. Defaults to 10.
    Returns:
        tuple[Model,Model,Model]: (generator,discriminator,adversarial)
    """
    #initial parameters
    lr=2e-4
    decay=6e-8
    #build discriminator
    inputs=Input((image_size,image_size,1),name="discriminator_input")
    labels=Input((num_labels,),name="class_labels")
    discriminator=build_discriminator(inputs,labels,image_size)
    discriminator.compile(loss=tf.keras.losses.binary_crossentropy,
                          optimizer=tf.keras.optimizers.RMSprop(lr=lr,decay=decay),
                          metrics=[tf.keras.metrics.Accuracy(),])
    plot_model(discriminator,
               to_file=os.path.join("model_structure","discriminator.png"),
               show_shapes=True)
    #build generator
    inputs=Input((latent_size,),name="discriminator_input")
    generator=build_generator(inputs,labels,image_size)
    plot_model(generator,
               to_file=os.path.join("model_structure","generator.png"),
               show_shapes=True)
    #adversarial model=generator+discriminator
    discriminator.trainable=False
    outputs=discriminator([generator([inputs,labels]),labels])
    adversarial=Model([inputs,labels],outputs,name="adversarial")
    adversarial.compile(loss=tf.keras.losses.binary_crossentropy,
                        optimizer=tf.keras.optimizers.RMSprop(lr=lr*0.5,decay=decay*0.5),
                        metrics=[tf.keras.metrics.Accuracy(),])
    plot_model(adversarial,
               to_file=os.path.join("model_structure","adversarial.png"),
               show_shapes=True)
    models=(generator,discriminator,adversarial)
    return models

def train(train_steps:int=40000,
          batch_size:int=64,
          save_interval:int=100,
          model_name="cgan.h5"):
    """_summary_

    Args:
        train_steps (int, optional): back propagation number. Defaults to 40000.
        batch_size (int, optional): batch size. Defaults to 64.
        save_interval (int, optional): every interval for saving image. Defaults to 100.
        model_name (str, optional): save model name. Defaults to "cgan.h5".
    """
    #data preprocessing
    (x_train,y_train),(_,_)=mnist.load_data()
    x_train=np.reshape(x_train,[-1,28,28,1])
    x_train=x_train.astype(np.float32)/255
    y_train=to_categorical(y_train)
    #initial parameters
    train_size=x_train.shape[0]
    image_size=x_train.shape[1]
    latent_size=100
    num_labels=y_train.shape[1]
    #model loading
    generator,discriminator,adversarial=build_models(image_size,latent_size,num_labels)
    for i in range(train_steps):
        #random generate index
        rand_indexes=np.random.randint(0,train_size,size=batch_size)
        real_images=x_train[rand_indexes]
        real_labels=y_train[rand_indexes]
        #generate fake image
        noise=np.random.uniform(-1,1,size=(batch_size,latent_size))
        fake_labels=np.eye(num_labels)[np.random.choice(num_labels,batch_size)]
        fake_images=generator.predict([noise,fake_labels])
        #real+fake images=1 batch of data
        x=np.concatenate((real_images,fake_images))
        labels=np.concatenate((real_labels,fake_labels))
        #label real and fake images
        #real=1,fake=0
        y=np.ones([2*batch_size,1])
        y[batch_size:,:]=0.
        #train discriminator
        loss,acc=discriminator.train_on_batch([x,labels],y)
        log1=f"{i}:[discriminator loss:{loss:.3},acc:{acc:.3}]"
        #train adversarial
        y=np.ones([batch_size,1])
        loss,acc=adversarial.train_on_batch([noise,fake_labels],y)
        log2=f"[adversarial loss:{loss:.3},acc:{acc:.3}]"
        print(log1+log2)
        if (i+1)%save_interval==0:
            plot_images(model=generator,
                        num_labels=num_labels,
                        step=(i+1))
        generator.save(os.path.join("model_saves",model_name))
def plot_images(model:Model,
                img_dir: str,
                num_labels:int,
                step:int=0,
                img_num:int=16,
                show_shape:tuple=(4,4),
                latent_size:int=100, 
                show:bool=False,
                ):
    """draw images

    Args:
        model (Model): generator
        num_labels (int): category number
        step (int, optional): the training step. Defaults to 0.
        img_num (int, optional): the digits number show in image. Defaults to 16.
        show_shape (tuple, optional): show shape multiplier of img_num. Defaults to (4,4).
        latent_size (int, optional): initial noize data size. Defaults to 100.
        img_dir (str, optional): image saving dir. Defaults to "cgan_image".
        show (bool, optional): whether show image in GUI. Defaults to False.
    """    
    #save image dir
    os.makedirs(img_dir,exist_ok=True)
    file_name=os.path.join(img_dir,f"mnist_{model.name}_{step}.png")
    #noise input and label
    noise_input=np.random.uniform(-1.,1.,size=(img_num,latent_size))
    noise_class=np.eye(num_labels)[np.random.choice(num_labels,img_num)]
    #predict
    predict_images=model.predict([noise_input,noise_class])
    predict_images=predict_images.reshape(*show_shape,28,28,1)
    imgs=np.vstack([np.hstack(i) for i in predict_images])
    #draw image
    plt.figure()
    plt.title(str(np.argmax(noise_class,axis=1)))
    plt.imshow(imgs,interpolation='none',cmap='gray')
    plt.axis('off')
    plt.savefig(file_name)
    if show:
        plt.show()
    else:
        plt.close("all")
    
if __name__=="__main__":
    train()