from tensorflow.keras.layers import Dense,Conv2D
from tensorflow.keras.layers import BatchNormalization,Activation
from tensorflow.keras.layers import AveragePooling2D,Input
from tensorflow.keras.layers import Flatten,add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model,to_categorical
import numpy as np
import os
import tensorflow as tf
from typing import *
#global variables
BASE=os.path.join(os.path.dirname(__file__))

def load_data(num_classes:int=10,substract_pixel_mean:bool=True):
    """load cifar10 data
    Args:
        num_classes (int, optional): _description_. Defaults to 10.
        substract_pixel_mean (bool, optional): _description_. Defaults to True.
    Returns:
        _type_: _description_
    """
    (x_train,y_train),(x_test,y_test)=cifar10.load_data()
    input_shape=x_train.shape[1:]
    x_train=x_train.astype(np.float32)/255
    x_test=x_test.astype(np.float32)/255
    if substract_pixel_mean:
        x_train_mean=np.mean(x_train,axis=0)
        x_train-=x_train_mean
        x_test-=x_train_mean
    y_train=to_categorical(y_train,num_classes)
    y_test=to_categorical(y_test,num_classes)
    return x_train,x_test,y_train,y_test,input_shape
def lr_schedule(epoch:int):
    """learning rate schedule
    reduce lr after 80 120 160 190
    call automatically as part of call back
    Args:
        epoch (int): the number of epochs
    Return:
        lr (float):learning rate
    """
    lr=1e-3
    if epoch>180:
        lr*=0.5e-3
    elif epoch>160:
        lr*=1e-3
    elif epoch>120:
        lr*=1e-2
    elif epoch>80:
        lr*=1e-1
    print(f"learning rate now:{lr}")
    return lr
def resnet_layer(inputs:Tuple[int,int,int],
                 num_filters:int=16,
                 kernel_size:int=3,
                 strides:int=1,
                 activation:str="relu",
                 batch_normalization:bool=True,
                 conv_first:bool=True):
    """resnet layer

    Args:
        input (Tuple[int,int,int]): _description_
        num_filters (int, optional): _description_. Defaults to 16.
        kernel_size (int, optional): _description_. Defaults to 3.
        strides (int, optional): _description_. Defaults to 1.
        activation (str, optional): _description_. Defaults to "relu".
        conv_first (bool, optional): _description_. Defaults to True.
    Returns:
        x(tensor):tensor as input to the next layer
    """
    conv=Conv2D(num_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))
    x=inputs
    if conv_first:
        x=conv(x)
        if batch_normalization:
            x=BatchNormalization()(x)
        if activation is not None:
            x=Activation(activation)(x)
    else:
        if batch_normalization:
            x=BatchNormalization()(x)
        if activation is not None:
            x=Activation(activation)(x)
        x=conv(x)
    return x

def resnet_v1(input_shape:Tuple[int,int,int],depth:int=3*6+2,num_classes:int=10):
    if (depth-2)%6!=0:
        raise ValueError("depth should be 6n+2")
    num_filters=16
    num_res_blocks=int((depth-2)/6)
    inputs=Input(shape=input_shape)
    x=resnet_layer(inputs=inputs)
    #instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides=1
            if stack>0 and res_block==0:
                strides=2 #downsampling
            y=resnet_layer(inputs=x,
                           num_filters=num_filters,
                           strides=strides)
            y=resnet_layer(inputs=y,
                           num_filters=num_filters,
                           activation=None)
            if stack>0 and res_block==0:
                x=resnet_layer(inputs=x,
                               num_filters=num_filters,
                               kernel_size=1,
                               strides=strides,
                               activation=None,
                               batch_normalization=False)
            x=add([x,y])
            x=Activation("relu")(x)
    num_filters*=2
    x=AveragePooling2D(pool_size=8)(x)
    y=Flatten()(x)
    outputs=Dense(num_classes,activation="softmax",kernel_initializer="he_normal")(y)
    model=Model(inputs=inputs,outputs=outputs)
    plot_model(model,to_file="resnet.png",show_shapes=True)
    return model
def model_training(model,x_train,x_test,y_train,y_test,epochs:int=1,batch_size:int=32):
    global lr_schedule
    model.compile(loss=tf.losses.categorical_crossentropy,
                  optimizer=Adam(lr_schedule(0)),
                  metrics=tf.metrics.Accuracy())
    model.summary()
    #save model to file
    save_dir=os.path.join(BASE,"model_saves")
    model_name=f"cifar10_resnet.{epochs:03}.h5"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath=os.path.join(save_dir,model_name)
    #checkpoint save
    checkpoint=ModelCheckpoint(filepath=filepath,
                               monitor="val_acc",
                               verbose=1,
                               save_best_only=True
                               )
    lr_schedule=LearningRateScheduler(lr_schedule)
    lr_reducer=ReduceLROnPlateau(factor=np.sqrt(0.1),
                                 cooldown=0,
                                 patience=5,
                                 min_lr=0.5e-6)
    callbacks=[checkpoint,lr_schedule,lr_reducer]
    #set data argumentation
    datagen=ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
    )
    datagen.fit(x_train)
    steps_per_epoch=np.ceil(len(x_train)/batch_size)
    model.fit(x=datagen.flow(x_train,y_train,batch_size=batch_size),
              verbose=1,
              epochs=epochs,
              validation_data=(x_test,y_test),
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)
    scores=model.evaluate(x_test,y_test,batch_size,verbose=0)
    print("test loss:",scores[0])
    print("test accuracy:",scores[1])
if __name__== "__main__":
    x_train,x_test,y_train,y_test,input_shape=load_data()
    model=resnet_v1(input_shape=input_shape)
    model_training(model,x_train,x_test,y_train,y_test)