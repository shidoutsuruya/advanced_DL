import numpy as np
from typing import *
from tensorflow.keras.layers import Dense,Dropout,Input,Conv2D,\
MaxPooling2D,Flatten,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical,plot_model
import tensorflow as tf
#global variable
image_size=28
def data_load():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    num_labels=len(np.unique(y_train))
    #convert the number to label
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    #resize and normalize
    global image_size
    x_train=np.reshape(x_train,[-1,image_size,image_size])
    x_test=np.reshape(x_test,[-1,image_size,image_size])
    x_train=x_train.astype(np.float32)/255
    x_test=x_test.astype(np.float32)/255
    return x_train,x_test,y_train,y_test

def y_network(
    input_shape:Tuple[int,int,int]=(image_size,image_size,1),
    batch_size:int=32,
    kernel_size:int=3,
    dropout:float=0.4,
    n_filters:int=32,
    num_labels:int=10
):
    #left side
    left_inputs=Input(shape=input_shape)
    x=left_inputs
    filters=n_filters
    for i in range(3):
        x=Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 padding="same",
                 activation="relu")(x)
        x=Dropout(dropout)(x)
        x=MaxPooling2D()(x)
        filters*=2
    #right side
    right_inputs=Input(shape=input_shape)
    y=right_inputs
    filters=n_filters
    for i in range(3):
        y=Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 padding="same",
                 activation="relu",
                 dilation_rate=2)(y)
        y=Dropout(dropout)(y)
        y=MaxPooling2D()(y)
        filters*=2
    #concatenate
    y=concatenate([x,y])
    y=Flatten()(y)
    y=Dropout(dropout)(y)
    outputs=Dense(num_labels,activation="softmax")(y)
    #package model
    model=Model([left_inputs,right_inputs],outputs)
    model.summary()
    return model
if __name__=="__main__":
    #load function
    model=y_network()
    x_train,x_test,y_train,y_test=data_load()
    #model trainingy
    plot_model(model,to_file="hello.png",show_shapes=True)
    model.compile(loss=tf.losses.categorical_crossentropy,
                  optimizer=tf.optimizers.Adam(),
                  metrics=["accuracy"])
    model.fit([x_train,x_train],y_train,
              validation_data=([x_test,x_test],y_test),
              epochs=20)
    score=model.evaluate([x_test,x_test],y_test,batch_size=128)