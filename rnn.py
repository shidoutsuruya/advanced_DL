import numpy as np
from typing import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,SimpleRNN
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.datasets import mnist
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
#create model
def rnn_model(input_shape:Tuple[int,int],
        num_labels:int,     
        units:int=256,
        dropout:float=0.2,
        ):
    model=Sequential()
    model.add(SimpleRNN(units=units,
                        dropout=dropout,
                        input_shape=input_shape))
    model.add(Dense(num_labels))
    model.add(Activation("softmax"))
    model.summary()
    return model
if __name__=="__main__":
    image_size=28
    x_train,x_test,y_train,y_test=data_load()
    my_model=rnn_model(input_shape=(image_size,image_size),num_labels=10)
    my_model.compile(loss="categorical_crossentropy",
                    optimizer="sgd",
                    metrics=["accuracy"])
    my_model.fit(x_train,y_train,epochs=20,batch_size=128)
    _,acc=my_model.evaluate(x_test,y_test,batch_size=128,verbose=1)
    print("test accuracy:",acc)

    
