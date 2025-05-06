import numpy as np
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import winsound
import re
import os


def Modded_CNN(X_train, X_test, y_train, y_test,num_classes,pool,path_pickel,key):
    
    input_shape =(28,28,1)
    input_layer = keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (5,5), activation='swish',data_format='channels_last')(input_layer) 
    max_pool1 = layers.MaxPooling2D((3, 3))(conv1)
    
    conv2 = layers.Conv2D(50, (3, 3), activation='swish',data_format='channels_last')(max_pool1)
    max_pool2 = layers.MaxPooling2D((2, 2))(conv2) 
    
    conv3 = layers.Conv2D(80,(2, 2), activation='swish',data_format='channels_last')(max_pool2)
    max_pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = layers.Conv2D(120,(1, 1), activation='swish',data_format='channels_last')(max_pool3)
    max_pool4 = layers.MaxPooling2D((1, 1))(conv4)
    
    
    global_avg_pool = layers.GlobalAveragePooling2D()(max_pool4)
    dense1 = layers.Dense(120, activation='relu')(global_avg_pool)
    dropout = layers.Dropout(0.4)(dense1)  

    skip_connection = layers.Add()([global_avg_pool, dropout]) 

    output_layer = layers.Dense(num_classes, activation='softmax')(skip_connection)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    pickel_name = path_pickel+"\\"+ key+".keras"
    model.save(pickel_name)

    return model



