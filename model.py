from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf


def cnn_small(pretrained_weights=None, image_size=(128, 256, 3), input_chan=3, output_chan=3):
    print("Generating CNN-Small Model using settings: ")
    print("\t- Pretrained Weights= " + str(pretrained_weights))
    print("\t- image_size= " + str(image_size))

    inputs = Input(image_size)
    norm = BatchNormalization()(inputs)
    conv1 = Conv2D(16, 3, activation='relu')(norm)
    conv1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu')(pool1)
    conv2 = Dropout(0.5)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(64, 3, activation='relu')(conv3)
    conv3 = Dropout(0.5)(conv3)

    flat = Flatten()(conv3)
  
    dense1 = Dense(32,activation='relu')(flat)
    dense1 = Dropout(0.2)(dense1)
    out1 = Dense(4)(dense1)

    model = Model(inputs=inputs, output=out1)

    model.compile(loss='mae',  metrics=['mae', 'acc'], optimizer='sgd')

    print(model.summary())

    if(pretrained_weights):
        print('Loading Weights from ' + pretrained_weights)
        model.load_weights(pretrained_weights)
    print("Saving to JSON")
    model_json = model.to_json()
    with open("cnn-small.json", "w") as json_file:
        json_file.write(model_json)
    return model


from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.applications.xception import Xception  

def xception(pretrained_weights=None, image_size=(128, 256, 3), input_chan=3, output_chan=3):
    print("Generating CNN-Small Model using settings: ")
    print("\t- Pretrained Weights= " + str(pretrained_weights))
    print("\t- image_size= " + str(image_size))

    inputs = Input(image_size)
    base_model = Xception(include_top=False, input_tensor=inputs, weights='imagenet')
    y = base_model.layers[-1].output
    y = GlobalAveragePooling2D()(y)
    y = Dense(60, activation='sigmoid')(y)
    model = Model(inputs=inputs, outputs=y)
    model.compile(loss='mae',  metrics=['mae', 'acc'], optimizer='sgd')

    print(model.summary())

    if(pretrained_weights):
        print('Loading Weights from ' + pretrained_weights)
        model.load_weights(pretrained_weights)
    print("Saving to JSON")
    model_json = model.to_json()
    with open("cnn-small.json", "w") as json_file:
        json_file.write(model_json)
    return model
