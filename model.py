from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



def cnn_small(pretrained_weights=None, image_size=(128, 256, 3), input_chan=3, output_chan=3):
    print("Generating CNN-Small Model using settings: ")
    print("\t- Pretrained Weights= " + str(pretrained_weights))
    print("\t- image_size= " + str(image_size))

    inputs = Input(image_size)
    norm = BatchNormalization()(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(norm)
    conv1 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)


    flat = Flatten()(conv5)
    dense1 = RepeatVector(50)(flat)
    dense1 = Dense(150)(dense1)
    out1 = Dense(2,input_dim=3)(dense1)

    model = Model(input=inputs, output=out1)

    model.compile(optimizer=Adam(lr=0.01),
        loss='mae', metrics=["mae"])

    print(model.summary())

    if(pretrained_weights):
        print('Loading Weights from ' + pretrained_weights)
        model.load_weights(pretrained_weights)
    print("Saving to JSON")
    model_json = model.to_json()
    with open("cnn-small.json", "w") as json_file:
        json_file.write(model_json)
    return model
