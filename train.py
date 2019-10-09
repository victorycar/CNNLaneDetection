import cv2
import pandas
import numpy as np
import keras
import glob
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow import keras
from skimage.draw import line
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
from model import *

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


PATH_BASE = "data/"
VAL_COUNT = 300
TRAIN_COUNT = 800
BATCH = 16
MODEL_VERSION = "xception_v1"

model = xception()



def lane_generatoe(files, batch_size=4):

    while True:
        # Select files (paths/indices) for the batch
        batch_paths = np.random.choice(a=files,
                                       size=batch_size)
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            entry = np.load(input_path, allow_pickle=True)
            input = entry[0]
          
            x_coords = []
            y_coords = []

            for point in entry[1]:
                x_coords.append(point[0] / 1280)
            for point in entry[2]:
                x_coords.append(point[0] / 1280)
            for point in entry[3]:
                x_coords.append(point[0] / 1280)
            for point in entry[4]:
                x_coords.append(point[0] / 1280)
  

            batch_input += [input]
            batch_output += [x_coords]
            
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield(batch_x, batch_y)

print("LOADING MODEL")

try:
    model = multi_gpu_model(model)
except:
    print("~NOT USING MUTLI GPU")
    pass

model_checkpoint_best = ModelCheckpoint('weights/'+MODEL_VERSION+'.hdf5', monitor='loss',verbose=1,save_best_only=True)

def predict_model(index=0):
    image = cv2.imread("test.jpg")
    
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image = cv2.resize(image, (256,128))
    test = np.array([image])
    test = test.reshape(len(test),128,256,3)
    lanes = model.predict(test, verbose=1)
    return lanes

class Predict(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
      np.save("training/"+MODEL_VERSION+"/-predict_"+str(epoch)+".npy", predict_model())
      return
predict_cb = Predict()


from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./log/'+MODEL_VERSION)


trainFiles = glob.glob(PATH_BASE + "/train/*.npy")
valFiles = glob.glob(PATH_BASE + "/val/*.npy")

trainGen = lane_generatoe(trainFiles,BATCH)
valGen = lane_generatoe(valFiles, BATCH)

trainFiles = trainFiles[:TRAIN_COUNT]
valFiles = valFiles[:VAL_COUNT]

H = model.fit_generator(
	trainGen,
	steps_per_epoch=TRAIN_COUNT // BATCH,
	validation_data=valGen,
	validation_steps=VAL_COUNT // BATCH,
    epochs=1000,
    verbose=1, 
    callbacks=[ model_checkpoint_best, predict_cb])
