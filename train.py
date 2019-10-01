import cv2
import pandas
import numpy as np
import keras
import glob
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import multi_gpu_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as keras
from skimage.draw import line
from tqdm import tqdm_notebook as tqdm
from keras_tqdm import TQDMNotebookCallback
from datetime import datetime
from model import *

PATH_BASE = "data/"
VAL_COUNT = 300
TRAIN_COUNT = 800
BATCH = 8


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
           

            input = cv2.resize(input, (256,128))
            x_coords = []
            y_coords = []

            for point in entry[1]:
                x_coords.append(point[0])
                y_coords.append(point[1])
            
            batch_input += [input]
            batch_output += [entry[1]]
            
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield(batch_x, batch_y)

print("LOADING MODEL")

model = cnn_small()
try:
    model = multi_gpu_model(model)
except:
    print("~NOT USING MUTLI GPU")
    pass

model_checkpoint_best = ModelCheckpoint('weights/cnn-small.hdf5', monitor='loss',verbose=1,save_best_only=True)


from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./log/cnn-small/',
                         write_images=True,
                         batch_size=BATCH,
                         update_freq="batch"
                         )


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
    epochs=100,
    verbose=1, 
    callbacks=[ model_checkpoint_best])
