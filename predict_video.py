import cv2
import pandas
import numpy as np
import keras
import glob
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as keras
from skimage.draw import line
from tqdm import tqdm_notebook as tqdm
from keras_tqdm import TQDMNotebookCallback

from model import *

import os

model = cnn_small("weights/cnn-small.hdf5")
#model = unet_small("weights/lane-small.hdf5")

Y_MAP = [310.0, 360.0, 260.0, 310.0, 360.0, 410.0, 460, 510.0, 560.0, 610.0]
def draw_lane(image, lane_array):
    index = 0

    for point in lane_array:
    
        x = point * 1280
        y = Y_MAP[index]
        
        
        cv2.circle(image, (int(x),int(y)), 3, [0, 255, 100], 3)
        index += 1
        if index > 9:
            index = 0

        
    
    return image
        

def predict_model(image):
    #files = glob.glob(PATH_TEST + "*.jpg")
    #image = cv2.imread(files[index])
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    image = cv2.resize(image, (256,128))
    test = np.array([image])
    test = test.reshape(len(test),128,256,3)
    lanes = model.predict(test, verbose=0)
    return lanes[0]

import numpy as np
import cv2

cap = cv2.VideoCapture("drive-1.mp4")


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        continue;

    frame = cv2.resize(frame, (1280,720))
    #frame = cv2.cvtColor(cv2.BGR2RGB)
    lanes = predict_model(frame)
    frame = draw_lane(frame, lanes)
   
    cv2.imshow("Result", frame)
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()