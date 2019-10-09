import cv2
import pandas
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.path import Path
from skimage.draw import line, bezier_curve

Y_MAP = []

def draw_lane(image, lane_array):
    for point in lane_array:
        if point[0] < 0.1:
            continue

        x = point[0]
        y = point[1]

        Y_MAP.append(y)
        #print((int(x),int(y)))
        cv2.circle(image, (int(x),int(y)), 3, [0, 255, 100], 3)

def draw_label(data):
    image = data[0]
    image = cv2.resize(image, (1280,720))
    image *= 255
    image = cv2.normalize(image.astype('uint8'), None, 0, 255, cv2.NORM_MINMAX)

    lane_1 = data[1]
    lane_2 = data[2]
    lane_3 = data[3]
    lane_4 = data[4]

    
    draw_lane(image, lane_1)
    draw_lane(image, lane_2)
    draw_lane(image, lane_3)
    draw_lane(image, lane_4)
   
   

    cv2.imshow("test",image)
    

    

files = glob.glob("data/val/*.npy")

for file in files:
    data = np.load(file, allow_pickle=True)
    draw_label(data)
    cv2.waitKey(100)
