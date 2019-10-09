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

Y_MAP = [310.0, 360.0, 260.0, 310.0, 360.0, 410.0, 460, 510.0, 560.0, 610.0]
def draw_lane(image, lane_array):
    index = 0
    print(lane_array)
    for point in lane_array:
    
        x = point * 1280
        y = Y_MAP[index]
        
        
        cv2.circle(image, (int(x),int(y)), 3, [0, 255, 100], 3)
        index += 1
        if index > 9:
            index = 0
        

def draw_label(data):
    image = cv2.imread("test.jpg")
    image = cv2.resize(image, (1280,720))
    lane_1 = data[0]
    #lane_2 = data[1]
    #lane_3 = data[2]
    #lane_4 = data[3]

    
    draw_lane(image, lane_1)
   

    cv2.imshow("reu",image)
    
while True:
    files = glob.glob("training/xception_v1/*.npy")
    
    for file in files:
        data = np.load(file, allow_pickle=True)
        draw_label(data)
        cv2.waitKey(10)



