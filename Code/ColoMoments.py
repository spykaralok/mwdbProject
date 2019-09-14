import configparser
import cv2
import os
import numpy as np
from PIL import Image
from scipy.stats import skew


# Function to extract color moments of a given image
def colorFeature(img_out):
    y_len, x_len, channel = img_out.shape
    print(y_len)
    y, u, v = cv2.split(img_out)
    meanOfY=[]
    meanofU=[]
    meanofV=[]
    sdOfY = []
    sdofU = []
    sdofV = []
    skewOfY = []
    skewofU = []
    skewofV = []
    color_feature_vector = []


    l, w = 100, 100
    for i in range(0, y_len, 100):
        temp_mean = []
        temp_sd = []
        temp_skew = []

        for j in range(0, x_len, 100):
            # Slicing image into 100*100 matrix

            meanimg = np.nanmean(img_out[i:i + 100, j:j + 100], axis=tuple(range(img_out[i:i + 100, j:j + 100].ndim-1)))
            deviationimg = np.std(img_out[i:i + 100, j:j + 100], axis=tuple(range(img_out[i:i + 100, j:j + 100].ndim-1)))

            arr_y = y[i:i + 100, j:j + 100]
            arr_u = u[i:i + 100, j:j + 100]
            arr_v = v[i:i + 100, j:j + 100]

            # appending mean of each color channel of every 100*100 sub matrix
            meanOfY.append(meanimg[0])
            meanofU.append(meanimg[1])
            meanofV.append(meanimg[2])

            # appending standard deviation of each color channel aof every 100*100 sub matrix
            sdOfY.append(deviationimg[0])
            sdofU.append(deviationimg[1])
            sdofV.append(deviationimg[2])
            # appending Skewness of each color channel aof every 100*100 sub matrix
            skewOfY.append(skew(arr_y.flatten()))
            skewofU.append(skew(arr_u.flatten()))
            skewofV.append(skew(arr_v.flatten()))
    color_feature_vector.extend([meanOfY, meanofU, meanofV, sdOfY, sdofU, sdofV,skewOfY, skewofU, skewofV])
    return color_feature_vector

