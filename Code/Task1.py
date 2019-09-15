# Implemented a program which, given an image ID and one of the models(Color Moments, Local Binary Pattern), extracts
# and saves the corresponding feature descriptors in file.

import cv2
import json
import configparser

from Code import LBP
from Code import ColoMoments


def main():
    # Reading data-set from a Configuration file
    config_file = open('C:/Users/spykar/PycharmProjects/MWDB/Config.cfg')
    config = configparser.RawConfigParser(allow_no_value=True)
    config.readfp(config_file)

    HAND_DATASET = config.get('PATH', 'hand_dataset')
    OUTPUT_DIR = config.get('PATH', 'output_dir')

    # Taking imageid as an input from end user
    given_image = input("Enter an image id to get its feature vector:")

    # Fetching path of a source image to read
    imgpath = HAND_DATASET + given_image+".jpg";
    img_bgr = cv2.imread(imgpath)
    # converting RGB Channel to YUV Channel so as to extract Color Moments
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    # converting RGB Channel to GRAY so as to extract LBP
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Calling feature extracting methods and storing them into respective dictionaries
    lbp_vector = {
        "name": imgpath,
        "lbp": LBP.lbp(img_gray)
    }

    color_vector = {
        "name": imgpath,
        "colormoments": ColoMoments.colorFeature(img_yuv)
    }

    # Dumping JSON feature vector dictionaries into output files
    with open(OUTPUT_DIR+given_image+"_color.json", "w") as fp:
        json.dump(color_vector, fp, indent=4, sort_keys=True)
    with open(OUTPUT_DIR+given_image+"_lbp.json", "w") as fp:
        json.dump(lbp_vector, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
	main()
