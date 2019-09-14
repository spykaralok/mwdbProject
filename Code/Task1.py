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

    # Taking imageid input from end user
    given_image = input("Enter and image id to get its feature vector:")


    imgpath = HAND_DATASET + given_image+".jpg";
    print(imgpath)
    img_bgr = cv2.imread(imgpath)
    # converting RGB Channel to YUV Channel for extracting Color Moments
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    # converting RGB Channel to GRAY for extracting LBP
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    lbp_vector = {
        "name": imgpath,
        "lbp": LBP.lbp(img_gray)
    }
    print(lbp_vector)
    color_vector = {
        "name": imgpath,
        "colormoments": ColoMoments.colorFeature(img_yuv)
    }

    # Writing feature vector dictionaries into output files converting them to Json
    with open(OUTPUT_DIR+given_image+"_color.json", "w") as fp:
        json.dump(color_vector, fp, indent=4, sort_keys=True)
    with open(OUTPUT_DIR+given_image+"_lbp.json", "w") as fp:
        json.dump(lbp_vector, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
	main()