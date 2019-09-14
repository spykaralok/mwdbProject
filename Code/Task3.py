import cv2
import json
import configparser
import os

from Code.ColoMoments import colorFeature
from Code.LBP import lbp

config_file = open('C:/Users/spykar/PycharmProjects/MWDB/Config.cfg')
config = configparser.RawConfigParser(allow_no_value=True)
config.readfp(config_file)

HAND_DATASET = config.get('PATH', 'hand_dataset')
OUTPUT_DIR = config.get('PATH', 'output_dir')

imgs = []


def findksimilarImages():
    # Taking necessary inputs from user
    sourceimage = input("Enter a source image: ")
    modelname = input("Enter the feature descriptor modal:")
    k = input("number of most similar images: ")

    # checking for type of feature vector to apply similarity on
    if modelname == "colormodal":
        imgpath = HAND_DATASET + sourceimage + ".jpg";
        print(imgpath)
        img_bgr = cv2.imread(imgpath)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

        src_feature_vector = colorFeature(img_yuv)

        for left_out_images in os.listdir(HAND_DATASET):
            if left_out_images != sourceimage:
                print("Processing color moments for image ", left_out_images)

                destimgpath = HAND_DATASET + sourceimage + ".jpg";

                img_bgr = cv2.imread(destimgpath)
                img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
                dest_feature_vector = colorFeature(img_yuv)
                # To implement similarity vector

                similarity_vector = []
                with open(OUTPUT_DIR + sourceimage + "_CMSimilarity.json", "w") as fp:
                    json.dump(similarity_vector, fp, indent=4, sort_keys=True)

    elif modelname == "lbp":
        imgpath = HAND_DATASET + sourceimage + ".jpg";
        print(imgpath)
        img_bgr = cv2.imread(imgpath)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        src_feature_vector = lbp(img_gray)

        for left_out_images in os.listdir(HAND_DATASET):
            if left_out_images != sourceimage:
                print("Processing color moments for image ", left_out_images)
                destimgpath = HAND_DATASET + sourceimage + ".jpg";

                img_bgr = cv2.imread(destimgpath)
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                dest_feature_vector = lbp(img_gray)

                # To implement similarity vector
                similarity_vector = []
                with open(OUTPUT_DIR + sourceimage + "_LBPSimilaarity.json", "w") as fp:
                    json.dump(similarity_vector, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    findksimilarImages()
