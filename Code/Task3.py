# Implemented a program which, given an image ID, a model, and a value k returns and visualizes the most
# similar k images based on the corresponding visual descriptors. For each match, also list the overall matching score.

import cv2
import json
import configparser
import os
import numpy as np
from Code.ColoMoments import colorFeature
from Code.LBP import lbp

config_file = open('C:/Users/spykar/PycharmProjects/MWDB/Config.cfg')
config = configparser.RawConfigParser(allow_no_value=True)
config.readfp(config_file)

HAND_DATASET = config.get('PATH', 'hand_dataset')
OUTPUT_DIR = config.get('PATH', 'output_dir')

imgs = []


def square_rooted(x):
    return np.round(np.sqrt(sum([a*a for a in x])), 3)

# Function to find Cosine similarity of given two vectors
def findsimilarity(source_vector, dest_vector):

    dot_product = np.dot(source_vector, dest_vector)
    norm_a = np.linalg.norm(source_vector)
    norm_b = np.linalg.norm(dest_vector)
    return dot_product / (norm_a * norm_b)

# Function to find k similiar images to a given image
def findksimilarImages():
    # Taking necessary inputs from user
    sourceimage = input("Enter a source image: ")
    modelname = input("Enter a feature descriptor modal:")
    k= int(input("Enter a value of most k similar images to store: "))

    # checking type of feature vector entered,only valid values are('CM','LBP')
    if modelname == "CM":
        # Fetching path of a source image to read
        imgpath = HAND_DATASET + sourceimage + ".jpg";
        # Reading image using OpenCV Library
        img_bgr = cv2.imread(imgpath)
        # converting image from RGB to YUV channel so as to extract Color moments
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

        src_feature_vector = colorFeature(img_yuv)
        #dictionary variable to store different image names and their similarity score to a given image
        similarity_vector = {}
        for left_out_images in os.listdir(HAND_DATASET):
            if left_out_images != sourceimage:
                # Fetching path of an image to read
                destimgpath = HAND_DATASET + left_out_images;
                #Reading image using OpenCV Library
                img_bgr = cv2.imread(destimgpath)
                img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
                dest_feature_vector = colorFeature(img_yuv)
                # calling function to find similarity score between pair of images

                similarity_vector_id = findsimilarity(src_feature_vector, dest_feature_vector)
                similarity_vector[left_out_images] = similarity_vector_id
        # Storing only required top k similar images in sorted form
        decresing_similarity_vector = sorted(similarity_vector.items(), key=lambda kv: kv[1], reverse=True)[:k]
        # Writing similarity as a Json dump to a file
        with open(OUTPUT_DIR + sourceimage + "_CMSimilarity.json", "w") as fp:
            json.dump(decresing_similarity_vector, fp, indent=4, sort_keys=True)

    elif modelname == "LBP":
        imgpath = HAND_DATASET + sourceimage + ".jpg";
        # Reading image using OpenCV Library
        img_bgr = cv2.imread(imgpath)
        # converting image from RGB to Gray so as to extract Local Binary Pattern
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        src_feature_vector = lbp(img_gray)
        # dictionary variable to store different image names and their similarity score to a given image
        similarity_vector = {}
        for left_out_images in os.listdir(HAND_DATASET):
            if left_out_images != sourceimage:
                # Fetching path of an image to read
                destimgpath = HAND_DATASET + left_out_images;
                # Reading image using OpenCV Library
                img_bgr = cv2.imread(destimgpath)
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                dest_feature_vector = lbp(img_gray)
                # calling function to find similarity score between pair of images
                similarity_vector_id = findsimilarity(src_feature_vector, dest_feature_vector)
                similarity_vector[left_out_images] = similarity_vector_id
        # Storing only required top k similar images in sorted form
        decresing_similarity_vector = sorted(similarity_vector.items(), key=lambda kv: kv[1], reverse=True)[:k]
        # Writing similarity as a Json dump to a file
        with open(OUTPUT_DIR + sourceimage + "_LbpSimilarity.json", "w") as fp:
            json.dump(decresing_similarity_vector, fp, indent=4, sort_keys=True)

    else: print("Valid values for models are : 'CM' and 'LBP'")

if __name__ == '__main__':
    findksimilarImages()
