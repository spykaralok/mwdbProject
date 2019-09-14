import cv2
import json
import os
import configparser
from Code import LBP
from Code import ColoMoments
from pymongo import MongoClient

# MongoDb Related Database ,Collection Names
DATABASE_NAME = "multimedia_db"

# Reading data-set from a Configuration file
config_file = open('C:/Users/spykar/PycharmProjects/MWDB/Config.cfg')
config = configparser.RawConfigParser(allow_no_value=True)
config.readfp(config_file)

HAND_DATASET = config.get('PATH', 'hand_dataset')
OUTPUT_DIR = config.get('PATH', 'output_dir')


# Function to push feature vectors into MongoDB
def push_all_features_into_mango(collectionName):
    files = os.listdir(HAND_DATASET)

    for file in files:
        img_bgr = cv2.imread(HAND_DATASET + file)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        feature_vector = {
            "name": file,
            "lbp": LBP.lbp(img_gray),
            "colormoments": ColoMoments.colorFeature(img_yuv)
        }
        # Writing feature vectors into output files
        with open(OUTPUT_DIR + file + ".json", "w") as fp:
            json.dump(feature_vector, fp, indent=4, sort_keys=True)

        # inserting both feature vectors combined into database
        collectionName.hands.insert_one(feature_vector)


# Function written to make MongoDB connectivity
def main():
    try:
        mongoConnection = MongoClient('mongodb://localhost:27017/')
        push_all_features_into_mango(mongoConnection.multimedia_db)


    except:
        print("Failure connecting to DB")


if __name__ == '__main__':
    main()
