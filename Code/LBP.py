# Program to extract Local binary patterns of a given Image using available libraries.
import numpy as np
from skimage import feature


# Function to extract Local binary patterns of a given image
def lbp(img_gray):
    # Saving height and width of a given image
    height, width = img_gray.shape

    data = []

    for i in range(0, height, 100):

        for j in range(0, width, 100):
            # using Scikit-image library to extract local binary pattern of an image
            lbp = feature.local_binary_pattern(img_gray[i:i + 100, j:j + 100], 24, 8)
            # For feature vector extraction we are taking only y values into account
            hist, bins = np.histogram(lbp, bins=range(256), density=True)

            data = np.concatenate([data, hist])
            # lbp_feature_vector.extend(data)

    return data.tolist()
