import numpy as np
from skimage import feature


# Function to extract Linear binary pattern of a given image
def lbp(img_gray):

    height, width = img_gray.shape

    img_lbp = np.zeros((height, width, 3), np.uint8)
    data = []

    for i in range(0, height, 100):

        for j in range(0, width, 100):

            lbp = feature.local_binary_pattern(img_gray[i:i + 100, j:j + 100], 24, 8)
            hist, bins = np.histogram(lbp, bins=range(256), density=True)

            data.append(hist.tolist())
    # lbp_feature_vector.extend(data)

    return data
