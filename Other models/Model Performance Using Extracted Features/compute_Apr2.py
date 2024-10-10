import numpy as np
import pandas as pd
import imageio
from skimage import feature
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
import os

def ensure_uint8(image):
    # Check if the image is floating-point and adjust its scale if necessary
    if image.dtype in [np.float32, np.float64]:
        # Scale the image to the 0-1 range if it's not already
        if image.max() > 1 or image.min() < 0:
            # Normalize images with values outside the 0-1 range
            image -= image.min()  # Shift to 0-... range
            image /= image.max()  # Scale to 0-1 range
        # Convert to 8-bit unsigned integer
        image_uint8 = img_as_ubyte(image)
    elif image.dtype != np.uint8:
        # For other non-float, non-uint8 types, convert directly to uint8
        image_uint8 = image.astype(np.uint8)
    else:
        # If already uint8, no conversion is needed
        image_uint8 = image
    return image_uint8


def extract_lbp_features(image):
    # Local Binary Patterns 
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(257), density=True)
    return lbp_hist.mean()

def extract_glcm_contrast_homogeneity(image):
    # Ensure the image is in the correct format
    image_uint8 = ensure_uint8(image)
    # GLCM calculation and extraction of contrast and homogeneity
    glcm = feature.graycomatrix(image_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
    return contrast, homogeneity

def extract_canny_edges(image):
    # Canny Edge
    edges = feature.canny(image)
    return np.mean(edges) 

def extract_diversity_index(image):
    unique_values = len(np.unique(image))
    total_values = image.size
    diversity_index = unique_values / total_values
    return diversity_index

def extract_features(img_path):
    img = imageio.v2.imread(img_path)
    features = {}

    for ch in range(1, img.shape[2] + 1):
        channel_img = img[:, :, ch - 1]
        prefix = f"channel{ch}"

        features[f"{prefix}_lbp_mean"] = extract_lbp_features(channel_img)
        contrast, homogeneity = extract_glcm_contrast_homogeneity(channel_img)
        features[f"{prefix}_glcm_contrast"] = contrast
        features[f"{prefix}_glcm_homogeneity"] = homogeneity
        features[f"{prefix}_canny_edge_ratio"] = extract_canny_edges(channel_img)
        features[f"{prefix}_diversity_index"] = extract_diversity_index(channel_img)


    return features

img_dir = './train/images'  # directory where your images are stored
n_images = 225  # number of images, adjust accordingly

# Initialize a list to store the computed values for each image
rows_list = []

# Process each image
for i in range(1, n_images + 1):
    img_file_name = f'{i}.tiff'
    path = os.path.join(img_dir, img_file_name)
    img_features = extract_features(path)
    img_features['id'] = i
    rows_list.append(img_features)
    print(i)
 
# Create and save DataFrame
df = pd.DataFrame(rows_list)
cols = ['id'] + [col for col in df if col != 'id']
df = df[cols]
output_csv_path = './2_Apr_result_#5.csv'
df.to_csv(output_csv_path, index=False)

print(df.head())
print('end')