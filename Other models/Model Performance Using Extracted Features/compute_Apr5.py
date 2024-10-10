import numpy as np
import pandas as pd
import imageio
from skimage import measure
from scipy import ndimage
from skimage.filters import sobel
import os
import imageio

def extract_circularity(perimeter, area):
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)

def extract_entropy_of_laplacian(image):
    laplacian = ndimage.laplace(image)
    histogram, _ = np.histogram(laplacian, bins=256, range=(-256, 256))
    histogram = histogram / np.sum(histogram)
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))

def extract_sobel_edge_count(image):
    edges = sobel(image)
    return np.sum(edges > np.mean(edges))

def extract_shape_features(image):
    thresh = np.mean(image)
    binary_image = image > thresh
    regions = measure.regionprops(measure.label(binary_image))
    if not regions:
        return 0, 0, 0, 0, 0, 0
    region = regions[0]
    area = region.area
    perimeter = region.perimeter
    circularity = extract_circularity(perimeter, area)
    eccentricity = region.eccentricity
    orientation = region.orientation
    convex_area = region.convex_area
    solidity = region.solidity
    return circularity, eccentricity, orientation, perimeter, area / perimeter, convex_area, solidity


def extract_features(img_path):
    img = imageio.v2.imread(img_path)
    features = {}

    for ch in range(1, img.shape[2] + 1):
        channel_img = img[:, :, ch - 1]
        prefix = f"channel{ch}"

        features[f"{prefix}_entropy_of_laplacian"] = extract_entropy_of_laplacian(channel_img)
        features[f"{prefix}_sobel_edge_count"] = extract_sobel_edge_count(channel_img)
        circularity, eccentricity, orientation, perimeter, area_perimeter_ratio, convex_area, solidity = extract_shape_features(channel_img)
        features[f"{prefix}_circularity"] = circularity
        features[f"{prefix}_eccentricity"] = eccentricity
        features[f"{prefix}_orientation"] = orientation
        features[f"{prefix}_perimeter"] = perimeter
        features[f"{prefix}_area_perimeter_ratio"] = area_perimeter_ratio
        features[f"{prefix}_convex_area"] = convex_area
        features[f"{prefix}_solidity"] = solidity

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
output_csv_path = './5_Apr_result_#9.csv'
df.to_csv(output_csv_path, index=False)

print(df.head())
print('end')