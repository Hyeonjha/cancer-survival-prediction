from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import pandas as pd
import imageio
from skimage import feature, measure
from skimage.feature import hog
import cv2
import os
import mahotas.features
import mahotas as mh


def extract_zernike_moments(image):
    if image.ndim >= 3:
        image = rgb2gray(image)
    image_resized = resize(image, (30, 30), anti_aliasing=True, mode='reflect')
    moments = mahotas.features.zernike_moments(image_resized, radius=15, degree=8)
    return moments


def extract_haralick_features(image):
    if image.ndim >= 3:
        image = rgb2gray(image)
    
    if image.dtype == np.float32:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Haralick
    features = mh.features.haralick(image).mean(0)
    return features

def extract_sift_features(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # SIFT 
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return len(keypoints)


def extract_hog_features(image):
    if image.ndim >= 3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image
    hog_features = hog(gray_image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, feature_vector=True)
    return np.mean(hog_features)



def extract_features(img_path):
    img = imageio.v2.imread(img_path)
    features = {}

    for ch in range(1, img.shape[2] + 1):
        channel_img = img[:, :, ch - 1]
        prefix = f"channel{ch}"

        # 채널별 특징 계산
        features[f"{prefix}_zernike_moments_mean"] = np.mean(extract_zernike_moments(channel_img))
        features[f"{prefix}_haralick_mean"] = np.mean(extract_haralick_features(channel_img))
        features[f"{prefix}_sift_keypoints"] = extract_sift_features(channel_img)
        features[f"{prefix}_hog_mean"] = extract_hog_features(channel_img)


    return features

img_dir = './train/images'  
n_images = 225 

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
output_csv_path = './1_Apr_result_#4.csv'
df.to_csv(output_csv_path, index=False)

print(df.head())
print('end')