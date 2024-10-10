import numpy as np
import pandas as pd
import imageio
from skimage import color
from skimage.feature import hog, local_binary_pattern, blob_log
from skimage.filters import gabor, gaussian
from scipy.ndimage import gaussian_laplace
from scipy.stats import kurtosis
import cv2
import os

# Setup
img_dir = './train/images'  
n_images = 225  

# Initialize a list to store the computed values for each image
rows_list = []

# Function to compute Gabor texture features
def compute_gabor_features(img):
    filt_real, filt_imag = gabor(img, frequency=0.6)
    energy = np.sqrt(filt_real**2 + filt_imag**2).mean()
    return energy

# Process each image
for i in range(1, n_images + 1):
    img_file_name = f'{i}.tiff'
    path = os.path.join(img_dir, img_file_name)
    img = imageio.v2.imread(path)

    image_stats = {'id': i}

    for channel in range(img.shape[2]):
        channel_data = img[:, :, channel]
        gray_img = color.rgb2gray(channel_data) if channel_data.ndim == 3 else channel_data

        # Gabor Texture Features
        gabor_energy = compute_gabor_features(gray_img)
        
        # HOG Descriptor
        hog_features = hog(gray_img, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        hog_feature = np.mean(hog_features)
        
        # Color Histograms
        hist, _ = np.histogram(gray_img, bins=256, range=(0, 255))
        color_hist_feature = np.mean(hist)
        
        # Image Moments
        moments = cv2.moments(gray_img.astype(np.uint8))
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moment_feature = np.mean(hu_moments)
        
        # Laplacian of Gaussian (using SciPy's gaussian_laplace)
        log_image = gaussian_laplace(gray_img, sigma=3)
        log_feature = np.mean(log_image)
        
        # Blob Detection
        blobs = blob_log(gray_img, max_sigma=30, num_sigma=10, threshold=.1)
        blob_count = len(blobs)
        
        # Kurtosis of Image Histogram
        kurt_feature = kurtosis(hist)

        # Update stats
        image_stats[f'channel{channel+1}_gabor_energy'] = gabor_energy
        image_stats[f'channel{channel+1}_hog_feature'] = hog_feature
        image_stats[f'channel{channel+1}_color_hist_feature'] = color_hist_feature
        image_stats[f'channel{channel+1}_hu_moment'] = hu_moment_feature
        image_stats[f'channel{channel+1}_log_feature'] = log_feature
        image_stats[f'channel{channel+1}_blob_count'] = blob_count
        image_stats[f'channel{channel+1}_kurt_feature'] = kurt_feature

    rows_list.append(image_stats)
    print(i)

# Create and save DataFrame
df = pd.DataFrame(rows_list)
output_csv_path = './26_1_result_#7.csv'
df.to_csv(output_csv_path, index=False)

print('end')
