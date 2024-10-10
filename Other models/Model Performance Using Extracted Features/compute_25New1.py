import numpy as np
import pandas as pd
import imageio
from skimage import color, feature, filters
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage
import cv2
import mahotas
from scipy.stats import kurtosis
import os

# Setup
img_dir = './train/images'  
n_images = 225  

# Initialize a list to store the computed values for each image
rows_list = []

# Function to compute Gabor texture features
def compute_gabor_features(img):
    kernel = gabor_kernel(frequency=0.6)
    filtered_real, filtered_imag = ndimage.convolve(img, np.real(kernel)), ndimage.convolve(img, np.imag(kernel))
    energy = np.sqrt(filtered_real**2 + filtered_imag**2).mean()
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
        hog_features, hog_image = hog(gray_img, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        hog_feature = np.mean(hog_features)
        
        # Color Histograms (Using only one channel for demonstration)
        if img.ndim == 3:
            hist, _ = np.histogram(channel_data, bins=256, range=(0, 255))
            color_hist_feature = np.mean(hist)
        else:
            color_hist_feature = np.mean(gray_img)  # Placeholder value for grayscale
        
        # Image Moments
        moments = cv2.moments(channel_data.astype(np.uint8))
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moment_feature = np.mean(hu_moments)
        
        # Update stats
        image_stats[f'channel{channel+1}_gabor_energy'] = gabor_energy
        image_stats[f'channel{channel+1}_hog_feature'] = hog_feature
        image_stats[f'channel{channel+1}_color_hist_feature'] = color_hist_feature
        image_stats[f'channel{channel+1}_hu_moment'] = hu_moment_feature

    rows_list.append(image_stats)
    print(i)

# Create and save DataFrame
df = pd.DataFrame(rows_list)
output_csv_path = './25_1_result_#4.csv'
df.to_csv(output_csv_path, index=False)

print('end')