import numpy as np
import pandas as pd
import imageio
from skimage import feature, filters
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import os

def compute_gabor_features(img):
    # Define Gabor filter parameters
    frequencies = [0.1, 0.2, 0.4]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    energies = []
    for frequency in frequencies:
        for theta in thetas:
            filt_real, filt_imag = filters.gabor(img, frequency=frequency, theta=theta)
            energy = np.sqrt(filt_real**2 + filt_imag**2).mean()
            energies.append(energy)
    return np.mean(energies)  # Return the average energy

def extract_glcm_properties(img):
    # Compute GLCM and extract properties
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return contrast, correlation

def extract_features(img_path):
    img = imageio.v2.imread(img_path)
    features = {}

    # Iterate over each channel in the image
    for ch in range(1, img.shape[2] + 1):
        channel_img = img[:, :, ch - 1]
        prefix = f"channel{ch}"

        # Statistical features
        features[f"{prefix}_mean"] = np.mean(channel_img)
        features[f"{prefix}_std"] = np.std(channel_img)
        features[f"{prefix}_skew"] = skew(channel_img.flatten())
        features[f"{prefix}_kurt"] = kurtosis(channel_img.flatten())

        # Gabor filter energy
        gabor_energy = compute_gabor_features(channel_img)
        features[f"{prefix}_gabor_energy"] = gabor_energy

        # GLCM properties
        contrast, correlation = extract_glcm_properties((channel_img * 255).astype(np.uint8))  # Ensure image is in the correct format
        features[f"{prefix}_contrast"] = contrast
        features[f"{prefix}_correlation"] = correlation

        #print(ch)

    return features

img_dir = './train/images' 
n_images = 225  

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
output_csv_path = './29_1_result#2+4.csv'
df.to_csv(output_csv_path, index=False)

print(df.head())
