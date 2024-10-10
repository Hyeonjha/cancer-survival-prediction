from skimage.transform import radon
from skimage.color import rgb2lab
from scipy.fftpack import fft2
from skimage.filters import scharr
from scipy.ndimage import laplace
import numpy as np
import pandas as pd
import os
import imageio

def extract_radon_features(image):
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    return np.mean(sinogram), np.std(sinogram)

def extract_fft_energy_spectrum(image):
    f_transform = fft2(image)
    f_abs = np.abs(f_transform) ** 2
    energy_spectrum = np.sum(f_abs)
    return energy_spectrum

def extract_scharr_edges(image):
    edges = scharr(image)
    return np.mean(edges), np.std(edges)

def extract_laplacian_variance(image):
    laplacian_img = laplace(image)
    return np.var(laplacian_img)

def extract_cielab_color_histogram(image):
    if image.ndim == 3 and image.shape[2] == 3:
        lab_image = rgb2lab(image)
        l_hist, _ = np.histogram(lab_image[:, :, 0], bins=32, range=(0, 100))
        a_hist, _ = np.histogram(lab_image[:, :, 1], bins=32, range=(-128, 127))
        b_hist, _ = np.histogram(lab_image[:, :, 2], bins=32, range=(-128, 127))
        return np.mean(l_hist), np.mean(a_hist), np.mean(b_hist)
    else:
        return 0, 0, 0



def extract_features(img_path):
    img = imageio.v2.imread(img_path)
    features = {}

    for ch in range(1, img.shape[2] + 1):
        channel_img = img[:, :, ch - 1]
        prefix = f"channel{ch}"

        r_mean, r_std = extract_radon_features(channel_img)
        features[f"{prefix}_radon_mean"] = r_mean
        features[f"{prefix}_radon_std"] = r_std

        features[f"{prefix}_energy_spect"] = extract_fft_energy_spectrum(channel_img)
        
        s_mean, s_std = extract_scharr_edges(channel_img)
        features[f"{prefix}_scharr_mean"] = s_mean
        features[f"{prefix}_scharr_std"] = s_std

        features[f"{prefix}_laplacian_var"] = extract_laplacian_variance(channel_img)
        hist_1_mean, a_hist_mean, b_hist_mean = extract_cielab_color_histogram(channel_img)
        features[f"{prefix}_1_hist_mean"] = hist_1_mean
        features[f"{prefix}_a_hist_mean"] = a_hist_mean
        features[f"{prefix}_b_hist_mean"] = b_hist_mean


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
output_csv_path = './3_Apr_result_#9.csv'
df.to_csv(output_csv_path, index=False)

print(df.head())
print('end')