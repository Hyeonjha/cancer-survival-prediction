import numpy as np
import pandas as pd
import imageio
from skimage import color, feature, measure
from skimage.filters import threshold_otsu
from scipy import ndimage
from scipy.stats import skew
from scipy.fftpack import dct
import mahotas
import os
from skimage.transform import resize
from sklearn.decomposition import PCA
from pywt import dwt2


img_dir = './train/images'  
n_images = 225  

# Initialize a list to store the computed values for each image
rows_list = []

# Process each image
for i in range(1, n_images + 1):
    img_file_name = f'{i}.tiff'
    path = os.path.join(img_dir, img_file_name)
    img = imageio.v2.imread(path)

    # Initialize a dictionary for the current image statistics
    image_stats = {'id': i}

    # Calculate statistics for each channel and add them to the dictionary
    for channel in range(img.shape[2]):
        channel_data = img[:,:,channel]
        channel_data_flat = channel_data.flatten()
        gray_img = channel_data if channel_data.ndim == 2 else color.rgb2gray(channel_data)

        # 1. Wavelet Features
        coeffs = dwt2(channel_data, 'haar')
        cA, (cH, cV, cD) = coeffs
        wavelet_feature = np.mean(cA.flatten())
        
        # 2. Principal Component Analysis (PCA) Feature
        pca = PCA(n_components=1)
        pca_feature = pca.fit_transform(channel_data_flat.reshape(-1, 1))
        
        # 3. Graph-Based Features (Clustering Coefficient)
        thresh = threshold_otsu(channel_data)
        bw = channel_data > thresh
        G = feature.graycomatrix(bw.astype(int), [1], [0], 256, symmetric=True, normed=True)
        clustering_coef = feature.graycoprops(G, 'ASM')[0, 0]
        
        # 4. Chromaticity Features (assuming the image has 3 channels)
        if img.ndim == 3:
            sum_channels = np.sum(img, axis=2)
            chromaticity = channel_data / (sum_channels + 1e-7)  # Avoid division by zero
            chroma_feature = np.mean(chromaticity)
        else:
            chroma_feature = np.mean(channel_data)  # For grayscale, use the mean intensity
        
        # Add features to the statistics dictionary
        image_stats[f'channel{channel+1}_wavelet_feature'] = wavelet_feature
        image_stats[f'channel{channel+1}_pca_feature'] = pca_feature[0, 0]  # Assuming pca returns a 2D array
        image_stats[f'channel{channel+1}_clustering_coef'] = clustering_coef
        image_stats[f'channel{channel+1}_chroma_feature'] = chroma_feature
        
    # Append the dictionary to the rows list
    rows_list.append(image_stats)
    print(i)

# Create a DataFrame to store the results
df = pd.DataFrame(rows_list)

# Save the DataFrame as a CSV file
output_csv_path = './24_3_result_#4.csv'
df.to_csv(output_csv_path, index=False)

print('end')
