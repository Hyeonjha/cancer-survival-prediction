import numpy as np
import pandas as pd
import imageio
from skimage import filters, color, feature, measure, util
from scipy import ndimage, stats
from scipy.ndimage import center_of_mass
from scipy.stats import kurtosis
import mahotas
import os

def fractal_dimension(Z, threshold=0.9):
    # Only for 2d image
    assert(len(Z.shape) == 2)

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the counts to the sizes
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


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
        channel_data = img[:, :, channel]
        channel_data_flat = channel_data.flatten()
        gray_img = channel_data if channel_data.ndim == 2 else color.rgb2gray(channel_data)

        # 1. Radial Distribution Feature (simplified version)
        center_of_mass = ndimage.measurements.center_of_mass(gray_img)
        radial_dist = ndimage.distance_transform_edt(gray_img != 0)
        radial_feature = radial_dist[int(center_of_mass[0]), int(center_of_mass[1])]
        
        # 2. Fourier Descriptors
        contours = measure.find_contours(gray_img, 0.8)
        # Simplified: just the length of the longest contour
        fourier_feature = max([len(contour) for contour in contours]) if contours else 0
        
        # 3. Fractal Dimension
        fd = fractal_dimension(gray_img)
        image_stats[f'channel{channel+1}_fractal_dim'] = fd
        
        # 4. Variance of Laplacian
        laplacian_var = np.var(ndimage.laplace(gray_img))
        
        # 5. Co-occurrence Matrix Contrast
        glcm = feature.graycomatrix((gray_img * 255).astype(np.uint8), [1], [0], 256, symmetric=True, normed=True)
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        
        # 6. Autocorrelation Feature
        autocorr = np.correlate(gray_img.flatten(), gray_img.flatten(), mode='same')
        autocorrelation_feature = np.mean(autocorr)
        
        # Add features to the statistics dictionary
        image_stats[f'channel{channel+1}_radial_feature'] = radial_feature
        image_stats[f'channel{channel+1}_fourier_feature'] = fourier_feature
        image_stats[f'channel{channel+1}_fractal_dim'] = fd
        image_stats[f'channel{channel+1}_laplacian_var'] = laplacian_var
        image_stats[f'channel{channel+1}_contrast'] = contrast
        image_stats[f'channel{channel+1}_autocorrelation'] = autocorrelation_feature

    # Append the dictionary to the rows list
    rows_list.append(image_stats)
    print(i)

# Create a DataFrame to store the results
df = pd.DataFrame(rows_list)

# Save the DataFrame as a CSV file
output_csv_path = './24_2_result_#6.csv'
df.to_csv(output_csv_path, index=False)

print('end')
