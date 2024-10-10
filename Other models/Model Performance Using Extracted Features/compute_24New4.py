import numpy as np
import pandas as pd
import imageio
from skimage import color, feature, measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu, sobel  # Corrected import here
from scipy import ndimage, stats
from scipy.stats import entropy
import os

# Define constants for LBP
RADIUS = 3
N_POINTS = 8 * RADIUS


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
        if channel_data.dtype.kind == 'f':
            # Convert to an integer type if it's floating point
            channel_data = (channel_data * 255).astype(np.uint8)
        gray_img = channel_data if channel_data.ndim == 2 else color.rgb2gray(channel_data)

        # Compute GLCM and properties
        glcm = graycomatrix(channel_data, distances=[5], angles=[0], symmetric=True, normed=True)
        texture_entropy = entropy(glcm.ravel())
        asm = graycoprops(glcm, 'ASM')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        # Compute LBP variance
        lbp = local_binary_pattern(gray_img, N_POINTS, RADIUS, method='uniform')
        lbp_var = np.var(lbp)

        # Compute gradient (Sobel)
        sobel_img = sobel(gray_img)
        gradient_mean = np.mean(sobel_img)
        gradient_variance = np.var(sobel_img)

        # Euler number
        euler_num = measure.euler_number(gray_img > threshold_otsu(gray_img))

        # Update image_stats dictionary
        image_stats[f'channel{channel+1}_texture_entropy'] = texture_entropy
        image_stats[f'channel{channel+1}_asm'] = asm
        image_stats[f'channel{channel+1}_dissimilarity'] = dissimilarity
        image_stats[f'channel{channel+1}_correlation'] = correlation
        image_stats[f'channel{channel+1}_lbp_var'] = lbp_var
        image_stats[f'channel{channel+1}_gradient_mean'] = gradient_mean
        image_stats[f'channel{channel+1}_gradient_variance'] = gradient_variance
        image_stats[f'channel{channel+1}_euler_num'] = euler_num

    # Append the dictionary to the rows list
    rows_list.append(image_stats)
    print(i)

# Create a DataFrame to store the results
df = pd.DataFrame(rows_list)


# Save the DataFrame as a CSV file
output_csv_path = './24_4_result_#8.csv'
df.to_csv(output_csv_path, index=False)

print('end')

