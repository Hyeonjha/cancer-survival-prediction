import numpy as np
import pandas as pd
import imageio
from scipy.stats import kurtosis
from skimage.measure import shannon_entropy

img_dir = './train/images/'

# Function to calculate the median intensity for each channel
def median_intensity(image):
    return np.median(image, axis=(1, 2))

# Function to calculate the kurtosis for each channel
def kurtosis_intensity(image):
    return kurtosis(image, axis=(1, 2), fisher=False)

# Function to calculate the entropy for each channel
def entropy_intensity(image):
    # Flatten the image to 1D for entropy calculation
    return [shannon_entropy(channel) for channel in image]

# Function to calculate the total area for each channel (cells expressing the protein)
# Here we assume that cell expression is above a certain intensity threshold
def area_intensity(image, threshold=0.1):
    # Threshold the image and calculate the area (number of pixels above threshold)
    return [(channel > threshold).sum() for channel in image]

# List to collect all the features for each image
all_features = []

# Process each TIFF image in the directory
for i in range(1, 226):  # Assuming we have 225 TIFF images
    # Construct the image path
    img_path = f'{img_dir}{i}.tiff'
    
    # Read the image
    img = imageio.imread(img_path)

    # Calculate the features
    medians = median_intensity(img)
    kurtosis_vals = kurtosis_intensity(img)
    entropy_vals = entropy_intensity(img)
    area_vals = area_intensity(img)

    # Append the features to the list
    all_features.append([i] + list(medians) + list(kurtosis_vals) + list(entropy_vals) + list(area_vals))

# Convert the list of all features to a dataframe
df_all_features = pd.DataFrame(all_features)

# Create column names
column_names = ['id']
for feature in ['median_intensity', 'kurtosis', 'entropy', 'area']:
    for channel in range(1, 53):  # Assuming there are 52 channels/proteins
        column_names.append(f'{feature}_channel{channel}')

# Assign the column names to the dataframe
df_all_features.columns = column_names

# Save the dataframe to a CSV file
csv_output_path = 'medianKurtosisEntropyArea.csv'
df_all_features.to_csv(csv_output_path, index=False)

# Return the path to the saved CSV file
csv_output_path
