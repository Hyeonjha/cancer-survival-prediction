import numpy as np
import pandas as pd
import imageio
from PIL import Image
from IPython.display import display, Image
from scipy.stats import skew
import time
from tqdm import tqdm
import os

img_dir = './images' # dir that saves your images
n_train = 225 # number of images
n_protein = 52 # number of proteins 

avg_list = [] 
std_list = []
skewness_list = []
for i in tqdm(np.arange(1, n_train+1), total=n_train, desc="Processing"):
    img_file_name = f'{i}.tiff'
    path = os.path.join(img_dir, img_file_name)
    img = imageio.v2.imread(path)

    # Assuming img is a 3D array where the first dimension corresponds to channels (proteins)
    channel_means = np.mean(img, axis=(1, 2))  # Compute mean across each channel
    channel_stds = np.std(img, axis=(1, 2))    # Compute std across each channel
    channel_skewness = [skew(img[channel].flatten()) for channel in range(img.shape[0])]  # Compute skewness for each channel
    
    # Append the statistics for this image to the lists
    avg_list.append([i] + channel_means.tolist())
    std_list.append([i] + channel_stds.tolist())
    skewness_list.append([i] + channel_skewness)

# Create dataframes for the calculated statistics
avg_df = pd.DataFrame(avg_list, columns=['id'] + [f'channel{i}_mean_intensity' for i in range(1, n_protein+1)])
std_df = pd.DataFrame(std_list, columns=['id'] + [f'channel{i}_std_intensity' for i in range(1, n_protein+1)])
skew_df = pd.DataFrame(skewness_list, columns=['id'] + [f'channel{i}_skewness' for i in range(1, n_protein+1)])

# Combine the dataframes into a single one for easier handling
stats_df = pd.concat([avg_df, std_df.drop('id', axis=1), skew_df.drop('id', axis=1)], axis=1)

# Save the combined dataframe to a csv file
stats_df.to_csv('new_meanStdSkew.csv', index=False)