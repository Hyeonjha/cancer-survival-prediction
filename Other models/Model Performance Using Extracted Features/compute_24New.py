import numpy as np
import pandas as pd
import imageio
from skimage.feature.texture import graycomatrix, graycoprops  # 수정됨
from scipy.stats import skew, kurtosis
from skimage.filters import gabor
from skimage.measure import shannon_entropy
import os


img_dir = './train/images'  
n_images = 225  
rows_list = []

for i in range(1, n_images + 1):
    img_file_name = f'{i}.tiff'
    path = os.path.join(img_dir, img_file_name)
    img = imageio.v2.imread(path)
    
    image_stats = {'id': i}
    
    for channel in range(img.shape[2]):
        channel_data = img[:,:,channel].flatten()
        

        kurt = kurtosis(channel_data)
        entropy = shannon_entropy(channel_data)
        
        # Haralick feature
        if channel_data.dtype.kind == 'f':
            # scale 0-1 -> 0-255 & uint8
            channel_data = (channel_data * 255).astype(np.uint8)

        glcm = graycomatrix(channel_data.reshape(img.shape[:2]), distances=[5], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        
        # Gabor
        gabor_filtration = gabor(img[:,:,channel], frequency=0.6)[0]
        gabor_std = np.std(gabor_filtration)
        
    
        image_stats[f'channel{channel+1}_kurtosis'] = kurt
        image_stats[f'channel{channel+1}_entropy'] = entropy
        image_stats[f'channel{channel+1}_contrast'] = contrast
        image_stats[f'channel{channel+1}_gabor_std'] = gabor_std
    
    rows_list.append(image_stats)
    print(i)

df = pd.DataFrame(rows_list)
output_csv_path = './24_1_result_#4.csv'
df.to_csv(output_csv_path, index=False)

print('end')
