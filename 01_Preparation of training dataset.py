#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install plotly


# In[2]:


pip install scikit-image


# In[3]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io

import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import data, filters, measure, morphology
from skimage import data
from skimage.color import rgb2gray


# # C1 Rice 

# In[4]:


Img_Small_C1 = cv2.imread("D:\\Test_Al\\Images_Datasets\\01_Training_and_Validate_images\\01_Small_C1.jpg")


# In[5]:


Img_fig_Small_C1 = px.imshow(Img_Small_C1, binary_string=True)
Img_fig_Small_C1.update_traces(hoverinfo='skip')


# In[6]:


Img_Small_C1_Gray=rgb2gray(Img_Small_C1)


# In[7]:


threshold_Small_C1 = filters.threshold_otsu(Img_Small_C1_Gray)


# In[8]:


img_mask_Small_C1 = Img_Small_C1_Gray > threshold_Small_C1
img_mask_Small_C1 = morphology.remove_small_objects(img_mask_Small_C1, 15)
img_mask_Small_C1 = morphology.remove_small_holes(img_mask_Small_C1, 15)


# In[9]:


Img_fig_Small_C1 = px.imshow(img_mask_Small_C1, binary_string=True)
Img_fig_Small_C1.update_traces(hoverinfo='skip')


# In[10]:


labels_Small_C1 = measure.label(img_mask_Small_C1)


# In[11]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

# Small_C1
props_Small_C1 = regionprops_table(labels_Small_C1, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []

# Iterate over regions for Small_C1
for idx, region in enumerate(regionprops(labels_Small_C1)):
    # Calculate properties for each region
    equiv_diameter = np.sqrt(4 * region.area / np.pi)
    aspect_ratio = region.major_axis_length / region.minor_axis_length
    compactness = (region.perimeter ** 2) / (4 * np.pi * region.area)
    roundness = (4 * region.area) / (np.pi * (region.major_axis_length ** 2))
    
    # Append the calculated values to the respective lists
    equiv_diameters.append(equiv_diameter)
    aspect_ratios.append(aspect_ratio)
    compactnesses.append(compactness)
    roundnesses.append(roundness)
    categories.append('Small Broke C1')  # Category 1 for Small_C1

# Create a dictionary with all the properties
props_Small_C1 = {
    'area': props_Small_C1['area'],
    'major_axis_length': props_Small_C1['major_axis_length'],
    'minor_axis_length': props_Small_C1['minor_axis_length'],
    'perimeter': props_Small_C1['perimeter'],
    'eccentricity': props_Small_C1['eccentricity'],
    'solidity': props_Small_C1['solidity'],
    'extent': props_Small_C1['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Small_C1 = pd.DataFrame(props_Small_C1)


# In[12]:


df_Small_C1


# # Small Broken

# In[13]:


Img_Small_Broke = cv2.imread("D:\\Test_Al\\Images_Datasets\\01_Training_and_Validate_images\\02_Smallbroke.jpg")


# In[14]:


Img_fig_Small_Broke = px.imshow(Img_Small_Broke, binary_string=True)
Img_fig_Small_Broke.update_traces(hoverinfo='skip')


# In[15]:


Img_Small_Broke_Gray=rgb2gray(Img_Small_Broke)


# In[16]:


threshold_Small_Broke = filters.threshold_otsu(Img_Small_Broke_Gray)


# In[17]:


img_mask_Small_Broke = Img_Small_Broke_Gray > threshold_Small_Broke
img_mask_Small_Broke = morphology.remove_small_objects(img_mask_Small_Broke, 15)
img_mask_Small_Broke = morphology.remove_small_holes(img_mask_Small_Broke, 15)


# In[18]:


Img_fig_Small_Broke = px.imshow(img_mask_Small_Broke, binary_string=True)
Img_fig_Small_Broke.update_traces(hoverinfo='skip')


# In[19]:


labels_Small_Broke = measure.label(img_mask_Small_Broke)


# In[20]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

# Small_Broke
props_Small_Broke = regionprops_table(labels_Small_Broke, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []

# Iterate over regions for Small_C1
for idx, region in enumerate(regionprops(labels_Small_Broke)):
    # Calculate properties for each region
    equiv_diameter = np.sqrt(4 * region.area / np.pi)
    aspect_ratio = region.major_axis_length / region.minor_axis_length
    compactness = (region.perimeter ** 2) / (4 * np.pi * region.area)
    roundness = (4 * region.area) / (np.pi * (region.major_axis_length ** 2))
    
    # Append the calculated values to the respective lists
    equiv_diameters.append(equiv_diameter)
    aspect_ratios.append(aspect_ratio)
    compactnesses.append(compactness)
    roundnesses.append(roundness)
    categories.append('Small Broke')  # Category 2 for Small_Broke

# Create a dictionary with all the properties
props_Small_Broke = {
    'area': props_Small_Broke['area'],
    'major_axis_length': props_Small_Broke['major_axis_length'],
    'minor_axis_length': props_Small_Broke['minor_axis_length'],
    'perimeter': props_Small_Broke['perimeter'],
    'eccentricity': props_Small_C1['eccentricity'],
    'solidity': props_Small_Broke['solidity'],
    'extent': props_Small_Broke['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Small_Broke = pd.DataFrame(props_Small_Broke)


# In[21]:


df_Small_Broke


# # Big Broke

# In[22]:


Img_Big_Broke = cv2.imread("D:\\Test_Al\\Images_Datasets\\01_Training_and_Validate_images\\03_Bigbroke.jpg")


# In[23]:


Img_fig_Big_Broke = px.imshow(Img_Big_Broke, binary_string=True)
Img_fig_Big_Broke.update_traces(hoverinfo='skip')


# In[24]:


Img_Big_Broke_Gray=rgb2gray(Img_Big_Broke)


# In[25]:


threshold_Big_Broke = filters.threshold_otsu(Img_Big_Broke_Gray)


# In[26]:


img_mask_Big_Broke = Img_Big_Broke_Gray > threshold_Big_Broke
img_mask_Big_Broke = morphology.remove_small_objects(img_mask_Big_Broke, 15)
img_mask_Big_Broke = morphology.remove_small_holes(img_mask_Big_Broke, 15)


# In[27]:


Img_fig_Big_Broke = px.imshow(img_mask_Big_Broke, binary_string=True)
Img_fig_Big_Broke.update_traces(hoverinfo='skip')


# In[28]:


labels_Big_Broke = measure.label(img_mask_Big_Broke)


# In[29]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

# Big_Broke
props_Big_Broke = regionprops_table(labels_Big_Broke, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []

# Iterate over regions for Big Broke
for idx, region in enumerate(regionprops(labels_Big_Broke)):
    # Calculate properties for each region
    equiv_diameter = np.sqrt(4 * region.area / np.pi)
    aspect_ratio = region.major_axis_length / region.minor_axis_length
    compactness = (region.perimeter ** 2) / (4 * np.pi * region.area)
    roundness = (4 * region.area) / (np.pi * (region.major_axis_length ** 2))
    
    # Append the calculated values to the respective lists
    equiv_diameters.append(equiv_diameter)
    aspect_ratios.append(aspect_ratio)
    compactnesses.append(compactness)
    roundnesses.append(roundness)
    categories.append('Big Broke')  # Category 3 for Big_Broke

# Create a dictionary with all the properties
props_Big_Broke = {
    'area': props_Big_Broke['area'],
    'major_axis_length': props_Big_Broke['major_axis_length'],
    'minor_axis_length': props_Big_Broke['minor_axis_length'],
    'perimeter': props_Big_Broke['perimeter'],
    'eccentricity': props_Big_Broke['eccentricity'],
    'solidity': props_Big_Broke['solidity'],
    'extent': props_Big_Broke['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Big_Broke = pd.DataFrame(props_Big_Broke)


# In[30]:


df_Big_Broke


# # Head Rice

# In[31]:


Img_Head_Rice = cv2.imread("D:\\Test_Al\\Images_Datasets\\01_Training_and_Validate_images\\04_Head_Rice.jpg")


# In[32]:


Img_fig_Head_Rice = px.imshow(Img_Head_Rice, binary_string=True)
Img_fig_Head_Rice.update_traces(hoverinfo='skip')


# In[33]:


Img_Head_Rice_Gray=rgb2gray(Img_Head_Rice)


# In[34]:


threshold_Head_Rice = filters.threshold_otsu(Img_Head_Rice_Gray)


# In[35]:


img_mask_Head_Rice = Img_Head_Rice_Gray > threshold_Head_Rice
img_mask_Head_Rice = morphology.remove_small_objects(img_mask_Head_Rice, 15)
img_mask_Head_Rice = morphology.remove_small_holes(img_mask_Head_Rice, 15)


# In[36]:


Img_fig_Head_Rice = px.imshow(img_mask_Head_Rice, binary_string=True)
Img_fig_Head_Rice.update_traces(hoverinfo='skip')


# In[37]:


labels_Head_Rice = measure.label(img_mask_Head_Rice)


# In[38]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

# Head Rice
props_Head_Rice = regionprops_table(labels_Head_Rice, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []

# Iterate over regions for Small_Broke
for idx, region in enumerate(regionprops(labels_Head_Rice)):
    # Calculate properties for each region
    equiv_diameter = np.sqrt(4 * region.area / np.pi)
    aspect_ratio = region.major_axis_length / region.minor_axis_length
    compactness = (region.perimeter ** 2) / (4 * np.pi * region.area)
    roundness = (4 * region.area) / (np.pi * (region.major_axis_length ** 2))
    
    # Append the calculated values to the respective lists
    equiv_diameters.append(equiv_diameter)
    aspect_ratios.append(aspect_ratio)
    compactnesses.append(compactness)
    roundnesses.append(roundness)
    categories.append('Head rice')  # Category 4 for Head Rice

# Create a dictionary with all the properties
props_Head_Rice = {
    'area': props_Head_Rice['area'],
    'major_axis_length': props_Head_Rice['major_axis_length'],
    'minor_axis_length': props_Head_Rice['minor_axis_length'],
    'perimeter': props_Head_Rice['perimeter'],
    'eccentricity': props_Head_Rice['eccentricity'],
    'solidity': props_Head_Rice['solidity'],
    'extent': props_Head_Rice['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Head_Rice = pd.DataFrame(props_Head_Rice)


# In[39]:


df_Head_Rice


# # Whole Rice

# In[40]:


Img_Whole_Rice = cv2.imread("D:\\Test_Al\\Images_Datasets\\01_Training_and_Validate_images\\05_Whole_Grains.jpg")


# In[41]:


Img_fig_Whole_Rice = px.imshow(Img_Whole_Rice, binary_string=True)
Img_fig_Whole_Rice.update_traces(hoverinfo='skip')


# In[42]:


Img_Whole_Rice_Gray=rgb2gray(Img_Whole_Rice)


# In[43]:


threshold_Whole_Rice = filters.threshold_otsu(Img_Whole_Rice_Gray)


# In[44]:


img_mask_Whole_Rice = Img_Whole_Rice_Gray > threshold_Whole_Rice
img_mask_Whole_Rice = morphology.remove_small_objects(img_mask_Whole_Rice, 15)
img_mask_Whole_Rice = morphology.remove_small_holes(img_mask_Whole_Rice, 15)


# In[45]:


Img_fig_Whole_Rice = px.imshow(img_mask_Whole_Rice, binary_string=True)
Img_fig_Whole_Rice.update_traces(hoverinfo='skip')


# In[46]:


labels_Whole_Rice = measure.label(img_mask_Whole_Rice)


# In[47]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

# Small_Broke
props_Whole_Rice = regionprops_table(labels_Whole_Rice, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []

# Iterate over regions for Small_Broke
for idx, region in enumerate(regionprops(labels_Whole_Rice)):
    # Calculate properties for each region
    equiv_diameter = np.sqrt(4 * region.area / np.pi)
    aspect_ratio = region.major_axis_length / region.minor_axis_length
    compactness = (region.perimeter ** 2) / (4 * np.pi * region.area)
    roundness = (4 * region.area) / (np.pi * (region.major_axis_length ** 2))
    
    # Append the calculated values to the respective lists
    equiv_diameters.append(equiv_diameter)
    aspect_ratios.append(aspect_ratio)
    compactnesses.append(compactness)
    roundnesses.append(roundness)
    categories.append('Whole Rice')  # Category 5 for Whole Rice

# Create a dictionary with all the properties
props_Whole_Rice = {
    'area': props_Whole_Rice['area'],
    'major_axis_length': props_Whole_Rice['major_axis_length'],
    'minor_axis_length': props_Whole_Rice['minor_axis_length'],
    'perimeter': props_Whole_Rice['perimeter'],
    'eccentricity': props_Whole_Rice['eccentricity'],
    'solidity': props_Whole_Rice['solidity'],
    'extent': props_Whole_Rice['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Whole_Rice = pd.DataFrame(props_Whole_Rice)


# In[48]:


df_Whole_Rice


# # Combine to CSV file

# In[49]:


csv_file_path = 'Small_C1.csv'
df_Small_C1.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Small_Broke.csv'
df_Small_Broke.to_csv(csv_file_path, index=False) 
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Big_Broke.csv'
df_Big_Broke.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Head_Rice.csv'
df_Head_Rice.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Whole_Rice.csv'
df_Whole_Rice.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export


# In[50]:


import pandas as pd

# List of CSV file names
csv_files = [
    "Small_C1.csv",
    "Small_Broke.csv",
    "Big_Broke.csv",
    "Head_Rice.csv",
    "Whole_Rice.csv",
]

# Create an empty DataFrame to store the combined data
combined_prop_data = pd.DataFrame()

# Loop through the CSV files and append their data to the combined_data DataFrame
for file in csv_files:
    df = pd.read_csv(file)  # Read each CSV file
    combined_prop_data = combined_prop_data.append(df, ignore_index=True)  # Append data to the combined DataFrame

# Save the combined data to a new CSV file
combined_prop_data.to_csv("Training_Data.csv", index=False)


# In[51]:


Training_Data = pd.read_csv("Training_Data.csv")


# In[52]:


Training_Data.info()


# In[53]:


pd.DataFrame(Training_Data)


# In[54]:


unique_categories = Training_Data['category'].unique()
print(unique_categories)


# In[ ]:




