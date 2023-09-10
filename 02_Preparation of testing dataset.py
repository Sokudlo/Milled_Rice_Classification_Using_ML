#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install plotly


# In[3]:


pip install scikit-image


# In[4]:


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


# In[ ]:





# # Sample1 Rice Grade 100%

# In[5]:


Img_Sample1 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample1\\Sample1.jpg")


# In[6]:


plt.imshow(Img_Sample1)


# In[7]:


Img_Sample1_1 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample1\\Sample1_1.jpg")
Img_Sample1_2 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample1\\Sample1_2.jpg")
Img_Sample1_3 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample1\\Sample1_3.jpg")
Img_Sample1_4 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample1\\Sample1_4.jpg")


# In[8]:


Img_Sample1_1_Gray=rgb2gray(Img_Sample1_1)
Img_Sample1_2_Gray=rgb2gray(Img_Sample1_2)
Img_Sample1_3_Gray=rgb2gray(Img_Sample1_3)
Img_Sample1_4_Gray=rgb2gray(Img_Sample1_4)


# In[9]:


# Create a figure with subplots to display the images
plt.figure(figsize=(10, 4))


# Display the second image sample
plt.subplot(2, 3, 1)
plt.imshow(Img_Sample1_1_Gray, cmap='gray')
plt.title('Sample1_1 Whole_Rice')

# Display the third image sample
plt.subplot(2, 3, 2)
plt.imshow(Img_Sample1_2_Gray, cmap='gray')
plt.title('Sample1_2 Head_Rice')

# Display the fourth image sample
plt.subplot(2, 3, 3)
plt.imshow(Img_Sample1_3_Gray, cmap='gray')
plt.title('Sample1_3 Bigbroke_Rice')

# Display the fourth image sample
plt.subplot(2, 3, 4)
plt.imshow(Img_Sample1_4_Gray, cmap='gray')
plt.title('Sample1_4 Smallbroke_Rice')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[10]:


# Apply Otsu's thresholding to each image
threshold_Img_Sample1_1 = filters.threshold_otsu(Img_Sample1_1_Gray)
threshold_Img_Sample1_2 = filters.threshold_otsu(Img_Sample1_2_Gray)
threshold_Img_Sample1_3 = filters.threshold_otsu(Img_Sample1_3_Gray)
threshold_Img_Sample1_4 = filters.threshold_otsu(Img_Sample1_4_Gray)


# In[11]:


# Sample1_1
img_mask_Sample1_1 = Img_Sample1_1_Gray > threshold_Img_Sample1_1
img_mask_Sample1_1 = morphology.remove_small_objects(img_mask_Sample1_1, 15)
img_mask_Sample1_1 = morphology.remove_small_holes(img_mask_Sample1_1, 15)

# Sample1_2
img_mask_Sample1_2 = Img_Sample1_2_Gray > threshold_Img_Sample1_2
img_mask_Sample1_2 = morphology.remove_small_objects(img_mask_Sample1_2, 15)
img_mask_Sample1_2 = morphology.remove_small_holes(img_mask_Sample1_2, 15)

# Sample1_3
img_mask_Sample1_3 = Img_Sample1_3_Gray > threshold_Img_Sample1_3
img_mask_Sample1_3 = morphology.remove_small_objects(img_mask_Sample1_3, 15)
img_mask_Sample1_3 = morphology.remove_small_holes(img_mask_Sample1_3, 15)

# Sample1_4
img_mask_Sample1_4 = Img_Sample1_4_Gray > threshold_Img_Sample1_4
img_mask_Sample1_4 = morphology.remove_small_objects(img_mask_Sample1_4, 15)
img_mask_Sample1_4 = morphology.remove_small_holes(img_mask_Sample1_4, 15)


# In[12]:


img_labels_Sample1_1 = measure.label(img_mask_Sample1_1)
img_labels_Sample1_2 = measure.label(img_mask_Sample1_2)
img_labels_Sample1_3 = measure.label(img_mask_Sample1_3)
img_labels_Sample1_4 = measure.label(img_mask_Sample1_4)


# Sample1_1

# In[15]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample1_1 = regionprops_table(img_labels_Sample1_1, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample1_1)):
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
    categories.append('Whole Rice') 

# Create a dictionary with all the properties
props_Sample1_1 = {
    'area': props_Sample1_1['area'],
    'major_axis_length': props_Sample1_1['major_axis_length'],
    'minor_axis_length': props_Sample1_1['minor_axis_length'],
    'perimeter': props_Sample1_1['perimeter'],
    'eccentricity': props_Sample1_1['eccentricity'],
    'solidity': props_Sample1_1['solidity'],
    'extent': props_Sample1_1['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample1_1 = pd.DataFrame(props_Sample1_1)


# In[16]:


df_Sample1_1


# In[ ]:





# Sample1_2

# In[17]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample1_2 = regionprops_table(img_labels_Sample1_2, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample1_2)):
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
    categories.append('Head Rice')  

# Create a dictionary with all the properties
props_Sample1_2 = {
    'area': props_Sample1_2['area'],
    'major_axis_length': props_Sample1_2['major_axis_length'],
    'minor_axis_length': props_Sample1_2['minor_axis_length'],
    'perimeter': props_Sample1_2['perimeter'],
    'eccentricity': props_Sample1_2['eccentricity'],
    'solidity': props_Sample1_2['solidity'],
    'extent': props_Sample1_2['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample1_2 = pd.DataFrame(props_Sample1_2)


# In[18]:


df_Sample1_2


# In[ ]:





# Sample1_3

# In[19]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample1_3 = regionprops_table(img_labels_Sample1_3, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample1_3)):
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
    categories.append('Big Broke')

# Create a dictionary with all the properties
props_Sample1_3 = {
    'area': props_Sample1_3['area'],
    'major_axis_length': props_Sample1_3['major_axis_length'],
    'minor_axis_length': props_Sample1_3['minor_axis_length'],
    'perimeter': props_Sample1_3['perimeter'],
    'eccentricity': props_Sample1_3['eccentricity'],
    'solidity': props_Sample1_3['solidity'],
    'extent': props_Sample1_3['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample1_3 = pd.DataFrame(props_Sample1_3)


# In[20]:


df_Sample1_3


# In[ ]:





# Sample1_4

# In[21]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample1_4 = regionprops_table(img_labels_Sample1_4, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample1_4)):
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
    categories.append('Small Broke')

# Create a dictionary with all the properties
props_Sample1_4 = {
    'area': props_Sample1_4['area'],
    'major_axis_length': props_Sample1_4['major_axis_length'],
    'minor_axis_length': props_Sample1_4['minor_axis_length'],
    'perimeter': props_Sample1_4['perimeter'],
    'eccentricity': props_Sample1_4['eccentricity'],
    'solidity': props_Sample1_4['solidity'],
    'extent': props_Sample1_4['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample1_4 = pd.DataFrame(props_Sample1_4)


# In[22]:


df_Sample1_4


# In[ ]:





# Combine

# In[23]:


csv_file_path = 'Sample1_1.csv'
df_Sample1_1.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample1_2.csv'
df_Sample1_2.to_csv(csv_file_path, index=False) 
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample1_3.csv'
df_Sample1_3.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample1_4.csv'
df_Sample1_4.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export


# In[24]:


import pandas as pd

# List of CSV file names
csv_files = [
    "Sample1_1.csv",
    "Sample1_2.csv",
    "Sample1_3.csv",
    "Sample1_4.csv",
]

# Create an empty DataFrame to store the combined data
combined_prop_data = pd.DataFrame()

# Loop through the CSV files and append their data to the combined_data DataFrame
for file in csv_files:
    df = pd.read_csv(file)  # Read each CSV file
    combined_prop_data = combined_prop_data.append(df, ignore_index=True)  # Append data to the combined DataFrame

# Save the combined data to a new CSV file
combined_prop_data.to_csv("Sample1_Data.csv", index=False)


# In[25]:


Sample1_Data = pd.read_csv("Sample1_Data.csv")


# In[26]:


Sample1_Data.info()


# In[27]:


Sample1_Data


# In[49]:


unique_categories = Sample1_Data['category'].unique()
print(unique_categories)


# In[28]:


#####################     Sample1: CSV file is Sample1_Data.csv     ##################### 


# In[ ]:





# # Sample 2 Rice Grade 5%

# In[29]:


Img_Sample2 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample2\\Sample2.jpg")


# In[30]:


plt.imshow(Img_Sample2)


# In[31]:


Img_Sample2_1 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample2\\Sample2_1.jpg")
Img_Sample2_2 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample2\\Sample2_2.jpg")
Img_Sample2_3 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample2\\Sample2_3.jpg")
Img_Sample2_4 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample2\\Sample2_4.jpg")


# In[32]:


Img_Sample2_1_Gray=rgb2gray(Img_Sample2_1)
Img_Sample2_2_Gray=rgb2gray(Img_Sample2_2)
Img_Sample2_3_Gray=rgb2gray(Img_Sample2_3)
Img_Sample2_4_Gray=rgb2gray(Img_Sample2_4)


# In[33]:


# Create a figure with subplots to display the images
plt.figure(figsize=(10, 6))


# Display the second image sample
plt.subplot(1, 4, 1)
plt.imshow(Img_Sample2_1_Gray, cmap='gray')
plt.title('Sample2_1 Whole_Rice')

# Display the third image sample
plt.subplot(1, 4, 2)
plt.imshow(Img_Sample2_2_Gray, cmap='gray')
plt.title('Sample2_2 Head_Rice')

# Display the fourth image sample
plt.subplot(1, 4, 3)
plt.imshow(Img_Sample2_3_Gray, cmap='gray')
plt.title('Sample2_3 Bigbroke_Rice')

# Display the fourth image sample
plt.subplot(1, 4, 4)
plt.imshow(Img_Sample2_4_Gray, cmap='gray')
plt.title('Sample2_4 Smallbroke_Rice')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[34]:


threshold_Img_Sample2_1 = filters.threshold_otsu(Img_Sample2_1_Gray)
threshold_Img_Sample2_2 = filters.threshold_otsu(Img_Sample2_2_Gray)
threshold_Img_Sample2_3 = filters.threshold_otsu(Img_Sample2_3_Gray)
threshold_Img_Sample2_4 = filters.threshold_otsu(Img_Sample2_4_Gray)


# In[35]:


# Sample2_1
img_mask_Sample2_1 = Img_Sample2_1_Gray > threshold_Img_Sample2_1
img_mask_Sample2_1 = morphology.remove_small_objects(img_mask_Sample2_1, 15)
img_mask_Sample2_1 = morphology.remove_small_holes(img_mask_Sample2_1, 15)

# Sample2_2
img_mask_Sample2_2 = Img_Sample2_2_Gray > threshold_Img_Sample2_2
img_mask_Sample2_2 = morphology.remove_small_objects(img_mask_Sample2_2, 15)
img_mask_Sample2_2 = morphology.remove_small_holes(img_mask_Sample2_2, 15)

# Sample2_3
img_mask_Sample2_3 = Img_Sample2_3_Gray > threshold_Img_Sample2_3
img_mask_Sample2_3 = morphology.remove_small_objects(img_mask_Sample2_3, 15)
img_mask_Sample2_3 = morphology.remove_small_holes(img_mask_Sample2_3, 15)

# Sample2_4
img_mask_Sample2_4 = Img_Sample2_4_Gray > threshold_Img_Sample2_4
img_mask_Sample2_4 = morphology.remove_small_objects(img_mask_Sample2_4, 15)
img_mask_Sample2_4 = morphology.remove_small_holes(img_mask_Sample2_4, 15)


# In[36]:


img_labels_Sample2_1 = measure.label(img_mask_Sample2_1)
img_labels_Sample2_2 = measure.label(img_mask_Sample2_2)
img_labels_Sample2_3 = measure.label(img_mask_Sample2_3)
img_labels_Sample2_4 = measure.label(img_mask_Sample2_4)


# In[ ]:





# Sample2_1

# In[37]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample2_1 = regionprops_table(img_labels_Sample2_1, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample2_1)):
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
    categories.append('Whole Rice')  # Category 5 for Whole

# Create a dictionary with all the properties
props_Sample2_1 = {
    'area': props_Sample2_1['area'],
    'major_axis_length': props_Sample2_1['major_axis_length'],
    'minor_axis_length': props_Sample2_1['minor_axis_length'],
    'perimeter': props_Sample2_1['perimeter'],
    'eccentricity': props_Sample2_1['eccentricity'],
    'solidity': props_Sample2_1['solidity'],
    'extent': props_Sample2_1['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample2_1 = pd.DataFrame(props_Sample2_1)


# In[38]:


df_Sample2_1


# In[ ]:





# Sample2_2

# In[39]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample2_2 = regionprops_table(img_labels_Sample2_2, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample2_2)):
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
    categories.append('Head rice')  # Category 4 for Head

# Create a dictionary with all the properties
props_Sample2_2 = {
    'area': props_Sample2_2['area'],
    'major_axis_length': props_Sample2_2['major_axis_length'],
    'minor_axis_length': props_Sample2_2['minor_axis_length'],
    'perimeter': props_Sample2_2['perimeter'],
    'eccentricity': props_Sample2_2['eccentricity'],
    'solidity': props_Sample2_2['solidity'],
    'extent': props_Sample2_2['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample2_2 = pd.DataFrame(props_Sample2_2)


# In[40]:


df_Sample2_2


# In[ ]:





# Sample2_3

# In[41]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample2_3 = regionprops_table(img_labels_Sample2_3, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample2_3)):
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
    categories.append('Big Broke')  # Category 3 for Big Broke

# Create a dictionary with all the properties
props_Sample2_3 = {
    'area': props_Sample2_3['area'],
    'major_axis_length': props_Sample2_3['major_axis_length'],
    'minor_axis_length': props_Sample2_3['minor_axis_length'],
    'perimeter': props_Sample2_3['perimeter'],
    'eccentricity': props_Sample2_3['eccentricity'],
    'solidity': props_Sample2_3['solidity'],
    'extent': props_Sample2_3['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample2_3 = pd.DataFrame(props_Sample2_3)


# In[42]:


df_Sample2_3


# In[ ]:





# Sample2_4

# In[43]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample2_4 = regionprops_table(img_labels_Sample2_4, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample2_4)):
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
    categories.append('Small Broke')  # Category 2 for Small

# Create a dictionary with all the properties
props_Sample2_4 = {
    'area': props_Sample1_4['area'],
    'major_axis_length': props_Sample2_4['major_axis_length'],
    'minor_axis_length': props_Sample2_4['minor_axis_length'],
    'perimeter': props_Sample2_4['perimeter'],
    'eccentricity': props_Sample2_4['eccentricity'],
    'solidity': props_Sample2_4['solidity'],
    'extent': props_Sample2_4['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample2_4 = pd.DataFrame(props_Sample2_4)


# In[44]:


df_Sample2_4


# In[ ]:





# Combine

# In[50]:


csv_file_path = 'Sample2_1.csv'
df_Sample2_1.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample2_2.csv'
df_Sample2_2.to_csv(csv_file_path, index=False) 
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample2_3.csv'
df_Sample2_3.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample2_4.csv'
df_Sample2_4.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export


# In[51]:


import pandas as pd

# List of CSV file names
csv_files = [
    "Sample2_1.csv",
    "Sample2_2.csv",
    "Sample2_3.csv",
    "Sample2_4.csv",
]

# Create an empty DataFrame to store the combined data
combined_prop_data = pd.DataFrame()

# Loop through the CSV files and append their data to the combined_data DataFrame
for file in csv_files:
    df = pd.read_csv(file)  # Read each CSV file
    combined_prop_data = combined_prop_data.append(df, ignore_index=True)  # Append data to the combined DataFrame

# Save the combined data to a new CSV file
combined_prop_data.to_csv("Sample2_Data.csv", index=False)


# In[52]:


Sample2_Data = pd.read_csv("Sample2_Data.csv")


# In[53]:


Sample2_Data.info()


# In[54]:


Sample2_Data


# In[55]:


unique_categories = Sample2_Data['category'].unique()
print(unique_categories)


# In[47]:


#####################     Sample2: CSV file is Sample2_Data.csv     ##################### 


# In[ ]:





# # Sample 3 Rice Grade 10%

# In[56]:


Img_Sample3 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample3\\Sample3.jpg")


# In[57]:


plt.imshow(Img_Sample3)


# In[58]:


Img_Sample3_1 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample3\\Sample3_1.jpg")
Img_Sample3_2 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample3\\Sample3_2.jpg")
Img_Sample3_3 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample3\\Sample3_3.jpg")
Img_Sample3_4 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample3\\Sample3_4.jpg")
Img_Sample3_5 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample3\\Sample3_5.jpg")


# In[59]:


Img_Sample3_1_Gray=rgb2gray(Img_Sample3_1)
Img_Sample3_2_Gray=rgb2gray(Img_Sample3_2)
Img_Sample3_3_Gray=rgb2gray(Img_Sample3_3)
Img_Sample3_4_Gray=rgb2gray(Img_Sample3_4)
Img_Sample3_5_Gray=rgb2gray(Img_Sample3_5)


# In[60]:


# Create a figure with subplots to display the images
plt.figure(figsize=(12, 5))


# Display the second image sample
plt.subplot(2, 3, 1)
plt.imshow(Img_Sample3_1_Gray, cmap='gray')
plt.title('Sample3_1 Whole_Rice')

# Display the third image sample
plt.subplot(2, 3, 2)
plt.imshow(Img_Sample3_2_Gray, cmap='gray')
plt.title('Sample3_2 Head_Rice')

# Display the fourth image sample
plt.subplot(2, 3, 3)
plt.imshow(Img_Sample3_3_Gray, cmap='gray')
plt.title('Sample3_3 Bigbroke_Rice')

# Display the fourth image sample
plt.subplot(2, 3, 4)
plt.imshow(Img_Sample3_4_Gray, cmap='gray')
plt.title('Sample3_4 Smallbroke_Rice')

# Display the fifth  image sample
plt.subplot(2, 3, 5)
plt.imshow(Img_Sample3_5_Gray, cmap='gray')
plt.title('Sample3_5 Smallbroke_C1')


# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[61]:


threshold_Img_Sample3_1 = filters.threshold_otsu(Img_Sample3_1_Gray)
threshold_Img_Sample3_2 = filters.threshold_otsu(Img_Sample3_2_Gray)
threshold_Img_Sample3_3 = filters.threshold_otsu(Img_Sample3_3_Gray)
threshold_Img_Sample3_4 = filters.threshold_otsu(Img_Sample3_4_Gray)
threshold_Img_Sample3_5 = filters.threshold_otsu(Img_Sample3_5_Gray)


# In[62]:


# Sample3_1 
img_mask_Sample3_1 = Img_Sample3_1_Gray > threshold_Img_Sample3_1
img_mask_Sample3_1 = morphology.remove_small_objects(img_mask_Sample3_1, 15)
img_mask_Sample3_1 = morphology.remove_small_holes(img_mask_Sample3_1, 15)

# Sample3_2
img_mask_Sample3_2 = Img_Sample3_2_Gray > threshold_Img_Sample3_2
img_mask_Sample3_2 = morphology.remove_small_objects(img_mask_Sample3_2, 15)
img_mask_Sample3_2 = morphology.remove_small_holes(img_mask_Sample3_2, 15)

# Sample3_3
img_mask_Sample3_3 = Img_Sample3_3_Gray > threshold_Img_Sample3_3
img_mask_Sample3_3 = morphology.remove_small_objects(img_mask_Sample3_3, 15)
img_mask_Sample3_3 = morphology.remove_small_holes(img_mask_Sample3_3, 15)

# Sample3_4
img_mask_Sample3_4 = Img_Sample3_4_Gray > threshold_Img_Sample3_4
img_mask_Sample3_4 = morphology.remove_small_objects(img_mask_Sample3_4, 15)
img_mask_Sample3_4 = morphology.remove_small_holes(img_mask_Sample3_4, 15)

# Sample3_5
img_mask_Sample3_5 = Img_Sample3_5_Gray > threshold_Img_Sample3_5
img_mask_Sample3_5 = morphology.remove_small_objects(img_mask_Sample3_5, 15)
img_mask_Sample3_5 = morphology.remove_small_holes(img_mask_Sample3_5, 15)


# In[63]:


img_labels_Sample3_1 = measure.label(img_mask_Sample3_1)
img_labels_Sample3_2 = measure.label(img_mask_Sample3_2)
img_labels_Sample3_3 = measure.label(img_mask_Sample3_3)
img_labels_Sample3_4 = measure.label(img_mask_Sample3_4)
img_labels_Sample3_5 = measure.label(img_mask_Sample3_5)


# In[ ]:





# Sample3_1

# In[64]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample3_1 = regionprops_table(img_labels_Sample3_1, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample3_1)):
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
    categories.append('Whole Rice')  # Category 5 for Whole

# Create a dictionary with all the properties
props_Sample3_1 = {
    'area': props_Sample3_1['area'],
    'major_axis_length': props_Sample3_1['major_axis_length'],
    'minor_axis_length': props_Sample3_1['minor_axis_length'],
    'perimeter': props_Sample3_1['perimeter'],
    'eccentricity': props_Sample3_1['eccentricity'],
    'solidity': props_Sample3_1['solidity'],
    'extent': props_Sample3_1['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample3_1 = pd.DataFrame(props_Sample3_1)


# In[65]:


df_Sample3_1


# In[ ]:





# Sample3_2

# In[66]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample3_2 = regionprops_table(img_labels_Sample3_2, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample3_2)):
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
    categories.append('Head rice')  # Category 4 for Head

# Create a dictionary with all the properties
props_Sample3_2 = {
    'area': props_Sample3_2['area'],
    'major_axis_length': props_Sample3_2['major_axis_length'],
    'minor_axis_length': props_Sample3_2['minor_axis_length'],
    'perimeter': props_Sample3_2['perimeter'],
    'eccentricity': props_Sample3_2['eccentricity'],
    'solidity': props_Sample3_2['solidity'],
    'extent': props_Sample3_2['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample3_2 = pd.DataFrame(props_Sample3_2)


# In[67]:


df_Sample3_2


# In[ ]:





# Sample3_3

# In[68]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample3_3 = regionprops_table(img_labels_Sample3_3, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample3_3)):
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
    categories.append('Big Broke')  # Category 3 for Big Broke

# Create a dictionary with all the properties
props_Sample3_3 = {
    'area': props_Sample3_3['area'],
    'major_axis_length': props_Sample3_3['major_axis_length'],
    'minor_axis_length': props_Sample3_3['minor_axis_length'],
    'perimeter': props_Sample3_3['perimeter'],
    'eccentricity': props_Sample3_3['eccentricity'],
    'solidity': props_Sample3_3['solidity'],
    'extent': props_Sample3_3['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample3_3 = pd.DataFrame(props_Sample3_3)


# In[69]:


df_Sample3_3 


# In[ ]:





# Sample3_4

# In[70]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample3_4 = regionprops_table(img_labels_Sample3_4, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample3_4)):
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
    categories.append('Small Broke')  # Category 2 for Small

# Create a dictionary with all the properties
props_Sample3_4 = {
    'area': props_Sample3_4['area'],
    'major_axis_length': props_Sample3_4['major_axis_length'],
    'minor_axis_length': props_Sample3_4['minor_axis_length'],
    'perimeter': props_Sample3_4['perimeter'],
    'eccentricity': props_Sample3_4['eccentricity'],
    'solidity': props_Sample3_4['solidity'],
    'extent': props_Sample3_4['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample3_4 = pd.DataFrame(props_Sample3_4)


# In[71]:


df_Sample3_4 


# In[ ]:





# Sample3_5

# In[72]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample3_5 = regionprops_table(img_labels_Sample3_5, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample3_5)):
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
    categories.append('Small Broke C1')  # Category 1 for C1

# Create a dictionary with all the properties
props_Sample3_5 = {
    'area': props_Sample3_5['area'],
    'major_axis_length': props_Sample3_5['major_axis_length'],
    'minor_axis_length': props_Sample3_5['minor_axis_length'],
    'perimeter': props_Sample3_5['perimeter'],
    'eccentricity': props_Sample3_5['eccentricity'],
    'solidity': props_Sample3_5['solidity'],
    'extent': props_Sample3_5['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample3_5 = pd.DataFrame(props_Sample3_5)


# In[73]:


df_Sample3_5


# In[ ]:





# Combine

# In[74]:


csv_file_path = 'Sample3_1.csv'
df_Sample3_1.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample3_2.csv'
df_Sample3_2.to_csv(csv_file_path, index=False) 
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample3_3.csv'
df_Sample3_3.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample3_4.csv'
df_Sample3_4.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample3_5.csv'
df_Sample3_5.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export


# In[75]:


import pandas as pd

# List of CSV file names
csv_files = [
    "Sample3_1.csv",
    "Sample3_2.csv",
    "Sample3_3.csv",
    "Sample3_4.csv",
    "Sample3_5.csv",
]

# Create an empty DataFrame to store the combined data
combined_prop_data = pd.DataFrame()

# Loop through the CSV files and append their data to the combined_data DataFrame
for file in csv_files:
    df = pd.read_csv(file)  # Read each CSV file
    combined_prop_data = combined_prop_data.append(df, ignore_index=True)  # Append data to the combined DataFrame

# Save the combined data to a new CSV file
combined_prop_data.to_csv("Sample3_Data.csv", index=False)


# In[76]:


Sample3_Data = pd.read_csv("Sample3_Data.csv")


# In[77]:


Sample3_Data.info()


# In[78]:


Sample3_Data


# In[79]:


unique_categories = Sample3_Data['category'].unique()
print(unique_categories)


# In[80]:


#####################     Sample3: CSV file is Sample3_Data.csv     ##################### 


# In[ ]:





# # Sample 4 Rice Grade 15%

# In[81]:


Img_Sample4 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample4\\Sample4.jpg")


# In[82]:


plt.imshow(Img_Sample4)


# In[83]:


Img_Sample4_1 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample4\\Sample4_1.jpg")
Img_Sample4_2 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample4\\Sample4_2.jpg")
Img_Sample4_3 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample4\\Sample4_3.jpg")
Img_Sample4_4 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample4\\Sample4_4.jpg")
Img_Sample4_5 = cv2.imread("D:\\Test_Al\\Images_Datasets\\02_Testing_images\\Sample4\\Sample4_5.jpg")


# In[84]:


Img_Sample4_1_Gray=rgb2gray(Img_Sample4_1)
Img_Sample4_2_Gray=rgb2gray(Img_Sample4_2)
Img_Sample4_3_Gray=rgb2gray(Img_Sample4_3)
Img_Sample4_4_Gray=rgb2gray(Img_Sample4_4)
Img_Sample4_5_Gray=rgb2gray(Img_Sample4_5)


# In[85]:


# Create a figure with subplots to display the images
plt.figure(figsize=(12, 8))


# Display the second image sample
plt.subplot(2, 3, 1)
plt.imshow(Img_Sample4_1_Gray, cmap='gray')
plt.title('Sample4_1 Whole_Rice')

# Display the third image sample
plt.subplot(2, 3, 2)
plt.imshow(Img_Sample4_2_Gray, cmap='gray')
plt.title('Sample4_2 Head_Rice')

# Display the fourth image sample
plt.subplot(2, 3, 3)
plt.imshow(Img_Sample4_3_Gray, cmap='gray')
plt.title('Sample4_3 Bigbroke_Rice')

# Display the fourth image sample
plt.subplot(2, 3, 4)
plt.imshow(Img_Sample4_4_Gray, cmap='gray')
plt.title('Sample4_4 Smallbroke_Rice')

# Display the fifth  image sample
plt.subplot(2, 3, 5)
plt.imshow(Img_Sample4_5_Gray, cmap='gray')
plt.title('Sample4_5 Smallbroke_C1')


# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[86]:


threshold_Img_Sample4_1 = filters.threshold_otsu(Img_Sample4_1_Gray)
threshold_Img_Sample4_2 = filters.threshold_otsu(Img_Sample4_2_Gray)
threshold_Img_Sample4_3 = filters.threshold_otsu(Img_Sample4_3_Gray)
threshold_Img_Sample4_4 = filters.threshold_otsu(Img_Sample4_4_Gray)
threshold_Img_Sample4_5 = filters.threshold_otsu(Img_Sample4_5_Gray)


# In[87]:


# Sample4_1
img_mask_Sample4_1 = Img_Sample4_1_Gray > threshold_Img_Sample4_1
img_mask_Sample4_1 = morphology.remove_small_objects(img_mask_Sample4_1, 15)
img_mask_Sample4_1 = morphology.remove_small_holes(img_mask_Sample4_1, 15)

# Sample4_2
img_mask_Sample4_2 = Img_Sample4_2_Gray > threshold_Img_Sample4_2
img_mask_Sample4_2 = morphology.remove_small_objects(img_mask_Sample4_2, 15)
img_mask_Sample4_2 = morphology.remove_small_holes(img_mask_Sample4_2, 15)

# Sample4_3
img_mask_Sample4_3 = Img_Sample4_3_Gray > threshold_Img_Sample4_3
img_mask_Sample4_3 = morphology.remove_small_objects(img_mask_Sample4_3, 15)
img_mask_Sample4_3 = morphology.remove_small_holes(img_mask_Sample4_3, 15)

# Sample4_4
img_mask_Sample4_4 = Img_Sample4_4_Gray > threshold_Img_Sample4_4
img_mask_Sample4_4 = morphology.remove_small_objects(img_mask_Sample4_4, 15)
img_mask_Sample4_4 = morphology.remove_small_holes(img_mask_Sample4_4, 15)

# Sample4_5
img_mask_Sample4_5 = Img_Sample4_5_Gray > threshold_Img_Sample4_5
img_mask_Sample4_5 = morphology.remove_small_objects(img_mask_Sample4_5, 15)
img_mask_Sample4_5 = morphology.remove_small_holes(img_mask_Sample4_5, 15)


# In[88]:


img_labels_Sample4_1 = measure.label(img_mask_Sample4_1)
img_labels_Sample4_2 = measure.label(img_mask_Sample4_2)
img_labels_Sample4_3 = measure.label(img_mask_Sample4_3)
img_labels_Sample4_4 = measure.label(img_mask_Sample4_4)
img_labels_Sample4_5 = measure.label(img_mask_Sample4_5)


# In[ ]:





# Sample4_1

# In[89]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample4_1 = regionprops_table(img_labels_Sample4_1, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample4_1)):
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
    categories.append('Whole Rice')  # Category 5 for Whole

# Create a dictionary with all the properties
props_Sample4_1 = {
    'area': props_Sample4_1['area'],
    'major_axis_length': props_Sample4_1['major_axis_length'],
    'minor_axis_length': props_Sample4_1['minor_axis_length'],
    'perimeter': props_Sample4_1['perimeter'],
    'eccentricity': props_Sample4_1['eccentricity'],
    'solidity': props_Sample4_1['solidity'],
    'extent': props_Sample4_1['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample4_1 = pd.DataFrame(props_Sample4_1)


# In[90]:


df_Sample4_1


# In[ ]:





# Sample4_2

# In[91]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample4_2 = regionprops_table(img_labels_Sample4_2, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample4_2)):
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
    categories.append('Head rice')  # Category 4 for Head

# Create a dictionary with all the properties
props_Sample4_2 = {
    'area': props_Sample4_2['area'],
    'major_axis_length': props_Sample4_2['major_axis_length'],
    'minor_axis_length': props_Sample4_2['minor_axis_length'],
    'perimeter': props_Sample4_2['perimeter'],
    'eccentricity': props_Sample4_2['eccentricity'],
    'solidity': props_Sample4_2['solidity'],
    'extent': props_Sample4_2['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample4_2 = pd.DataFrame(props_Sample4_2)


# In[92]:


df_Sample4_2


# In[ ]:





# Sample4_3

# In[93]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample4_3 = regionprops_table(img_labels_Sample4_3, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample4_3)):
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
    categories.append('Big Broke')  # Category 3 for Big Broke

# Create a dictionary with all the properties
props_Sample4_3 = {
    'area': props_Sample4_3['area'],
    'major_axis_length': props_Sample4_3['major_axis_length'],
    'minor_axis_length': props_Sample4_3['minor_axis_length'],
    'perimeter': props_Sample4_3['perimeter'],
    'eccentricity': props_Sample4_3['eccentricity'],
    'solidity': props_Sample4_3['solidity'],
    'extent': props_Sample4_3['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample4_3 = pd.DataFrame(props_Sample4_3)


# In[94]:


df_Sample4_3


# In[ ]:





# Sample4_4

# In[97]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample4_4 = regionprops_table(img_labels_Sample4_4, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample4_4)):
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
    categories.append('Small Broke')  # Category 2 for Small

# Create a dictionary with all the properties
props_Sample4_4 = {
    'area': props_Sample4_4['area'],
    'major_axis_length': props_Sample4_4['major_axis_length'],
    'minor_axis_length': props_Sample4_4['minor_axis_length'],
    'perimeter': props_Sample4_4['perimeter'],
    'eccentricity': props_Sample4_4['eccentricity'],
    'solidity': props_Sample4_4['solidity'],
    'extent': props_Sample4_4['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample4_4 = pd.DataFrame(props_Sample4_4)


# In[98]:


df_Sample4_4


# In[ ]:





# Sample4_5

# In[99]:


import pandas as pd
import numpy as np
from skimage.measure import regionprops, regionprops_table

props_Sample4_5 = regionprops_table(img_labels_Sample4_5, properties=('area', 'major_axis_length', 'minor_axis_length', 'perimeter', 'eccentricity', 'solidity', 'extent'))

# Create empty lists to store property values
equiv_diameters = []
aspect_ratios = []
compactnesses = []
roundnesses = []
categories = []


for idx, region in enumerate(regionprops(img_labels_Sample4_5)):
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
    categories.append('Small Broke C1')  # Category 1 for C1

# Create a dictionary with all the properties
props_Sample4_5 = {
    'area': props_Sample4_5['area'],
    'major_axis_length': props_Sample4_5['major_axis_length'],
    'minor_axis_length': props_Sample4_5['minor_axis_length'],
    'perimeter': props_Sample4_5['perimeter'],
    'eccentricity': props_Sample4_5['eccentricity'],
    'solidity': props_Sample4_5['solidity'],
    'extent': props_Sample4_5['extent'],
    'equiv_diameter': equiv_diameters,
    'aspect_ratio': aspect_ratios,
    'compactness': compactnesses,
    'roundness': roundnesses,
    'category': categories,
}

# Create a DataFrame from the dictionary
df_Sample4_5 = pd.DataFrame(props_Sample4_5)


# In[100]:


df_Sample4_5


# In[ ]:





# Combine

# In[101]:


csv_file_path = 'Sample4_1.csv'
df_Sample4_1.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample4_2.csv'
df_Sample4_2.to_csv(csv_file_path, index=False) 
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample4_3.csv'
df_Sample4_3.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample4_4.csv'
df_Sample4_4.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export

csv_file_path = 'Sample4_5.csv'
df_Sample4_5.to_csv(csv_file_path, index=False)
print(f"Data has been exported to {csv_file_path}") # print a message to confirm the export


# In[102]:


import pandas as pd

# List of CSV file names
csv_files = [
    "Sample4_1.csv",
    "Sample4_2.csv",
    "Sample4_3.csv",
    "Sample4_4.csv",
    "Sample4_5.csv",
]

# Create an empty DataFrame to store the combined data
combined_prop_data = pd.DataFrame()

# Loop through the CSV files and append their data to the combined_data DataFrame
for file in csv_files:
    df = pd.read_csv(file)  # Read each CSV file
    combined_prop_data = combined_prop_data.append(df, ignore_index=True)  # Append data to the combined DataFrame

# Save the combined data to a new CSV file
combined_prop_data.to_csv("Sample4_Data.csv", index=False)


# In[103]:


Sample4_Data = pd.read_csv("Sample4_Data.csv")


# In[104]:


Sample4_Data.info()


# In[105]:


Sample4_Data


# In[106]:


unique_categories = Sample4_Data['category'].unique()
print(unique_categories)


# In[ ]:


#####################     Sample4: CSV file is Sample4_Data.csv     ##################### 


# In[ ]:




