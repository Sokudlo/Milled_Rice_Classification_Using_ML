#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[45]:


df=pd.read_csv('Training_Data.csv')
df.info()


# In[46]:


df.columns


# In[47]:


df.describe()


# In[48]:


X=df[['area', 'major_axis_length', 'minor_axis_length', 'perimeter',
       'eccentricity', 'solidity', 'extent', 'equiv_diameter', 'aspect_ratio',
       'compactness', 'roundness']]


# In[49]:


unique_categories = df['category'].unique()
print(unique_categories)


# In[50]:


y=df[['category']]
y.describe()


# In[51]:


import pandas as pd

# Assuming 'df' is your DataFrame with the 'Category' column
df['category'] = df['category'].replace({
    'Small Broke C1': '1',
    'Small Broke': '2',
    'Big Broke': '3',
    'Head rice': '4',
    'Whole Rice': '5'
})

# Verify the updated 'Category' column


# In[52]:


print(df['category'])


# In[53]:


y=df[['category']]


# In[54]:


import pandas as pd
import numpy as np

np.random.seed(42)

X = pd.DataFrame(X)  # Convert X to a DataFrame

X_shadow = X.apply(np.random.permutation)
X_shadow.columns = ['shadow_' + str(feat) for feat in X.columns]
X_boruta = pd.concat([X, X_shadow], axis=1)


# In[55]:


X_shadow.columns


# In[56]:


X_boruta


# In[57]:


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(max_depth=10, random_state=42)
forest.fit(X_boruta, y)
# Store feature importances
feat_imp_X = forest.feature_importances_[:len(X.columns)]
feat_imp_shadow = forest.feature_importances_[len(X.columns):]
# Compute hits
hits= feat_imp_X > feat_imp_shadow.max()


# In[58]:


import numpy as np

arr = np.array([feat_imp_X])

np.set_printoptions(formatter={'float': lambda x: format(x, '10.10f')})
print(arr)


# In[59]:


hits


# In[60]:


feat_imp_shadow


# In[61]:


fs = pd.Series(data=hits, index=X.columns)
fs


# In[62]:


fss = pd.Series(data=feat_imp_shadow, index=X.columns)#[hits] #.sort_values(ascending=True)
fss


# In[63]:


#fs = pd.Series(feat_imp_X, index = X.columns).sort_values (ascending= True)
fs = pd.Series(data=feat_imp_X, index=X.columns)[hits].sort_values(ascending=True)

fs


# In[ ]:




