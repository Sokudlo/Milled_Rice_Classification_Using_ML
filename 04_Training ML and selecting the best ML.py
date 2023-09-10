#!/usr/bin/env python
# coding: utf-8

# # Training ML and selecting the best ML

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('Training_Data.csv')
df.info()


# In[3]:


df.columns


# In[4]:


X=df[['area', 'major_axis_length', 'minor_axis_length', 'perimeter',
       'eccentricity', 'solidity', 'extent', 'equiv_diameter', 'aspect_ratio',
       'compactness', 'roundness']]


# In[5]:


import pandas as pd


df['category'] = df['category'].replace({
    'Small Broke C1': '1',
    'Small Broke': '2',
    'Big Broke': '3',
    'Head rice': '4',
    'Whole Rice': '5'
})


# In[6]:


y=df[['category']]


# In[7]:


y


# In[ ]:





# In[8]:


X1=df[['area', 'major_axis_length',  'perimeter','equiv_diameter' ]] # According to the result of Boruta's feature selection


# In[9]:


X1


# In[10]:


from sklearn.model_selection import train_test_split

#Splitting training and validating data
X1_train, X1_test, y_train, y_test = train_test_split (X1,y, test_size= 0.3, random_state=7, shuffle=True)


# In[11]:


import sklearn
from sklearn.model_selection import train_test_split

# classifiers algorithm
from sklearn.neighbors import KNeighborsClassifier                      # 1. K-Neighbors Classifier
from sklearn.linear_model import LogisticRegression                    # 2. Logistic Regression Classifier
from sklearn.tree import DecisionTreeClassifier                            # 3. Decision Tree Classifier            
from sklearn.ensemble import GradientBoostingClassifier              # 4. Gradient Boosting Classifier
from sklearn.ensemble import RandomForestClassifier                 # 5. Random Forest Classifier
from sklearn.ensemble import BaggingClassifier                          # 6. Bagging Classifier
from sklearn.ensemble import AdaBoostClassifier                        # 7. Ada Boost Classifier
from sklearn.naive_bayes import GaussianNB                             # 8. Gaussian NB Classifier
from sklearn.neural_network import MLPClassifier                       # 9. Multilayer Perceptron 
from sklearn.svm import SVC                                                    # 10. Support Vector Classifier
from sklearn.gaussian_process import GaussianProcessClassifier   # 11. Gaussian Process Classifier


from sklearn import metrics
import time


# In[12]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

algo = [
    [KNeighborsClassifier(n_neighbors=4), 'KNeighborsClassifier'],
    [LogisticRegression(solver='liblinear'), 'LogisticRegression'],
    [DecisionTreeClassifier(min_samples_split=3), 'DecisionTreeClassifier'],
    [GradientBoostingClassifier(), 'GradientBoostingClassifier'],
    [RandomForestClassifier(), 'RandomForestClassifier'],
    [BaggingClassifier(), 'BaggingClassifier'],
    [AdaBoostClassifier(n_estimators=5), 'AdaBoostClassifier'],
    [GaussianNB(), 'GaussianNB'],
    [MLPClassifier(), 'MLPClassifier'],
    [SVC(kernel='linear'), 'SVC_linear'],
    [GaussianProcessClassifier(), 'GaussianProcessClassifier']
]

top_models = []  # List to store the top models
top_scores = []  # List to store the corresponding scores

for a in algo:
    model = a[0]
    start_time = time.time()
    model.fit(X1_train, y_train)
    end_time = time.time()
    total_time = end_time - start_time
    y_pred = model.predict(X1_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Model: {a[1]}")
    print(f"Total Computation Time: {total_time:.4f} seconds")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    
    # Calculate a combined score 
    combined_score = (accuracy + precision + recall + f1) / 4
    
    # Add the model and its score to the top_models and top_scores lists
    top_models.append(a[1])
    top_scores.append(combined_score)

# Sort the models based on scores in descending order
sorted_models = [x for _, x in sorted(zip(top_scores, top_models), reverse=True)]




# In[13]:


# Print the top 3 models
print("Top 3 Models:")
for i, model in enumerate(sorted_models[:3]):
    print(f"{i + 1}. {model}")


# In[ ]:




