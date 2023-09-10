#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


df_0 = pd.read_csv('Training_Data.csv')


# In[3]:


X_train=df_0[['area', 'major_axis_length', 'perimeter','equiv_diameter']]
y_train=df_0[['category']]


# In[4]:


import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[5]:


model=RandomForestClassifier()


# In[6]:


model.fit(X_train, y_train)


# In[7]:


model.score(X_train, y_train) #R-squared


# In[ ]:





# # Sample1

# In[113]:


df_1 = pd.read_csv('Sample1_Data.csv')


# In[114]:


X_test1=df_1[['area', 'major_axis_length',  'perimeter','equiv_diameter']]
y_test1=df_1[['category']]


# In[115]:


import time

start_time = time.time()

y_pred1=model.predict(X_test1)

end_time = time.time()

Computation_time = end_time - start_time


# In[116]:


test1=pd.concat([X_test1, y_test1], axis='columns')


# In[117]:


dc1=pd.concat([test1.reset_index(), pd.Series(y_pred1, name='predicted')], axis='columns')


# In[118]:


dc1


# In[119]:


Testing_accuracy = accuracy_score(y_test1, y_pred1)
print('Accuracy on testing data: {:.2f}%'.format(Testing_accuracy * 100))


# In[120]:


category_labels = sorted(y_test1['category'].unique())

cm = confusion_matrix(y_test1, y_pred1)
plt.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar()

plt.xticks(np.arange(len(category_labels)), category_labels, rotation='vertical')
plt.yticks(np.arange(len(category_labels)), category_labels)

plt.xlabel('Predicted')
plt.ylabel('True')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.show()


# In[121]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(f"Total Computation Time: {Computation_time:.4f} seconds")
print("Accuracy Score:", accuracy_score(y_test1, y_pred1) * 100)
print("Testing Report:\n", classification_report(y_test1, y_pred1, digits=4))


# In[122]:


# Calculate metrics
accuracy = accuracy_score(y_test1, y_pred1)
classification_rep = classification_report(y_test1, y_pred1, digits=4, output_dict=True)
macro_avg_precision = classification_rep['macro avg']['precision']
macro_avg_recall = classification_rep['macro avg']['recall']
macro_avg_f1_score = classification_rep['macro avg']['f1-score']

# Create the result1 dictionary with the calculated values
result1 = {
    'Total Computation Time': Computation_time,
    'Accuracy': accuracy * 100,
    'Macro Avg Precision': macro_avg_precision * 100,
    'Macro Avg Recall': macro_avg_recall * 100,
    'Macro Avg F1-Score': macro_avg_f1_score * 100
}

# Print the formatted values
for key, value in result1.items():
    if isinstance(value, float):
        # Format floats to display exactly four decimal places
        formatted_value = '{:.4f}'.format(value)
    else:
        formatted_value = value  # Leave non-float values as they are

    print(f'{key}: {formatted_value}')


# In[ ]:





# # Sample2

# In[123]:


df_2 = pd.read_csv('Sample2_Data.csv')


# In[124]:


X_test2=df_2[['area', 'major_axis_length',  'perimeter','equiv_diameter']]
y_test2=df_2[['category']]


# In[125]:


start_time = time.time()

y_pred2=model.predict(X_test2)

end_time = time.time()

Computation_time = end_time - start_time


# In[126]:


test2=pd.concat([X_test2, y_test2], axis='columns')


# In[127]:


dc2=pd.concat([test2.reset_index(), pd.Series(y_pred2, name='predicted')], axis='columns')


# In[128]:


dc2


# In[129]:


Testing_accuracy = accuracy_score(y_test2, y_pred2)
print('Accuracy on testing data: {:.2f}%'.format(Testing_accuracy * 100))


# In[130]:


category_labels = sorted(y_test2['category'].unique())

cm = confusion_matrix(y_test2, y_pred2)
plt.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar()

plt.xticks(np.arange(len(category_labels)), category_labels, rotation='vertical')
plt.yticks(np.arange(len(category_labels)), category_labels)

plt.xlabel('Predicted')
plt.ylabel('True')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.show()


# In[131]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(f"Total Computation Time: {Computation_time:.4f} seconds")
print("Accuracy Score:", accuracy_score(y_test2, y_pred2) * 100)
print("Testing Report:\n", classification_report(y_test2, y_pred2, digits=4))


# In[132]:


# Calculate metrics
accuracy = accuracy_score(y_test2, y_pred2)
classification_rep = classification_report(y_test2, y_pred2, digits=4, output_dict=True)
macro_avg_precision = classification_rep['macro avg']['precision']
macro_avg_recall = classification_rep['macro avg']['recall']
macro_avg_f1_score = classification_rep['macro avg']['f1-score']

# Create the result1 dictionary with the calculated values
result2 = {
    'Total Computation Time': Computation_time,
    'Accuracy': accuracy * 100,
    'Macro Avg Precision': macro_avg_precision * 100,
    'Macro Avg Recall': macro_avg_recall * 100,
    'Macro Avg F1-Score': macro_avg_f1_score * 100
}

# Print the formatted values
for key, value in result2.items():
    if isinstance(value, float):
        # Format floats to display exactly four decimal places
        formatted_value = '{:.4f}'.format(value)
    else:
        formatted_value = value  # Leave non-float values as they are

    print(f'{key}: {formatted_value}')


# In[ ]:





# # Sample3

# In[133]:


df_3 = pd.read_csv('Sample3_Data.csv')


# In[134]:


X_test3=df_3[['area', 'major_axis_length',  'perimeter','equiv_diameter']]
y_test3=df_3[['category']]


# In[135]:


start_time = time.time()

y_pred3=model.predict(X_test3)

end_time = time.time()

Computation_time = end_time - start_time


# In[136]:


test3=pd.concat([X_test3, y_test3], axis='columns')


# In[137]:


dc3=pd.concat([test3.reset_index(), pd.Series(y_pred3, name='predicted')], axis='columns')


# In[138]:


dc3


# In[139]:


Testing_accuracy = accuracy_score(y_test3, y_pred3)
print('Accuracy on testing data: {:.2f}%'.format(Testing_accuracy * 100))


# In[140]:


category_labels = sorted(y_test3['category'].unique())

cm = confusion_matrix(y_test3, y_pred3)
plt.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar()

plt.xticks(np.arange(len(category_labels)), category_labels, rotation='vertical')
plt.yticks(np.arange(len(category_labels)), category_labels)

plt.xlabel('Predicted')
plt.ylabel('True')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.show()


# In[141]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(f"Total Computation Time: {Computation_time:.4f} seconds")
print("Accuracy Score:", accuracy_score(y_test3, y_pred3) * 100)
print("Testing Report:\n", classification_report(y_test3, y_pred3, digits=4))


# In[142]:


# Calculate metrics
accuracy = accuracy_score(y_test3, y_pred3)
classification_rep = classification_report(y_test3, y_pred3, digits=4, output_dict=True)
macro_avg_precision = classification_rep['macro avg']['precision']
macro_avg_recall = classification_rep['macro avg']['recall']
macro_avg_f1_score = classification_rep['macro avg']['f1-score']

# Create the result1 dictionary with the calculated values
result3 = {
    'Total Computation Time': Computation_time,
    'Accuracy': accuracy * 100,
    'Macro Avg Precision': macro_avg_precision * 100,
    'Macro Avg Recall': macro_avg_recall * 100,
    'Macro Avg F1-Score': macro_avg_f1_score * 100
}

# Print the formatted values
for key, value in result3.items():
    if isinstance(value, float):
        # Format floats to display exactly four decimal places
        formatted_value = '{:.4f}'.format(value)
    else:
        formatted_value = value  # Leave non-float values as they are

    print(f'{key}: {formatted_value}')


# In[ ]:





# # Sample4

# In[143]:


df_4 = pd.read_csv('Sample4_Data.csv')


# In[144]:


X_test4=df_4[['area', 'major_axis_length',  'perimeter','equiv_diameter']]
y_test4=df_4[['category']]


# In[145]:


start_time = time.time()

y_pred4=model.predict(X_test4)

end_time = time.time()

Computation_time = end_time - start_time


# In[146]:


test4=pd.concat([X_test4, y_test4], axis='columns')


# In[147]:


dc4=pd.concat([test4.reset_index(), pd.Series(y_pred4, name='predicted')], axis='columns')


# In[148]:


dc4


# In[149]:


Testing_accuracy = accuracy_score(y_test4, y_pred4)
print('Accuracy on testing data: {:.2f}%'.format(Testing_accuracy * 100))


# In[150]:


category_labels = sorted(y_test4['category'].unique())

cm = confusion_matrix(y_test4, y_pred4)
plt.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar()

plt.xticks(np.arange(len(category_labels)), category_labels, rotation='vertical')
plt.yticks(np.arange(len(category_labels)), category_labels)

plt.xlabel('Predicted')
plt.ylabel('True')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.show()


# In[151]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(f"Total Computation Time: {Computation_time:.4f} seconds")
print("Accuracy Score:", accuracy_score(y_test4, y_pred4) * 100)
print("Testing Report:\n", classification_report(y_test4, y_pred4, digits=4))


# In[152]:


# Calculate metrics
accuracy = accuracy_score(y_test4, y_pred4)
classification_rep = classification_report(y_test4, y_pred4, digits=4, output_dict=True)
macro_avg_precision = classification_rep['macro avg']['precision']
macro_avg_recall = classification_rep['macro avg']['recall']
macro_avg_f1_score = classification_rep['macro avg']['f1-score']

# Create the result1 dictionary with the calculated values
result4 = {
    'Total Computation Time': Computation_time,
    'Accuracy': accuracy * 100,
    'Macro Avg Precision': macro_avg_precision * 100,
    'Macro Avg Recall': macro_avg_recall * 100,
    'Macro Avg F1-Score': macro_avg_f1_score * 100
}

# Print the formatted values
for key, value in result4.items():
    if isinstance(value, float):
        # Format floats to display exactly four decimal places
        formatted_value = '{:.4f}'.format(value)
    else:
        formatted_value = value  # Leave non-float values as they are

    print(f'{key}: {formatted_value}')


# In[ ]:




