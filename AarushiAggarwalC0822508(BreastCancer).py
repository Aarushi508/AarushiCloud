#!/usr/bin/env python
# coding: utf-8

# In[32]:


#Loading Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore")
import opendatasets as od

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[33]:


#Loading Dataset
data = pd.read_csv("breast-cancer-wisconsin-data/data.csv")


# In[34]:


#Displaying First five rows of dataset
data.head()


# In[35]:


#Dropping the features which we do not want
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)


# In[36]:


#Changing the title of the properties
data = data.rename(columns = {"diagnosis":"target"})


# In[37]:


#Displaying Columns of our dataset
data.columns


# In[38]:


# Displaying the total number of rows and columns of our dataset
print("Data Shape:", data.shape) 


# In[39]:


#Displaying the information of dataset
data.info()


# In[40]:


#Changing the data type from string to integer as it is required during trains.(B=0, M-1)
data["target"]=[1 if i.strip()=="M" else 0 for i in data.target]


# In[41]:


#Describing dataset
data.describe()


# In[42]:


#Finding corelation between the data
data.corr()


# In[43]:


#Plotting the dataset(Data Analysis)
data_melt=pd.melt(data,id_vars="target",
                 var_name="features",
                 value_name="value")


# In[44]:


plt.figure(figsize=(10,7))
sns.boxplot(x="features", y="value",hue="target", data=data_melt)
plt.xticks(rotation=90)
plt.show()


# In[45]:


#Defining x and y variables
y= data.target
x= data.drop(["target"],axis=1)


# In[46]:


columns=x.columns.tolist()


# In[47]:


#Loading some more important libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis


# In[48]:


#Splitting data in Test and Training Data
test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)


# In[49]:


#Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[50]:


#Describing Training and Testing dataset
X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df.describe()


# In[51]:


#Defining Y test variable
X_train_df["target"] = Y_train


# In[52]:


#Plotting graph after splitting
data_melted = pd.melt(X_train_df, id_vars = "target",
                       var_name = "features",
                       value_name = "value")


# In[53]:


plt.figure(figsize=(10,7))
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()


# In[54]:


#Simple k-nearest neighbors classifiers
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
score = knn.score(X_test, Y_test)

print("Basic KNN Accuracy: % {}".format(acc))
print("Score : ", score)
print("Confusion Matrix : ", cm)


# In[55]:


# #KNN Best Parameters
def KNN_Best_Params(x_train, x_test, y_train, y_test):
    
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
     
    knn = KNeighborsClassifier()
    
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy")
    grid.fit(x_train, y_train)
    
    print("Best training score : {} with paremeters : {}".format(grid.best_score_, grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_) # best paremetre olarak gelen değerlerimiz.
    knn.fit(x_train, y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test) 
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ",cm_test)
    print("CM Train: ",cm_train)
    
    return grid


# In[56]:


#Printing Training, Testing Scores and Confusion Matrix
grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test) 


# In[57]:


#Printing Accuracy
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
acc_test_nca = accuracy_score(y_pred,Y_test)
print("Accuracy Score --> {}".format(knn.score(X_test,Y_test)))

