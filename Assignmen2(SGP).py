#!/usr/bin/env python
# coding: utf-8

# In[91]:


#loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.utils import shuffle


# In[92]:


#loading datasets and inspecting it
df=pd.read_csv('D:\lambton\CBD 2214(Big Data)\phython\student-mat.csv')
#Filtering out 0 grades
df = df[~df['G3'].isin([0,1])]
print(df.info())
#Displaying the top rows of data
df.head()


# In[93]:


#Displaying last rows of data
df.tail()


# In[94]:


#Displaying total size of dataframe
df.shape


# In[95]:


#Displaying Column Names
df.columns


# In[96]:


#Displaying DataTypes
df.dtypes


# In[97]:


#Dropping Null Values
df.dropna


# In[98]:


#Dropping Duplicate data
df.drop_duplicates


# In[99]:


#Statistical Calculations of data
df.describe()


# In[100]:


#Coreorelation between different variables
df.corr()


# In[101]:


#boxplot visualization
plt.figure(figsize=(15,10))
sns.boxplot(data=df)


# In[102]:


#Displaying correlation of G3 with Failure using regplot
sns.regplot(x = 'failures',y ='G3',data=df).set_title('Effect of failures on final grade')


# In[103]:


#Displaying correlation of G3 with absences using regplot
sns.regplot(x = 'absences',y = 'G3', data=df).set_title('Effect of Absences on grade')


# In[104]:


#Displaying correlation of G3 with studytime using regplot
sns.regplot(x = 'studytime',y = 'G3', data=df).set_title('Effect of studytime on grades')


# In[105]:


#Displaying correlation of G3 with G2 using regplot
sns.regplot(x = 'G2',y = 'G3', data = df).set_title('G2 vs G3 grades')


# In[106]:


#selecting necessary columns 
df = df[['G1','G2','G3','studytime','failures','absences']]
predict = 'G3'
#Splitting dataset, 20% testing 80% training
#Defining independent variable
x = np.array(df.drop([predict],1))
#Defining dependent variable
y = np.array(df[predict])
#Importing train_test_split model to train and test data 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=100)


# In[107]:


# Size of Train and Test sets
x_train.shape


# In[108]:


x_test.shape


# In[109]:


#Using linear Regression methods to find accuracy of data
linear_regression = LinearRegression()
linear_regression.fit(xtrain,ytrain)
accuracy = linear_regression.score(xtest,ytest)
print(accuracy)


# In[110]:


#Standardizing the data((z = x - m)/std)
from sklearn.preprocessing import StandardScaler
#Creating object of StandardScaler class
sc = StandardScaler()
x_train = sc.fit_transform(xtrain)
x_test = sc.transform(xtest)


# In[111]:


xtrain


# In[112]:


xtest


# In[113]:


#Scaling the data into standardized form
for name,method in [('Linear Regression', LinearRegression(n_jobs=-1))]: 
    method.fit(x_train,ytrain)
    predict = method.predict(x_test)

print('Method: {}'.format(name))   

#Determining and Printing Intercept and coefficients
print('\nIntercept: {:0.2f}'.format(float(method.intercept_)))
coeff_table=pd.DataFrame(np.transpose(method.coef_),columns=['Coefficients'])
print('\n')
print(coeff_table)
    
#Determing MAE,MSE and RMSE
print('\nR2: {:0.2f}'.format(metrics.r2_score(ytest, predict)))
print('Mean Absolute Error: {:0.2f}'.format(metrics.mean_absolute_error(ytest, predict)))  
print('Mean Squared Error: {:0.2f}'.format(metrics.mean_squared_error(ytest, predict)))  
print('Root Mean Squared Error: {:0.2f}'.format(np.sqrt(metrics.mean_squared_error(ytest, predict)))) 


# In[114]:


#Forecast Table
predict2 = predict.T
diff = predict2-ytest
Forecast=pd.DataFrame({'Actual':ytest,'Predicted':predict2.round(1),'Difference':diff.round(1)})
print('\nForecast Table')
Forecast.head()

