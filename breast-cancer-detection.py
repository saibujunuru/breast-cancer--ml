#!/usr/bin/env python
# coding: utf-8

# In[38]:


#importing libraries
import pandas as pd
import numpy as np


# In[39]:


df = pd.read_csv('breast-cancer.csv')


# In[40]:


df


# In[41]:


df.shape


# In[42]:


df.info()


# In[43]:


#data cleaning 
df.drop(df.columns[[-1,0]], axis = 1 , inplace = True)#inplace = true updates the original dataset
df


# In[44]:


df['diagnosis'].value_counts() #to count no of values in a column


# In[45]:


#Feature selection
from sklearn.model_selection import train_test_split


# In[46]:


diag_map = {"M" : 1 , "B" : 0}


# In[47]:


df["diagnosis"] = df["diagnosis"].map(diag_map) #to map th values in dic with the values in dataset


# In[48]:


df


# In[49]:


df['diagnosis'].value_counts()


# In[53]:


x = df[['radius_mean' ,'perimeter_mean', 'area_mean' , 'concavity_mean','concave points_mean']]


# In[54]:


y = df[['diagnosis']]


# In[55]:


x_train , X_test , y_train , y_test = train_test_split(x ,y ,test_size = 0.2 , random_state = 42)


# In[57]:


#model- knn
from sklearn.neighbors import KNeighborsClassifier


# In[58]:


knn = KNeighborsClassifier()


# In[60]:


knn.fit(x_train , y_train)


# In[62]:


knn_y_pred = knn.predict(X_test)


# In[65]:


from sklearn.metrics import accuracy_score
accuracy_score(knn_y_pred , y_test)


# In[66]:


#model - logistic regression
from sklearn.linear_model import LogisticRegression


# In[73]:


lr = LogisticRegression(random_state = 0)
lr.fit(x_train ,y_train)
lr_y_pred = lr.predict(X_test)


# In[74]:


accuracy_score(lr_y_pred , y_test)


# In[79]:


#model - Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb =  GaussianNB()


# In[80]:


gnb.fit(x_train ,y_train)


# In[81]:


gnb_y_pred = gnb.predict(X_test)


# In[82]:


accuracy_score(lr_y_pred , y_test)


# In[84]:


# model - k cross validation
from sklearn.model_selection import cross_val_score
accuracy_all = []
cvs_all = []


# In[91]:


import warnings
warnings.filterwarnings("ignore")


# In[92]:


# naive bayes k cross validation
scores = cross_val_score(knn , x, y, cv= 10)
accuracy_all.append(accuracy_score(knn_y_pred,y_test))
cvs_all.append(np.mean(scores))
print("acc : {0:.2%} ".format(accuracy_score(knn_y_pred , y_test)))
print("cross val : {0:.2%} (+/- {1: .2%})".format(np.mean(scores),np.std(scores)*2))


# In[93]:


# knn k cross validation
scores = cross_val_score(knn , x, y, cv= 10)
accuracy_all.append(accuracy_score(gnb_y_pred,y_test))
cvs_all.append(np.mean(scores))
print("acc : {0:.2%} ".format(accuracy_score(gnb_y_pred , y_test)))
print("cross val : {0:.2%} (+/- {1: .2%})".format(np.mean(scores),np.std(scores)*2))


# In[94]:


#logistic regression k cross validation

scores = cross_val_score(knn , x, y, cv= 10)
accuracy_all.append(accuracy_score(lr_y_pred,y_test))
cvs_all.append(np.mean(scores))
print("acc : {0:.2%} ".format(accuracy_score(lr_y_pred , y_test)))
print("cross val : {0:.2%} (+/- {1: .2%})".format(np.mean(scores),np.std(scores)*2))


# In[ ]:




