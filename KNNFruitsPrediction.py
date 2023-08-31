#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Every fruit is described with four features:
#1) mass of fruit 
#2) width of fruit 
#3) height
#4) color score of fruit


# In[2]:


#KNN to predict fruits
#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#read txt document
df= pd.read_table('C:/Users/Georgina/Desktop/fruitsds.txt')


# In[4]:


#first few rows
df.head()


# In[5]:


print(df.shape)


# In[6]:


#descriptive statistics to summarize the central tendency, dispersion and shape of a dataset’s distribution
df.describe()


# In[7]:


#count number of fruits - how manyentries per fruit
print(df.groupby('fruit_name').size())


# In[8]:


#information about the data types & columns
df.info(verbose=True)


# In[9]:


#visualize distribution of fruits - count number #store data in 4 different data fra
sns.countplot(df['fruit_name'],label="Count")
plt.show()


# In[10]:


# giving unique labels to fruit name to make results easier to interpret
predct = dict(zip(df.fruit_label.unique(), df.fruit_name.unique()))   
predct


# In[11]:


#store data in 4 different dataframes
apples=df[df['fruit_name']=='apple']
oranges=df[df['fruit_name']=='orange']
lemons=df[df['fruit_name']=='lemon']
mandarins=df[df['fruit_name']=='mandarin']


# In[12]:


#contains information on apples
apples.head()
#unique label for apples is 1


# In[13]:


#contains information on oranges
oranges.head()
#unique label for oranges is 3


# In[14]:


#contains information on lemons
lemons.head()
#unique label for lemons is 4


# In[15]:


#contains information on mandarins
mandarins.head()
#unique label for mandarins is 2


# In[16]:


#scatter plot according to fruit labels
plt.scatter(df['width'],df['height'])


# In[17]:


plt.scatter(df['mass'],df['color_score'])


# In[18]:


#use KNN to predict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[19]:


#split into train and test
x=df[['mass','width','height']]
y=df['fruit_label']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[20]:


knn=KNeighborsClassifier()


# In[21]:


knn.fit(x_train,y_train)


# In[22]:


#checking accuracy of classifier
knn.score(x_test,y_test)


# In[23]:


#making a prediction using the parameters mass,width and height
#example1 - a fruit has a mass of 55, width of 5.0 and a height of 7
prediction1=knn.predict([['55','5.0','7']])
predct[prediction1[0]]


# In[24]:


#making a prediction using the parameters mass,width and height
#example2 - a fruit has a mass of 150, width of 10.0 and a height of 8
prediction1=knn.predict([['150','10.0','8']])
predct[prediction1[0]]


# In[25]:


#making a prediction using the parameters mass,width and height
#example3 - a fruit has a mass of 300, width of 7 and a height of 10
prediction2=knn.predict([['300','7','10']])
predct[prediction2[0]]


# In[26]:


#making a prediction using the parameters mass,width and height
#example4 - a fruit has a mass of 100, width of 6.5 and a height of 7
prediction1=knn.predict([['100','6.5','7']])
predct[prediction1[0]]


# In[ ]:




