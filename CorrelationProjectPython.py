#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12, 8)


# In[2]:


#create dataframe and store dataset there
data = pd.read_csv(r'C:/Users/Georgina/Desktop/movies.csv')


# In[3]:


#view data
data.head()


# In[4]:


#percentage of missing data using for loop
for col in data.columns:
    missinginfo = np.mean(data[col].isnull())
    print('{} - {}%' .format(col, missinginfo))


# In[5]:


#column data types
data.dtypes


# In[6]:


#identify the missing values
data.isnull()


# In[7]:


#fish out the missing values
data1 = data.dropna()


# In[8]:


#percentage of missing data using for loop in the new dataframe (data1)
for col in data1.columns:
    missinginfo = np.mean(data1[col].isnull())
    print('{} - {}%' .format(col, missinginfo))


# In[9]:


#change unnecessary float datatypes to integers
data1['budget'] = data1['budget'].astype('int64')
data1['gross'] = data1['gross'].astype('int64')


# In[10]:


#confirm the change in column data types
data1.dtypes


# In[12]:


#creating a new column for year from released date
#convert rleased column from object data type to string
data1['monthreleased'] = data1['released'].astype(str).str[:4]
data1


# In[13]:


#deleting repeated column 
del data1['newyear']


# In[14]:


data1


# In[15]:


#rank the movies by score
data1.sort_values(by=['score'], inplace = False, ascending =False )


# In[16]:


#Case 1
#H_0:There is no significant correlation / linear relation between budget and gross
#H_A: The is a significant correlation between budget and gross

#Case2
#H_0:There is no significant correlation / linear relation between score and gross
#H_A: The is a significant correlation between score and gross

#Case3
#H_0:There is no significant correlation / linear relation between votes and gross
#H_A: The is a significant correlation between votes and gross

#Case4
#H_0:There is no significant correlation / linear relation between company and budget
#H_A: The is a significant correlation between company and budget

#Case5
#H_0:There is no significant correlation / linear relation between monthreleased and gross
#H_A: The is a significant correlation between monthreleased and gross


# In[22]:


#correlation of (original) numeric data using spearman method
data1.corr(method = 'spearman')


# In[28]:


#visualize correlation matrix using heatmap
correlation_matrix = data1.corr(method = 'spearman')
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix HeatMap')
plt.show()
#darker colors represent a lower correlation


# In[18]:


#Case 1
#Scatter plot
plt.scatter(x=data1['budget'], y=data1['gross'])
plt.title('Film Budget VS Gross Earnings')
plt.xlabel('Film Budget')
plt.ylabel('Gross Earnings')
plt.show()


# In[20]:


#Case 1
#regression model fit
sns.regplot(x = 'budget', y = 'gross', data = data1, scatter_kws = {"color" : "yellow"}, line_kws = {"color" : "black"})


# In[24]:


#Case 2
#Scatter plot
plt.scatter(x=data1['score'], y=data1['gross'])
plt.title('Film score VS Gross Earnings')
plt.xlabel('Film score')
plt.ylabel('Gross Earnings')
plt.show()


# In[26]:


#Case 2
#regression model fit
sns.regplot(x = 'score', y = 'gross', data = data1, scatter_kws = {"color" : "pink"}, line_kws = {"color" : "black"})


# In[33]:


#budget and gross have a correlation of 0.6930 
#score and gross have a correlation of 0.1831
#votes and gross have a correlation of 0.7458
#Case1 Conclusion : Since the correlation between budget and gross is a strong correlation of 0.6930, there is significant evidence to reject the null hypothesis.
#Case 2 Conclusion : Since the correlation between score and gross is a weak correlation of 0.1831, there is significant evidence to fail to reject the null hypothesis.
#Case 3 Conclusion : Since the correlation between votes and gross is a strong correlation of 0.7458, there is significant evidence to reject the null hypothesis.


# In[30]:


#Giving object data types random unique numeric representatives (using unranked data)
#df - new data frame with unique numeric representations
df = data1
for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name] = df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes
df


# In[31]:


#correlation for all new data 
df.corr(method = 'pearson')


# In[32]:


#visualize correlation matrix using heatmap
correlation_matrix = df.corr(method = 'pearson')
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix HeatMap')
plt.show()
#darker colors represent a lower correlation


# In[ ]:


#weak correlations
#company and budget have a correlation of 0.1702
#month released and gross have a correlation of 0.0068
#Case 4 Conclusion : Since the correlation between company and budget is 0.1702, there is significant evidence to fail to reject the null hypothesis.
#Cse 5 Conclusion: Since the correlation between month released and gross is 0.0068, there is significant evidence to fail to reject the null hypothesis.

