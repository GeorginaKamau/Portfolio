#!/usr/bin/env python
# coding: utf-8

# In[1]:


#predicting probability of survival on the titanic ship using machine learning algorithms
#Logistic regression


# In[2]:


#step1
#data preparation
import numpy as np 
import pandas as pd 

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore')


# In[3]:


# Read CSV train data file into DataFrame
train = pd.read_csv(r'C:\Users\GKamau\Downloads\traint.csv')
# preview train data
train.head()


# In[4]:


# Read CSV test data file into DataFrame
test = pd.read_csv(r'C:\Users\GKamau\Downloads\testt.csv')
# preview test data
test.head()


# In[5]:


#entries in ytrain and test data, test should be significantly less
print('The number of samples into the train data is {}.'.format(train.shape[0]))
print('The number of samples into the test data is {}.'.format(test.shape[0]))


# In[6]:


# check for missing values in train data
train.isnull().sum()


# In[7]:


# missing values for test data
test.isnull().sum()


# In[8]:


# percent of missing "Age" 
print('Percent of missing "Age" records is %.2f%%' %((train['Age'].isnull().sum()/train.shape[0])*100))
# percent of missing "Cabin" 
print('Percent of missing "Cabin" records is %.2f%%' %((train['Cabin'].isnull().sum()/train.shape[0])*100))


# In[9]:


# less than 50% of age values are missing, we can replace the nulls with a value either mean or median
# more than 50% of cabin values is missing, replacing the nulls will more than likely change the entire shape of our data distribution
# for embarked we will use mode since there are only 2 missing values


# In[10]:


# age distribution
ax = train["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[11]:


# median is used to impute values for rightly skewed distributions, mean can give biased results
print('The median of "Age" is %.2f' %(train["Age"].median(skipna=True)))


# In[12]:


# Finding ,mode for embarked
print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(train['Embarked'].value_counts())
sns.countplot(x='Embarked', data=train, palette='Set2')
plt.show()


# In[13]:


print('The most common boarding port of embarkation is %s.' %train['Embarked'].value_counts().idxmax())


# In[14]:


#create a copy of the data where youll perform the imputations
train_data = train.copy()
train_data["Age"].fillna(train["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna(train['Embarked'].value_counts().idxmax(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)


# In[15]:


# check missing values in adjusted train data
train_data.isnull().sum()


# In[16]:


# compare distribution of imputed age and initial
plt.figure(figsize=(15,8))
ax = train["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train["Age"].plot(kind='density', color='teal')
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
train_data["Age"].plot(kind='density', color='orange')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[17]:


## Create categorical variable for traveling alone combining SibSp and Parch
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)


# In[18]:


#create categorical variables and drop some variables
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_male', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()
#pclass is divided into 3 columns for each passenger class


# In[19]:


#spliting into categorical variables means its either that or not, using 1 and 0 to indicate


# In[20]:


#do the same for test data
#imputing age variable with the median of age in test dataset
#remove cabin since it has too many missing values
#impute the fare with median


# In[21]:


print('The median of "Age" is %.2f' %(test["Fare"].median(skipna=True)))


# In[22]:


test_data = test.copy()
test_data["Age"].fillna(train["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)


# In[23]:


test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)
test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)


# In[24]:


testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_male', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()


# In[25]:


#exploratory data analysis


# In[26]:


#difference between the ages of those who survived 1 and those who didnt 0
plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[27]:


#a lot of youth didnt survive while many of those who survived were children


# In[28]:


#classifying people according to their age; minors are younger than 18 years
final_train['IsMinor']=np.where(final_train['Age']< 18, 1, 0)
#if age is less than 18, is a minor, 1 .....0 otherwise
final_test['IsMinor']=np.where(final_test['Age']<18, 1, 0)


# In[29]:


plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Fare"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Fare"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
ax.set(xlabel='Fare')
plt.xlim(-20,200)
plt.show()


# In[30]:


#majority of those who died paid a lower fare


# In[31]:


#correlation indexes using spearman
final_train.corr(method = 'spearman')


# In[32]:


#heatmap to visualize correlation between variables using spearman method
plt.figure(figsize=(15,8))
correlation_matrix = final_train.corr(method = 'spearman')
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Matrix HeatMap')
plt.show()


# In[33]:


# Create a box plot for survivers by class
plt.figure(figsize=(8, 6))  
sns.barplot(x='Pclass', y='Survived', data=train)
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.title('Bar Plot of Survived by Pclass')
plt.show()
#middle lines represent the mean of the individual bars


# In[34]:


#These lines are used in a "bar plot" created with Seaborn to represent the standard error of the mean (SEM) or confidence intervals for the data points


# In[35]:


# Create a box plot for where they boarded the ship
plt.figure(figsize=(8, 6))  
sns.barplot(x='Embarked', y='Survived', data=train)
plt.xlabel('Embarked')
plt.ylabel('Survived')
plt.title('Bar Plot of Survived by Pclass')
plt.show()


# In[36]:


# Create a box plot for those who traveled alone vs with family/friends
plt.figure(figsize=(8, 6))  
sns.barplot(x='TravelAlone', y='Survived', data=final_train)
plt.xlabel('TravelAlone')
plt.ylabel('Survived')
plt.title('Bar Plot of Survived by Pclass')
plt.show()


# In[37]:


# Create a box plot comparing genders that survived
plt.figure(figsize=(8, 6))  
sns.barplot(x='Sex', y='Survived', data=train)
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.title('Bar Plot of Survived by Pclass')
plt.show()


# In[38]:


#passengers with a higher survival rate were either from first class,children, female, paid a higher fare, traveled with family/friends or boarded the ship in Cherbourg, Franc


# In[39]:


#getting the important variables for the predictive model
#Feature selection
#Recursive Feature elimination



# In[117]:


#Step 2
#model initialization
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


# In[160]:


# Build a logreg and compute the feature importances
model = LogisticRegression()


# In[161]:


#Step 3
#Model training
cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Pclass_3","Embarked_C","Embarked_S","Embarked_Q","Sex_female","IsMinor"] 
X = final_train[cols]
y = final_train['Survived']


# In[164]:


# create the RFE model and select 10 attributes
rfe = RFE(estimator = model,n_features_to_select=10)
rfe = rfe.fit(X, y)


# In[165]:


# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))


# In[166]:


#get optimal number of features
#RFE in cross validation loop....RFECV
from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))


# In[123]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[167]:


Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2','Pclass_3', 'Embarked_C', 
                     'Embarked_S', 'Embarked_Q', 'Sex_female', 'IsMinor']


# In[168]:


#step 4
#Model evaluation
#review of model evaluation procedures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss


# In[169]:


# create X (features) and y (response)
X = final_train[Selected_features]
y = final_train['Survived']


# In[170]:


#train and test split (70/30) with random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


# In[171]:


# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))


# In[172]:


idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95
print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))


# In[173]:


plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()


# In[174]:


# model evaluation based on k fold cross validation
# 10-fold cross-validation logistic regression
logreg = LogisticRegression()


# In[175]:


# Use cross_val_score function
# We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the data
# cv=10 for 10 folds
# scoring = {'accuracy', 'neg_log_loss', 'roc_auc'} for evaluation metric - althought they are many


# In[176]:


scores_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
scores_log_loss = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')
scores_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')
print('K-fold cross-validation results:')
print(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())
print(logreg.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())
print(logreg.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())


# In[177]:


#model evaluation using cross validate function
from sklearn.model_selection import cross_validate

scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}

modelCV = LogisticRegression()

results = cross_validate(modelCV, X, y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))


# In[178]:


#adding the fare variable to see if it holds any useful information
cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Pclass_3","Embarked_C","Embarked_S","Embarked_Q","Sex_female","IsMinor"]
X = final_train[cols]

scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}

modelCV = LogisticRegression()

results = cross_validate(modelCV, final_train[cols], y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))


# In[179]:


#there isnt significance difference , hence the fare variable is considered noise


# In[180]:


#GridSearchCV evaluates the model using cross-validation while searching through a predefined hyperparameter grid
#list of scoring metrics : ACCURACY, LOG LOSS, ROC-AUC
#
#


# In[181]:


#Define the Hyperparameter Grid
from sklearn.model_selection import GridSearchCV

X = final_train[Selected_features]


# In[182]:


#Choose Scoring Metrics
param_grid = {'C': np.arange(1e-05, 3, 0.1)}
scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}



# In[183]:


#GridSearchCV Configuration
gs = GridSearchCV(LogisticRegression(), return_train_score=True,
                  param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy')

gs.fit(X, y)
results = gs.cv_results_

print('='*20)
print("best params: " + str(gs.best_estimator_))
print("best params: " + str(gs.best_params_))
print('best score:', gs.best_score_)
print('='*20)


# In[184]:


plt.figure(figsize=(15, 10))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)

plt.xlabel("Inverse of regularization strength: C")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, param_grid['C'].max()) 
ax.set_ylim(0.35, 0.95)
# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_C'].data, dtype=float)

for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]


# In[185]:


# Fit (train) the model with the training data
model.fit(X_train, y_train)


# In[191]:


#make predictions with the fitted model
y_pred = model.predict(X_test)


# In[192]:


#Step 5
#Prediction
# Input random values for age, sex, and class (1st, 2nd, or 3rd) 
#passenger 1
age = 30  # Replace with the desired age
travel_alone = 0  # Replace with 1 or 0
Pclass = 1  # Replace with 1, 2, or 3
embarked_c = 0  # Replace with 1 if the passenger boarded in Cherbourg, 0 otherwise
embarked_s = 1  # Replace with 1 if the passenger boarded in Southhampton, 0 otherwise
embarked_q = 0  # Replace with 1 if the passenger boarded in Queenstown, 0 otherwise
sex_female = 0  # Replace with 1 for female passengers, 0 for male passengers
is_minor = 0  # Replace with 1 if the passenger is a minor, 0 otherwise


# In[193]:


# Prepare the input data in the same format as the training data
# Create a DataFrame with feature names that match the model's training data
input_data = pd.DataFrame({'Age': [age],'TravelAlone': [travel_alone], 'Pclass_1': [0], 'Pclass_2': [0], 'Pclass_3': [0],
                           'Embarked_C': [embarked_c], 'Embarked_S': [embarked_s], 'Embarked_Q': [embarked_q], 'Sex_female': [sex_female],
                            'IsMinor': [is_minor]})


# In[199]:


input_data[f'Pclass_{Pclass}'] = 1


# In[195]:


# Make predictions for the input data
predicted_survival = model.predict(input_data)

if predicted_survival[0] == 1:
    print("The passenger is predicted to survive.")
else:
    print("The passenger is predicted not to survive.")


# In[197]:


#Step 5
#Prediction
# Input random values for age, sex, and class (1st, 2nd, or 3rd) 
#passenger 2
age = 30  # Replace with the desired age
travel_alone = 0  # Replace with 1 or 0
Pclass = 3  # Replace with 1, 2, or 3
embarked_c = 1  # Replace with 1 if the passenger boarded in Cherbourg, 0 otherwise
embarked_s = 0  # Replace with 1 if the passenger boarded in Southhampton, 0 otherwise
embarked_q = 0  # Replace with 1 if the passenger boarded in Queenstown, 0 otherwise
sex_female = 1  # Replace with 1 for female passengers, 0 for male passengers
is_minor = 1  # Replace with 1 if the passenger is a minor, 0 otherwise


# In[198]:


# Prepare the input data in the same format as the training data
# Create a DataFrame with feature names that match the model's training data
input_data = pd.DataFrame({'Age': [age],'TravelAlone': [travel_alone], 'Pclass_1': [0], 'Pclass_2': [0], 'Pclass_3': [0],
                           'Embarked_C': [embarked_c], 'Embarked_S': [embarked_s], 'Embarked_Q': [embarked_q], 'Sex_female': [sex_female],
                            'IsMinor': [is_minor]})


# In[200]:


# Make predictions for the input data
predicted_survival = model.predict(input_data)

if predicted_survival[0] == 1:
    print("The passenger is predicted to survive.")
else:
    print("The passenger is predicted not to survive.")


# In[ ]:




