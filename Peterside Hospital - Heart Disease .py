#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORT NECESSARY LIBRARIES

# FOR DATA ANALYSIS
import pandas as pd
import numpy as np

# FOR DATA VISUALISATION.
import matplotlib.pyplot as plt
import seaborn as sns

# FOR DATA PRE-PROCESSING.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# CLASSIFIER LIBRARIES
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.svm import LinearSVC, SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier

# EVALUATION METRICS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 
from sklearn.metrics import confusion_matrix

import warnings 
warnings.filterwarnings ("ignore")


# In[2]:


# LOAD DATASET
df = pd.read_csv (r"/Users/mamsie/Desktop/Data Science/WEEK 11/Supervised Machine Learning/heart.csv")
df.head()


# # Features in the dataset and meaning:
# * age - age in years,
# * sex - (1 = male; 0 = female),
# * cp - chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4; asymptomatic),
# * trestbos - resting blood pressure (in mm Hg on admission to the hospital),
# * chol - serum cholestoral in mg/dl,
# * fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false),
# * restecg - resting electrocardiographic results,
# * thalach - maximum heart rate achieved,
# * exang - exercise induced angina (1 = yes; 0 = no),
# * oldpeak - ST depression induced by exercise relative to rest,
# * slope - the slope of the peak exercise ST segment,
# * ca - number of major vessels (0-3) colored by flourosopy,
# * thal - 3 = normal; 6 = fixed defect: 7 = reversable defect,
# * target - have disease or not (1=yes, 0=no).

# In[3]:


# FOR BETTER UNDERSTANDING AND FLOW OF ANALYSIS, I WILL RENAME SOME OF THE COLUMNS
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol','fasting_blood_sugar','rest_ecg',
'max_heart_rate_achieved','exercise_induced_angina','st_depression','st_slope' , 'num_major_vessels' , 'thalassemia' , 'target']
df.head()


# In[4]:


# DATA VERIFICATION - DATA TYPE, NUMBER OF FEATURES AND ROWS, MISSING DATA, ETC  
df.info()


# In[5]:


# STATISTICAL ANALYSIS OF THE DATA  
df.describe()


# In[6]:


# CHECK FOR MISSING VALUES
print(df.isnull().sum())

# VISUALISING THE MISSING DATA
plt.figure(figsize = (10,3))
sns.heatmap(df.isnull(), cbar=True, cmap="Blues_r");


# ## EXPLORATORY DATA ANALYSIS
# 
# ### UNIVARIATE ANALYSIS

# In[7]:


# RETURN THE COLUMN LABELS (COLUMN NAMES) OF THE DATAFRAME 
df.columns


# In[8]:


# CHECK FOR OUTLIERS
sns.boxplot(x=df["thalassemia"]);


# In[9]:


# CHECK FOR OUTLIERS
sns.boxplot(x=df["cholesterol"]);


# In[10]:


# CHECK FOR OUTLIERS
sns.boxplot(x=df["resting_blood_pressure"]);


# In[11]:


# CHECK FOR OUTLIERS
sns.boxplot(x=df["max_heart_rate_achieved"]);


# * The plots above indicate the presence of outliers in several features of the dataset.

# In[12]:


# DATA VISUALISATION
# AGE BRACKET
def age_bracket(age): 
    if age <= 35:
        return "Young adults"
    elif age <= 55:
        return "Middle-aged adults" 
    elif age <= 65:
        return "Senior citizens" 
    else:
        return "Elderly"
df['age_bracket'] = df['age'].apply(age_bracket)

# INVESTIGATING THE AGE GROUP OF PATIENTS

# Sets the size of the plot to be 10 inches in width and 5 inches in height.
plt.figure(figsize = (10, 5))

# Creates a countplot using the Seaborn library, with 'age_bracket' on the x-axis 
sns.countplot (x='age_bracket', data=df) 

# Sets the label for the x-axis as 'Age Group'.
plt.xlabel('Age Group')

# Sets the label for the y-axis as 'Count of Age Group'.
plt.ylabel('Count of Age Group') 

# Sets the title of the plot as 'Total Number of Patients'.
plt.title('Total Number of Patients');


# * Among the patients in the hospital, the age group with the highest number of individuals are middle-aged and older adults, typically between the ages of 36 to 65. Conversely, young adults make up the smallest proportion of patients.

# In[13]:


df.head()


# In[14]:


# DATA VISUALISATION

# Investigate the gender distribution of the patients. 
def Gender (sex) :
    if sex == 1:
        return "Male"
    else:
        return "Female"
    
df['Gender'] = df['sex'].apply(Gender)


# In[15]:


# Convert the gender counts into a dataframe
gender_counts = df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

# Create a pie chart
plt.figure(figsize=(10, 5))
plt.pie(gender_counts['Count'], labels=gender_counts['Gender'], startangle=90, counterclock=False, autopct='%1.1f%%')
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Add a circle at the center to create a doughnut chart
plt.gca().axis('equal')
plt.title('Gender Distribution')
plt.tight_layout()
plt.show()


# * The hospital has twice as many male patients as female patients, with males accounting for 68.3% of the total patients and females comprising 31.7%.

# In[16]:


# DATA VISUALISATION

# Visualise the distribution of patients based on the different categories of chest pain.

def chest_pain(cp):
    if cp == 1:
        return "Typical Angina"
    elif cp == 2:
        return "Atypical Angina" 
    elif cp == 3:
        return "Non-anginal Pain"
    else:
        return "Asymptomatic"
    
df['cp_cat'] = df['chest_pain_type'].apply (chest_pain)

plt. figure(figsize = (10, 5))
sns.countplot(x='cp_cat', data=df)  
plt.xlabel ("Types of chest pain") 
plt.ylabel("Count of patient Gender") 
plt. title("Total Number of Patients");


# * Regarding chest pain categories, patients with the highest number are asymptomatic, followed by those with atypical angina, typical angina, and finally, those with non-anginal pain, who are the least represented.

# In[17]:


# DATA VISUALISATION

# target - have disease or not (1=yes, 0=no)
def label(tg):
    if tg == 1:
        return "Yes" 
    else:
        return "No"
df['label'] = df['target'].apply(label)

# Total patient in each category
print(df["label"].value_counts ())

# Investigate the gender distribution of patients in each category of 'label'.
plt.figure(figsize = (10, 5))
sns.countplot (x='label', data=df) 
plt.xlabel('Target')
plt.ylabel ('Count of patient Gender')
plt. title( 'Total Number of Patients');


# * There are slightly more patients who have been diagnosed with heart disease than those who have not.

# ## BIVARIATE ANALYSIS

# In[18]:


# Visualising the distribution of the age group of patients whether they have a disease (represented by the 'label' column)
plt.figure(figsize = (10, 5))
sns.countplot(x='age_bracket', data=df, hue='label') 
plt.xlabel('Age Group')
plt.ylabel ('Count of Age Group')
plt.title('Total Number of Patients');


# * In terms of age groups, middle-aged adults have a higher incidence of diagnosed heart disease compared to those without a diagnosis. Among senior citizens, the proportion of individuals without a heart disease diagnosis is greater than those with a diagnosis. In the elderly population, the number of patients with a heart disease diagnosis is slightly higher than those without. For young adults, there are more individuals with a heart disease diagnosis than without.

# In[19]:


# Visualising the distribution of gender of patients whether they have a disease (represented by the 'label' column)
plt.figure(figsize = (10, 5))
sns.countplot(x='Gender', data=df, hue='label') 
plt.xlabel('Gender')
plt.ylabel ('Count of patient Gender')
plt.title('Total Number of Patients');


# * The incidence of diagnosed heart disease is relatively higher in males than in females. Additionally, the number of males without a heart disease diagnosis is also higher than the number of females without a diagnosis.

# In[20]:


# Shows the distribution of the 'label' among different categories of chest pain ('cp_cat').
plt.figure(figsize = (10, 5))
sns.countplot(x='cp_cat', data=df, hue='label') 
plt.xlabel('Types of chest pain')
plt.ylabel ('Count of patient Gender')
plt.title('Total Number of Patients');


# # Multivariate analysis

# In[21]:


# Correlation between the features in the dataset
plt.figure(figsize = (10, 10))
hm = sns.heatmap(df.corr(), cbar=True, annot=True, square=True, fmt='.2f',
annot_kws={'size': 10})


# * In general, the dataset features show weak correlations. However, a moderate positive correlation of 0.43 is observed between chest pain type and target. On the other hand, the most pronounced negative correlation of -0.58 is found between st slope and st depression.

# ## Feature Engineering / Data pre - processing

# In[22]:


# create a copy of the data (Exclude target/Label alongside other columns that was created)
df1 = df[['age','chest_pain_type','resting_blood_pressure', 'cholesterol' , 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression','st_slope', 'num_major_vessels','thalassemia']]
label = df [ ['target' ]]


# In[23]:


df1.head()


# In[24]:


label.head()


# In[25]:


# Display the data type of each column in the pandas DataFrame df1
df1.dtypes


# In[26]:


# Dealing with Outliers - 'resting_blood pressure', 'cholesterol', 'thalassemia'

# Normalize the data
scaler = MinMaxScaler()

df1 ["Scaled RBP"] = scaler.fit_transform(df1['resting_blood_pressure']. values.reshape(-1, 1))
df1 ["Scaled chol"] = scaler.fit_transform(df1 [ ['cholesterol']].values.reshape (-1, 1))
df1 ["Scaled_thal"] = scaler.fit_transform(df1 [[ 'thalassemia']].values.reshape (-1, 1))
df1 ["Scaled_max_heart_rate"] = scaler.fit_transform(df1[[ 'max_heart_rate_achieved']].values.reshape(-1, 1))
                                              
df1. head()


# In[27]:


# Removes the columns 'resting_blood_pressure', 'thalassemia', 'cholesterol', 'max_heart_rate_achieved' from the DataFrame 'df1' and updates the DataFrame in place.
df1.drop(['resting_blood_pressure', 'thalassemia', 'cholesterol', 'max_heart_rate_achieved'], axis=1, inplace=True) 
df1. head()


# ## MACHINE LEARNING

# In[28]:


# Split the dataset into training and testing sets 

X_train, X_test, y_train, y_test = train_test_split(df1, label, test_size=0.2, random_state=42)


# In[29]:


# Model Building
# Logistic Regression

logreg = LogisticRegression()

logreg.fit (X_train, y_train)

ly_pred = logreg.predict (X_test)

print("Logistic Regression")
print ("Accuracy:", accuracy_score (y_test, ly_pred))
print("Precision:", precision_score(y_test, ly_pred))
print("Recall:", recall_score (y_test, ly_pred))
print("F1-score:" , f1_score (y_test, ly_pred))
print("AUC-ROC:", roc_auc_score(y_test, ly_pred))


# * The logistic regression model achieved an accuracy of 0.87, indicating that the model correctly predicted the presence or absence of heart disease in 87% of cases. The precision score of 0.88 indicates that out of all the positive predictions made by the model, 88% were actually true positives. The recall score of 0.88 indicates that the model was able to correctly identify 88% of all positive cases of heart disease. The F1-score of 0.88 indicates a good balance between precision and recall. The AUC-ROC score of 0.87 indicates that the model is good at distinguishing between positive and negative cases of heart disease.

# In[30]:


# Create a confusion matrix
lcm = confusion_matrix(y_test, ly_pred)

# Visualize the confusion matrix
sns.heatmap(lcm, annot=True, cmap="Blues", fmt="g") 
plt.xlabel("Predicted") 
plt.ylabel ("Actual") 
plt.title("Confusion Matrix") 
plt.show()


# * True Positive (TP): The model correctly predicted 25 individuals as having a heart disease.
# * False Positive (FP): The model incorrectly predicted 4 individuals as having a heart disease when they actually did not.
# * False Negative (FN): The model incorrectly predicted 4 individuals as not having a heart disease when they actually did.
# * True Negative (TN): The model correctly predicted 28 individuals as not having a heart disease.
# * Overall, the model appears to perform reasonably well in identifying positive cases.
# 

# # RANDOM FOREST

# In[31]:


# Model Building
# Random Forest Classifier
rfc = RandomForestClassifier ()
rfc.fit(X_train, y_train) 
rfy_pred = rfc.predict (X_test)
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, rfy_pred))
print("Precision:", precision_score(y_test, rfy_pred))
print("Recall:", recall_score (y_test, rfy_pred))
print("F1-score:", f1_score (y_test, rfy_pred))
print("AUC-ROC:", roc_auc_score(y_test, rfy_pred))


# In[32]:


# Create a confusion matrix
rcm = confusion_matrix(y_test, rfy_pred)

# Visualize the confusion matrix
sns.heatmap(rcm, annot=True, cmap="Blues", fmt="g") 
plt.xlabel("Predicted") 
plt.ylabel ("Actual") 
plt.title("Confusion Matrix") 
plt.show()


# * The confusion matrix shows that out of the total 61 test cases, 24 were true positive and 26 were true negative. The model predicted 5 cases as positive which were actually negative (false positive), and 6 cases as negative which were actually positive (false negative).
# 

# In[33]:


# 8 Machine learning Algorithms will be applied to the dataset
classifiers = [[XGBClassifier(), 'XGB Classifier'],
              [RandomForestClassifier(), 'Random Forest'],
              [KNeighborsClassifier(),'K-Nearest Neighbors'],
              [SGDClassifier(), 'SGD Classifier'],
              [SVC(), 'SVC'],
              [GaussianNB(),'Naive Bayes'],
              [DecisionTreeClassifier(random_state = 42), "Decision tree"],
              [LogisticRegression(),'Logistic Regression']
              ]


# In[34]:


classifiers


# In[35]:


acc_list = {}
precision_list = {}
recall_list = {}
roc_list = {}
cm_dict = {}
f1_list = {}

for classifier in classifiers:
    model = classifier[0]
    model.fit(X_train, y_train) 
    model_name = classifier[1]
    
    pred = model.predict(X_test)
    
    a_score = accuracy_score (y_test, pred)
    p_score = precision_score(y_test, pred)
    r_score = recall_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    acc_list[model_name] = ([str(round (a_score*100, 2)) + '%'])
    precision_list[model_name] = ([str(round(p_score*100, 2)) + '%'])
    recall_list[model_name] = ([str(round(r_score*100, 2)) + '%'])
    roc_list[model_name] = ([str(round(roc_score*100, 2)) + '%'])
    f1_list[model_name] = [str(round(f1*100, 2)) + '%']  
    
    cm = confusion_matrix(y_test, pred)
    cm_dict[model_name] = cm

    if model_name != classifiers[-1][1]:
       print('')        


# In[36]:


acc_list


# In[37]:


print("Accuracy Score")
s1 = pd.DataFrame (acc_list)
s1. head()


# In[38]:


print("Precision Score")
s2 = pd.DataFrame (precision_list)
s2. head()


# In[39]:


print("Recall")
s3 = pd.DataFrame (recall_list)
s3. head()


# In[40]:


print("ROC Score")
s4 = pd.DataFrame (roc_list)
s4. head()


# In[41]:


f1_list


# In[42]:


cm_dict


# * Based on the performance metrics, the SGD Classifier model outperformed the other models in terms of accuracy, precision, recall, and ROC score. Consequently, it can be concluded that the SGD Classifier model would be the most suitable option for predicting the probability of an individual having heart disease.
