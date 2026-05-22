#!/usr/bin/env python
# coding: utf-8
### TASKS
1. Data Exploration:
a. Load the dataset and perform exploratory data analysis (EDA).
b. Examine the features, their types, and summary statistics.
c. Create visualizations such as histograms, box plots, or pair plots to visualize the distributions and relationships between features.
Analyze any patterns or correlations observed in the data.
2. Data Preprocessing:
a. Handle missing values (e.g., imputation).
b. Encode categorical variables.
3. Model Building:
a. Build a logistic regression model using appropriate libraries (e.g., scikit-learn).
b. Train the model using the training data.
4. Model Evaluation:
a. Evaluate the performance of the model on the testing data using accuracy, precision, recall, F1-score, and ROC-AUC score.
Visualize the ROC curve.
5. Interpretation:
a. Interpret the coefficients of the logistic regression model.
b. Discuss the significance of features in predicting the target variable (survival probability in this case).

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import  OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_regression


# In[2]:


df=pd.read_csv('diabetes.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# ### As per the information collected from the given dataset all the columns in the data set are of numerical type

# In[6]:


### Satistical informaion of the dataset
df.describe()


# In[7]:


### creating Histograms of the features
sns.histplot(data=df,x='Pregnancies')
plt.show()


# In[8]:


sns.histplot(data=df,x='BloodPressure',kde=True,color='blue')
plt.show()


# In[9]:


### Getting pair plot for all the data given
sns.pairplot(data=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']],diag_kind='kde')
plt.show()


# In[10]:


### the above pairplot  is combination of Histograms along with density plot
## implies that all the columns in the dataset are of continuous type and the Outcome column is discrete type
## to check whether the data is normally distribut, we need to find the skewners value of particular column
df['Pregnancies'].skew(),df['BloodPressure'].skew(),df['Glucose'].skew(),df['SkinThickness'].skew(), df['Insulin'].skew(),df['BMI'].skew(),df['DiabetesPedigreeFunction'].skew(),df['Age'].skew()


# #### as no value of the skewners is equal to 0, it is not normally distributed data

# In[11]:


### Finding Correlation and visually represnet through heatmap
df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']].cov()


# In[12]:


### finding correlation() which means bringing the values of covariance in the range of -1 to +1
corr=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']].corr()
corr


# In[13]:


sns.heatmap(data=corr,annot=True)


# In[14]:


### performing EDA
## checking for missing values
df.isnull().sum()


# ##### No missing values are present in the dataset
# #####  checking for duplicates

# In[15]:


df.duplicated().sum()


# In[16]:


## selecting target and features
target=df[['Outcome']]
features=df.drop(columns=['Outcome'])


# In[17]:


features.head()


# In[18]:


## no duplicates are present in the dataset
### checking for outliers


# In[19]:


df.boxplot()


# In[20]:


### Outliers are present in the data
### Removing the outliers by capping


# In[21]:


def outlier_cappping(df,column):
    Q1= df[column].quantile(0.25)
    Q3= df[column].quantile(0.75)
    IQR= Q3-Q1
    lower_extreme= Q1-1.5*IQR
    upper_extreme= Q3+1.5*IQR
    df[column]=df[column].apply(lambda x: lower_extreme if x<lower_extreme else upper_extreme if x>upper_extreme else x)
for col in features.select_dtypes(['int','float']).columns:
    outlier_cappping(features,col)


# In[22]:


features.boxplot()
plt.show()


# ### No outliers are present

# In[23]:


### Feature Selection
x_train,x_test,y_train,y_test=train_test_split(features,target,train_size=0.8,random_state=100)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[24]:


### As no categorical data is present in data set , no need of labelling the data and no need of using Ordinal Encoder


# In[25]:


features.columns


# In[26]:


## Using f_regrr to check the relationship between the features and target
f_reg=f_regression(features,target)
pd.Series(f_reg[0],index=features.columns).sort_values(ascending=False).plot(kind='bar')
plt.show()


# ### As the relationship with the 'SkinThickness' and 'BloodPressure' column is very less . if needed we can drop these two columns, but dropping these columns could lead to loss of some data 

# In[27]:


target= df[['Outcome']]
features=df.drop(columns=['Outcome'])
features.head()


# In[28]:


features_1=features.drop(columns=['BloodPressure','SkinThickness'])
features_1.head()


# In[29]:


### Scaling using Standard Scaler by fit and transform
std_sca=StandardScaler()
x_train[[ 'Pregnancies', 'Glucose', 'Insulin',
         'BMI', 'DiabetesPedigreeFunction', 'Age','Glucose', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]=std_sca.fit_transform(x_train[['Pregnancies', 'Glucose', 'Insulin',
         'BMI', 'DiabetesPedigreeFunction', 'Age','Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])
x_test[['Pregnancies', 'Glucose', 'Insulin',
         'BMI', 'DiabetesPedigreeFunction', 'Age','Glucose', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]=std_sca.fit_transform(x_test[['Pregnancies', 'Glucose', 'Insulin',
         'BMI', 'DiabetesPedigreeFunction', 'Age','Glucose', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']])


# In[30]:


log_model=LogisticRegression()
log_model.fit(x_train,y_train)


# In[31]:


### finding out the coefficients of the logistic regression model
log_model.coef_


# In[32]:


log_model.intercept_


# In[33]:


y_pred=log_model.predict(x_test)
y_pred


# In[34]:


#### Changing the thershold

y_pred1= [1 if x[1]>=0.6 else 0 for x in log_model.predict_proba(x_test)]
print(y_pred1)
print('Accuracy_Score:',accuracy_score(y_test,y_pred1))


# In[35]:


accuracy_score(y_test,y_pred)


# In[36]:


y_test


# In[37]:


#### sigmoid values based on which predict 0,1,0,1 is given
log_model.predict_proba(x_test)


# In[38]:


### to get only the 2nd column(sigmoid) of the above array
sigmoid=log_model.predict_proba(x_test)[:,1]
sigmoid


# In[39]:


#### logloss
from sklearn.metrics import log_loss
log_loss(y_test,sigmoid)


# ### Performance Metrics

# In[40]:


from sklearn.metrics import confusion_matrix,classification_report


# In[41]:


conf=confusion_matrix(y_test,y_pred1)
conf


# In[42]:


sns.heatmap(conf,annot=True)
plt.xlabel('y_pred1')
plt.ylabel('y_test')
plt.title('confusion_matrix')
plt.show();


# In[43]:


print(classification_report(y_test,y_pred1))


# #### Finding ROC Curve

# In[44]:


from sklearn.metrics import roc_auc_score,roc_curve


# In[45]:


auc_score=roc_auc_score(y_test,sigmoid)
auc_score


# In[46]:


### to find tpr and fpr,and plot the same
fpr,tpr,thr=roc_curve(y_test,sigmoid)

plt.plot(fpr,tpr,color='red',lw=2,label=f'AUC_SCORE:{auc_score:.2f}')
plt.plot([0,1],linestyle='--',color='grey')
plt.xlabel('FPR',color='red',size=12)
plt.ylabel('TPR',color='red',size=12)
plt.title('ROC Curve',color='green',size=14)
plt.legend()
plt.grid()
plt.show()
plt.show()


# #### 5. Interpretation:
# a. Interpret the coefficients of the logistic regression model.
# 
# b. Discuss the significance of features in predicting the target variable (survival probability in this case).

# In[47]:


# Get the feature names and their corresponding coefficients
coefficients = pd.Series(log_model.coef_[0], features.columns) 
intercept = log_model.intercept_[0]

print("Intercept:", intercept)
print("\nCoefficients:\n", coefficients)


# In[48]:


# Sort by absolute value for significance
coefficients_sorted = coefficients.abs().sort_values(ascending=False)
print(coefficients_sorted)


# #### 6. Deployment with Streamlit:
# In this task, we have to deploy our logistic regression model using Streamlit. 
# The deployment can be done locally or online via Streamlit Share. 
# Task includes creating a Streamlit app in Python that involves loading your trained model and setting up user inputs for predictions.
# 
# For online deployment, use Streamlit Community Cloud, which supports deployment from GitHub repositories.
# 

# In[51]:


import pickle

with open("logistic_model.pkl", "wb") as file:
    pickle.dump(log_model, file)


# In[52]:


import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes")

# User inputs
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 300)
blood_pressure = st.number_input("Blood Pressure", 0, 200)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have diabetes")
    else:
        st.success("✅ The person is not likely to have diabetes")


# #### Interview Questions:
# ###### 1. What is the difference between precision and recall?
# ###### Precision
# Defn: The proportion of correctly predicted positive observations to the total predicted positives.
# Formula: Precision = True Positives(TP) /True Positives (TP)+False Positives (FP)
# Interpretation: Out of all the positive labels the model predicted, how many were actually correct?
# High Precision: Few false positives.
# 
# ###### Recall (Sensitivity or True Positive Rate)
# Definition: The proportion of correctly predicted positive observations to all actual positives.
# Formula: Recall = True Positives (TP) / True Positives (TP)+False Negatives (FN)
# Interpretation: Out of all the actual positive cases, how many did the model correctly identify?
# High Recall: Few false negatives.
# 
# ##### 2.What is cross-validation, and why is it important in binary classification?
# ###### Cross-Validation
# Cross-validation is a model evaluation technique used to assess how well a machine learning model will generalize to an independent dataset.
# The idea is to split the dataset into multiple subsets, train the model on some of them, and validate it on the remaining ones — repeating this process multiple times.
# Most Common Type: K-Fold Cross-Validation
# Split the dataset into K equal parts (folds).
# For each fold:
# Use K-1 folds for training.
# Use 1 fold for testing/validation.
# Repeat K times, each time with a different fold used as the validation set.
# Average the results to get a more reliable estimate of model performance.
# ##### 3.Why is Cross-Validation Important in Binary Classification?
# Avoids Overfitting
# Training on only one split may cause the model to perform well there but poorly on unseen data.
# Cross-validation tests on multiple splits, ensuring the model is not memorizing the data.
# Gives Reliable Performance Metrics
# For binary classification, metrics like accuracy, precision, recall, F1-score, and AUC can vary across splits.
# Cross-validation gives an average, which is more stable and trustworthy.
# Handles Imbalanced Datasets
# Using Stratified K-Fold ensures each fold has a similar proportion of positive and negative classes, which is crucial in binary classification (e.g., spam vs. not spam, disease vs. no disease).
# Model Selection and Hyperparameter Tuning
# When tuning models using techniques like GridSearchCV or RandomizedSearchCV, cross-validation helps compare models fairly and robustly.

# In[ ]:




