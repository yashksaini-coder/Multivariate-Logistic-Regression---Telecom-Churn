#!/usr/bin/env python
# coding: utf-8

# ## Telecom Churn Case Study
# With 21 predictor variables we need to predict whether a particular customer will switch to another telecom provider or not. In telecom terminology, this is referred to as churning and not churning, respectively.

# ### Step 1: Importing and Merging Data

# In[7]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[8]:


# Importing Pandas and NumPy
import pandas as pd, numpy as np


# In[169]:


data_details = pd.read_csv("Datasets/Dictionary.csv",encoding='latin-1')
data_details


# In[9]:


# Importing all datasets
churn_data = pd.read_csv("Datasets/churn_data.csv")
churn_data.head()


# In[10]:


customer_data = pd.read_csv("Datasets/customer_data.csv")
customer_data.head()


# In[11]:


internet_data = pd.read_csv("Datasets/internet_data.csv")
internet_data.head()


# #### Combining all data files into one consolidated dataframe

# In[12]:


# Merging on 'customerID'
df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')


# In[13]:


# Final dataframe with all predictor variables
telecom = pd.merge(df_1, internet_data, how='inner', on='customerID')


# ### Step 2: Inspecting the Dataframe

# In[14]:


# Let's see the head of our master dataset
telecom.head()


# In[15]:


# Let's check the dimensions of the dataframe
telecom.shape


# In[16]:


# let's look at the statistical aspects of the dataframe
telecom.describe()


# In[17]:


# Let's see the type of each column
telecom.info()


# ### Step 3: Data Preparation

# #### Converting some binary variables (Yes/No) to 0/1

# In[18]:


# List of variables to map

varlist =  ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
telecom[varlist] = telecom[varlist].apply(binary_map)


# In[19]:


telecom.head()


# #### For categorical variables with multiple levels, create dummy features (one-hot encoded)

# In[20]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(telecom[['Contract', 'PaymentMethod', 'gender', 'InternetService']], drop_first=True)

# Adding the results to the master dataframe
telecom = pd.concat([telecom, dummy1], axis=1)


# In[21]:


telecom.head()


# In[22]:


# Creating dummy variables for the remaining categorical variables and dropping the level with big names.

# Creating dummy variables for the variable 'MultipleLines'
ml = pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')
# Dropping MultipleLines_No phone service column
ml1 = ml.drop(['MultipleLines_No phone service'], axis=1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,ml1], axis=1)


# In[23]:


# Creating dummy variables for the variable 'OnlineSecurity'.
os = pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')
os1 = os.drop(['OnlineSecurity_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,os1], axis=1)


# In[24]:


# Creating dummy variables for the variable 'OnlineBackup'.
ob = pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')
ob1 = ob.drop(['OnlineBackup_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ob1], axis=1)


# In[25]:


# Creating dummy variables for the variable 'DeviceProtection'. 
dp = pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,dp1], axis=1)


# In[26]:


# Creating dummy variables for the variable 'TechSupport'. 
ts = pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ts1], axis=1)


# In[27]:


# Creating dummy variables for the variable 'StreamingTV'.
st =pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,st1], axis=1)


# In[28]:


# Creating dummy variables for the variable 'StreamingMovies'. 
sm = pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,sm1], axis=1)


# In[29]:


telecom.head()


# ## Dropping the repeated variables

# In[30]:


# We have created dummies for the below variables, so we can drop them
telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies'], axis=1)


# In[31]:


# Assuming 'TotalCharges' column contains strings that represent numeric values
telecom['TotalCharges'] = pd.to_numeric(telecom['TotalCharges'], errors='coerce')

# 'coerce' option will replace any parsing errors with NaN values


# In[32]:


telecom.info()


# Now you can see that you have all variables as numeric.

# ## Checking for Outliers

# In[33]:


# Checking for outliers in the continuous variables
num_telecom = telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]


# In[34]:


num_telecom.head()


# In[35]:


num_telecom.describe()


# ## ***Plotting boxplots for the variable***
# - ## **MonthlyCharges**
# - ## **TotalChagres**

# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(data=telecom,x=telecom['MonthlyCharges'])
plt.title('Boxplot for Selected Columns')
plt.ylabel('Values')
plt.xlabel('Columns')
plt.show()


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(data=telecom,x=telecom['TotalCharges'])
plt.title('Boxplot for Selected Columns')
plt.ylabel('Values')
plt.xlabel('Columns')
plt.show()


# In[38]:


# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
num_telecom.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# From the distribution shown above, you can see that there no outliers in your data. The numbers are gradually increasing.

# ## Checking for Missing Values and Imputing Them

# In[39]:


# Adding up the missing values (column-wise)
telecom.isnull().sum()


# It means that 11/7043 = 0.001561834 i.e 0.1%, best is to remove these observations from the analysis

# In[40]:


# Checking the percentage of missing values
round(100*(telecom.isnull().sum()/len(telecom.index)), 2)


# In[41]:


# Removing NaN TotalCharges rows
telecom = telecom[~np.isnan(telecom['TotalCharges'])]


# In[42]:


# Checking percentage of missing values after removing the missing values
round(100*(telecom.isnull().sum()/len(telecom.index)), 2)


# Now we don't have any missing values

# ## Step 4: Test-Train Split

# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


# Putting feature variable to X
X = telecom.drop(['Churn','customerID'], axis=1)

X.head()


# In[45]:


# Putting response variable to y
y = telecom['Churn']

y.head()


# In[46]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ## Step 5: Feature Scaling

# In[47]:


from sklearn.preprocessing import StandardScaler


# In[48]:


scaler = StandardScaler()

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()


# In[49]:


### Checking the Churn Rate
churn = (sum(telecom['Churn'])/len(telecom['Churn'].index))*100
churn


# We have almost 27% churn rate

# ## Step 6: Looking at Correlations

# In[50]:


# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[51]:


telecom.info()


# In[52]:


telecom.columns


# In[53]:


telecom['customerID']


# ## - **Dropping the ***customerID*** column from the telecom dataset.**
# ## - **and creating the ***tele_corr*** variableand creating the correlation table**

# In[54]:


tele_corr = telecom.drop('customerID',axis=1,inplace=True)


# In[55]:


tele_corr


# In[56]:


telecom_corr = telecom.corr()
telecom_corr


# In[57]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(telecom_corr,annot = True)
plt.show()


# ### Dropping highly correlated dummy variables

# In[58]:


X_test = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                       'StreamingTV_No','StreamingMovies_No'], axis=1)
X_train = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                         'StreamingTV_No','StreamingMovies_No'], axis=1)


# #### Checking the Correlation Matrix

# After dropping highly correlated variables now let's check the correlation matrix again.

# In[59]:


plt.figure(figsize = (20,10))
sns.heatmap(X_train.corr(),annot = True)
plt.show()


# ## Step 7: Model Building
# Let's start by splitting our data into a training set and a test set.

# #### Running Your First Training Model

# In[60]:


import statsmodels.api as sm


# In[61]:


y_train.info()


# ## **Here i have passed the ***X_train.astype(int)*** as I was getting the error assocaited with the**
# ## ***Panda data cast to numpy dtypoe of onject*** 

# In[62]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train.astype(int))), family = sm.families.Binomial())
logm1.fit().summary()


# ## Step 8: Feature Selection Using RFE

# In[63]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[64]:


from sklearn.feature_selection import RFE

# Assuming you have already initialized your logistic regression model (logreg),
# and X_train, y_train are your training data and labels respectively.

# Create RFE object without specifying the number of features initially
rfe = RFE(estimator=logreg, n_features_to_select=13)  # Choose 13 features OR Running RFE with 13 variables as output

# Fit RFE to your training data
rfe.fit(X_train, y_train)


# In[65]:


rfe.support_


# In[66]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[67]:


col = X_train.columns[rfe.support_]


# In[68]:


X_train.columns[~rfe.support_]


# ### Assessing the model with StatsModels

# In[69]:


X_train_sm = sm.add_constant(X_train[col].astype(int))
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[70]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[71]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# ### Creating a dataframe with the actual churn flag and the predicted probabilities

# In[72]:


y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})
y_train_pred_final['CustID'] = y_train.index
y_train_pred_final.head()


# ### Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

# In[73]:


y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[74]:


from sklearn import metrics


# In[75]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
print(confusion)


# In[76]:


# Predicted     not_churn    churn
# Actual
# not_churn        3270      365
# churn            579       708  


# In[77]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# ### Checking VIFs

# In[78]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ## **Here I have done the same type conversion**

# In[79]:


X_train[col] = X_train[col].astype(int)


# In[80]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# There are a few variables with high VIF. It's best to drop these variables as they aren't helping much with prediction and unnecessarily making the model complex. The variable 'PhoneService' has the highest VIF. So let's start by dropping that.

# In[81]:


# col = col.drop('PhoneService', 1)
col


# In[82]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[83]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[84]:


y_train_pred[:10]


# In[85]:


y_train_pred_final['Churn_Prob'] = y_train_pred


# In[86]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[87]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# So overall the accuracy hasn't dropped much.

# ### Let's check the VIFs again

# In[88]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[89]:


# Let's drop TotalCharges since it has a high VIF
col = col.drop('TotalCharges')
col


# In[90]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[91]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[92]:


y_train_pred[:10]


# In[93]:


y_train_pred_final['Churn_Prob'] = y_train_pred


# In[94]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[95]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# The accuracy is still practically the same.

# ### Let's now check the VIFs again

# In[96]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# All variables have a good value of VIF. So we need not drop any more variables and we can proceed with making predictions using this model only

# In[97]:


# Let's take a look at the confusion matrix again 
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
confusion


# In[98]:


# Actual/Predicted     not_churn    churn
        # not_churn        3269      366
        # churn            595       692  


# In[99]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted)


# ## Metrics beyond simply accuracy

# In[100]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[101]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[102]:


# Let us calculate specificity
TN / float(TN+FP)


# In[103]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[104]:


# positive predictive value 
print (TP / float(TP+FP))


# In[105]:


# Negative predictive value
print (TN / float(TN+ FN))


# ## Step 9: Plotting the ROC Curve

# An ROC curve demonstrates several things:
# 
# - It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# - The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# - The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[106]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[107]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Churn, y_train_pred_final.Churn_Prob, drop_intermediate = False )


# In[108]:


draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# ### Step 10: Finding Optimal Cutoff Point

# Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[109]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[110]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[111]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# #### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

# In[112]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[113]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.final_predicted)


# In[114]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.final_predicted )
confusion2


# In[115]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[116]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[117]:


# Let us calculate specificity
TN / float(TN+FP)


# In[118]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[119]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[120]:


# Negative predictive value
print (TN / float(TN+ FN))


# ## Precision and Recall

# In[121]:


#Looking at the confusion matrix again


# In[122]:


confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
confusion


# ##### Precision
# TP / TP + FP

# In[123]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# ##### Recall
# TP / TP + FN

# In[124]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# Using sklearn utilities for the same

# In[125]:


from sklearn.metrics import precision_score, recall_score


# In[165]:


# ?precision_score


# In[127]:


precision_score(y_train_pred_final.Churn, y_train_pred_final.predicted)


# In[128]:


recall_score(y_train_pred_final.Churn, y_train_pred_final.predicted)


# ### Precision and recall tradeoff

# In[129]:


from sklearn.metrics import precision_recall_curve


# In[130]:


y_train_pred_final['Churn']


# In[131]:


y_train_pred_final['predicted']


# In[132]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# In[133]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ## Step 11: Making predictions on the test set

# In[134]:


X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])


# In[135]:


X_test = X_test[col]
X_test.head()


# In[136]:


X_test_sm = sm.add_constant(X_test)


# Making predictions on the test set

# In[137]:


X_test_sm.info()


# ## **Here I have created the ***float_columns*** having the float type**

# In[138]:


float_columns = X_test_sm.select_dtypes(include=['float64'])


# In[139]:


float_columns.info()


# In[140]:


# X_test_sm[float_columns] = X_test_sm[float_columns].astype(int)
# running this code will throw an error


# ## **Here is the error I encountered while runnign this code**
# ![X_test_sm Error.png](<attachment:X_test_sm Error.png>)

# ## **Running the code for a sample_df and predicting the ***y_test_pred*** usng the ***res*** model** 

# In[141]:


sample_df = X_test_sm.astype(int)


# In[142]:


sample_df.info()


# In[143]:


sample_df_pred = res.predict(sample_df)


# ## **Converting the X_test_sm type into ***integer*****

# In[144]:


X_test_sm = X_test_sm.astype(int)


# In[145]:


X_test_sm.info()


# In[146]:


y_test_pred = res.predict(X_test_sm)


# In[147]:


y_test_pred[:10]


# In[148]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[149]:


# Let's see the head
y_pred_1.head()


# In[150]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[151]:


# Putting CustID to index
y_test_df['CustID'] = y_test_df.index


# In[152]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[153]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[154]:


y_pred_final.head()


# In[155]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Churn_Prob'})


# ## **Here changing the ***Reindex_axis*** method to ***reindex*** as the previous got outdated**

# In[156]:


# Rearranging the columns
y_pred_final = y_pred_final.reindex(['CustID','Churn','Churn_Prob'], axis=1)


# In[157]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[158]:


y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[159]:


y_pred_final.head()


# In[160]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)


# In[161]:


confusion2 = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted )
confusion2


# In[162]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[163]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[164]:


# Let us calculate specificity
TN / float(TN+FP)

