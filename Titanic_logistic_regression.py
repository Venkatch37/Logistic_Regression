#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[127]:


df=pd.read_csv("/Users/venka/OneDrive/Desktop/Data science course/titanic.csv")


# In[128]:


df


# In[129]:


df.shape


# In[130]:


df.head()


# # checking Null values in each column

# In[131]:


df.isnull().sum().sort_values(ascending=False)


# - Cabin column as more null values and it is not effecting the dataset we can drop it
# - Age is numerica we have replace null values with mean or median
# - Embarked column is categorical, replace null values with mode

# In[132]:


df.drop(["Cabin"],axis=1,inplace=True)
df


# In[133]:


df["Age"].describe()


# - Here mean and median are almost similar we can replace null values with mean or median

# In[134]:


plt.figure(figsize=(10,7))
sns.distplot(df["Age"])
plt.axvline(df["Age"].mean(),color='r')
plt.axvline(df["Age"].median(),color='b')
plt.title("Distplot")
plt.show()


# In[135]:


df['Age'].fillna(df['Age'].mean(),inplace=True)


# In[136]:


df.isnull().sum().sort_values(ascending=False)


# In[137]:


df["Embarked"].value_counts()


# In[138]:


df["Embarked"].mode()


# In[139]:


df["Embarked"].mode()[0]


# In[140]:


df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)


# In[141]:


df.isnull().sum().sort_values(ascending=False)


# # Feature Engineering
# - Reduce columns as much as possible

# In[142]:


df.head()


# - Slibsp : sibling or spouse are there or not
# - Parch : parents or child are there or not
# - 2 columns combined as family size
# - family size= slibsp+parch+1( 1 is the person itself)

# In[143]:


df["Famsize"]=df["SibSp"]+df["Parch"]+1


# In[144]:


df


# - column called genderclass here i am adding a age<15 as child

# In[145]:


df["GenderClass"]=df["Sex"]
df.loc[df["Age"]<=15,"GenderClass"]="Child"
df


# - or we can write with lamda function
# - df[GenderClass]=df.apply(lambda x: "Child" if df["Age"] <15 else df["Sex"], axis=1)

# In[146]:


df.shape


# In[147]:


sns.heatmap(df.corr(numeric_only=True))


# In[148]:


sns.heatmap(df.corr(numeric_only=True),annot=True)


# - Now drop unwanted columns 

# In[149]:


df.head()


# In[150]:


df.drop(["PassengerId","Name","Sex","SibSp","Parch","Ticket"],axis=1,inplace=True)


# In[151]:


df


# In[152]:


df.columns


# - Here Gender calss and embarked are categorical we need to convert into numerical 
# - Here we are using Dumpification

# # Dummification
# - There are two method 
#     - OneHot encoding
#     - Label Encoder
# 

# In[153]:


df=pd.get_dummies(df,columns=["GenderClass","Embarked"],drop_first=True,dtype=int)
df #one hot encoding. For dependent columns always we use one hot encoding


# # Rename Columns

# In[163]:


df.rename({"GenderClass_female":"GenderClass_Female","GenderClass_male":"GenderClass_Male"},axis=1,inplace=True)


# In[164]:


df


# In[166]:


X=df.iloc[:,df.columns!="Survived"]
X


# In[171]:


y=df["Survived"]
y.head()


#  - Or we can write
# - y=df[:,0]

# # Splitting into X and y training and test datasets

# In[172]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=750)


# In[173]:


X_train.head()


# In[174]:


X_test.head()


# In[175]:


y_train.head()


# In[176]:


y_test.head()


# In[177]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# - Here Age and Fare are Numerical columns we need standadise them

# In[182]:


num_col=["Age","Fare"]
num_col


# # standardization
# - We can standardize the numerical columns by 3 ways
#     - MinMax Scaler
#     - Standard Scaler
#     - Robust Scaler

# In[187]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
# sc=MinMaxScaler()
# sc=RobustScaler()
sc=StandardScaler()
X_train[num_col]=sc.fit_transform(X_train[num_col])
X_test[num_col]=sc.fit_transform(X_test[num_col])


# In[188]:


X_train.head()


# # Logistic Regression

# In[189]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)


# # Predictions

# In[190]:


y_pred_train=lr.predict(X_train)
y_pred_test=lr.predict(X_test)


# In[199]:


y_pred_df=pd.DataFrame({"Actual":y_test,"Expected":y_pred_test})


# In[200]:


y_pred_df


# # Accuracy

# In[202]:


from sklearn.metrics import accuracy_score
print("Training_Accuracy_Score : ",accuracy_score(y_train,y_pred_train))
print("\nTesting_Accuracy_Score : ",accuracy_score(y_test,y_pred_test))


# In[208]:


from sklearn.metrics import confusion_matrix
print("Training_confusion_matrix : ",confusion_matrix(y_train,y_pred_train))
print("\n\nTesting_confusion_matrix : ",confusion_matrix(y_test,y_pred_test))


# In[209]:


from sklearn.metrics import precision_score
print("Training_precision_Score : ",precision_score(y_train,y_pred_train))
print("\n\nTesting_precision_Score : ",precision_score(y_test,y_pred_test))


# In[210]:


from sklearn.metrics import recall_score
print("Training_recall_Score : ",recall_score(y_train,y_pred_train))
print("\n\nTesting_recall_Score : ",recall_score(y_test,y_pred_test))


# In[211]:


from sklearn.metrics import f1_score
print("Training_f1_Score : ",f1_score(y_train,y_pred_train))
print("\n\nTesting_f1_Score : ",f1_score(y_test,y_pred_test))


# In[213]:


from sklearn.metrics import classification_report
print("Training_classification_report : ",classification_report(y_train,y_pred_train))
print("\n\nTesting_classification_report : ",classification_report(y_test,y_pred_test))


# In[ ]:




