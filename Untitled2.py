
# coding: utf-8

# In[251]:


import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 


# In[252]:


from sklearn.neighbors import KNeighborsClassifier


# In[253]:


from sklearn.svm import SVC
from sklearn.linear_model import Perceptron


# In[254]:


from sklearn.tree import DecisionTreeClassifier


# In[255]:


from sklearn.ensemble import RandomForestClassifier


# In[256]:


train_df = pd.read_csv("D:/ML/titanic/train.csv")


# In[257]:


test_df = pd.read_csv("D:/ML/titanic/test.csv")


# In[258]:


train_df.head(5)


# In[259]:


test_df.head(5)


# In[260]:


print(train_df.isnull().sum())


# In[261]:


sns.heatmap(train_df.isnull(), annot  = False,yticklabels  = False, cbar = True,cmap = 'winter' )


# In[262]:


train_df.drop(labels='Cabin', inplace = True, axis = 1)


# In[263]:


sns.heatmap(train_df.isnull(), annot = False, cmap='winter', cbar = True)


# In[264]:


print(test_df.isnull().sum())


# In[265]:


test_df.drop(labels = 'Cabin', inplace=True, axis = 1)


# In[266]:


sns.heatmap(test_df.isnull(), annot = False, cmap = 'winter', cbar = True, yticklabels=False)


# In[267]:


print(test_df.isnull().sum())


# In[268]:


train_df


# In[269]:


test_df


# In[270]:


train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())


# In[271]:


print(train_df)


# In[272]:


test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())


# In[273]:


test_df


# In[274]:


sns.heatmap(train_df.isnull(), annot=False, yticklabels=False, cbar=True, cmap='winter')


# In[275]:


plt.figure(figsize=(10,6))
train_df['Age'].head(10).hist(alpha = 0.5, bins = 50)
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()


# In[276]:


plt.figure(figsize=(15,8))
plt.scatter(train_df.Survived, train_df.Age, color='green')
plt.ylabel('age')


# In[277]:


x = train_df.drop(['PassengerId','Survived','Name','Ticket','Fare','Embarked'], axis = 1).values
x


# In[278]:


y = train_df["Survived"].values
y


# In[279]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x[:,1] = encoder.fit_transform(x[:,1])
x


# In[280]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state =0)


# In[281]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
x_train, x_test


# In[282]:


logistic_regressor = LogisticRegression(random_state = 0)


# In[283]:


logistic_regressor.fit(x_train, y_train)


# In[284]:


a = logistic_regressor.predict(x_test)


# In[285]:


a


# In[286]:


x_test.shape, y_test.shape,y_train.shape


# In[287]:


logistic_accuracy = logistic_regressor.score(x_test, y_test)


# In[288]:


logistic_accuracy


# In[294]:


test_df.head()


# In[296]:


test_df1 = test_df
test_df1.head(5)

