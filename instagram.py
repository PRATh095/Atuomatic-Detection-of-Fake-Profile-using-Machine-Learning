import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
instadata=pd.read_csv('F:instagram_dataset.csv')
print(len(instadata))
#View the data using head function which returns top rows
print(instadata.head())
print(instadata.index)
print(instadata.columns)
print(instadata.info)
print(instadata.dtypes)
print(instadata.describe())
# **Data Analysis**
#Import Seaborn for visually analysing the data
#countplot of private vs not private
sns.countplot(x='is_private',data=instadata)
plt.show()
#Private Vs has channel
sns.countplot(x='is_private',data=instadata,hue='has_channel')
# Create histogram
instadata.hist()
plt.show()
# Correlation plot
plt.figure(figsize=(20, 10))
cm = instadata.corr()
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)
plt.show()
#Note: We will use displot method to see the histogram. However some records does not have age hence the method will throw an error. In order to avoid that we will use dropna method to eliminate null values from graph
#Check for null
print(instadata.isna())
#Check how many values are null
print(instadata.isna().sum())
#Visualize null values
sns.heatmap(instadata.isna())
plt.show()
#find the % of null values in is_Private column
print((instadata['is_private'].isna().sum()/len(instadata['is_private']))*100)
#find the % of null values in is_fake column
print((instadata['is_fake'].isna().sum()/len(instadata['is_fake']))*100)
#find the distribution for the is_fake column
sns.displot(x='is_fake',data=instadata)
plt.show()
#**Data Cleaning**
#Fill the missing values we will fill the missing values for is_private. In order to fill missing values we use fillna method. For now we will fill the missing is_private taking average of all age
#fill is_private column
print(instadata['is_private'].isna().sum())
#visualize null values
sns.heatmap(instadata.isna())
plt.show()
#Preaparing Data for Model
#No we will require to convert all non-numerical columns to numeric. Please note this is required for feeding data into model. Lets see which columns are non numeric info describe method
#Check for the non-numeric column
print(instadata.info())
print(instadata.dtypes)
#drop the columns which are not required - non numeric
print(instadata.head())
#separate dependent and independent variables
x=instadata[['edge_followed_by', 'edge_follow', 'username_length', 'username_has_number', 'full_name_has_number', 'full_name_length', 'is_private','is_joined_recently', 'has_channel', 'is_business_account', 'has_guides', 'has_external_url', 'is_fake']]
y=instadata['is_fake']
print(y)
#** Data Modelling**
#** Logestic Regression**
#import train test split method
from sklearn.model_selection import train_test_split
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#import Logistic  Regression
from sklearn.linear_model import LogisticRegression
#Fit  Logistic Regression 
lr=LogisticRegression()
lr.fit(x_train,y_train)
#predict
predict=lr.predict(x_test)
## Testing
#print confusion matrix 
from sklearn.metrics import confusion_matrix
datafr=pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])
print(datafr)
# Plot the confusion matrix
clasre=confusion_matrix(y_test,predict)
sns.heatmap(clasre, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
#import classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))
# Evaluate the model on the testing set
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predict)
print('Accuracy:', accuracy)
#**Precision is fine considering Model Selected and Available Data. Accuracy can be increased by further using more features (which we dropped earlier) and/or  by using other model**
#Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations <br>
#Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class
#F1 score - F1 Score is the weighted average of Precision and Recall.
#separate dependent and independent variables
x=instadata[['edge_followed_by', 'edge_follow', 'username_length', 'username_has_number', 'full_name_has_number', 'full_name_length', 'is_private','is_joined_recently', 'has_channel', 'is_business_account', 'has_guides', 'has_external_url']]
y=instadata['is_joined_recently']
#** **Random Forest Algorithm**
# train and test the model using train_test_split method
from sklearn.model_selection import train_test_split
## Define the features and target variables
X = instadata.drop('is_fake', axis=1)
y = instadata['is_fake']
# train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# shape of train and test
print(x_train.shape)
print(x_test.shape)
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
# Train the random forest model with the parameters n_estimators,max_depth,random_state
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(x_train, y_train)
#prediction of testing data
pred=model.predict(x_test)
print("pred",pred)
print("y_test",y_test)
#**# Testing**
#print confusion matrix 
from sklearn.metrics import confusion_matrix
dataf=pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])
print(dataf)
# Plot the confusion matrix
conf_mat=confusion_matrix(y_test,predict)
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
#import classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
# Evaluate the model on the testing set
accuracy = accuracy_score(y_test, pred)
print('Accuracy:', accuracy)
#**Precision is fine considering Model Selected and Available Data. Accuracy can be increased by further using more features (which we dropped earlier) and/or  by using other model**
#Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations <br>
#Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class
#F1 score - F1 Score is the weighted average of Precision and Recall.
#separate dependent and independent variables
#****Support Vector Machine**
# train and test the model using train_test_split method
from sklearn.model_selection import train_test_split
# Split the dataset into features and target variable
x=instadata[['edge_followed_by', 'edge_follow','full_name_has_number', 'full_name_length', 'is_joined_recently', 'is_business_account','has_external_url', 'is_fake']]
y=instadata['is_private']
print(y)
# train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#Support Vector Machine
from sklearn import svm
#train ths Support Vector classifier
clf = svm.SVC(kernel='linear', C=1)
# Fit the classifier to the training data
clf.fit(x_train, y_train)
## Make predictions on the testing data
pred=clf.predict(x_test)
#**# Testing
#print confusion matrix 
from sklearn.metrics import confusion_matrix
data=pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])
print(data)
# Plot the confusion matrix
conf_mat=confusion_matrix(y_test,predict)
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
#import classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
# Evaluate the model on the testing set
accuracy = accuracy_score(y_test, pred)
print('Accuracy:', accuracy)
#**Precision is fine considering Model Selected and Available Data. Accuracy can be increased by further using more features (which we dropped earlier) and/or  by using other model**
#Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations <br>
#Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class
#F1 score - F1 Score is the weighted average of Precision and Recall.

# In my project  compare three model, the logistics Regression has higher accuracy

