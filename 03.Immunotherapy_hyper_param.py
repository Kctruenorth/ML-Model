'''
#Aim: Hyper-parameter search
#Description: To search the optimal parameters for logistic regression model
'''

#Import Libraries
import numpy as np
import pandas as pd
#Libraries for data visualization
#We will use sklearn for building logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

#Loading and describing the data
data = pd.read_csv('Immunotherapy_dataset.csv')
data = data.dropna() #Removes missing values

#Split the data into dependent (X) and independent (y) variables
y = data.pop("BOR") #Pops this variable out of the dataset and stores it y
X = data #The remaining holds the dependent variables

#Spliting the data using an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.8)

print("train size X : ",X_train.shape)
print("train size y : ",y_train.shape)
print("test size X : ",X_test.shape)
print("test size y : ",y_test.shape)

#Feature scaling

scale=StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

#check for distribution of labels
print(y_train.value_counts(normalize=True))

#Train the model
lr_basemodel = LogisticRegression(class_weight='balanced', solver= 'liblinear')
lr_basemodel.fit(X_train, y_train)
y_pred_basemodel = lr_basemodel.predict(X_test)

#Model Evaluation
roc_auc_basemodel = roc_auc_score(y_test, y_pred_basemodel)
print("ROC AUC score for base model is:", roc_auc_basemodel)

#Hyperparamter tuning
lr = LogisticRegression(max_iter=100, solver= 'liblinear', class_weight = 'balanced')

param = {
    'C': [0.01, 0.1, 0.5, 0.7, 0.75, 1, 5, 10, 100],
    'penalty': ['l2', 'l1']
}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
#Gridsearch for hyperparam tuning
model= GridSearchCV(estimator= lr,param_grid=param,scoring="roc_auc",cv=folds,return_train_score=True)
#train model to learn relationships between x and y

print("Starting hyperparmeter tuning")
model.fit(X_train,y_train)
# print best hyperparameters
print("Best roc_auc score: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)

