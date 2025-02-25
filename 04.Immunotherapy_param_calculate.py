#Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Libraries for data visualization
#We will use sklearn for building logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, precision_score, recall_score, accuracy_score

#Loading and describing the data
data = pd.read_csv('Immunotherapy_dataset.csv')
data = data.dropna() #Removes missing values

#Split the data into dependent (X) and independent (y) variables
y = data.pop("BOR") #Pops this variable out of the dataset and stores it y
X = data #The remaining holds the dependent variables

#Spliting the data using an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.8)

#Feature scaling
scale=StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

#Building Model again with best params
lr2 = LogisticRegression(C=0.1, class_weight= 'balanced', penalty='l1', solver = 'liblinear')
lr2.fit(X_train,y_train)
# predict probabilities on Test and take probability for class 1([:1])
y_pred_prob_test = lr2.predict_proba(X_test)[:, 1]

# Set a custom threshold
threshold = 0.54
# Predict labels based on the custom threshold
y_pred_test_custom_threshold = (y_pred_prob_test >= threshold).astype(int)

#predict labels on test dataset
#y_pred_test = lr2.predict(X_test)


print(f"Total number of samples: {len(data)}")
print(f"Unique labels in y_test: {y_test.unique()}")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# create confusion matrix
cm = confusion_matrix(y_test, y_pred_test_custom_threshold, labels=[1, 0])
group_names = ['TP', 'FP', 'FN', 'TN']
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

# Create labels for each cell of the confusion matrix
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)

# Visualize confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 5))
ax = sns.heatmap(cm, annot=labels, fmt='', cmap="YlGnBu", xticklabels=["Positive (1)", "Negative (0)"], yticklabels=["Positive (1)", "Negative (0)"])
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# ROC-AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_test) 
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=1, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random classifier')  # Diagonal line (Random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


print(f"ROC-AUC score for test dataset: {roc_auc_score(y_test, y_pred_prob_test):.4f}")

# Precision score
print(f"Precision score for test dataset: {precision_score(y_test, y_pred_test_custom_threshold):.4f}")

# Recall score
print(f"Recall score for test dataset: {recall_score(y_test, y_pred_test_custom_threshold):.4f}")

# F1 score
print(f"F1 score for test dataset: {f1_score(y_test, y_pred_test_custom_threshold):.4f}")

print(f"Accuracy for the test dataset: {accuracy_score(y_test, y_pred_test_custom_threshold):.4f}")

