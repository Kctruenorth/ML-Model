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

#Create the app:

def get_user_input():
    user_input = {}
    user_input['Age'] = st.number_input("What is your Age?", 3, 94)
    user_input['Albumin'] = st.slider("What is your albumin (g/dL)?", 1.6, 5.3)
    user_input['PD-L1_score'] = st.slider("What is your PD-L1 score?", 0, 100)
    user_input['dNLR'] = st.slider("What is your Neutrophil-Lymphocyte Ratio (NLR)?", 0.0, 40.0)
    user_input['FGA'] = st.slider("What is your Fraction Genome Altered (FGA)?", 0.000, 1.000)
    user_input['MSI'] = st.slider("What is your Microsatelite Instability (MSI)?", 0.00, 30.00)
    user_input['Pack-year'] = st.slider("What is your Pack-year?", 0.0, 120.0)
    user_input['TMB'] = st.slider("What is your Tumor Mutational Burden (TMB)?", 0.0, 60.0)
    
    return user_input
    
st.title("Immunotherapy Prediction")
st.markdown("<h3 style='font-size: 20px;'>Enter your information below</h3>", unsafe_allow_html=True)
# Get user inputs
user_input = get_user_input()

if st.button ("Calculate"):
    # Convert the user input into a pandas DataFrame
    user_input_df = pd.DataFrame([user_input])

    # Scale the user's input to match the model's expected input format
    user_input_scaled = scale.transform(user_input_df)

    probability_bor_1 = lr2.predict_proba(user_input_scaled)[:, 1]
    
    st.markdown("<h3 style='font-size: 24px; color: #2a7f62;'>Your Results</h3>", unsafe_allow_html=True)
    st.markdown("### The likelihood of response to immune checkpoint blockade therapy:")
    
    st.markdown(f"<h2 style='font-size: 40px; color: #0d4a40; font-weight: bold; text-align: center;'>{probability_bor_1[0] * 100:.2f}%</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='font-size: 18px;'>This result is how likely you are to respond to PD-1/PD-L1 immune checkpoint inhibitors. This means that out of 100 NSCLC patients with similar characteristics, approximately {probability_bor_1[0] * 100:.0f} will show an objective response.</p>", unsafe_allow_html=True)
    
    st.markdown("### Check the diagram below for the benefits of immunotherapy over conventional cancer treatments.")
    
    st.image("Image1.jpeg", use_container_width=True)