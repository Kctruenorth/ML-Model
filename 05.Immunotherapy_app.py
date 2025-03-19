import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE
import time
import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPE = ['https://www.googleapis.com/auth/spreadsheets']

SERVICE_ACCOUNT_FILE = "streamlit_key.json"

credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes = SCOPE)

service = build("sheets", "v4", credentials = credentials)

sheet = service.spreadsheets()

sheet_id = "1vqfxPaw2Ke_pRkGB-VugpPwnej-WlMt6XMRGRLxRV88"

# Read CSV file
data = pd.read_csv('Immunotherapy_dataset.csv')
data = data.dropna()  # Remove missing values

# Split the data into dependent and independent variables
y = data.pop("BOR")  # Pops this variable out of the dataset and stores it as independent
X = data  # The remaining holds the dependent variables

# Split the data using an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

# Feature scaling
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# Oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Build Logistic Regression model
lr2 = LogisticRegression(C=0.1, penalty='l1', solver='liblinear')
lr2.fit(X_train_resampled, y_train_resampled)

# Ideal threshold determined by model
threshold = 0.4163

# Function to get user input
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

#Function to get user response rates
def get_user_response():
    user_response = {}
    user_response['Age'] = st.number_input("What is your Age?", 3, 94)
    user_response['Albumin'] = st.slider("What is your albumin (g/dL)?", 1.6, 5.3)
    user_response['PD-L1_score'] = st.slider("What is your PD-L1 score?", 0, 100)
    user_response['dNLR'] = st.slider("What is your Neutrophil-Lymphocyte Ratio (NLR)?", 0.0, 40.0)
    user_response['FGA'] = st.slider("What is your Fraction Genome Altered (FGA)?", 0.000, 1.000)
    user_response['MSI'] = st.slider("What is your Microsatelite Instability (MSI)?", 0.00, 30.00)
    user_response['Pack-year'] = st.slider("What is your Pack-year?", 0.0, 120.0)
    user_response['TMB'] = st.slider("What is your Tumor Mutational Burden (TMB)?", 0.0, 60.0)
    user_response['BOR'] = st.selectbox("What was your BOR (0 for non-responder, 1 for responder)", [0, 1])

   
    return user_response
# Function to append data to Google Sheets
def append_to_google_sheet(user_data):
    values = [
        [
            user_data['Age'],
            user_data['Albumin'],
            user_data['PD-L1_score'],
            user_data['dNLR'],
            user_data['FGA'],
            user_data['MSI'],
            user_data['Pack-year'],
            user_data['TMB'],
            user_data['BOR']
        ]
    ]
    
    body = {
        'values': values
    }
    
    # Append data to the next available row in the sheet
    result = sheet.values().append(
        spreadsheetId=sheet_id,
        range="Immunotherapy_dataset!A:I",  # Searches for the next available row (column A)
        valueInputOption="RAW", 
        body=body
    ).execute()

    return result

# CSS code
st.markdown("""
    <style>
        .stButton button {
            background-color: #2a7f62;
            color: white;
            border-radius: 10px;
            padding: 12px;
            font-size: 18px;
        }
        .stButton button:hover {
            background-color: white;
            color: black;
        }
       
        .stButton button:active {
            background-color: #2a7f62;
            color: white
        }
       
        .stSlider>div>div>input {
            border-radius: 5px;
        }
        .stMarkdown h3 {
            font-size: 20px;
            color: #2a7f62;
        }
        .stMarkdown p {
            font-size: 18px;
            line-height: 1.5;
            color: #333;
        }
        .stImage {
            border-radius: 10px;
        }
        
        .stColumn {
            background-color: #f4f4f4;
            border-radius: 10px;
            padding: 20px;
        } 
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size: 40px; color: white; background-color: #2a7f62; padding: 20px; border-radius: 10px; margin-bottom: 10px;'>Immunotherapy Response Predictor</h1>", unsafe_allow_html=True)

# Sidebar for navigation
page = st.sidebar.radio("Select a page", ["Predictor", "Dataset", "Contribute"])

# Display different content based on the page selected
if page == "Predictor":
    # Split into columns for input and results
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='font-size: 20px;'>Enter your information below</h3>", unsafe_allow_html=True)
        user_input = get_user_input()

    with col2:
        st.markdown("<h3 style='font-size: 24px;'>Your Results</h3>", unsafe_allow_html=True)
        if st.button("Calculate"):
            with st.spinner('Calculating...'):
                time.sleep(2)  # Processing time
                # Convert the user input into a pandas DataFrame
                user_input_df = pd.DataFrame([user_input])

                # Scale user's input
                user_input_scaled = scale.transform(user_input_df)

                # Predict probability of response 
                probability_bor_1 = lr2.predict_proba(user_input_scaled)[:, 1]
               
                # Set the color based on response
                if probability_bor_1[0] >= threshold:
                    color = "#0d4a40"  # Green for responder
                    image = "responder.jpeg"
                    caption = "Example of an objective response towards ICI"
                else:
                    color = "#ab270f"  # Red for non-responder
                    image = "non_responder.gif"
                    caption = "Example of a non-response towards ICI"

                st.markdown(f"<h2 style='font-size: 36px; color: {color};'>{probability_bor_1[0] * 100:.2f}%</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 16px;'>This result is how likely you are to respond to PD-1/PD-L1 immune checkpoint inhibitors. This means that out of 100 NSCLC patients with similar characteristics, approximately {probability_bor_1[0] * 100:.0f} will show an objective response.</p>", unsafe_allow_html=True)
                st.image(image, caption = caption, use_container_width=True)
                st.image("Image1.jpeg", caption="Immunotherapy Benefits", use_container_width=True)

elif page == "Dataset":
    st.markdown("<h3 style='font-size: 24px;'>Immunotherapy Dataset</h3>", unsafe_allow_html=True)
    st.write("Here is the dataset used for training the model:")

    # Display the dataset
    st.dataframe(data)
    st.markdown("<p style='font-size: 16px;'>You can also access the dataset in Google Sheets here:</p>", unsafe_allow_html=True)
    st.markdown("<a href='https://docs.google.com/spreadsheets/d/1vqfxPaw2Ke_pRkGB-VugpPwnej-WlMt6XMRGRLxRV88/edit?gid=1209406464#gid=1209406464' target='_blank'>Click here to open the dataset in Google Sheets</a>", unsafe_allow_html=True)

elif page == "Contribute":
    st.markdown("<h3 style='font-size: 24px;'>Contribute to the database</h3>", unsafe_allow_html=True)
    st.write("By entering your clinical features and results, it helps update the model with new data")
    user_response = get_user_response()
    if st.button("Submit Data"):
        with st.spinner('Submitting data...'):
            time.sleep(2) 
            # Append data to Google Sheets
            append_result = append_to_google_sheet(user_response)
            st.success("Your data has been successfully submitted to the database!")
    
    

    
    


