'''
#Aim: Describe the data
#Description: For the data section of the poster
'''

#Import Libraries
import numpy as np
from tabulate import tabulate
import pandas as pd
#Libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
#We will use sklearn for building logistic regression model
from sklearn.linear_model import LogisticRegression

#Loading and describing the data
data = pd.read_csv('Immunotherapy_dataset.csv')
data = data.dropna() #Removes missing values

desc_data = data.describe().round(2)
print(tabulate(desc_data, headers='keys', tablefmt='pretty'))
