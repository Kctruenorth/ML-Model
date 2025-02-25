'''
#Aim: Correlation between features
#Description: Displays a heatmap showing correlation between continuous features
'''

#Import Libraries
import numpy as np
import pandas as pd
#Libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
#We will use sklearn for building logistic regression model
from sklearn.linear_model import LogisticRegression

#Loading and describing the data
data = pd.read_csv('Immunotherapy_dataset.csv')
data = data.dropna() #Removes missing values 

# Compute correlation with only BOR and sort by absolute value (most important first)
corr_with_bor = data.corr(method='spearman')[["BOR"]].dropna()  # Drop NaN values if any
corr_with_bor["abs_corr"] = corr_with_bor["BOR"].abs()  # Get absolute correlation for sorting
corr_with_bor = corr_with_bor.sort_values(by="abs_corr", ascending=False).drop(columns=["abs_corr"])  # Sort and remove helper column

palette_length = 400 #The number of colors
my_color = sns.color_palette ("RdBu_r", n_colors = palette_length)

# Plot heatmap
plt.figure(figsize=(4, 6))
sns.heatmap(corr_with_bor, annot=True, cmap=my_color, center=0, vmin=-1, vmax=1)
plt.title("Correlation with BOR")
plt.show()
