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

#Data cleaning / Data Preprocessing

#correlation between features heatmap

#1. Obtain the correlation coefficents
corr_matrix = data.corr(method = 'spearman') #Use the spearman method to calculate correlation coefficents
corr_matrix.columns = data.columns #The columns are assigned the headers of the CSV file
corr_matrix.index = data.columns #The rows are also assigned the headers of the CSV file

#2. Create a mask to hide the upperhalf of the matrix
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True #The code so that the upperhalf is hidden

#3. Create a color palette for the matrix
palette_length = 400 #The number of colors
my_color = sns.color_palette ("RdBu_r", n_colors = palette_length) #Creates the color palette from blue to red

#4. Create the heatmap
fig, ax = plt.subplots(figsize=(3.5, 3.2)) #Figure size
plt.subplots_adjust(left=0.02, bottom=0.02, right=0.9, top=0.95)
heatmap = sns.heatmap(corr_matrix, mask = mask, cmap = my_color, center = 0,
                      vmin = -1, vmax = 1, xticklabels = True, yticklabels = True, annot = True, fmt = ".2f",
                      cbar = True, cbar_kws = {"shrink": 0.5, "label": "Spearman correlation coefficient"},
                      cbar_ax=ax.inset_axes([0.72, 0.5, 0.04, 0.5]),
                      linewidths=0.1, linecolor='white', square=True, ax=ax)
# display the column names at the diagonal
continuous_features_full = ['Age', 'Albumin', 'PD-L1 TPS','NLR', 'FGA', 'MSI', 'Pack Year', 'TMB', 'BOR']
for i in range(len(corr_matrix.columns)):
    plt.text(i + 0.5, i + 0.5, continuous_features_full[i], ha='left', va='bottom', rotation=45, fontsize=8)
# show the plot
plt.xticks([])
plt.yticks([])
plt.title("Correlation Heatmap")
plt.show()

