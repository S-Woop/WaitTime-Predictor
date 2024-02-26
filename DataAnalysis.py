import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np

# Load the dataset
data = pd.read_csv('waitlist_data.csv')

# Preprocess the data
data['Holiday'] = data['Holiday'].astype(int)
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'])
data['DateTime'] = data['Date'] + data['Time'].dt.time

# Univariate Analysis: Plot a histogram of 'SeatingTime'
plt.hist(data['SeatingTime'], bins=30)
plt.xlabel('Seating Time')
plt.ylabel('Frequency')
plt.title('Distribution of Seating Time')
plt.show()

# Bivariate Analysis: Plot a boxplot of 'SeatingTime' for each 'Holiday' category
data.boxplot(column='SeatingTime', by='Holiday')
plt.title('Seating Time by Holiday')
plt.show()

# Multivariate Analysis: Create a pairplot of all variables
sns.pairplot(data[['Holiday', 'DateTime', 'Number', 'SeatingTime']])
plt.show()

# Visualizing the Data: Plot 'SeatingTime' over 'DateTime'
plt.plot_date(data['DateTime'], data['SeatingTime'])
plt.xlabel('DateTime')
plt.ylabel('Seating Time')
plt.title('Seating Time over DateTime')
plt.show()

# Correlation Analysis: Plot heatmap of correlation matrix
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Univariate Analysis: Plot a histogram of 'Number'
plt.hist(data['Number'], bins=30)
plt.xlabel('Number of People')
plt.ylabel('Frequency')
plt.title('Distribution of Number of People')
plt.show()

# Univariate Analysis: Plot a bar chart of 'Holiday'
data['Holiday'].value_counts().plot(kind='bar')
plt.xlabel('Holiday')
plt.ylabel('Count')
plt.title('Distribution of Holiday')
plt.show()

# Bivariate Analysis: Scatter plot of 'Number' vs 'SeatingTime'
plt.scatter(data['Number'], data['SeatingTime'])
plt.xlabel('Number of People')
plt.ylabel('Seating Time')
plt.title('Seating Time vs Number of People')
plt.show()

# Feature Engineering: Create a new feature 'Hour' from 'DateTime'
data['Hour'] = data['DateTime'].dt.hour

# Univariate Analysis: Plot a histogram of 'Hour'
plt.hist(data['Hour'], bins=24)
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.title('Distribution of Hour of the Day')
plt.show()

# Bivariate Analysis: Box plot of 'SeatingTime' for each 'Hour' category
data.boxplot(column='SeatingTime', by='Hour')
plt.title('Seating Time by Hour of the Day')
plt.show()

# Anomaly Detection: Detect outliers in 'SeatingTime'
def detect_outliers(data):
    outliers = data[(np.abs(zscore(data)) > 3).all(axis=1)]
    return outliers

outliers = detect_outliers(data['SeatingTime'])
print('Outliers in Seating Time:', outliers)