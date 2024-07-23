# Optimizing-Numerical-Weather-Prediction-Model-Performance-Using-Machine-Learning-Techniques


This project aims to improve the accuracy of Numerical Weather Prediction (NWP) models through the application of machine learning techniques, specifically Linear Regression. Using Austin Weather dataset that includes various meteorological parameters, the project focuses on predicting precipitation levels.


### Importing Libraries
First, we import several libraries that help us work with data, create machine learning models, and plot graphs:
python
import pandas as pd  # for handling data
import sklearn  # for machine learning
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for creating plots
from sklearn.linear_model import LinearRegression  # for creating a linear regression model


### Reading the Data
We read a dataset from a CSV file named "austin_final.csv":
python
data = pd.read_csv("austin_final.csv")


### Preparing the Data
We separate the data into features (X) and labels (Y):
- X contains all columns except the "PrecipitationSumInches" column.
- Y contains only the "PrecipitationSumInches" column, which is the amount of rainfall.
python
X = data.drop(['PrecipitationSumInches'], axis=1)
Y = data['PrecipitationSumInches'].values.reshape(-1, 1)  # reshaping Y to be a 2-D array


### Selecting a Random Day
We select a random day from the dataset to highlight in our plots:
python
day_index = 798  # example day index
days = [i for i in range(Y.size)]  # list of all day indices


### Training the Model
We create a linear regression model and train it using the features (X) and labels (Y):
python
clf = LinearRegression()
clf.fit(X, Y)


### Testing the Model
We test our model with a sample input to predict the amount of rainfall:
python
inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45], 
                [57], [29.68], [10], [7], [2], [0], [20], [4], [31]]) 
inp = inp.reshape(1, -1)
print('The precipitation in inches for the input is:', clf.predict(inp))

This input is a set of values representing different weather parameters for one day.

### Plotting the Results
We create two types of plots to visualize the data and the model's performance:

1. *Precipitation Trend Graph*:
   - We plot the amount of rainfall (Y) over all days.
   - We highlight one specific day (in red) to observe its rainfall:
   python
   plt.scatter(days, Y, color='g')  # all days in green
   plt.scatter(days[day_index], Y[day_index], color='r')  # one specific day in red
   plt.title("Precipitation level")
   plt.xlabel("Days")
   plt.ylabel("Precipitation in inches")
   plt.show()
   

2. *Precipitation vs Selected Attributes*:
   - We select a few columns from X to see how they relate to rainfall.
   - We plot these attributes against the number of days:
   python
   x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 
                     'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 
                     'WindAvgMPH'], axis=1)

   for i in range(x_vis.columns.size):
       plt.subplot(3, 2, i + 1)
       plt.scatter(days, x_vis[x_vis.columns.values[i]][:100], color='g')
       plt.scatter(days[day_index], x_vis[x_vis.columns.values[i]][day_index], color='r')
       plt.title(x_vis.columns.values[i])
   plt.show()
   

### Summary
- We read weather data from a CSV file.
- We separate the data into features (X) and labels (Y).
- We train a linear regression model to predict rainfall.
- We test the model with a sample input.
- We plot graphs to visualize rainfall trends and how various weather attributes relate to rainfall.

This code helps us understand how weather parameters can be used to predict rainfall using a linear regression model.
