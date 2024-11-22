#------------------------------------------------------------------------#
'''          Project 4: California Housing Price Prediction         '''
#------------------------------------------------------------------------#

# Step1: Import all libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score  # Import r2_score
import numpy as np

# Step2: Load the data

# Step2.1: Read the “housing.csv” file from the folder into the program
housingData = pd.read_csv('housing.csv')

# Step2.2: Print first few rows of this data
print('Print first few rows of this data - ')
print(housingData.head())

# Step2.3: Extract input (X) and output (y) data from the dataset
X = housingData.iloc[:, :-1].values
y = housingData.iloc[:, -1].values  # Adjust to 1D array for regression

# Step3: Handle missing values
# Fill the missing values with the mean of the respective column
imputer = SimpleImputer(strategy='mean')
X[:, :-1] = imputer.fit_transform(X[:, :-1])

# Step4: Encode categorical data
# Convert categorical column in the dataset to numerical data
labelencoder = LabelEncoder()
X[:, -1] = labelencoder.fit_transform(X[:, -1])

# Step5: Split the dataset
# 80% training dataset and 20% test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step6: Standardize data
# Standardize training and test datasets
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

####################################################################
'''Task1: Perform Linear Regression'''
####################################################################

# Task1.1: Perform Linear Regression on training data
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)

# Task1.2: Predict output for test dataset using the fitted model
predictionLinear = linearRegression.predict(X_test)

# Task1.3: Print root mean squared error (RMSE) and R² from Linear Regression
rmseLinear = np.sqrt(mean_squared_error(y_test, predictionLinear))
r2Linear = r2_score(y_test, predictionLinear)
print('Root mean squared error (RMSE) from Linear Regression =', rmseLinear)
print('R² from Linear Regression =', r2Linear)

####################################################################
'''Task2: Perform Decision Tree Regression'''
####################################################################

# Task2.1: Perform Decision Tree Regression on training data
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, y_train)

# Task2.2: Predict output for test dataset using the fitted model
predictionDT = DTregressor.predict(X_test)

# Task2.3: Print root mean squared error and R² from Decision Tree Regression
rmseDT = np.sqrt(mean_squared_error(y_test, predictionDT))
r2DT = r2_score(y_test, predictionDT)
print('Root mean squared error from Decision Tree Regression =', rmseDT)
print('R² from Decision Tree Regression =', r2DT)

####################################################################
'''Task3: Perform Random Forest Regression'''
####################################################################

# Task3.1: Perform Random Forest Regression on training data
RFregressor = RandomForestRegressor()
RFregressor.fit(X_train, y_train)

# Task3.2: Predict output for test dataset using the fitted model
predictionRF = RFregressor.predict(X_test)

# Task3.3: Print root mean squared error and R² from Random Forest Regression
rmseRF = np.sqrt(mean_squared_error(y_test, predictionRF))
r2RF = r2_score(y_test, predictionRF)
print('Root mean squared error from Random Forest Regression =', rmseRF)
print('R² from Random Forest Regression =', r2RF)

####################################################################
'''Task4: Perform K-Nearest Neighbors (KNN) Regression'''
####################################################################

# Task4.1: Perform KNN Regression on training data
KNNregressor = KNeighborsRegressor(n_neighbors=5)  # You can adjust n_neighbors as needed
KNNregressor.fit(X_train, y_train)

# Task4.2: Predict output for test dataset using the fitted model
predictionKNN = KNNregressor.predict(X_test)

# Task4.3: Print root mean squared error and R² from KNN Regression
rmseKNN = np.sqrt(mean_squared_error(y_test, predictionKNN))
r2KNN = r2_score(y_test, predictionKNN)
print('Root mean squared error from KNN Regression =', rmseKNN)
print('R² from KNN Regression =', r2KNN)

####################################################################
'''Task5: Bonus exercise: Perform Linear Regression with one independent variable'''
####################################################################

# Task5.1: Extract just the median_income column from the independent variables (from X_train and X_test)
X_train_median_income = X_train[:, [7]]
X_test_median_income = X_test[:, [7]]

# Task5.2: Perform Linear Regression to predict housing values based on median_income
linearRegression2 = LinearRegression()
linearRegression2.fit(X_train_median_income, y_train)

# Task5.3: Predict output for test dataset using the fitted model
predictionLinear2 = linearRegression2.predict(X_test_median_income)

# Task5.4: Plot the fitted model for training data as well as for test data

# Task5.4.1: Visualize the Training set
plt.scatter(X_train_median_income, y_train, color='green')
plt.plot(X_train_median_income, linearRegression2.predict(X_train_median_income), color='red')
plt.title('Training result - median_income vs median_house_value')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()

# Task5.4.2: Visualize the Testing set
plt.scatter(X_test_median_income, y_test, color='blue')
plt.plot(X_train_median_income, linearRegression2.predict(X_train_median_income), color='red')
plt.title('Testing result - median_income vs median_house_value')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()

####################################################################
'''                          End                          '''
####################################################################
