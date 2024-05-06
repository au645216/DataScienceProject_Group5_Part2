# Importing pandas.
import pandas as pd
# Importing matplotlib.pyplot.
import matplotlib.pyplot as plt
# Importing the ExponentialSmoothing function.
    #pip install statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Importing the ParameterGrid function.
from sklearn.model_selection import ParameterGrid
# Importing the mean squared error funciton.
from sklearn.metrics import mean_squared_error
# Importing the square root funciton.
from math import sqrt

#-----------------------------------------------------------------------------------------------------------------------------------
# Exploratory Data Analysis and Feature Engineering
#-----------------------------------------------------------------------------------------------------------------------------------
# Importing the datafile.
data = pd.read_excel('flightprices.xlsx')

# Displaying the dimensions of the data.
data.shape

# The first and last date.
min(data['Timestamp'])
max(data['Timestamp'])

# Displaying the first and last five rows of the data.
data
# Displaying the first five rows of the data.
data.head()
# Displaying the last five rows of the data.
data.tail()

# Checking if there's any missing values.
data.isna().any()
# Checking the data type of the variables.
data.dtypes
# Displaying the variable types and if any of the variabels contains missing value.
data.info()
# Checkig if there's any duplicated values.
data.duplicated()

# Removing all columns expect på Timestamp and Price.
data = data[['Price (Raw)', 'Timestamp']]
# The first and last five rows.
data
# The dimensions.
data.shape

# Summary statistics of the Price variable.
data['Price (Raw)'].describe().round(2)
# Extracting the date and time witht the highest price.
data.loc[data['Price (Raw)'].idxmax()]
# Extracting the date and time witht the highest price.
data.loc[data['Price (Raw)'].idxmin()]

# Saving the min, max, and mean value of price for plotting.
min_price = data['Price (Raw)'].min()
max_price = data['Price (Raw)'].max()
mean_price = data['Price (Raw)'].mean()
# Saving the timestamp for the mix and max price for plotting
min_price_date = data['Timestamp'][data['Price (Raw)'].idxmin()]
max_price_date = data['Timestamp'][data['Price (Raw)'].idxmax()]

# Plotting Price over time.
plt.plot(data['Timestamp'], # x-values.
         data['Price (Raw)'], # y-values.
         color = '#0603c0', # Color of line.
         zorder = 1) # Specifying that this is the first layer of the graph.
# Assigning a titel to the plot.
plt.title('Price over time',
          fontsize = 15,
          fontweight = 'bold', # Bold text.
          pad = 15)  # Space between graph and title.
# Labeling the X- and Y-axis.
plt.xlabel('Date',
           fontsize = 13,
           fontweight = 'bold', # Bold text.
           labelpad = 15)  # Space between axis and label.
plt.ylabel('Price', 
           rotation = 0, # No rotation.
           fontsize = 13,
           fontweight = 'bold', # Bold text.
           labelpad = 35)  # Space between axis and label.
# Adding a dotted line representing the mean price.
plt.axhline(mean_price, 
            label = 'Mean Price',
            color = '#3d3afb',  # Color of line.
            linestyle = '--', # Dotted line.
            zorder = 2) # Specifying that this is the second layer of the graph.
# Adding a mark for the mix and max price.
plt.scatter(min_price_date, # Where on the x-axis.
            min_price, # Where on the y-axis.
            label = 'Min Price', 
            color = '#bcbafd', # Color of mark.
            s = 100, # Size of the mark.
            zorder = 2) # Specifying that this is the second layer of the graph.
plt.scatter(max_price_date, # Where on the x-axis.
            max_price, # Where on the y-axis.
            label = 'Max Price',
            color = '#010030', # Color of mark.
            s = 100, # Size of the mark.
            zorder = 2) # Specifying that this is the second layer of the graph.
# Ensure the plot elements fit within the figure area without overlapping.
plt.tight_layout()
# Adding legend to the plot.
plt.legend(loc = 'lower left',
           fontsize = 11)
# Displayingt the plot.
plt.show()

# Create boxplots to visualize outliers in numerical variables
fig, axs = plt.subplots(1, 2)
axs[0].boxplot(data["Price (Raw)"])
axs[0].set_title("Boxplot of Price")
plt.show()

# Box-plot to identify missing values.
boxplot = plt.boxplot(data['Price (Raw)'])
# Assigning a titel to the plot.
plt.title('Box Plot of Price',
          fontsize = 15,
          fontweight = 'bold', # Bold text.
          pad = 15)  # Space between graph and title.
# Labeling the X- and Y-axis.
plt.xlabel('Price',
           fontsize = 13,
           fontweight = 'bold', # Bold text.
           labelpad = 15)  # Space between axis and label.
plt.ylabel('Distribution', 
           rotation = 0, # No rotation.
           fontsize = 13,
           fontweight = 'bold', # Bold text.
           labelpad = 35)  # Space between axis and label.
# Accessing the median line and setting the color so it matches the repport.
median_line = boxplot['medians'][0]
median_line.set_color('#0603c0')  # Color of median line.
plt.show()

# Calculating and checking for outliers in Prce.
q1 = data['Price (Raw)'].quantile(0.25)
q3 = data['Price (Raw)'].quantile(0.75)
IQR = q3 - q1
data['Price (Raw)'][((data['Price (Raw)'] < (q1 - 1.5 * IQR)) | (data['Price (Raw)'] >(q3 + 1.5 * IQR)))]

# Converting the Price variabel into a timeseries.
# Setting the Timestamp column as the index with a specific frequency.
data.set_index('Timestamp', 
               inplace = True)
# Specify the frequency as every 6 hours ('6H')
data.index.freq = '6H'

#-----------------------------------------------------------------------------------------------------------------------------------
# Model Development
#-----------------------------------------------------------------------------------------------------------------------------------
# Splittig the data 80/20
split_index = int(len(data) * 0.8)
train = data.iloc[:split_index]
test = data.iloc[split_index:]

# Dimensions of the train and test set and the time and date for where the data was split.
train.shape
test.shape
train.index.max()
test.index.min()

# Extracting Price.
y_train = train['Price (Raw)'] 
y_test = test['Price (Raw)'] 

# Defining a parameter grid for the grid search.
param_grid = {
    'seasonal_periods': [4, # daily pattern.
                         28], # Weekly pattern.
    'trend': ['add', # Additive trend
              'mul'],  # Multiplicative trend
    'seasonal': ['add', # Additive seasonal component
                 'mul'],  # Multiplicative seasonal component
    'use_boxcox': [True, # Box-Cox transformation.
                   False] # No transformation.
}
# ParameterGrid(param_grid) generates a combinations of parameters.
# **params syntax is used to unpack the dictionary of parameters and pass them as keyword arguments to the ExponentialSmoothing constructor.

# Performing grid search to determine the best combination of parameters.
model_BIC = float('inf')
best_model = None
model_parameters = None

for params in ParameterGrid(param_grid): 
    model = ExponentialSmoothing(y_train, **params) 
    fitted_model = model.fit()
    BIC = fitted_model.bic 
    
    if BIC < model_BIC:
        best_model = fitted_model
        model_BIC = BIC
        model_parameters = params

# Use the best model for forecasting
print(f"Model parameters: {model_parameters}")
print(f"Best model parameters: {best_model.params}")
print(f"Training BIC: {model_BIC}")
train_RMSE = sqrt(mean_squared_error(y_train, best_model.fittedvalues))
print(f"Train RMSE: {train_RMSE}")

#-----------------------------------------------------------------------------------------------------------------------------------
# Results and Interpretation
#-----------------------------------------------------------------------------------------------------------------------------------
# Forecasting 29 periods ahead because this is the length of the test data.
forecast = best_model.forecast(steps = len(y_test))
forecast
# The RMSE of the forecast.
test_RMSE = sqrt(mean_squared_error(y_test, forecast))
print(f"Test RMSE: {test_RMSE}")

# Plotting the model and forecast over the true values of Price.
# Assigning a titel to the plot.
plt.title('Winter´s Exponential Smoothing Forecast',
          fontsize = 15,
          fontweight = 'bold',
          pad = 15)
# Labeling the X- and Y-axis.
plt.xlabel('Date',
           fontsize = 13,
           fontweight = 'bold',
           labelpad = 15)
plt.ylabel('Price',
           fontsize = 13,
           fontweight = 'bold',
           labelpad = 35)
# The true price.
plt.plot(data.index,
         data['Price (Raw)'], 
         label = 'Data', 
         color = 'black')
# The model fit.
plt.plot(y_train.index, 
         best_model.fittedvalues, 
         label = 'Model Fit', 
         color = '#3d3afb')
# The forecast.
plt.plot(y_test.index, 
         forecast, 
         label='Forecast', 
         color='#bcbafd')
# Adding legend to the plot.
plt.legend(loc = 'lower left',
           fontsize = 11)
# Displayingt the plot.
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------
# Results and Interpretation (Prediction)
#-----------------------------------------------------------------------------------------------------------------------------------
#Create a combined dataset
combined_dataset = pd.concat([y_train, y_test])

# Retrain the model on the combined dataset
predict_model = ExponentialSmoothing(combined_dataset, **model_parameters)

# Define the parameters that can be adjusted during fit
adjustable_params = ['smoothing_level', 'smoothing_trend', 'smoothing_seasonal', 'damping_trend', 'lamda_', 'remove_bias']

# Filter out the parameters that can't be adjusted during fit
fit_params = {k: v for k, v in best_model.params.items() if k in adjustable_params}

# Fit the model on the combined dataset with the parameters from best_model.params
fitted_predict_model = predict_model.fit(**fit_params)

# Predict 40 steps ahead
forecast_future = fitted_predict_model.forecast(steps=40)

# Define predict timestamp array
from datetime import datetime, timedelta
start_datetime = datetime(2024, 4, 27, 5, 0) # Define the start datetime
num_timestamps = 40 # number of timestamps (steps)
frequency = timedelta(hours=6) # Fequency of timestamps (6 hours)

# Create the timestamp array
timestamps = [start_datetime + i * frequency for i in range(num_timestamps)]
timestamp_strings = [timestamp.strftime("%d-%m-%Y %H:%M") for timestamp in timestamps] # Format the timestamps as strings in the desired format

# Assigning a titel to the plot.
plt.title('Winter´s Exponential Smoothing Prediction',
          fontsize = 15,
          fontweight = 'bold',
          pad = 15)
# Labeling the X- and Y-axis.
plt.xlabel('Date',
           fontsize = 13,
           fontweight = 'bold',
           labelpad = 15)
plt.ylabel('Price',
           fontsize = 13,
           fontweight = 'bold',
           labelpad = 35)
# The true price.
plt.plot(data.index,
         data['Price (Raw)'], 
         label = 'Data', 
         color = 'black')
# The model fit.
plt.plot(combined_dataset.index, 
         fitted_predict_model.fittedvalues, 
         label = 'Model Fit', 
         color = '#3d3afb')
# The forecast.
plt.plot(forecast_future.index, 
         forecast_future, 
         label='Forecast', 
         color='#bcbafd')
# Adding legend to the plot.
plt.legend(loc = 'lower left',
           fontsize = 11)
# Displayingt the plot.
plt.show()