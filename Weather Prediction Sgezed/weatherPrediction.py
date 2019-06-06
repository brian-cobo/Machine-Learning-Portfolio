#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:24:59 2019

@author: brian
"""

import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings('ignore')

data = pd.read_csv('weatherHistory.csv')
data.info()
data.head()
data.describe()
data = data.dropna()
data.columns
data.rename(columns = {'Temperature (C)':'temperature', 
                       'Apparent Temperature (C)':'apparentTemp', 
                       'Wind Speed (km/h)': 'windSpeed', 
                       'Visibility (km)':'visibility', 
                       'Pressure (millibars)':'pressure'}, 
            inplace = True)
data.columns

le = LabelEncoder()
precip = data['Precip Type'].values
data['PrecipType2']= le.fit_transform(precip)

summary = data['Summary'].values
data['Summary2'] = le.fit_transform(summary)

y = data['apparentTemp']
x = data[['Humidity', 'windSpeed', 'visibility', 
          'pressure', 'PrecipType2', 'Summary2', 
          'Wind Bearing (degrees)']]
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.8, 
                                                    test_size=0.2, 
                                                    shuffle = True)

lr = LinearRegression()
lr.fit(x_train, y_train)
print('\n\nCoefficients:', lr.coef_)
print('\n\nIntercepts:', lr.intercept_*100)

yPrediction = lr.predict(x_test)
score = round(lr.score(x_test, y_test), 4)
print('\n\nAccuracy of Linear Regression Model:', score*100,'%')

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print('\n\nTraining Neural Networks now:')
iterations = 1000
mlp = MLPRegressor(hidden_layer_sizes=(7, 7, 7, 7, 7),
                   max_iter = iterations, 
                   early_stopping = True)

mlp.fit(x_train, y_train)
pd.DataFrame(mlp.loss_curve_).plot(title = 'Learning Loss')
predictions = mlp.predict(x_test)
score = round(mlp.score(x_test, y_test), 4)
print('\n\nAccuracy using Neural Networks after', 
      iterations,'iterations:', 
      score*100, '%')



