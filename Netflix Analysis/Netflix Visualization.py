import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv('Netflix_5_Year_Stocks.csv', skiprows = range(1,2))

print('\n\nData Shape:', data.shape)
print('\n\nData Head:\n', data.head())
print('\n\nData Info:\n', data.info())
print('\n\nData Description:\n', data.describe())
print('\n\nData Columns:\n', data.columns)

data = data.sort_values(by='date')

data['close'].plot()
plt.show()

print('Finished')